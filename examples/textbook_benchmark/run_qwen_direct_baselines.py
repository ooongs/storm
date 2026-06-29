"""Direct Qwen textbook baselines for the outline-blind chapter benchmark.

This runner implements two simple baselines over
chapter_benchmark_final_outline_blind.jsonl:

1. llm_only: generate the whole chapter from the benchmark outline only.
2. rag_serper: generate search queries, retrieve Serper snippets, then generate
   the chapter from the benchmark outline plus retrieved context.

Outputs are resumable. Existing successful chapters are skipped by default, so
rerunning the command retries only failed, interrupted, or not-yet-run chapters.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_storm.lm import QwenModel
from examples.storm_examples.run_storm_textbook_benchmark import (
    QWEN_DEFAULT_API_BASE,
    attach_output_keys,
    chapter_leakage_phrases,
    chapter_output_id,
    clean_heading,
    count_words,
    format_knowledge_units,
    format_objectives,
    length_budget,
    load_env_file,
    make_chapter_source_filter,
    matches_leakage_phrase,
    max_heading_depth,
    normalize_outline_blind_chapter,
    ordered_sections,
    ordered_subsections,
    output_structure_metrics,
    read_json,
    source_to_match_text,
    write_json,
)


DEFAULT_BENCHMARK_PATH = Path("chapter_benchmark_final_outline_blind.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/qwen_textbook_direct_baselines")
DEFAULT_BASELINES = ["llm_only", "rag_serper"]
BASELINE_ALIASES = {
    "llm": "llm_only",
    "llm_only": "llm_only",
    "rag": "rag_serper",
    "rag_serper": "rag_serper",
    "llm_rag": "rag_serper",
}


def qwen_extra_body(thinking_mode: str, thinking_budget: Optional[int]) -> Optional[dict]:
    if thinking_mode == "default":
        return None
    extra_body = {"enable_thinking": thinking_mode == "on"}
    if thinking_mode == "on" and thinking_budget:
        extra_body["thinking_budget"] = thinking_budget
    return extra_body


def iter_chapters(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            chapter = normalize_outline_blind_chapter(json.loads(line), index)
            chapter["_benchmark_index"] = index
            yield chapter


def normalize_baselines(values: List[str]) -> List[str]:
    baselines = []
    for value in values:
        normalized = BASELINE_ALIASES.get(value)
        if not normalized:
            raise RuntimeError(
                f"Unsupported baseline '{value}'. Choose llm_only and/or rag_serper."
            )
        if normalized not in baselines:
            baselines.append(normalized)
    return baselines


def select_chapters(chapters: List[dict], args) -> List[dict]:
    selected = chapters
    if args.chapter_id:
        wanted = set(args.chapter_id)
        selected = [
            chapter
            for chapter in selected
            if chapter.get("chapter_id") in wanted
            or chapter.get("dataset_id") in wanted
            or chapter_output_id(chapter) in wanted
        ]
    if args.start_index:
        selected = selected[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def baseline_dir(output_dir: Path, baseline: str) -> Path:
    return output_dir / baseline


def chapter_markdown_path(output_dir: Path, baseline: str, output_id: str) -> Path:
    return baseline_dir(output_dir, baseline) / "chapters" / f"{output_id}.md"


def chapter_log_path(output_dir: Path, baseline: str, output_id: str) -> Path:
    return baseline_dir(output_dir, baseline) / "logs" / f"{output_id}.json"


def chapter_raw_dir(output_dir: Path, baseline: str, output_id: str) -> Path:
    return baseline_dir(output_dir, baseline) / "raw-artifacts" / output_id


def markdown_from_response(text: str, chapter_title: str) -> str:
    markdown = (text or "").strip()
    fence = re.match(r"^```(?:markdown|md)?\s*(.*?)\s*```$", markdown, flags=re.DOTALL)
    if fence:
        markdown = fence.group(1).strip()

    markdown = re.sub(r"^\s*#{1,6}\s*Chapter\s+\d+\s*[:.-]\s*", "# ", markdown)
    expected_title = clean_heading(chapter_title)
    first_heading = re.match(r"^\s*#\s+(.+?)\s*$", markdown, flags=re.MULTILINE)
    if not first_heading:
        markdown = f"# {expected_title}\n\n{markdown}"
    elif clean_heading(first_heading.group(1)).lower() != expected_title.lower():
        markdown = f"# {expected_title}\n\n{markdown}"

    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"
    return markdown


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def chapter_requirement_markdown(chapter: dict) -> str:
    length = length_budget(chapter)
    lines = [
        f"Book title: {chapter.get('book_title') or chapter.get('metadata', {}).get('book_title', '')}",
        f"Chapter number: {chapter.get('chapter_number', '')}",
        f"Chapter title: {chapter.get('chapter_title', '')}",
        f"Target words: {length['target_words']}",
        f"Allowed word range: {length['word_range']}",
        "",
        "Benchmark outline requirements:",
    ]
    for section_index, section in enumerate(ordered_sections(chapter), start=1):
        lines.append(f"\nSection {section_index} ({section.get('section_id', '')})")
        objectives = format_objectives(section)
        if objectives:
            lines.append("Learning objectives:")
            lines.extend(f"- {objective}" for objective in objectives)
        for group_index, subsection in enumerate(ordered_subsections(section), start=1):
            units = format_knowledge_units(subsection)
            lines.append(f"Knowledge group {group_index} ({subsection.get('subsection_id', '')}):")
            if units:
                lines.extend(f"- {unit}" for unit in units)
            else:
                lines.append("- No explicit knowledge units.")
    return "\n".join(lines)


def build_article_prompt(chapter: dict, baseline: str, source_context: str = "") -> str:
    section_count = len(ordered_sections(chapter))
    length = length_budget(chapter)
    citation_instruction = (
        "When using retrieved facts, cite the supporting source with inline labels like [S1]. "
        "Do not invent citations and do not add a separate references section."
        if baseline == "rag_serper"
        else "Do not include citations, source lists, or references."
    )
    context_block = (
        "\nRetrieved context from independent web search:\n"
        f"{source_context}\n"
        if source_context
        else ""
    )
    return f"""You are an expert university textbook author.

Write a self-contained textbook chapter in Markdown from the requirements below.

Hard output constraints:
- Output Markdown only. Do not wrap it in code fences.
- Start with exactly one H1 heading: # {clean_heading(chapter.get('chapter_title', 'Untitled Chapter'))}
- Create exactly {section_count} H2 section heading(s), one per input section, in the same order.
- Under each H2, create H3 headings for the knowledge groups as needed.
- Do not use headings deeper than H4.
- Do not add extra H2 headings for introduction, conclusion, summary, exercises, references, or sources.
- Cover every learning objective and knowledge unit.
- Use clear textbook prose, equations, definitions, worked examples, and interpretation where useful.
- Aim for {length['target_words']} words and stay within {length['word_range']} words when possible.
- {citation_instruction}
- Do not mention the benchmark, JSONL input, prompt, hidden outline, or source textbook.
{context_block}
{chapter_requirement_markdown(chapter)}
"""


def build_query_prompt(chapter: dict, max_queries: int) -> str:
    requirements = chapter_requirement_markdown(chapter)
    return f"""Generate web search queries for writing a university textbook chapter.

Return a JSON array of {max_queries} or fewer concise search query strings.

Rules:
- Search for independent educational or reference sources about the concepts.
- Do not search for the exact book title, benchmark id, source page files, PDFs of the source textbook, or mirrored textbook pages.
- Prefer queries that combine the chapter topic with formulas, definitions, examples, or standard terminology.
- Output only valid JSON.

{requirements}
"""


def parse_json_array(text: str) -> Optional[List[str]]:
    candidate = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    match = re.search(r"\[[\s\S]*\]", candidate)
    if match:
        candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(parsed, list):
        return None
    queries = []
    for item in parsed:
        query = clean_heading(str(item))
        if query and query not in queries:
            queries.append(query)
    return queries


def heuristic_queries(chapter: dict, max_queries: int) -> List[str]:
    queries = []
    title = clean_heading(chapter.get("chapter_title", ""))
    if title:
        queries.append(f"{title} textbook definitions examples")
    for section in ordered_sections(chapter):
        for objective in format_objectives(section):
            query = compact_text(objective, 120)
            if query and query not in queries:
                queries.append(query)
            if len(queries) >= max_queries:
                return queries
        for subsection in ordered_subsections(section):
            units = format_knowledge_units(subsection)
            if not units:
                continue
            query = compact_text(" ".join(units[:2]), 140)
            if query and query not in queries:
                queries.append(query)
            if len(queries) >= max_queries:
                return queries
    return queries[:max_queries]


def call_qwen(
    prompt: str,
    args,
    max_tokens: int,
    purpose: str,
    raw_dir: Path,
) -> Tuple[str, dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{purpose}_prompt.md").write_text(prompt, encoding="utf-8")

    lm = QwenModel(
        model=args.model,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        api_base=os.getenv("DASHSCOPE_API_BASE", QWEN_DEFAULT_API_BASE),
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        cache=args.cache_mode != "off",
        provider_cache=args.cache_mode == "explicit",
        cache_prefix_chars=args.explicit_cache_prefix_chars or None,
        extra_body=qwen_extra_body(args.qwen_thinking, args.qwen_thinking_budget),
    )

    last_error = None
    for attempt in range(1, args.max_model_retries + 1):
        started = time.time()
        try:
            output = lm(prompt=prompt)[0]
            elapsed = time.time() - started
            response_meta = {}
            if lm.history:
                response = lm.history[-1].get("response") or {}
                choices = response.get("choices") or []
                response_meta = {
                    "id": response.get("id"),
                    "model": response.get("model"),
                    "finish_reason": (
                        choices[0].get("finish_reason") if choices else None
                    ),
                    "usage": response.get("usage"),
                }
            (raw_dir / f"{purpose}_response.md").write_text(output or "", encoding="utf-8")
            return output or "", {
                "purpose": purpose,
                "attempts": attempt,
                "elapsed_seconds": elapsed,
                "token_usage": lm.get_usage_and_reset(),
                "response": response_meta,
            }
        except Exception as exc:
            last_error = exc
            if attempt >= args.max_model_retries:
                break
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Qwen {purpose} call failed: {last_error}")


def generate_queries(chapter: dict, args, raw_dir: Path) -> Tuple[List[str], dict]:
    fallback = heuristic_queries(chapter, args.max_search_queries)
    if args.query_generation == "heuristic":
        return fallback, {"mode": "heuristic", "fallback_used": False}

    prompt = build_query_prompt(chapter, args.max_search_queries)
    try:
        response, usage = call_qwen(
            prompt=prompt,
            args=args,
            max_tokens=args.query_max_tokens,
            purpose="query_generation",
            raw_dir=raw_dir,
        )
        parsed = parse_json_array(response)
        if parsed:
            return parsed[: args.max_search_queries], {
                "mode": "llm",
                "fallback_used": False,
                "model_call": usage,
            }
        return fallback, {
            "mode": "llm",
            "fallback_used": True,
            "parse_error": "Query generation did not return a JSON array.",
            "model_call": usage,
        }
    except Exception as exc:
        return fallback, {
            "mode": "llm",
            "fallback_used": True,
            "error": str(exc),
        }


class SerperClient:
    def __init__(self, api_key: str, top_k: int, timeout: int):
        self.api_key = api_key
        self.top_k = top_k
        self.timeout = timeout
        self.base_url = "https://google.serper.dev/search"
        self._lock = threading.Lock()
        self.usage = 0

    def search(self, query: str) -> dict:
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "type": "search",
            "autocorrect": True,
            "num": self.top_k,
            "page": 1,
        }
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        with self._lock:
            self.usage += 1
        return response.json()


def search_with_serper(chapter: dict, queries: List[str], args, raw_dir: Path) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is required for rag_serper.")

    client = SerperClient(
        api_key=api_key, top_k=args.search_top_k, timeout=args.search_timeout
    )
    source_filter = make_chapter_source_filter(chapter)
    blocked_phrases = chapter_leakage_phrases(chapter)
    raw_results = []
    accepted_sources = []
    seen_urls = set()
    blocked_queries = []
    filtered_sources = []

    for query in queries[: args.max_search_queries]:
        if matches_leakage_phrase(query, blocked_phrases):
            blocked_queries.append(query)
            continue
        result = client.search(query)
        raw_results.append({"query": query, "result": result})
        knowledge_graph = result.get("knowledgeGraph") or {}
        for organic in result.get("organic") or []:
            url = organic.get("link") or organic.get("url")
            if not url or url in seen_urls:
                continue
            snippets = [snippet for snippet in [organic.get("snippet")] if snippet]
            source = {
                "url": url,
                "title": organic.get("title") or "",
                "description": knowledge_graph.get("description") or "",
                "snippets": snippets,
                "meta": {"query": query},
            }
            if not source_filter(source):
                filtered_sources.append(source)
                continue
            seen_urls.add(url)
            accepted_sources.append(source)
            if len(accepted_sources) >= args.max_sources:
                break
        if len(accepted_sources) >= args.max_sources:
            break

    payload = {
        "queries": queries,
        "blocked_queries": blocked_queries,
        "raw_results": raw_results,
        "accepted_sources": accepted_sources,
        "filtered_sources": filtered_sources,
        "metrics": {
            "requested_query_count": len(queries[: args.max_search_queries]),
            "actual_query_count": client.usage,
            "blocked_query_count": len(blocked_queries),
            "accepted_source_count": len(accepted_sources),
            "filtered_source_count": len(filtered_sources),
        },
    }
    write_json(raw_dir / "serper_search_results.json", payload)
    return payload


def source_context(search_payload: dict) -> str:
    lines = []
    for index, source in enumerate(search_payload.get("accepted_sources") or [], start=1):
        snippets = " ".join(source.get("snippets") or [])
        lines.extend(
            [
                f"[S{index}] {clean_heading(source.get('title', 'Untitled source'))}",
                f"URL: {source.get('url', '')}",
                f"Snippet: {compact_text(snippets or source.get('description', ''), 1000)}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def validate_markdown(chapter: dict, markdown: str) -> Tuple[dict, List[str]]:
    warnings = []
    actual_word_count = count_words(markdown)
    heading_depth = max_heading_depth(markdown)
    structure = output_structure_metrics(chapter, markdown, ignore_summary=False)
    if actual_word_count < 50:
        raise RuntimeError(f"Generated markdown is too short ({actual_word_count} words).")
    if heading_depth > 4:
        raise RuntimeError(f"Generated markdown has headings deeper than H4 ({heading_depth}).")
    if not structure["section_count_matches"]:
        raise RuntimeError(
            "Generated section count does not match dataset structure: "
            f"expected {structure['expected_section_count']} H2 sections, "
            f"got {structure['actual_section_count']} H2 sections."
        )

    length = length_budget(chapter)
    word_range = length.get("word_range") or [0, 0]
    if word_range[0] and actual_word_count < word_range[0]:
        warnings.append(
            f"Generated chapter is below requested word range ({actual_word_count} < {word_range[0]})."
        )
    if len(word_range) > 1 and word_range[1] and actual_word_count > word_range[1]:
        warnings.append(
            f"Generated chapter is above requested word range ({actual_word_count} > {word_range[1]})."
        )

    return {
        "actual_word_count": actual_word_count,
        "max_heading_depth": heading_depth,
        "structure": structure,
    }, warnings


def existing_success(output_dir: Path, baseline: str, chapter: dict) -> Optional[dict]:
    output_id = chapter_output_id(chapter)
    log_path = chapter_log_path(output_dir, baseline, output_id)
    md_path = chapter_markdown_path(output_dir, baseline, output_id)
    if not log_path.exists() or not md_path.exists():
        return None
    try:
        log = read_json(log_path)
        markdown = md_path.read_text(encoding="utf-8")
        metrics, warnings = validate_markdown(chapter, markdown)
    except Exception:
        return None
    if log.get("status") != "success":
        return None
    log.update(metrics)
    if warnings:
        log["warnings"] = sorted(set((log.get("warnings") or []) + warnings))
    log["skipped_existing"] = True
    return log


def task_metadata(chapter: dict, baseline: str, output_dir: Path) -> dict:
    output_id = chapter_output_id(chapter)
    return {
        "baseline": baseline,
        "chapter_id": chapter.get("chapter_id"),
        "dataset_id": chapter.get("dataset_id"),
        "output_id": output_id,
        "benchmark_index": chapter.get("_benchmark_index"),
        "book_slug": chapter.get("book_slug"),
        "book_title": chapter.get("book_title"),
        "chapter_number": chapter.get("chapter_number"),
        "chapter_title": chapter.get("chapter_title"),
        "length_budget": length_budget(chapter),
        "output_paths": {
            "markdown": str(chapter_markdown_path(output_dir, baseline, output_id)),
            "log": str(chapter_log_path(output_dir, baseline, output_id)),
            "raw_artifacts": str(chapter_raw_dir(output_dir, baseline, output_id)),
        },
    }


def dry_run_task(chapter: dict, baseline: str, args, output_dir: Path) -> dict:
    result = task_metadata(chapter, baseline, output_dir)
    result.update(
        {
            "status": "dry_run",
            "model": args.model,
            "parallel_chapters": args.parallel_chapters,
            "expected_h2_sections": len(ordered_sections(chapter)),
        }
    )
    if baseline == "rag_serper":
        result["planned_queries"] = heuristic_queries(chapter, args.max_search_queries)
    return result


def run_task(chapter: dict, baseline: str, args, output_dir: Path) -> dict:
    output_id = chapter_output_id(chapter)
    if not args.overwrite:
        existing = existing_success(output_dir, baseline, chapter)
        if existing is not None:
            return existing

    md_path = chapter_markdown_path(output_dir, baseline, output_id)
    log_path = chapter_log_path(output_dir, baseline, output_id)
    raw_dir = chapter_raw_dir(output_dir, baseline, output_id)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    log = task_metadata(chapter, baseline, output_dir)
    log.update(
        {
            "status": "running",
            "model": args.model,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    write_json(log_path, log)

    started = time.time()
    try:
        search_payload = None
        model_calls = []
        context = ""
        if baseline == "rag_serper":
            queries, query_log = generate_queries(chapter, args, raw_dir)
            queries = [
                query
                for query in queries
                if not matches_leakage_phrase(
                    source_to_match_text(query), chapter_leakage_phrases(chapter)
                )
            ][: args.max_search_queries]
            write_json(raw_dir / "queries.json", {"queries": queries, "query_log": query_log})
            if query_log.get("model_call"):
                model_calls.append(query_log["model_call"])
            search_payload = search_with_serper(chapter, queries, args, raw_dir)
            context = source_context(search_payload)

        prompt = build_article_prompt(chapter, baseline, source_context=context)
        response, usage = call_qwen(
            prompt=prompt,
            args=args,
            max_tokens=args.max_model_output_tokens,
            purpose="article_generation",
            raw_dir=raw_dir,
        )
        model_calls.append(usage)
        markdown = markdown_from_response(response, chapter.get("chapter_title", output_id))
        metrics, warnings = validate_markdown(chapter, markdown)
        md_path.write_text(markdown, encoding="utf-8")

        elapsed = time.time() - started
        log.update(
            {
                "status": "success",
                "elapsed_seconds": elapsed,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_calls": model_calls,
                "warnings": warnings,
                **metrics,
            }
        )
        if search_payload is not None:
            log["search_metrics"] = search_payload.get("metrics", {})
            log["search_result_path"] = str(raw_dir / "serper_search_results.json")
            log["query_path"] = str(raw_dir / "queries.json")
    except Exception as exc:
        elapsed = time.time() - started
        log.update(
            {
                "status": "failed",
                "elapsed_seconds": elapsed,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    write_json(log_path, log)
    return log


def summarize(results: List[dict]) -> dict:
    statuses: Dict[str, int] = {}
    by_baseline: Dict[str, Dict[str, int]] = {}
    for result in results:
        status = result.get("status", "unknown")
        baseline = result.get("baseline", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        by_baseline.setdefault(baseline, {})
        by_baseline[baseline][status] = by_baseline[baseline].get(status, 0) + 1

    successful = [result for result in results if result.get("status") == "success"]
    return {
        "task_count": len(results),
        "statuses": statuses,
        "statuses_by_baseline": by_baseline,
        "skipped_existing_count": sum(1 for result in results if result.get("skipped_existing")),
        "total_elapsed_seconds": sum(result.get("elapsed_seconds", 0) for result in results),
        "total_actual_words": sum(result.get("actual_word_count", 0) for result in successful),
        "total_search_queries": sum(
            result.get("search_metrics", {}).get("actual_query_count", 0)
            for result in successful
        ),
        "total_accepted_sources": sum(
            result.get("search_metrics", {}).get("accepted_source_count", 0)
            for result in successful
        ),
    }


def save_progress(output_dir: Path, ordered_results: List[Optional[dict]]) -> List[dict]:
    available = [result for result in ordered_results if result is not None]
    write_json(output_dir / "all_results.json", available)
    write_json(output_dir / "run_summary.json", summarize(available))

    baselines = sorted({result.get("baseline", "unknown") for result in available})
    for baseline in baselines:
        baseline_results = [
            result for result in available if result.get("baseline") == baseline
        ]
        base_dir = baseline_dir(output_dir, baseline)
        write_json(base_dir / "all_results.json", baseline_results)
        write_json(base_dir / "run_summary.json", summarize(baseline_results))
    return available


def progress_write(args, message: str) -> None:
    if not args.quiet:
        print(message, flush=True)


def run_tasks(chapters: List[dict], baselines: List[str], args) -> List[dict]:
    tasks = [(chapter, baseline) for chapter in chapters for baseline in baselines]
    ordered_results: List[Optional[dict]] = [None] * len(tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def execute_one(chapter: dict, baseline: str) -> dict:
        if args.dry_run:
            return dry_run_task(chapter, baseline, args, args.output_dir)
        return run_task(chapter, baseline, args, args.output_dir)

    if args.parallel_chapters <= 1:
        for index, (chapter, baseline) in enumerate(tasks, start=1):
            output_id = chapter_output_id(chapter)
            progress_write(args, f"[{index}/{len(tasks)}] start {baseline}/{output_id}")
            result = execute_one(chapter, baseline)
            ordered_results[index - 1] = result
            progress_write(
                args,
                f"[{index}/{len(tasks)}] done {baseline}/{output_id}: {result.get('status')}",
            )
            save_progress(args.output_dir, ordered_results)
            if result.get("status") == "failed" and args.stop_on_error:
                break
        return [result for result in ordered_results if result is not None]

    progress_write(
        args,
        f"Running {len(tasks)} baseline/chapter tasks with {args.parallel_chapters} workers.",
    )
    executor = ThreadPoolExecutor(max_workers=args.parallel_chapters)
    future_to_index = {
        executor.submit(execute_one, chapter, baseline): index
        for index, (chapter, baseline) in enumerate(tasks)
    }
    completed = 0
    try:
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            chapter, baseline = tasks[index]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                result = task_metadata(chapter, baseline, args.output_dir)
                result.update(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            ordered_results[index] = result
            progress_write(
                args,
                f"[{completed}/{len(tasks)}] done {baseline}/{chapter_output_id(chapter)}: {result.get('status')}",
            )
            save_progress(args.output_dir, ordered_results)
            if result.get("status") == "failed" and args.stop_on_error:
                for pending in future_to_index:
                    pending.cancel()
                break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return [result for result in ordered_results if result is not None]


def validate_keys_for_run(baselines: List[str]) -> None:
    missing = []
    if not os.getenv("DASHSCOPE_API_KEY"):
        missing.append("DASHSCOPE_API_KEY")
    if "rag_serper" in baselines and not os.getenv("SERPER_API_KEY"):
        missing.append("SERPER_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--chapter-id", action="append")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model", default=os.getenv("QWEN_PLUS_MODEL", "qwen3.7-plus"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    parser.add_argument("--max-model-output-tokens", type=int, default=24000)
    parser.add_argument("--query-max-tokens", type=int, default=1200)
    parser.add_argument(
        "--parallel-chapters",
        type=int,
        default=8,
        help="Number of baseline/chapter tasks to run concurrently.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["implicit", "explicit", "off"],
        default="implicit",
        help="implicit keeps LiteLLM local disk cache; explicit also adds DashScope cache_control markers.",
    )
    parser.add_argument(
        "--explicit-cache-prefix-chars",
        type=int,
        default=0,
        help="With --cache-mode explicit, mark only the first N prompt chars as cacheable. 0 marks the full prompt.",
    )
    parser.add_argument(
        "--qwen-thinking",
        choices=["default", "on", "off", "disable"],
        default="disable",
        help="Qwen thinking mode for both query generation and article generation.",
    )
    parser.add_argument("--qwen-thinking-budget", type=int, default=4096)
    parser.add_argument(
        "--query-generation",
        choices=["llm", "heuristic"],
        default="llm",
        help="How rag_serper search queries are generated.",
    )
    parser.add_argument("--max-search-queries", type=int, default=8)
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--max-sources", type=int, default=24)
    parser.add_argument("--search-timeout", type=int, default=30)
    parser.add_argument("--max-model-retries", type=int, default=3)
    args = parser.parse_args(argv)

    if args.parallel_chapters < 1:
        raise RuntimeError("--parallel-chapters must be at least 1.")

    load_env_file(args.env_file)
    baselines = normalize_baselines(args.baselines)
    if not args.dry_run:
        validate_keys_for_run(baselines)

    chapters = attach_output_keys(list(iter_chapters(args.benchmark_path)))
    selected = select_chapters(chapters, args)
    run_tasks(selected, baselines, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
