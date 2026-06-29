"""OmniThink textbook baseline for the outline-blind chapter benchmark.

This runner adapts the cloned OmniThink repository under external/OmniThink to
the same benchmark/output contract used by the STORM textbook runner. It keeps
the textbook benchmark structure fixed, while using OmniThink-style outline and
section generation over a lightweight textbook mind-map facade.
"""

import argparse
import copy
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import dspy
import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_storm.lm import LitellmModel, QwenModel
from knowledge_storm.storm_wiki.modules.article_generation import (
    WriteSection as StormWriteSection,
)
from knowledge_storm.storm_wiki.modules.outline_generation import (
    WriteOutline as StormWriteOutline,
)
from examples.storm_examples.run_storm_textbook_benchmark import (
    QWEN_DEFAULT_API_BASE,
    attach_output_keys,
    build_section_contexts,
    chapter_leakage_phrases,
    chapter_needs_generated_headings,
    chapter_output_id,
    clean_heading,
    count_words,
    format_knowledge_units,
    format_objectives,
    format_textbook_markdown,
    length_budget,
    load_env_file,
    make_chapter_source_filter,
    matches_leakage_phrase,
    max_heading_depth,
    normalize_outline_blind_chapter,
    ordered_sections,
    ordered_subsections,
    output_structure_metrics,
    qwen_extra_body,
    read_json,
    source_to_match_text,
    strip_numeric_citations,
    write_json,
)


DEFAULT_BENCHMARK_PATH = Path("chapter_benchmark_final_outline_blind.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/omnithink_textbook_benchmark")
DEFAULT_OMNITHINK_DIR = REPO_ROOT / "external" / "OmniThink"
DEFAULT_BASELINES = ["omnithink"]


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
        normalized = value.lower().replace("-", "_")
        if normalized not in {"omnithink", "omni_think"}:
            raise RuntimeError(
                f"Unsupported baseline '{value}'. This runner only supports omnithink."
            )
        if "omnithink" not in baselines:
            baselines.append("omnithink")
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


def compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def simple_tokens(text: str) -> set:
    stopwords = {
        "about",
        "above",
        "after",
        "also",
        "and",
        "are",
        "because",
        "between",
        "chapter",
        "define",
        "describe",
        "explain",
        "for",
        "from",
        "how",
        "into",
        "section",
        "that",
        "the",
        "their",
        "this",
        "through",
        "under",
        "using",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (text or "").lower())
        if token not in stopwords
    }


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
        section_heading = clean_heading(section.get("heading")) or f"Section {section_index}"
        lines.append(f"\n{section_heading}")
        objectives = format_objectives(section)
        if objectives:
            lines.append("Learning objectives:")
            lines.extend(f"- {objective}" for objective in objectives)
        for group_index, subsection in enumerate(ordered_subsections(section), start=1):
            subsection_heading = clean_heading(subsection.get("heading")) or (
                f"Knowledge group {group_index}"
            )
            units = format_knowledge_units(subsection)
            lines.append(f"{subsection_heading}:")
            lines.extend(f"- {unit}" for unit in units or ["No explicit knowledge units."])
    return "\n".join(lines)


def build_mindmap_queries(chapter: dict, max_queries: int) -> List[str]:
    queries = []
    title = clean_heading(chapter.get("chapter_title", ""))
    if title:
        queries.append(f"{title} textbook definitions examples")
    for section in ordered_sections(chapter):
        for objective in format_objectives(section):
            query = compact_text(f"{title} {objective}", 140)
            if query and query not in queries:
                queries.append(query)
            if len(queries) >= max_queries:
                return queries
        for subsection in ordered_subsections(section):
            units = format_knowledge_units(subsection)
            if not units:
                continue
            query = compact_text(f"{title} {' '.join(units[:2])}", 160)
            if query and query not in queries:
                queries.append(query)
            if len(queries) >= max_queries:
                return queries
    return queries[:max_queries]


class SerperTextbookRetriever:
    def __init__(
        self,
        api_key: str,
        top_k: int,
        timeout: int,
        max_sources: int,
        source_filter,
        blocked_phrases: List[str],
    ):
        self.api_key = api_key
        self.top_k = top_k
        self.timeout = timeout
        self.max_sources = max_sources
        self.source_filter = source_filter
        self.blocked_phrases = blocked_phrases
        self.base_url = "https://google.serper.dev/search"
        self.usage = {
            "actual_query_count": 0,
            "accepted_source_count": 0,
            "raw_search_result_count": 0,
            "leakage_blocked_count": 0,
            "leakage_query_blocked_count": 0,
            "duplicate_blocked_count": 0,
        }

    def __call__(self, query_or_queries, exclude_urls: Optional[List[str]] = None):
        return self.forward(query_or_queries, exclude_urls=exclude_urls or [])

    def forward(self, query_or_queries, exclude_urls: Optional[List[str]] = None):
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else list(query_or_queries or [])
        )
        exclude_urls = set(exclude_urls or [])
        accepted_sources = []
        seen_urls = set()
        raw_results = []
        blocked_queries = []

        for query in queries:
            query_text = source_to_match_text(query)
            if matches_leakage_phrase(query_text, self.blocked_phrases):
                blocked_queries.append(query)
                self.usage["leakage_query_blocked_count"] += 1
                continue
            response = requests.post(
                self.base_url,
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "q": query,
                    "type": "search",
                    "autocorrect": True,
                    "num": self.top_k,
                    "page": 1,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            raw_results.append({"query": query, "result": payload})
            self.usage["actual_query_count"] += 1
            knowledge_graph = payload.get("knowledgeGraph") or {}
            for organic in payload.get("organic") or []:
                self.usage["raw_search_result_count"] += 1
                url = organic.get("link") or organic.get("url")
                if not url:
                    continue
                if url in seen_urls or url in exclude_urls:
                    self.usage["duplicate_blocked_count"] += 1
                    continue
                source = {
                    "url": url,
                    "title": organic.get("title") or "",
                    "description": knowledge_graph.get("description") or "",
                    "snippets": [
                        snippet
                        for snippet in [organic.get("snippet"), organic.get("attributes")]
                        if isinstance(snippet, str) and snippet.strip()
                    ],
                    "meta": {"query": query},
                }
                if not self.source_filter(source):
                    self.usage["leakage_blocked_count"] += 1
                    continue
                seen_urls.add(url)
                accepted_sources.append(source)
                if len(accepted_sources) >= self.max_sources:
                    break
            if len(accepted_sources) >= self.max_sources:
                break

        self.usage["accepted_source_count"] += len(accepted_sources)
        return {
            "queries": queries,
            "blocked_queries": blocked_queries,
            "raw_results": raw_results,
            "accepted_sources": accepted_sources,
            "metrics": dict(self.usage),
        }


@dataclass
class MindMapNode:
    category: str
    concept: List[str] = field(default_factory=list)
    info: List[dict] = field(default_factory=list)
    children: Dict[str, "MindMapNode"] = field(default_factory=dict)


class TextbookMindMap:
    def __init__(self, chapter: dict, retrieve_top_k: int):
        self.chapter = chapter
        self.retrieve_top_k = retrieve_top_k
        self.root: Optional[MindMapNode] = None
        self.all_infos: List[dict] = []
        self.snippet_records: List[dict] = []

    def build_from_sources(self, sources: List[dict]) -> None:
        concepts = []
        for section in ordered_sections(self.chapter):
            concepts.extend(format_objectives(section))
            for subsection in ordered_subsections(section):
                concepts.extend(format_knowledge_units(subsection)[:2])
        concepts = [compact_text(concept, 180) for concept in concepts if concept]
        self.all_infos = sources
        self.root = MindMapNode(
            category=clean_heading(self.chapter.get("chapter_title", "Untitled Chapter")),
            concept=concepts[:80],
            info=sources,
        )

    def export_categories_and_concepts(self) -> str:
        lines = [clean_heading(self.chapter.get("chapter_title", "Untitled Chapter"))]
        if self.root:
            lines.append("Required textbook concepts:")
            lines.extend(f"- {concept}" for concept in self.root.concept)
        if self.all_infos:
            lines.append("Retrieved independent source signals:")
            for source in self.all_infos[:12]:
                snippets = " ".join(source.get("snippets") or [])
                label = clean_heading(source.get("title", "Untitled source"))
                lines.append(f"- {label}: {compact_text(snippets, 220)}")
        return "\n".join(lines)

    def prepare_table_for_retrieval(self) -> None:
        records = []
        for source in self.all_infos:
            snippets = source.get("snippets") or []
            if source.get("description"):
                snippets = snippets + [source["description"]]
            for snippet in snippets:
                if not snippet:
                    continue
                records.append(
                    {
                        "url": source.get("url", ""),
                        "title": source.get("title", ""),
                        "snippet": snippet,
                        "tokens": simple_tokens(
                            " ".join(
                                [
                                    source.get("title", ""),
                                    source.get("description", ""),
                                    snippet,
                                ]
                            )
                        ),
                    }
                )
        self.snippet_records = records

    def retrieve_information(self, queries, search_top_k: Optional[int] = None) -> List[dict]:
        if isinstance(queries, str):
            queries = [queries]
        queries = [query for query in queries or [] if clean_heading(str(query))]
        if not self.snippet_records:
            self.prepare_table_for_retrieval()
        if not self.snippet_records:
            return []

        query_tokens = simple_tokens(" ".join(str(query) for query in queries))
        scored = []
        for record in self.snippet_records:
            overlap = len(record["tokens"] & query_tokens)
            if overlap <= 0:
                overlap = 1 if query_tokens else 0
            scored.append((overlap, record))
        scored.sort(key=lambda item: item[0], reverse=True)

        limit = max(1, search_top_k or self.retrieve_top_k)
        grouped: Dict[str, dict] = {}
        for _, record in scored[: max(limit * 3, limit)]:
            url = record["url"]
            if url not in grouped:
                grouped[url] = {
                    "url": url,
                    "title": record.get("title", ""),
                    "description": "",
                    "snippets": [],
                }
            if record["snippet"] not in grouped[url]["snippets"]:
                grouped[url]["snippets"].append(record["snippet"])
            if len(grouped) >= limit:
                break
        return list(grouped.values())

    def to_dict(self) -> dict:
        return {
            "chapter_title": self.chapter.get("chapter_title"),
            "concept_count": len(self.root.concept if self.root else []),
            "source_count": len(self.all_infos),
            "snippet_count": len(self.snippet_records),
            "sources": self.all_infos,
        }


@dataclass
class OutlineDialogueTurn:
    user_utterance: str
    agent_utterance: str


class TextbookConvToSection(dspy.Module):
    def __init__(self, engine, article_text_processing):
        super().__init__()
        self.engine = engine
        self.article_text_processing = article_text_processing
        self.write_section = dspy.Predict(StormWriteSection)

    def forward(
        self,
        topic: str,
        outline: str,
        section: str,
        collected_info: List[dict],
        section_context: str = "",
    ):
        all_info = ""
        for index, info in enumerate(collected_info or [], start=1):
            snippets = info.get("snippets") or []
            all_info += f"[{index}]\n" + "\n".join(snippets)
            all_info += "\n\n"
        all_info = self.article_text_processing.limit_word_count_preserve_newline(
            all_info, 1500
        )

        with dspy.settings.context(lm=self.engine):
            section_text = self.write_section(
                topic=topic,
                info=all_info,
                outline=outline,
                section=section,
                section_context=section_context,
            ).output
        section_text = self.article_text_processing.clean_up_section(section_text)
        section_text = section_text.replace("\\[", "[").replace("\\]", "]")
        return dspy.Prediction(section=section_text)


class TextbookArticleGenerationModule:
    def __init__(
        self,
        article_gen_lm,
        article_text_processing,
        section_contexts: Dict[str, str],
        retrieve_top_k: int,
        max_thread_num: int,
    ):
        self.section_gen = TextbookConvToSection(
            engine=article_gen_lm,
            article_text_processing=article_text_processing,
        )
        self.section_contexts = section_contexts
        self.retrieve_top_k = retrieve_top_k
        self.max_thread_num = max_thread_num

    def generate_section(self, topic, section_name, mindmap, section_outline, section_query):
        collected_info = mindmap.retrieve_information(
            queries=section_query,
            search_top_k=self.retrieve_top_k,
        )
        output = self.section_gen(
            topic=topic,
            outline=section_outline,
            section=section_name,
            collected_info=collected_info,
            section_context=self.section_contexts.get(section_name, ""),
        )
        return {
            "section_name": section_name,
            "section_content": output.section,
            "collected_info": collected_info,
        }

    def generate_article(self, topic: str, mindmap: TextbookMindMap, article_with_outline):
        mindmap.prepare_table_for_retrieval()
        sections_to_write = article_with_outline.get_first_level_section_names()
        output_by_section: Dict[str, dict] = {}

        def submit_section(section_title: str):
            section_query = article_with_outline.get_outline_as_list(
                root_section_name=section_title,
                add_hashtags=False,
            )
            section_outline = "\n".join(
                article_with_outline.get_outline_as_list(
                    root_section_name=section_title,
                    add_hashtags=True,
                )
            )
            return self.generate_section(
                topic,
                section_title,
                mindmap,
                section_outline,
                section_query,
            )

        if self.max_thread_num <= 1:
            for section_title in sections_to_write:
                output = submit_section(section_title)
                output_by_section[section_title] = output
        else:
            with ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
                future_to_section = {
                    executor.submit(submit_section, section_title): section_title
                    for section_title in sections_to_write
                }
                for future in as_completed(future_to_section):
                    section_title = future_to_section[future]
                    output_by_section[section_title] = future.result()

        article = copy.deepcopy(article_with_outline)
        parent_section_name = article.root.section_name
        for section_title in sections_to_write:
            output = output_by_section[section_title]
            article.update_section(
                parent_section_name=parent_section_name,
                current_section_content=output["section_content"],
                current_section_info_list=output["collected_info"],
            )
        article.post_processing()
        return article


def load_omnithink(omnithink_dir: Path) -> dict:
    omnithink_dir = omnithink_dir.resolve()
    if not omnithink_dir.exists():
        raise RuntimeError(
            "OmniThink checkout not found at "
            f"{omnithink_dir}. Clone it with: git clone https://github.com/zjunlp/OmniThink.git external/OmniThink"
        )
    if not (omnithink_dir / "src").exists():
        raise RuntimeError(f"OmniThink checkout is missing src/: {omnithink_dir}")
    if str(omnithink_dir) not in sys.path:
        sys.path.insert(0, str(omnithink_dir))

    from src.actions.article_polish import ArticlePolishingModule
    from src.dataclass.Article import Article
    from src.utils.ArticleTextProcessing import ArticleTextProcessing

    return {
        "Article": Article,
        "ArticlePolishingModule": ArticlePolishingModule,
        "ArticleTextProcessing": ArticleTextProcessing,
    }


def provider_from_args(args) -> str:
    if args.provider != "auto":
        return args.provider
    normalized = (args.model or "").lower()
    return "qwen" if normalized.startswith("qwen") or normalized.startswith("dashscope/qwen") else "openai"


def create_lm(args, max_tokens: int):
    provider = provider_from_args(args)
    if provider == "qwen":
        extra_body = qwen_extra_body(args.qwen_thinking, args.qwen_thinking_budget)
        return QwenModel(
            model=args.model,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            api_base=os.getenv("DASHSCOPE_API_BASE", QWEN_DEFAULT_API_BASE),
            max_tokens=max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            cache=args.cache_mode != "off",
            provider_cache=args.cache_mode == "explicit",
            cache_prefix_chars=args.explicit_cache_prefix_chars or None,
            extra_body=extra_body,
        )

    kwargs = {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": args.temperature,
        "max_tokens": max_tokens,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["api_base"] = base_url
    return LitellmModel(model=args.model, **kwargs)


def model_usage(lms: Dict[str, object]) -> dict:
    usage = {}
    for name, lm in lms.items():
        if hasattr(lm, "get_usage_and_reset"):
            usage[name] = lm.get_usage_and_reset()
        history = getattr(lm, "history", None)
        if history is not None:
            usage.setdefault(name, {})
            usage[name]["call_count"] = len(history)
    return usage


def synthetic_section_heading(section: dict, index: int) -> str:
    heading = clean_heading(section.get("heading"))
    if heading:
        return heading
    candidate = ""
    objectives = format_objectives(section)
    if objectives:
        candidate = objectives[0]
    else:
        for subsection in ordered_subsections(section):
            units = format_knowledge_units(subsection)
            if units:
                candidate = units[0]
                break
    suffix = compact_text(candidate, 70) if candidate else ""
    return f"Section {index}: {suffix}" if suffix else f"Section {index}"


def synthetic_subsection_heading(subsection: dict, index: int) -> str:
    heading = clean_heading(subsection.get("heading"))
    if heading:
        return heading
    units = format_knowledge_units(subsection)
    suffix = compact_text(units[0], 70) if units else ""
    return f"Knowledge Group {index}: {suffix}" if suffix else f"Knowledge Group {index}"


def build_synthetic_outline(
    chapter: dict, section_headings: Optional[List[str]] = None
) -> str:
    lines = []
    for section_index, section in enumerate(ordered_sections(chapter), start=1):
        if section_headings and section_index <= len(section_headings):
            section_heading = clean_heading(section_headings[section_index - 1])
        else:
            section_heading = synthetic_section_heading(section, section_index)
        lines.append(f"# {section_heading}")
        for subsection_index, subsection in enumerate(
            ordered_subsections(section), start=1
        ):
            subsection_heading = synthetic_subsection_heading(subsection, subsection_index)
            lines.append(f"## {subsection_heading}")
    return "\n".join(lines)


def first_level_headings(outline: str, chapter_title: str) -> List[str]:
    headings = []
    for line in (outline or "").splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line.strip())
        if not match:
            continue
        level = len(match.group(1))
        heading = clean_heading(match.group(2))
        if level == 1 and heading.lower() == clean_heading(chapter_title).lower():
            continue
        if level == 1:
            headings.append(heading)
    return headings


def build_storm_outline_dialogue(
    chapter: dict, mindmap: TextbookMindMap
) -> List[OutlineDialogueTurn]:
    return [
        OutlineDialogueTurn(
            user_utterance=(
                "Please gather information for this textbook chapter.\n\n"
                f"{chapter_requirement_markdown(chapter)}"
            ),
            agent_utterance=mindmap.export_categories_and_concepts(),
        )
    ]


def resolve_outline(
    chapter: dict,
    mindmap: TextbookMindMap,
    outline_lm,
    args,
    raw_dir: Path,
) -> Tuple[str, dict]:
    mode = args.outline_mode
    if mode == "auto":
        mode = "omnithink" if chapter_needs_generated_headings(chapter) else "fixed"

    if mode in {"fixed", "synthetic"}:
        outline = build_synthetic_outline(chapter)
        return outline, {"mode": mode, "source": "benchmark_or_synthetic"}

    outline_module = StormWriteOutline(outline_lm)
    outline_result = outline_module(
        topic=clean_heading(chapter.get("chapter_title", "Untitled Chapter")),
        dlg_history=build_storm_outline_dialogue(chapter, mindmap),
    )
    raw_outline = outline_result.outline
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "omnithink_raw_outline.md").write_text(raw_outline or "", encoding="utf-8")
    (raw_dir / "storm_aligned_draft_outline.md").write_text(
        outline_result.old_outline or "",
        encoding="utf-8",
    )
    generated_headings = first_level_headings(
        raw_outline, chapter.get("chapter_title", "")
    )
    expected_count = len(ordered_sections(chapter))
    warnings = []
    if len(generated_headings) != expected_count:
        warnings.append(
            "OmniThink generated outline section count did not match benchmark "
            f"({len(generated_headings)} != {expected_count}); using synthetic benchmark-shaped outline."
        )
        outline = build_synthetic_outline(chapter)
        source = "storm_aligned_synthetic_fallback"
    else:
        outline = build_synthetic_outline(chapter, section_headings=generated_headings)
        source = "storm_aligned_top_level_headings"
    (raw_dir / "textbook_outline.md").write_text(outline, encoding="utf-8")
    return outline, {
        "mode": "omnithink",
        "source": source,
        "prompt_source": "knowledge_storm.storm_wiki.modules.outline_generation.WriteOutline",
        "raw_outline_path": str(raw_dir / "omnithink_raw_outline.md"),
        "draft_outline_path": str(raw_dir / "storm_aligned_draft_outline.md"),
        "outline_path": str(raw_dir / "textbook_outline.md"),
        "generated_first_level_headings": generated_headings,
        "warnings": warnings,
    }


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
            f"got {structure['actual_section_count']} H2 sections "
            f"({structure['section_headings']})."
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
    planned_mode = args.outline_mode
    if planned_mode == "auto":
        planned_mode = "omnithink" if chapter_needs_generated_headings(chapter) else "fixed"
    outline = build_synthetic_outline(chapter)
    result = task_metadata(chapter, baseline, output_dir)
    result.update(
        {
            "status": "dry_run",
            "model": args.model,
            "provider": provider_from_args(args),
            "omnithink_dir": str(args.omnithink_dir),
            "parallel_chapters": args.parallel_chapters,
            "outline_mode": planned_mode,
            "expected_h2_sections": len(ordered_sections(chapter)),
            "planned_queries": build_mindmap_queries(chapter, args.max_search_queries),
            "synthetic_outline_preview": outline.splitlines()[:20],
        }
    )
    return result


def run_task(chapter: dict, baseline: str, args, output_dir: Path, omni: dict) -> dict:
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
            "provider": provider_from_args(args),
            "omnithink_dir": str(args.omnithink_dir),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    write_json(log_path, log)

    started = time.time()
    try:
        outline_lm = create_lm(args, args.outline_max_tokens)
        section_lm = create_lm(args, args.section_max_tokens)
        polish_lm = create_lm(args, args.polish_max_tokens)
        lms = {"outline": outline_lm, "section": section_lm, "polish": polish_lm}

        query_candidates = build_mindmap_queries(chapter, args.max_search_queries)
        blocked_phrases = chapter_leakage_phrases(chapter)
        query_candidates = [
            query
            for query in query_candidates
            if not matches_leakage_phrase(source_to_match_text(query), blocked_phrases)
        ][: args.max_search_queries]
        write_json(raw_dir / "queries.json", {"queries": query_candidates})

        retriever = SerperTextbookRetriever(
            api_key=os.getenv("SERPER_API_KEY"),
            top_k=args.search_top_k,
            timeout=args.search_timeout,
            max_sources=args.max_sources,
            source_filter=make_chapter_source_filter(chapter),
            blocked_phrases=blocked_phrases,
        )
        search_payload = retriever.forward(query_candidates)
        write_json(raw_dir / "serper_search_results.json", search_payload)

        mindmap = TextbookMindMap(chapter=chapter, retrieve_top_k=args.retrieve_top_k)
        mindmap.build_from_sources(search_payload.get("accepted_sources") or [])
        mindmap.prepare_table_for_retrieval()
        write_json(raw_dir / "mindmap.json", mindmap.to_dict())
        (raw_dir / "requirements.md").write_text(
            chapter_requirement_markdown(chapter),
            encoding="utf-8",
        )

        outline, outline_log = resolve_outline(
            chapter=chapter,
            mindmap=mindmap,
            outline_lm=outline_lm,
            args=args,
            raw_dir=raw_dir,
        )
        article_topic = clean_heading(chapter.get("chapter_title", output_id))
        article_with_outline = omni["Article"].from_outline_str(
            topic=article_topic,
            outline_str=outline,
        )
        section_names = article_with_outline.get_first_level_section_names()
        section_contexts = build_section_contexts(chapter, outline, section_names)
        write_json(raw_dir / "section_contexts.json", section_contexts)

        writer_topic = "\n".join(
            [
                f"Chapter title: {chapter.get('chapter_title', '')}",
                f"Book title: {chapter.get('book_title') or chapter.get('metadata', {}).get('book_title', '')}",
                f"Length budget: {length_budget(chapter)}",
                "Task: write a university textbook chapter section using the supplied benchmark coverage requirements.",
            ]
        )
        article_generator = TextbookArticleGenerationModule(
            article_gen_lm=section_lm,
            article_text_processing=omni["ArticleTextProcessing"],
            section_contexts=section_contexts,
            retrieve_top_k=args.retrieve_top_k,
            max_thread_num=args.max_thread_num,
        )
        draft_article = article_generator.generate_article(
            topic=writer_topic,
            mindmap=mindmap,
            article_with_outline=article_with_outline,
        )
        final_article = draft_article
        if args.do_polish_article:
            polisher = omni["ArticlePolishingModule"](
                article_gen_lm=section_lm,
                article_polish_lm=polish_lm,
            )
            final_article = polisher.polish_article(
                topic=article_topic,
                draft_article=draft_article,
            )

        raw_article = final_article.to_string()
        (raw_dir / "article_raw.md").write_text(raw_article, encoding="utf-8")
        markdown_with_citations = format_textbook_markdown(
            chapter_title=chapter.get("chapter_title", output_id),
            raw_article=raw_article,
        )
        (raw_dir / "chapter_with_citations.md").write_text(
            markdown_with_citations,
            encoding="utf-8",
        )
        markdown = strip_numeric_citations(markdown_with_citations)
        metrics, warnings = validate_markdown(chapter, markdown)
        warnings.extend(outline_log.get("warnings") or [])
        if not search_payload.get("accepted_sources"):
            warnings.append(
                "No accepted search sources were available; OmniThink generation used benchmark context only."
            )
        md_path.write_text(markdown, encoding="utf-8")

        elapsed = time.time() - started
        log.update(
            {
                "status": "success",
                "elapsed_seconds": elapsed,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "outline": outline_log,
                "search_metrics": search_payload.get("metrics", {}),
                "mindmap": {
                    "concept_count": len(mindmap.root.concept if mindmap.root else []),
                    "source_count": len(mindmap.all_infos),
                    "snippet_count": len(mindmap.snippet_records),
                },
                "model_usage": model_usage(lms),
                "polishing": {"enabled": args.do_polish_article},
                "warnings": sorted(set(warnings)),
                **metrics,
            }
        )
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


def run_tasks(chapters: List[dict], baselines: List[str], args, omni: Optional[dict]) -> List[dict]:
    tasks = [(chapter, baseline) for chapter in chapters for baseline in baselines]
    ordered_results: List[Optional[dict]] = [None] * len(tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def execute_one(chapter: dict, baseline: str) -> dict:
        if args.dry_run:
            return dry_run_task(chapter, baseline, args, args.output_dir)
        return run_task(chapter, baseline, args, args.output_dir, omni)

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
        f"Running {len(tasks)} OmniThink baseline/chapter tasks with {args.parallel_chapters} workers.",
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


def validate_keys_for_run(args) -> None:
    provider = provider_from_args(args)
    required_model_key = "DASHSCOPE_API_KEY" if provider == "qwen" else "OPENAI_API_KEY"
    missing = [
        name
        for name in [required_model_key, "SERPER_API_KEY"]
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--omnithink-dir", type=Path, default=DEFAULT_OMNITHINK_DIR)
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--chapter-id", action="append")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "qwen"],
        default="auto",
    )
    parser.add_argument("--model", default=os.getenv("QWEN_PLUS_MODEL", "qwen3.7-plus"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    parser.add_argument("--outline-max-tokens", type=int, default=2400)
    parser.add_argument("--section-max-tokens", type=int, default=9000)
    parser.add_argument("--polish-max-tokens", type=int, default=24000)
    parser.add_argument("--max-model-output-tokens", type=int, default=24000)
    parser.add_argument(
        "--parallel-chapters",
        type=int,
        default=1,
        help="Number of baseline/chapter tasks to run concurrently.",
    )
    parser.add_argument("--max-thread-num", type=int, default=3)
    parser.add_argument(
        "--outline-mode",
        choices=["auto", "omnithink", "fixed", "synthetic"],
        default="auto",
        help="auto uses OmniThink outline generation when benchmark headings are missing.",
    )
    parser.add_argument("--do-polish-article", action="store_true")
    parser.add_argument("--max-search-queries", type=int, default=10)
    parser.add_argument("--search-top-k", type=int, default=5)
    parser.add_argument("--retrieve-top-k", type=int, default=5)
    parser.add_argument("--max-sources", type=int, default=30)
    parser.add_argument("--search-timeout", type=int, default=30)
    parser.add_argument(
        "--cache-mode",
        choices=["implicit", "explicit", "off"],
        default="implicit",
    )
    parser.add_argument("--explicit-cache-prefix-chars", type=int, default=0)
    parser.add_argument(
        "--qwen-thinking",
        choices=["default", "on", "off", "disable"],
        default="disable",
    )
    parser.add_argument("--qwen-thinking-budget", type=int, default=4096)
    args = parser.parse_args(argv)

    if args.parallel_chapters < 1:
        raise RuntimeError("--parallel-chapters must be at least 1.")
    if args.max_thread_num < 1:
        raise RuntimeError("--max-thread-num must be at least 1.")

    if args.max_model_output_tokens != parser.get_default("max_model_output_tokens"):
        args.polish_max_tokens = args.max_model_output_tokens

    load_env_file(args.env_file)
    baselines = normalize_baselines(args.baselines)
    omni = None if args.dry_run else load_omnithink(args.omnithink_dir)
    if not args.dry_run:
        validate_keys_for_run(args)

    chapters = attach_output_keys(list(iter_chapters(args.benchmark_path)))
    selected = select_chapters(chapters, args)
    run_tasks(selected, baselines, args, omni)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
