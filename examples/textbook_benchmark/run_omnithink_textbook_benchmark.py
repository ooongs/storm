"""OmniThink textbook baseline for the outline-blind chapter benchmark.

This runner adapts the cloned OmniThink repository under external/OmniThink to
the same benchmark/output contract used by the STORM textbook runner. It keeps
the textbook benchmark structure fixed, while using OmniThink-style outline and
section generation over a lightweight textbook mind-map facade.
"""

import argparse
import copy
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import traceback
import types
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
    checkpoint_path,
    checkpoint_stage,
    clean_heading,
    count_words,
    invalidate_checkpoint_stage,
    format_knowledge_units,
    format_objectives,
    format_section_knowledge_units,
    format_textbook_markdown,
    length_budget,
    load_checkpoint,
    load_env_file,
    make_chapter_source_filter,
    matches_leakage_phrase,
    max_heading_depth,
    module_io_dir,
    normalize_outline_blind_chapter,
    ordered_sections,
    ordered_subsections,
    output_structure_metrics,
    qwen_extra_body,
    read_json,
    read_json_if_exists,
    read_text_if_exists,
    safe_args_snapshot,
    stage_invalidated,
    stage_succeeded,
    source_to_match_text,
    strip_numeric_citations,
    update_checkpoint_stage,
    write_json,
    write_module_io,
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
    section_count = len(ordered_sections(chapter))
    lines = [
        f"Book title: {chapter.get('book_title') or chapter.get('metadata', {}).get('book_title', '')}",
        f"Chapter number: {chapter.get('chapter_number', '')}",
        f"Chapter title: {chapter.get('chapter_title', '')}",
        f"Target words: {length['target_words']}",
        f"Allowed word range: {length['word_range']}",
        "",
        "Outline constraints:",
        f"- Create exactly {section_count} top-level outline section(s) using single-# headings.",
        "- Create exactly one top-level # heading for each Input section below, in the same order.",
        "- Infer concise textbook section titles from each Input section's learning objectives and knowledge units.",
        "- Do not use the chapter title, book title, metadata labels, or target word budget as outline headings.",
        "- Do not add extra top-level # headings for learning objectives, introduction, conclusion, summary, exercises, references, or sources unless they correspond to an Input section.",
        "- Use ## or ### only for lower-level structure inside those required top-level sections.",
        "- Output only the markdown outline; do not add explanatory prose before or after it.",
        "",
        "Input sections:",
    ]
    for section_index, section in enumerate(ordered_sections(chapter), start=1):
        section_heading = clean_heading(section.get("heading")) or f"Section {section_index}"
        lines.append(f"\nInput section {section_index}: {section_heading}")
        objectives = format_objectives(section)
        if objectives:
            lines.append("Learning objectives:")
            lines.extend(f"- {objective}" for objective in objectives)
        section_units = format_section_knowledge_units(section)
        if section_units:
            lines.append("Knowledge units:")
            lines.extend(f"- {unit}" for unit in section_units)
            continue
        for group_index, subsection in enumerate(ordered_subsections(section), start=1):
            subsection_heading = clean_heading(subsection.get("heading")) or (
                f"Knowledge group {group_index}"
            )
            units = format_knowledge_units(subsection)
            lines.append(f"{subsection_heading}:")
            lines.extend(f"- {unit}" for unit in units or ["No explicit knowledge units."])
    return "\n".join(lines)


def outline_generation_topic(chapter: dict) -> str:
    return chapter_requirement_markdown(chapter)


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
        section_units = format_section_knowledge_units(section)
        if section_units:
            query = compact_text(f"{title} {' '.join(section_units[:2])}", 160)
            if query and query not in queries:
                queries.append(query)
            if len(queries) >= max_queries:
                return queries
            continue
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
            concepts.extend(format_section_knowledge_units(section)[:2])
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


def _ensure_omnithink_package_aliases(src_dir: Path) -> None:
    for package_name, package_dir in {
        "src": src_dir,
        "src.actions": src_dir / "actions",
        "src.dataclass": src_dir / "dataclass",
        "src.utils": src_dir / "utils",
    }.items():
        module = sys.modules.get(package_name)
        if module is None:
            module = types.ModuleType(package_name)
            module.__path__ = [str(package_dir)]
            module.__package__ = package_name
            sys.modules[package_name] = module
        elif not hasattr(module, "__path__"):
            module.__path__ = [str(package_dir)]


def _load_omnithink_module(module_name: str, path: Path):
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load OmniThink module {module_name} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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

    src_dir = omnithink_dir / "src"
    _ensure_omnithink_package_aliases(src_dir)

    text_processing_module = _load_omnithink_module(
        "src.utils.ArticleTextProcessing",
        src_dir / "utils" / "ArticleTextProcessing.py",
    )
    _load_omnithink_module(
        "src.utils.FileIOHelper",
        src_dir / "utils" / "FileIOHelper.py",
    )
    _load_omnithink_module(
        "src.dataclass.interface",
        src_dir / "dataclass" / "interface.py",
    )
    article_module = _load_omnithink_module(
        "src.dataclass.Article",
        src_dir / "dataclass" / "Article.py",
    )
    polish_module = _load_omnithink_module(
        "src.actions.article_polish",
        src_dir / "actions" / "article_polish.py",
    )

    return {
        "Article": article_module.Article,
        "ArticlePolishingModule": polish_module.ArticlePolishingModule,
        "ArticleTextProcessing": text_processing_module.ArticleTextProcessing,
    }


def provider_from_args(args) -> str:
    if args.provider != "auto":
        return args.provider
    normalized = (args.model or "").lower()
    return "qwen" if normalized.startswith("qwen") or normalized.startswith("dashscope/qwen") else "openai"


def create_lm(args, max_tokens: int, model: Optional[str] = None):
    provider = provider_from_args(args)
    model = model or args.model
    if provider == "qwen":
        extra_body = qwen_extra_body(args.qwen_thinking, args.qwen_thinking_budget)
        return QwenModel(
            model=model,
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
    return LitellmModel(model=model, **kwargs)


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


def collect_lm_stage_metrics(lms: Dict[str, object], names: List[str]) -> dict:
    usage = {}
    for name in names:
        lm = lms.get(name)
        if lm is None:
            continue
        usage[name] = {}
        if hasattr(lm, "get_usage_and_reset"):
            usage[name].update(lm.get_usage_and_reset())
        history = getattr(lm, "history", None)
        if history is not None:
            usage[name]["call_count"] = len(history)
    return {"model_usage": usage}


def append_lm_call_history(raw_dir: Path, lms: Dict[str, object], stage: str) -> str:
    history_path = raw_dir / "llm_call_history.jsonl"
    wrote = False
    with history_path.open("a", encoding="utf-8") as handle:
        for lm_name, lm in lms.items():
            history = getattr(lm, "history", None)
            if not history:
                continue
            for call in history:
                call = dict(call)
                call["checkpoint_stage"] = stage
                call["lm_name"] = lm_name
                call.pop("kwargs", None)
                handle.write(json.dumps(call, default=str) + "\n")
                wrote = True
            lm.history = []
    if not wrote and not history_path.exists():
        history_path.touch()
    return str(history_path)


def aggregate_checkpoint_model_usage(checkpoint: dict) -> Tuple[dict, dict]:
    stage_usage = {}
    aggregate = {}
    for stage_name, stage in (checkpoint.get("stages") or {}).items():
        usage_by_lm = (stage.get("metrics") or {}).get("model_usage") or {}
        if not usage_by_lm:
            continue
        stage_usage[stage_name] = usage_by_lm
        for lm_name, usage in usage_by_lm.items():
            aggregate.setdefault(lm_name, {})
            for model_name, tokens in usage.items():
                if model_name == "call_count":
                    aggregate[lm_name]["call_count"] = (
                        aggregate[lm_name].get("call_count", 0) + int(tokens or 0)
                    )
                    continue
                if not isinstance(tokens, dict):
                    continue
                target = aggregate[lm_name].setdefault(
                    model_name, {"prompt_tokens": 0, "completion_tokens": 0}
                )
                target["prompt_tokens"] += int(tokens.get("prompt_tokens", 0) or 0)
                target["completion_tokens"] += int(tokens.get("completion_tokens", 0) or 0)
    return aggregate, stage_usage


def omnithink_final_export_invalidates_article(error: str) -> bool:
    markers = [
        "Generated markdown is too short",
        "Generated markdown has headings deeper than H4",
        "Generated section count does not match dataset structure",
    ]
    return any(marker in (error or "") for marker in markers)


def article_from_markdown(omni: dict, topic: str, article_text: str):
    article = omni["Article"](topic)
    article_dict = omni["ArticleTextProcessing"].parse_article_into_dict(article_text)
    article.insert_or_create_section(article_dict=article_dict)
    return article


def build_fixed_outline(chapter: dict) -> str:
    lines = []
    for section in ordered_sections(chapter):
        section_heading = clean_heading(section.get("heading"))
        if not section_heading:
            continue
        lines.append(f"# {section_heading}")
        for subsection in ordered_subsections(section):
            subsection_heading = clean_heading(subsection.get("heading"))
            if subsection_heading:
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

    if mode == "fixed":
        outline = build_fixed_outline(chapter)
        (raw_dir / "textbook_outline.md").write_text(outline, encoding="utf-8")
        return outline, {
            "mode": mode,
            "source": "benchmark_fixed_headings",
            "outline_path": str(raw_dir / "textbook_outline.md"),
            "warnings": [],
        }

    outline_module = StormWriteOutline(outline_lm)
    outline_topic = outline_generation_topic(chapter)
    outline_result = outline_module(
        topic=outline_topic,
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
    if len(generated_headings) != expected_count:
        (raw_dir / "textbook_outline.md").write_text(raw_outline or "", encoding="utf-8")
        raise RuntimeError(
            "OmniThink generated outline section count does not match dataset "
            f"structure: expected {expected_count} top-level # sections, "
            f"got {len(generated_headings)} ({generated_headings})."
        )
    outline = raw_outline or ""
    (raw_dir / "textbook_outline.md").write_text(outline, encoding="utf-8")
    return outline, {
        "mode": "omnithink",
        "source": "omnithink_generated_outline",
        "prompt_source": "knowledge_storm.storm_wiki.modules.outline_generation.WriteOutline",
        "outline_topic": outline_topic,
        "raw_outline_path": str(raw_dir / "omnithink_raw_outline.md"),
        "draft_outline_path": str(raw_dir / "storm_aligned_draft_outline.md"),
        "outline_path": str(raw_dir / "textbook_outline.md"),
        "generated_first_level_headings": generated_headings,
        "warnings": [],
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
    fixed_outline = build_fixed_outline(chapter)
    result = task_metadata(chapter, baseline, output_dir)
    result.update(
        {
            "status": "dry_run",
            "model": args.model,
            "article_model": args.model,
            "outline_model": args.model,
            "aux_model": args.aux_model,
            "provider": provider_from_args(args),
            "omnithink_dir": str(args.omnithink_dir),
            "parallel_chapters": args.parallel_chapters,
            "outline_mode": planned_mode,
            "expected_h2_sections": len(ordered_sections(chapter)),
            "planned_queries": build_mindmap_queries(chapter, args.max_search_queries),
            "fixed_outline_preview": fixed_outline.splitlines()[:20],
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
    previous_log = None if args.overwrite else read_json_if_exists(log_path)
    previous_article_invalid = bool(
        previous_log
        and previous_log.get("status") == "failed"
        and omnithink_final_export_invalidates_article(previous_log.get("error", ""))
    )
    if args.overwrite:
        for stale_path in [
            checkpoint_path(raw_dir),
            raw_dir / "llm_call_history.jsonl",
            module_io_dir(raw_dir),
        ]:
            if stale_path.is_dir():
                shutil.rmtree(stale_path)
            elif stale_path.exists():
                stale_path.unlink()

    checkpoint = load_checkpoint(raw_dir)
    resume_events = []
    log = task_metadata(chapter, baseline, output_dir)
    log.update(
        {
            "status": "running",
            "model": args.model,
            "article_model": args.model,
            "outline_model": args.model,
            "aux_model": args.aux_model,
            "provider": provider_from_args(args),
            "omnithink_dir": str(args.omnithink_dir),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoint_path": str(checkpoint_path(raw_dir)),
            "resume_events": resume_events,
        }
    )
    write_json(log_path, log)
    write_module_io(
        raw_dir,
        "run",
        "input",
        {
            "args": safe_args_snapshot(args),
            "chapter": chapter,
            "baseline": baseline,
        },
    )

    started = time.time()
    current_stage = None
    lms: Dict[str, object] = {}
    try:
        outline_lm = create_lm(args, args.outline_max_tokens, model=args.model)
        section_lm = create_lm(args, args.section_max_tokens, model=args.model)
        polish_lm = create_lm(args, args.polish_max_tokens, model=args.aux_model)
        lms = {"outline": outline_lm, "section": section_lm, "polish": polish_lm}

        blocked_phrases = chapter_leakage_phrases(chapter)
        queries_path = raw_dir / "queries.json"
        current_stage = "query_planning"
        if (
            not args.overwrite
            and queries_path.exists()
            and (
                stage_succeeded(checkpoint, current_stage)
                or not checkpoint_stage(checkpoint, current_stage)
            )
        ):
            query_payload = read_json(queries_path)
            query_candidates = query_payload.get("queries") or []
            resume_events.append("loaded query_planning from queries.json")
            input_path = write_module_io(
                raw_dir,
                current_stage,
                "input",
                {"chapter": chapter, "max_search_queries": args.max_search_queries},
            )
            output_path = write_module_io(
                raw_dir,
                current_stage,
                "output",
                {
                    "queries": query_candidates,
                    "queries_path": str(queries_path),
                    "resumed_from_existing_artifact": True,
                },
            )
            if not stage_succeeded(checkpoint, current_stage):
                checkpoint = update_checkpoint_stage(
                    raw_dir,
                    current_stage,
                    {
                        "status": "success",
                        "resumed_from_existing_artifact": True,
                        "input_path": input_path,
                        "output_path": output_path,
                        "outputs": {"queries": str(queries_path)},
                    },
                )
        else:
            input_path = write_module_io(
                raw_dir,
                current_stage,
                "input",
                {"chapter": chapter, "max_search_queries": args.max_search_queries},
            )
            query_candidates = build_mindmap_queries(chapter, args.max_search_queries)
            query_candidates = [
                query
                for query in query_candidates
                if not matches_leakage_phrase(source_to_match_text(query), blocked_phrases)
            ][: args.max_search_queries]
            write_json(queries_path, {"queries": query_candidates})
            output_path = write_module_io(
                raw_dir,
                current_stage,
                "output",
                {"queries": query_candidates, "queries_path": str(queries_path)},
            )
            checkpoint = update_checkpoint_stage(
                raw_dir,
                current_stage,
                {
                    "status": "success",
                    "input_path": input_path,
                    "output_path": output_path,
                    "outputs": {"queries": str(queries_path)},
                },
            )

        search_path = raw_dir / "serper_search_results.json"
        current_stage = "retrieval"
        if (
            not args.overwrite
            and search_path.exists()
            and (
                stage_succeeded(checkpoint, current_stage)
                or not checkpoint_stage(checkpoint, current_stage)
            )
        ):
            search_payload = read_json(search_path)
            resume_events.append("loaded retrieval from serper_search_results.json")
            input_path = write_module_io(
                raw_dir,
                current_stage,
                "input",
                {
                    "queries": query_candidates,
                    "search_top_k": args.search_top_k,
                    "max_sources": args.max_sources,
                    "blocked_phrases": blocked_phrases,
                },
            )
            output_path = write_module_io(
                raw_dir,
                current_stage,
                "output",
                {
                    "serper_search_results": str(search_path),
                    "search_payload": search_payload,
                    "resumed_from_existing_artifact": True,
                },
            )
            if not stage_succeeded(checkpoint, current_stage):
                checkpoint = update_checkpoint_stage(
                    raw_dir,
                    current_stage,
                    {
                        "status": "success",
                        "resumed_from_existing_artifact": True,
                        "input_path": input_path,
                        "output_path": output_path,
                        "outputs": {"serper_search_results": str(search_path)},
                        "search_metrics": search_payload.get("metrics", {}),
                    },
                )
        else:
            input_path = write_module_io(
                raw_dir,
                current_stage,
                "input",
                {
                    "queries": query_candidates,
                    "search_top_k": args.search_top_k,
                    "max_sources": args.max_sources,
                    "blocked_phrases": blocked_phrases,
                },
            )
            retriever = SerperTextbookRetriever(
                api_key=os.getenv("SERPER_API_KEY"),
                top_k=args.search_top_k,
                timeout=args.search_timeout,
                max_sources=args.max_sources,
                source_filter=make_chapter_source_filter(chapter),
                blocked_phrases=blocked_phrases,
            )
            search_payload = retriever.forward(query_candidates)
            write_json(search_path, search_payload)
            output_path = write_module_io(
                raw_dir,
                current_stage,
                "output",
                {"serper_search_results": str(search_path), "search_payload": search_payload},
            )
            checkpoint = update_checkpoint_stage(
                raw_dir,
                current_stage,
                {
                    "status": "success",
                    "input_path": input_path,
                    "output_path": output_path,
                    "outputs": {"serper_search_results": str(search_path)},
                    "search_metrics": search_payload.get("metrics", {}),
                },
            )

        current_stage = "mindmap"
        input_path = write_module_io(
            raw_dir,
            current_stage,
            "input",
            {
                "accepted_sources": search_payload.get("accepted_sources") or [],
                "retrieve_top_k": args.retrieve_top_k,
            },
        )
        mindmap = TextbookMindMap(chapter=chapter, retrieve_top_k=args.retrieve_top_k)
        mindmap.build_from_sources(search_payload.get("accepted_sources") or [])
        mindmap.prepare_table_for_retrieval()
        mindmap_path = raw_dir / "mindmap.json"
        requirements_path = raw_dir / "requirements.md"
        requirements_text = chapter_requirement_markdown(chapter)
        write_json(mindmap_path, mindmap.to_dict())
        requirements_path.write_text(requirements_text, encoding="utf-8")
        output_path = write_module_io(
            raw_dir,
            current_stage,
            "output",
            {
                "mindmap": str(mindmap_path),
                "mindmap_data": mindmap.to_dict(),
                "requirements": str(requirements_path),
                "requirements_text": requirements_text,
            },
        )
        checkpoint = update_checkpoint_stage(
            raw_dir,
            current_stage,
            {
                "status": "success",
                "input_path": input_path,
                "output_path": output_path,
                "outputs": {
                    "mindmap": str(mindmap_path),
                    "requirements": str(requirements_path),
                },
            },
        )

        outline_path = raw_dir / "textbook_outline.md"
        current_stage = "outline_generation"
        outline = None
        outline_log = {}
        outline_reused = False
        if (
            not args.overwrite
            and outline_path.exists()
            and (
                stage_succeeded(checkpoint, current_stage)
            )
            and checkpoint_stage(checkpoint, current_stage).get("model") == args.model
        ):
            candidate_outline = outline_path.read_text(encoding="utf-8")
            candidate_headings = first_level_headings(
                candidate_outline, chapter.get("chapter_title", "")
            )
            if len(candidate_headings) == len(ordered_sections(chapter)):
                outline = candidate_outline
                outline_reused = True
                outline_log = {
                    "mode": args.outline_mode,
                    "source": "resumed_existing_textbook_outline",
                    "outline_path": str(outline_path),
                    "generated_first_level_headings": candidate_headings,
                    "warnings": [],
                }
                resume_events.append("loaded outline_generation from textbook_outline.md")
                input_path = write_module_io(
                    raw_dir,
                    current_stage,
                    "input",
                    {
                        "outline_mode": args.outline_mode,
                        "model": args.model,
                        "mindmap": str(mindmap_path),
                        "requirements": str(requirements_path),
                        "outline_topic": outline_generation_topic(chapter),
                    },
                )
                output_path = write_module_io(
                    raw_dir,
                    current_stage,
                    "output",
                    {
                        "outline": outline,
                        "outline_path": str(outline_path),
                        "first_level_headings": candidate_headings,
                        "model": args.model,
                        "resumed_from_existing_artifact": True,
                    },
                )
                if not stage_succeeded(checkpoint, current_stage):
                    checkpoint = update_checkpoint_stage(
                        raw_dir,
                        current_stage,
                        {
                            "status": "success",
                            "resumed_from_existing_artifact": True,
                            "model": args.model,
                            "input_path": input_path,
                            "output_path": output_path,
                            "outputs": {"outline": str(outline_path)},
                        },
                    )

        if outline is None:
            input_path = write_module_io(
                raw_dir,
                current_stage,
                "input",
                {
                    "outline_mode": args.outline_mode,
                    "model": args.model,
                    "mindmap": str(mindmap_path),
                    "requirements": str(requirements_path),
                    "outline_topic": outline_generation_topic(chapter),
                    "mindmap_summary": mindmap.export_categories_and_concepts(),
                },
            )
            outline, outline_log = resolve_outline(
                chapter=chapter,
                mindmap=mindmap,
                outline_lm=outline_lm,
                args=args,
                raw_dir=raw_dir,
            )
            metrics = collect_lm_stage_metrics(lms, ["outline"])
            history_path = append_lm_call_history(raw_dir, {"outline": outline_lm}, current_stage)
            output_path = write_module_io(
                raw_dir,
                current_stage,
                "output",
                {
                    "outline": outline,
                    "outline_path": str(outline_path),
                    "raw_outline": read_text_if_exists(raw_dir / "omnithink_raw_outline.md"),
                    "draft_outline": read_text_if_exists(raw_dir / "storm_aligned_draft_outline.md"),
                    "outline_log": outline_log,
                    "model": args.model,
                    "metrics": metrics,
                    "llm_call_history": history_path,
                },
            )
            checkpoint = update_checkpoint_stage(
                raw_dir,
                current_stage,
                {
                    "status": "success",
                    "model": args.model,
                    "input_path": input_path,
                    "output_path": output_path,
                    "outputs": {
                        "outline": str(outline_path),
                        "raw_outline": str(raw_dir / "omnithink_raw_outline.md"),
                        "draft_outline": str(raw_dir / "storm_aligned_draft_outline.md"),
                    },
                    "metrics": metrics,
                    "llm_call_history": history_path,
                },
            )

        current_stage = "outline_validation"
        outline_headings = first_level_headings(outline, chapter.get("chapter_title", ""))
        outline_structure = {
            "expected_section_count": len(ordered_sections(chapter)),
            "actual_section_count": len(outline_headings),
            "section_count_matches": len(outline_headings) == len(ordered_sections(chapter)),
            "section_headings": outline_headings,
        }
        output_path = write_module_io(
            raw_dir,
            current_stage,
            "output",
            {"outline": outline, "outline_structure": outline_structure},
        )
        if not outline_structure["section_count_matches"]:
            checkpoint = update_checkpoint_stage(
                raw_dir,
                current_stage,
                {
                    "status": "failed",
                    "output_path": output_path,
                    "outline_structure": outline_structure,
                },
            )
            raise RuntimeError(
                "Generated outline section count does not match dataset structure: "
                f"expected {outline_structure['expected_section_count']} top-level "
                f"# sections, got {outline_structure['actual_section_count']} "
                f"({outline_structure['section_headings']})."
            )
        checkpoint = update_checkpoint_stage(
            raw_dir,
            current_stage,
            {
                "status": "success",
                "output_path": output_path,
                "outline_structure": outline_structure,
                "outputs": {"outline": str(outline_path)},
            },
        )

        article_topic = clean_heading(chapter.get("chapter_title", output_id))
        article_with_outline = omni["Article"].from_outline_str(
            topic=article_topic,
            outline_str=outline,
        )
        section_names = article_with_outline.get_first_level_section_names()
        section_contexts = build_section_contexts(chapter, outline, section_names)
        section_contexts_path = raw_dir / "section_contexts.json"
        write_json(section_contexts_path, section_contexts)

        writer_topic = "\n".join(
            [
                f"Chapter title: {chapter.get('chapter_title', '')}",
                f"Book title: {chapter.get('book_title') or chapter.get('metadata', {}).get('book_title', '')}",
                f"Length budget: {length_budget(chapter)}",
                "Task: write a university textbook chapter section using the supplied benchmark coverage requirements.",
            ]
        )
        article_raw_path = raw_dir / "article_raw.md"
        current_stage = "article_generation"
        article_cache_usable = (
            not args.overwrite
            and outline_reused
            and not previous_article_invalid
            and not stage_invalidated(checkpoint, current_stage)
            and article_raw_path.exists()
            and (
                stage_succeeded(checkpoint, current_stage)
                or not checkpoint_stage(checkpoint, current_stage)
            )
        )
        draft_article = None
        if article_cache_usable:
            raw_article = article_raw_path.read_text(encoding="utf-8")
            resume_events.append("loaded article_generation from article_raw.md")
            input_path = write_module_io(
                raw_dir,
                current_stage,
                "input",
                {
                    "writer_topic": writer_topic,
                    "outline": outline,
                    "section_contexts": section_contexts,
                    "mindmap": str(mindmap_path),
                },
            )
            output_path = write_module_io(
                raw_dir,
                current_stage,
                "output",
                {
                    "article_raw": str(article_raw_path),
                    "article_raw_text": raw_article,
                    "resumed_from_existing_artifact": True,
                },
            )
            if not stage_succeeded(checkpoint, current_stage):
                checkpoint = update_checkpoint_stage(
                    raw_dir,
                    current_stage,
                    {
                        "status": "success",
                        "resumed_from_existing_artifact": True,
                        "input_path": input_path,
                        "output_path": output_path,
                        "outputs": {"article_raw": str(article_raw_path)},
                    },
                )
        else:
            input_path = write_module_io(
                raw_dir,
                current_stage,
                "input",
                {
                    "writer_topic": writer_topic,
                    "outline": outline,
                    "section_names": section_names,
                    "section_contexts": section_contexts,
                    "section_contexts_path": str(section_contexts_path),
                    "mindmap": str(mindmap_path),
                },
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
            raw_article = draft_article.to_string()
            article_raw_path.write_text(raw_article, encoding="utf-8")
            metrics = collect_lm_stage_metrics(lms, ["section"])
            history_path = append_lm_call_history(raw_dir, {"section": section_lm}, current_stage)
            output_path = write_module_io(
                raw_dir,
                current_stage,
                "output",
                {
                    "article_raw": str(article_raw_path),
                    "article_raw_text": raw_article,
                    "metrics": metrics,
                    "llm_call_history": history_path,
                },
            )
            checkpoint = update_checkpoint_stage(
                raw_dir,
                current_stage,
                {
                    "status": "success",
                    "input_path": input_path,
                    "output_path": output_path,
                    "outputs": {"article_raw": str(article_raw_path)},
                    "metrics": metrics,
                    "llm_call_history": history_path,
                },
            )

        final_raw_article = raw_article
        if args.do_polish_article:
            polished_raw_path = raw_dir / "article_polished_raw.md"
            current_stage = "article_polishing"
            polish_cache_usable = (
                not args.overwrite
                and article_cache_usable
                and polished_raw_path.exists()
                and stage_succeeded(checkpoint, current_stage)
            )
            if polish_cache_usable:
                final_raw_article = polished_raw_path.read_text(encoding="utf-8")
                resume_events.append("loaded article_polishing from article_polished_raw.md")
                input_path = write_module_io(
                    raw_dir,
                    current_stage,
                    "input",
                    {"article_raw": str(article_raw_path), "article_topic": article_topic},
                )
                output_path = write_module_io(
                    raw_dir,
                    current_stage,
                    "output",
                    {
                        "article_polished_raw": str(polished_raw_path),
                        "article_polished_raw_text": final_raw_article,
                        "resumed_from_existing_artifact": True,
                    },
                )
                checkpoint = update_checkpoint_stage(
                    raw_dir,
                    current_stage,
                    {
                        "status": "success",
                        "resumed_from_existing_artifact": True,
                        "input_path": input_path,
                        "output_path": output_path,
                        "outputs": {"article_polished_raw": str(polished_raw_path)},
                    },
                )
            else:
                if draft_article is None:
                    draft_article = article_from_markdown(omni, article_topic, raw_article)
                input_path = write_module_io(
                    raw_dir,
                    current_stage,
                    "input",
                    {
                        "article_raw": str(article_raw_path),
                        "article_raw_text": raw_article,
                        "article_topic": article_topic,
                    },
                )
                polisher = omni["ArticlePolishingModule"](
                    article_gen_lm=section_lm,
                    article_polish_lm=polish_lm,
                )
                final_article = polisher.polish_article(
                    topic=article_topic,
                    draft_article=draft_article,
                )
                final_raw_article = final_article.to_string()
                polished_raw_path.write_text(final_raw_article, encoding="utf-8")
                metrics = collect_lm_stage_metrics(lms, ["polish"])
                history_path = append_lm_call_history(raw_dir, {"polish": polish_lm}, current_stage)
                output_path = write_module_io(
                    raw_dir,
                    current_stage,
                    "output",
                    {
                        "article_polished_raw": str(polished_raw_path),
                        "article_polished_raw_text": final_raw_article,
                        "metrics": metrics,
                        "llm_call_history": history_path,
                    },
                )
                checkpoint = update_checkpoint_stage(
                    raw_dir,
                    current_stage,
                    {
                        "status": "success",
                        "input_path": input_path,
                        "output_path": output_path,
                        "outputs": {"article_polished_raw": str(polished_raw_path)},
                        "metrics": metrics,
                        "llm_call_history": history_path,
                    },
                )

        current_stage = "final_export"
        input_path = write_module_io(
            raw_dir,
            current_stage,
            "input",
            {
                "chapter_title": chapter.get("chapter_title", output_id),
                "raw_article": final_raw_article,
                "do_polish_article": args.do_polish_article,
            },
        )
        markdown_with_citations = format_textbook_markdown(
            chapter_title=chapter.get("chapter_title", output_id),
            raw_article=final_raw_article,
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
        output_path = write_module_io(
            raw_dir,
            current_stage,
            "output",
            {
                "markdown": str(md_path),
                "markdown_text": markdown,
                "markdown_with_citations": str(raw_dir / "chapter_with_citations.md"),
                "markdown_with_citations_text": markdown_with_citations,
                **metrics,
            },
        )
        checkpoint = update_checkpoint_stage(
            raw_dir,
            current_stage,
            {
                "status": "success",
                "input_path": input_path,
                "output_path": output_path,
                "outputs": {
                    "markdown": str(md_path),
                    "markdown_with_citations": str(raw_dir / "chapter_with_citations.md"),
                },
                **metrics,
            },
        )

        elapsed = time.time() - started
        checkpoint = load_checkpoint(raw_dir)
        aggregate_usage, stage_model_usage = aggregate_checkpoint_model_usage(checkpoint)
        log.update(
            {
                "status": "success",
                "elapsed_seconds": elapsed,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "outline": outline_log,
                "outline_structure": outline_structure,
                "search_metrics": search_payload.get("metrics", {}),
                "mindmap": {
                    "concept_count": len(mindmap.root.concept if mindmap.root else []),
                    "source_count": len(mindmap.all_infos),
                    "snippet_count": len(mindmap.snippet_records),
                },
                "model_usage": aggregate_usage,
                "stage_model_usage": stage_model_usage,
                "polishing": {"enabled": args.do_polish_article},
                "warnings": sorted(set(warnings)),
                **metrics,
            }
        )
    except Exception as exc:
        elapsed = time.time() - started
        failure_metrics = {}
        if current_stage in {"outline_generation", "article_generation", "article_polishing"}:
            failure_metrics = collect_lm_stage_metrics(lms, list(lms.keys()))
        history_path = append_lm_call_history(raw_dir, lms, current_stage or "unknown")
        failure_payload = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "llm_call_history": history_path,
        }
        if current_stage:
            if failure_metrics:
                failure_payload["metrics"] = failure_metrics
            update_checkpoint_stage(raw_dir, current_stage, failure_payload)
            if current_stage == "final_export" and omnithink_final_export_invalidates_article(
                str(exc)
            ):
                invalidate_checkpoint_stage(
                    raw_dir,
                    "article_generation",
                    f"Invalid final export output: {exc}",
                )
                resume_events.append(
                    "invalidated article_generation after final_export validation failure"
                )
        checkpoint = load_checkpoint(raw_dir)
        aggregate_usage, stage_model_usage = aggregate_checkpoint_model_usage(checkpoint)
        log.update(
            {
                "status": "failed",
                "elapsed_seconds": elapsed,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_stage": current_stage,
                "model_usage": aggregate_usage,
                "stage_model_usage": stage_model_usage,
            }
        )

    log["resume_events"] = resume_events
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


def task_progress_message(index: int, total: int, baseline: str, output_id: str, result: dict) -> str:
    message = f"[{index}/{total}] done {baseline}/{output_id}: {result.get('status')}"
    if result.get("status") == "failed":
        details = []
        if result.get("failed_stage"):
            details.append(f"stage={result.get('failed_stage')}")
        if result.get("error"):
            details.append(f"error={compact_text(result.get('error'), 240)}")
        if details:
            message = f"{message} ({'; '.join(details)})"
    return message


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
                task_progress_message(index, len(tasks), baseline, output_id, result),
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
                task_progress_message(
                    completed,
                    len(tasks),
                    baseline,
                    chapter_output_id(chapter),
                    result,
                ),
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
    parser.add_argument(
        "--aux-model",
        default=os.getenv("QWEN_FLASH_MODEL", "qwen3.6-flash"),
        help="Model for non-article-generation LM calls such as outline and polishing.",
    )
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
        choices=["auto", "omnithink", "fixed"],
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
