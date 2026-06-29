"""
Run a minimal STORM baseline for textbook chapter generation.

This runner adapts STORM Wiki with the benchmark-provided chapter outline as the
fixed outline. It keeps the normal STORM research and section generation flow,
but exports final Markdown as a textbook chapter.
"""

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
from knowledge_storm.lm import LitellmModel, QwenModel
from knowledge_storm.rm import SerperRM
from knowledge_storm.storm_wiki.modules.callback import BaseCallbackHandler
from knowledge_storm.storm_wiki.modules.prevent_data_leakage import is_allowed_source
from knowledge_storm.storm_wiki.modules.storm_dataclass import StormArticle
from knowledge_storm.utils import FileIOHelper

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_OUTPUT_DIR = Path("results/textbook_benchmark")
DEFAULT_BENCHMARK_PATH = Path("chapter_benchmark_final_outline_blind.jsonl")
DEFAULT_QUERY_PARAMS = {"autocorrect": True, "page": 1}
QWEN_DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def is_qwen_model(model: str) -> bool:
    normalized = (model or "").lower()
    return normalized.startswith("qwen") or normalized.startswith("dashscope/qwen")


def resolve_model_pair(args) -> Tuple[str, str]:
    if args.models:
        strong_model = args.models[0]
        weak_model = args.models[1] if len(args.models) > 1 else args.models[0]
        return weak_model, strong_model
    return args.weak_model, args.strong_model


def resolve_provider(args, weak_model: str, strong_model: str) -> str:
    if args.provider != "auto":
        return args.provider
    return "qwen" if is_qwen_model(weak_model) or is_qwen_model(strong_model) else "openai"


def qwen_extra_body(thinking_mode: str, thinking_budget: Optional[int]) -> Optional[dict]:
    if thinking_mode == "default":
        return None

    extra_body = {"enable_thinking": thinking_mode == "on"}
    if thinking_mode == "on" and thinking_budget:
        extra_body["thinking_budget"] = thinking_budget
    return extra_body


def normalize_source_match_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def source_to_match_text(source) -> str:
    if isinstance(source, dict):
        parts = [
            source.get("url", ""),
            source.get("link", ""),
            source.get("title", ""),
            source.get("description", ""),
        ]
        parts.extend(source.get("snippets") or [])
        return normalize_source_match_text(" ".join(str(part or "") for part in parts))
    return normalize_source_match_text(str(source or ""))


def matches_leakage_phrase(text: str, leakage_phrases: List[str]) -> bool:
    normalized = normalize_source_match_text(text)
    compact = normalized.replace(" ", "")
    for phrase in leakage_phrases:
        if phrase in normalized:
            return True
        if phrase.replace(" ", "") in compact:
            return True
    return False


def chapter_leakage_phrases(chapter: dict) -> List[str]:
    metadata = chapter.get("metadata") or {}
    phrases = [
        chapter.get("book_title"),
        metadata.get("book_title"),
        chapter.get("book_slug"),
        metadata.get("book_slug"),
        chapter.get("dataset_id"),
        chapter.get("id"),
        metadata.get("id"),
    ]
    for source_file in (metadata.get("chapter") or {}).get("source_page_files", []):
        phrases.append(Path(source_file).stem)

    normalized = []
    for phrase in phrases:
        text = normalize_source_match_text(str(phrase or ""))
        if len(text) >= 12 and text not in normalized:
            normalized.append(text)
    return normalized


def make_chapter_source_filter(chapter: dict):
    leakage_phrases = chapter_leakage_phrases(chapter)

    def is_valid(source) -> bool:
        if not is_allowed_source(source):
            return False
        return not matches_leakage_phrase(source_to_match_text(source), leakage_phrases)

    return is_valid


def chapter_needs_generated_headings(chapter: dict) -> bool:
    for section in ordered_sections(chapter):
        if not clean_heading(section.get("heading")):
            return True
        for subsection in ordered_subsections(section):
            if not clean_heading(subsection.get("heading")):
                return True
    return False


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def slugify(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", clean_heading(text).lower()).strip("_")
    return slug or fallback


def normalize_outline_blind_chapter(chapter: dict, index: int) -> dict:
    if "sections" in chapter or "section_blocks" not in chapter:
        return chapter

    chapter_id = chapter.get("chapter_id") or f"ch_{slugify(chapter.get('chapter_title'), f'{index:04d}')}"
    dataset_id = chapter.get("dataset_id") or chapter.get("id") or chapter_id
    normalized = {
        "chapter_id": chapter_id,
        "dataset_id": dataset_id,
        "source_chapter_id": chapter.get("source_chapter_id"),
        "book_slug": chapter.get("book_slug"),
        "book_title": chapter.get("book_title"),
        "chapter_number": chapter.get("chapter_number"),
        "chapter_title": chapter.get("chapter_title", ""),
        "sections": [],
        "metadata": dict(chapter.get("metadata") or {}),
    }
    knowledge_unit_count = 0
    for section_index, block in enumerate(chapter.get("section_blocks", [])):
        section_id = block.get("section_id") or f"sec_{section_index + 1:02d}"
        section = {
            "section_id": section_id,
            "heading": None,
            "order_index": section_index,
            "learning_objectives": [
                {
                    "lo_id": f"{section_id}_lo_{lo_index + 1:02d}",
                    "text": objective,
                }
                for lo_index, objective in enumerate(
                    block.get("learning_objectives", [])
                )
            ],
            "subsections": [],
        }
        for group_index, group in enumerate(block.get("ku_groups", [])):
            subsection_id = f"{section_id}_{group.get('group_id') or f'g_{group_index + 1:02d}'}"
            knowledge_units = [
                {
                    "ku_id": f"{subsection_id}_ku_{ku_index + 1:02d}",
                    "text": unit,
                    "section_id": section_id,
                    "subsection_id": subsection_id,
                    "order_index": ku_index,
                }
                for ku_index, unit in enumerate(group.get("knowledge_units", []))
            ]
            knowledge_unit_count += len(knowledge_units)
            section["subsections"].append(
                {
                    "subsection_id": subsection_id,
                    "heading": None,
                    "order_index": group_index,
                    "knowledge_units": knowledge_units,
                }
            )
        normalized["sections"].append(section)

    section_count = len(normalized["sections"])
    target_words = max(1200, knowledge_unit_count * 110)
    normalized["metadata"]["stats"] = {
        "section_count": section_count,
        "knowledge_unit_count": knowledge_unit_count,
    }
    normalized["length_budget"] = chapter.get("length_budget") or {
        "chapter_budget": {
            "target_words": target_words,
            "word_range": [round(target_words * 0.64), round(target_words * 1.55)],
        }
    }
    return normalized


def iter_chapters(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                chapter = json.loads(line)
                chapter = normalize_outline_blind_chapter(chapter, index)
                chapter["_benchmark_index"] = index
                yield chapter


def attach_output_keys(chapters: List[dict]) -> List[dict]:
    id_counts = Counter(chapter["chapter_id"] for chapter in chapters)
    dataset_id_counts = Counter(
        chapter.get("dataset_id") for chapter in chapters if chapter.get("dataset_id")
    )
    dataset_id_seen = Counter()
    for chapter in chapters:
        if chapter.get("_output_id"):
            continue
        dataset_id = chapter.get("dataset_id")
        if dataset_id:
            dataset_id_seen[dataset_id] += 1
            if dataset_id_counts[dataset_id] > 1:
                chapter["_output_id"] = f"{dataset_id}_{dataset_id_seen[dataset_id]:02d}"
            else:
                chapter["_output_id"] = dataset_id
            continue
        chapter_id = chapter["chapter_id"]
        if id_counts[chapter_id] > 1:
            chapter["_output_id"] = f"{chapter['_benchmark_index']:04d}_{chapter_id}"
        else:
            chapter["_output_id"] = chapter_id
    return chapters


def chapter_output_id(chapter: dict) -> str:
    return chapter.get("_output_id") or chapter["chapter_id"]


def ordered_sections(chapter: dict) -> List[dict]:
    return sorted(chapter.get("sections", []), key=lambda item: item.get("order_index", 0))


def ordered_subsections(section: dict) -> List[dict]:
    return sorted(section.get("subsections", []), key=lambda item: item.get("order_index", 0))


def chapter_counts(chapter: dict) -> Tuple[int, int]:
    sections = ordered_sections(chapter)
    section_count = chapter.get("metadata", {}).get("stats", {}).get("section_count")
    if section_count is None:
        section_count = len(sections)
    knowledge_unit_count = chapter.get("metadata", {}).get("stats", {}).get(
        "knowledge_unit_count"
    )
    if knowledge_unit_count is None:
        knowledge_unit_count = sum(
            len(subsection.get("knowledge_units", []))
            for section in sections
            for subsection in ordered_subsections(section)
        )
    return int(section_count), int(knowledge_unit_count)


def compute_budgets(chapter: dict) -> Dict[str, int]:
    section_count, knowledge_unit_count = chapter_counts(chapter)
    query_budget = 2 * section_count + math.ceil(knowledge_unit_count / 5)
    source_budget = min(3 * query_budget, 40)
    return {
        "section_count": section_count,
        "knowledge_unit_count": knowledge_unit_count,
        "query_budget": query_budget,
        "source_budget": source_budget,
    }


def length_budget(chapter: dict) -> Dict[str, object]:
    budget = chapter.get("length_budget", {}).get("chapter_budget", {})
    return {
        "target_words": int(budget.get("target_words", 0) or 0),
        "word_range": budget.get("word_range", [0, 0]),
    }


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


def format_objectives(section: dict) -> List[str]:
    return [
        clean_heading(item.get("text"))
        for item in section.get("learning_objectives", [])
        if clean_heading(item.get("text"))
    ]


def format_knowledge_units(subsection: dict) -> List[str]:
    units = sorted(
        subsection.get("knowledge_units", []), key=lambda item: item.get("order_index", 0)
    )
    return [clean_heading(unit.get("text")) for unit in units if clean_heading(unit.get("text"))]


def build_research_context(chapter: dict, budgets: Dict[str, int]) -> str:
    length = length_budget(chapter)
    parts = [
        f"Chapter title: {chapter.get('chapter_title', '')}",
        f"Book title: {chapter.get('metadata', {}).get('book_title', '')}",
        f"Required word budget: target {length['target_words']} words, range {length['word_range']}",
        "Task: gather information for a university textbook chapter. Use the learning objectives and knowledge units as requirements.",
        "Data leakage rule: do not search for, cite, quote, or rely on the source textbook, exact book title, source chapter pages, mirror PDFs, or benchmark source files. Use independent sources for general concepts only.",
        f"Search budget: at most {budgets['query_budget']} queries and {budgets['source_budget']} accepted sources.",
    ]
    for section in ordered_sections(chapter):
        section_heading = clean_heading(section.get("heading"))
        section_label = section_heading or section.get("section_id", "Untitled section")
        parts.append(f"Input section: {section_label}")
        objectives = format_objectives(section)
        if objectives:
            parts.append("Learning objectives:")
            parts.extend(f"- {objective}" for objective in objectives)
        for subsection in ordered_subsections(section):
            subsection_heading = clean_heading(subsection.get("heading"))
            units = format_knowledge_units(subsection)
            if units:
                subsection_label = subsection_heading or subsection.get(
                    "subsection_id", "Untitled subsection"
                )
                parts.append(f"Knowledge units for {subsection_label}:")
                parts.extend(f"- {unit}" for unit in units)
    return "\n".join(parts)


def build_writer_topic(chapter: dict) -> str:
    length = length_budget(chapter)
    parts = [
        f"Chapter title: {chapter.get('chapter_title', '')}",
        f"Book title: {chapter.get('metadata', {}).get('book_title', '')}",
        f"Required word budget: target {length['target_words']} words, range {length['word_range']}",
        "Task: write a university textbook chapter section. Use only the section-specific requirements provided in the writing context.",
    ]
    return "\n".join(parts)


def section_knowledge_unit_count(section: dict) -> int:
    return sum(
        len(subsection.get("knowledge_units", [])) for subsection in ordered_subsections(section)
    )


def build_section_context(
    chapter: dict,
    section: Optional[dict],
    section_heading: str,
    outline: str,
    section_target: int,
) -> str:
    length = length_budget(chapter)
    parts = [
        "Write this as university textbook prose, not a Wikipedia article.",
        "Use the provided section outline exactly. In the raw STORM section, start with '# {0}', use '##' for listed subsections, and use '###' only for necessary lower-level headings. The final exporter will demote headings so the chapter title is the only level-1 heading.".format(
            section_heading
        ),
        "Do not use raw headings deeper than ###. The final Markdown must not exceed ####.",
        "Use equations, examples, and explanatory paragraphs when useful. Keep inline citations like [1] when source information supports a claim.",
        f"Approximate section target: {section_target} words as part of a chapter target of {length['target_words']} words.",
        "Full chapter outline:",
        outline,
        f"Current section: {section_heading}",
    ]
    if section is None:
        parts.append(
            "No benchmark learning objectives or knowledge units are assigned to this generated section."
        )
    else:
        parts.append("Cover every listed learning objective and knowledge unit for this section.")
        objectives = format_objectives(section)
        if objectives:
            parts.append("Learning objectives for this section:")
            parts.extend(f"- {objective}" for objective in objectives)
        for subsection in ordered_subsections(section):
            subsection_heading = clean_heading(subsection.get("heading"))
            units = format_knowledge_units(subsection)
            subsection_label = subsection_heading or subsection.get(
                "subsection_id", "Untitled subsection"
            )
            parts.append(f"Subsection: {subsection_label}")
            if units:
                parts.append("Knowledge units:")
                parts.extend(f"- {unit}" for unit in units)
    return "\n".join(parts)


def build_section_contexts(
    chapter: dict, outline: str, section_names: Optional[List[str]] = None
) -> Dict[str, str]:
    length = length_budget(chapter)
    sections = ordered_sections(chapter)
    total_ku = max(1, sum(section_knowledge_unit_count(section) for section in sections))
    contexts = {}
    if section_names is None:
        for section in sections:
            section_heading = clean_heading(section.get("heading"))
            if not section_heading:
                continue
            section_ku = section_knowledge_unit_count(section)
            section_target = max(400, round(length["target_words"] * section_ku / total_ku))
            contexts[section_heading] = build_section_context(
                chapter=chapter,
                section=section,
                section_heading=section_heading,
                outline=outline,
                section_target=section_target,
            )
        return contexts

    for index, section_name in enumerate(section_names):
        section_heading = clean_heading(section_name)
        if not section_heading:
            continue
        section = sections[index] if index < len(sections) else None
        section_ku = section_knowledge_unit_count(section) if section else 0
        section_target = (
            max(400, round(length["target_words"] * section_ku / total_ku))
            if section
            else max(300, round(length["target_words"] / max(1, len(section_names))))
        )
        contexts[section_heading] = build_section_context(
            chapter=chapter,
            section=section,
            section_heading=section_heading,
            outline=outline,
            section_target=section_target,
        )
    return contexts


def count_words(markdown: str) -> int:
    text = re.sub(r"`[^`]*`", " ", markdown)
    text = re.sub(r"#+\s*", " ", text)
    text = re.sub(r"\[[0-9]+\]", " ", text)
    return len(re.findall(r"\b[\w'-]+\b", text))


def format_textbook_markdown(chapter_title: str, raw_article: str) -> str:
    output = [f"# {clean_heading(chapter_title)}", ""]
    for line in raw_article.splitlines():
        match = re.match(r"^(#{1,})\s+(.*)$", line.strip())
        if match:
            title = clean_heading(match.group(2))
            level = min(len(match.group(1)) + 1, 4)
            output.append(f"{'#' * level} {title}")
        else:
            output.append(line.rstrip())
    markdown = "\n".join(output).strip() + "\n"
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown


NUMERIC_CITATION_PATTERN = re.compile(r"\[\d+\]")


def strip_numeric_citations(markdown: str) -> str:
    markdown = re.sub(r"(?:\[\d+\])+", "", markdown)
    markdown = re.sub(r"[ \t]+([.,;:!?])", r"\1", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip() + "\n"


def max_heading_depth(markdown: str) -> int:
    depths = [
        len(match.group(1))
        for match in re.finditer(r"^(#{1,})\s+", markdown, flags=re.MULTILINE)
    ]
    return max(depths, default=0)


def markdown_headings(markdown: str) -> List[Tuple[int, str]]:
    headings = []
    for match in re.finditer(r"^(#{1,6})\s+(.*?)\s*$", markdown, flags=re.MULTILINE):
        headings.append((len(match.group(1)), clean_heading(match.group(2))))
    return headings


def output_structure_metrics(
    chapter: dict, markdown: str, ignore_summary: bool = False
) -> Dict[str, object]:
    headings = markdown_headings(markdown)
    h2_headings = [title for level, title in headings if level == 2]
    counted_h2_headings = [
        title
        for title in h2_headings
        if not (ignore_summary and title.lower() == "summary")
    ]
    expected_section_count = len(ordered_sections(chapter))
    actual_section_count = len(counted_h2_headings)
    return {
        "expected_section_count": expected_section_count,
        "actual_section_count": actual_section_count,
        "section_count_matches": actual_section_count == expected_section_count,
        "section_headings": counted_h2_headings,
        "raw_h2_headings": h2_headings,
        "max_heading_depth": max((level for level, _ in headings), default=0),
        "heading_level_counts": {
            str(level): count for level, count in Counter(level for level, _ in headings).items()
        },
    }


class BudgetedSerperRM(SerperRM):
    def __init__(
        self,
        query_budget: int,
        source_budget: int,
        *args,
        blocked_query_phrases: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.query_budget = query_budget
        self.source_budget = source_budget
        self.query_count = 0
        self.accepted_source_count = 0
        self.skipped_query_count = 0
        self.blocked_leakage_query_count = 0
        self.skipped_source_budget_count = 0
        self.blocked_query_phrases = blocked_query_phrases or []
        self._budget_lock = threading.Lock()

    def forward(self, query_or_queries, exclude_urls: List[str] = []):
        requested_queries = (
            [query_or_queries] if isinstance(query_or_queries, str) else list(query_or_queries)
        )
        queries = []
        for query in requested_queries:
            if matches_leakage_phrase(str(query), self.blocked_query_phrases):
                self.blocked_leakage_query_count += 1
                continue
            queries.append(query)

        if not queries:
            return []

        with self._budget_lock:
            if self.accepted_source_count >= self.source_budget:
                self.skipped_query_count += len(queries)
                return []
            remaining_queries = self.query_budget - self.query_count
            if remaining_queries <= 0:
                self.skipped_query_count += len(queries)
                return []
            allowed_queries = queries[:remaining_queries]
            self.skipped_query_count += len(queries) - len(allowed_queries)
            self.query_count += len(allowed_queries)

        results = super().forward(allowed_queries, exclude_urls=exclude_urls)

        with self._budget_lock:
            remaining_sources = self.source_budget - self.accepted_source_count
            if remaining_sources <= 0:
                self.skipped_source_budget_count += len(results)
                return []
            if len(results) > remaining_sources:
                self.skipped_source_budget_count += len(results) - remaining_sources
                results = results[:remaining_sources]
            self.accepted_source_count += len(results)
        return results

    def budget_log(self) -> Dict[str, int]:
        return {
            "query_budget": self.query_budget,
            "source_budget": self.source_budget,
            "actual_query_count": self.query_count,
            "accepted_source_count": self.accepted_source_count,
            "raw_search_result_count": self.raw_result_count,
            "leakage_blocked_count": self.filtered_by_source_count,
            "leakage_query_blocked_count": self.blocked_leakage_query_count,
            "exclude_url_blocked_count": self.filtered_by_exclude_count,
            "skipped_query_count": self.skipped_query_count,
            "skipped_source_budget_count": self.skipped_source_budget_count,
        }


def create_lm_configs(args) -> STORMWikiLMConfigs:
    weak_model, strong_model = resolve_model_pair(args)
    provider = resolve_provider(args, weak_model, strong_model)

    if provider == "qwen":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required in the environment or .env file.")

        qwen_kwargs = {
            "api_key": api_key,
            "api_base": os.getenv("DASHSCOPE_API_BASE", QWEN_DEFAULT_API_BASE),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "cache": args.cache_mode != "off",
            "provider_cache": args.cache_mode == "explicit",
            "cache_prefix_chars": args.explicit_cache_prefix_chars or None,
        }
        weak_extra_body = qwen_extra_body(
            args.qwen_weak_thinking, args.qwen_weak_thinking_budget
        )
        strong_extra_body = qwen_extra_body(
            args.qwen_strong_thinking, args.qwen_strong_thinking_budget
        )
        weak_lm = QwenModel(
            model=weak_model,
            max_tokens=args.weak_max_tokens,
            extra_body=weak_extra_body,
            **qwen_kwargs,
        )
        article_lm = QwenModel(
            model=strong_model,
            max_tokens=args.article_max_tokens,
            extra_body=strong_extra_body,
            **qwen_kwargs,
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required in the environment or .env file.")

        openai_kwargs = {
            "api_key": api_key,
            "temperature": args.temperature,
            "reasoning_effort": args.openai_reasoning_effort,
        }
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            openai_kwargs["base_url"] = base_url

        weak_lm = LitellmModel(
            model=weak_model,
            max_tokens=args.weak_max_tokens,
            **openai_kwargs,
        )
        article_lm = LitellmModel(
            model=strong_model,
            max_tokens=args.article_max_tokens,
            **openai_kwargs,
        )

    lm_configs = STORMWikiLMConfigs()
    lm_configs.set_conv_simulator_lm(weak_lm)
    lm_configs.set_question_asker_lm(weak_lm)
    lm_configs.set_outline_gen_lm(weak_lm)
    lm_configs.set_article_gen_lm(article_lm)
    lm_configs.set_article_polish_lm(weak_lm)
    return lm_configs


def raw_artifact_dir(output_dir: Path, output_id: str) -> Path:
    return output_dir / "raw-artifacts" / output_id


def chapter_markdown_path(output_dir: Path, output_id: str) -> Path:
    return output_dir / "chapters" / f"{output_id}.md"


def chapter_log_path(output_dir: Path, output_id: str) -> Path:
    return output_dir / "logs" / f"{output_id}.json"


def existing_has_leaked_sources(output_dir: Path, output_id: str, chapter: dict) -> bool:
    reference_path = raw_artifact_dir(output_dir, output_id) / "url_to_info.json"
    if not reference_path.exists():
        return True
    try:
        references = read_json(reference_path)
    except Exception:
        return True

    source_filter = make_chapter_source_filter(chapter)
    for url, info in (references.get("url_to_info") or {}).items():
        source = dict(info or {})
        source["url"] = url
        if not source_filter(source):
            return True
    return False


def existing_success(output_dir: Path, chapter: dict) -> Optional[dict]:
    output_id = chapter_output_id(chapter)
    log_path = chapter_log_path(output_dir, output_id)
    md_path = chapter_markdown_path(output_dir, output_id)
    if not log_path.exists() or not md_path.exists():
        return None
    log = read_json(log_path)
    if log.get("status") != "success":
        return None
    markdown = md_path.read_text(encoding="utf-8")
    if NUMERIC_CITATION_PATTERN.search(markdown):
        return None
    actual_word_count = count_words(markdown)
    heading_depth = max_heading_depth(markdown)
    if actual_word_count < 50 or heading_depth > 4:
        return None
    if log.get("actual_word_count", 0) < 50 or log.get("max_heading_depth", 99) > 4:
        return None
    structure_metrics = output_structure_metrics(
        chapter,
        markdown,
        ignore_summary=log.get("polishing", {}).get("enabled", False),
    )
    if not structure_metrics["section_count_matches"]:
        return None
    if existing_has_leaked_sources(output_dir, output_id, chapter):
        return None
    log["markdown"] = markdown
    log["structure"] = structure_metrics
    log["skipped_existing"] = True
    return log


def build_runner(
    args, budgets: Dict[str, int], raw_dir: Path, chapter: dict
) -> Tuple[STORMWikiRunner, BudgetedSerperRM]:
    query_params = dict(DEFAULT_QUERY_PARAMS)
    rm = BudgetedSerperRM(
        query_budget=budgets["query_budget"],
        source_budget=budgets["source_budget"],
        blocked_query_phrases=chapter_leakage_phrases(chapter),
        serper_search_api_key=os.getenv("SERPER_API_KEY"),
        k=args.search_top_k,
        query_params=query_params,
        is_valid_source=make_chapter_source_filter(chapter),
        ENABLE_EXTRA_SNIPPET_EXTRACTION=args.extra_snippet_extraction,
    )
    max_conv_turn = (
        args.max_conv_turns_per_chapter
        if args.max_conv_turns_per_chapter is not None
        else max(1, math.ceil(budgets["query_budget"] / args.max_search_queries_per_turn))
    )
    engine_args = STORMWikiRunnerArguments(
        output_dir=str(raw_dir.parent),
        max_conv_turn=max(1, max_conv_turn),
        max_perspective=0,
        max_search_queries_per_turn=args.max_search_queries_per_turn,
        disable_perspective=True,
        search_top_k=args.search_top_k,
        retrieve_top_k=args.retrieve_top_k,
        max_thread_num=args.max_thread_num,
    )
    runner = STORMWikiRunner(engine_args, create_lm_configs(args), rm)
    runner.article_output_dir = str(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return runner, rm


def run_chapter(chapter: dict, args, output_dir: Path) -> dict:
    chapter_id = chapter["chapter_id"]
    output_id = chapter_output_id(chapter)
    if not args.overwrite:
        existing = existing_success(output_dir, chapter)
        if existing is not None:
            return existing

    raw_dir = raw_artifact_dir(output_dir, output_id)
    md_path = chapter_markdown_path(output_dir, output_id)
    log_path = chapter_log_path(output_dir, output_id)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    use_storm_outline = chapter_needs_generated_headings(chapter)
    fixed_outline = "" if use_storm_outline else build_fixed_outline(chapter)
    budgets = compute_budgets(chapter)
    length = length_budget(chapter)
    research_context = build_research_context(chapter, budgets)
    writer_topic = build_writer_topic(chapter)
    section_contexts = {}
    weak_model, strong_model = resolve_model_pair(args)

    log = {
        "chapter_id": chapter_id,
        "dataset_id": chapter.get("dataset_id"),
        "output_id": output_id,
        "benchmark_index": chapter.get("_benchmark_index"),
        "chapter_title": chapter.get("chapter_title"),
        "status": "running",
        "article_model": strong_model,
        "aux_model": weak_model,
        "model_assignments": {
            "article_generation": strong_model,
            "knowledge_curation": weak_model,
            "question_asking": weak_model,
            "outline_generation": weak_model,
            "article_polishing": weak_model,
        },
        "outline_source": "storm_generated" if use_storm_outline else "benchmark_fixed",
        "length_budget": length,
        "budgets": budgets,
        "output_paths": {
            "markdown": str(md_path),
            "log": str(log_path),
            "raw_artifacts": str(raw_dir),
            "markdown_with_citations": str(raw_dir / "chapter_with_citations.md"),
        },
    }
    write_json(log_path, log)

    try:
        runner, rm = build_runner(args, budgets, raw_dir, chapter)
        runner.topic = research_context
        runner.storm_article_generation.set_writer_topic(writer_topic)
        runner.storm_article_generation.set_skip_introduction_and_conclusion(False)

        information_table = runner.run_knowledge_curation_module(
            ground_truth_url="None", callback_handler=BaseCallbackHandler()
        )
        if use_storm_outline:
            article_outline = runner.run_outline_generation_module(
                information_table=information_table,
                callback_handler=BaseCallbackHandler(),
            )
        else:
            article_outline = StormArticle.from_outline_str(
                topic=research_context, outline_str=fixed_outline
            )
            article_outline.dump_outline_to_file(str(raw_dir / "storm_gen_outline.txt"))
        outline_for_writer = "\n".join(
            article_outline.get_outline_as_list(add_hashtags=True, include_root=False)
        )
        section_contexts = build_section_contexts(
            chapter,
            outline_for_writer,
            article_outline.get_first_level_section_names() if use_storm_outline else None,
        )
        runner.storm_article_generation.set_section_contexts(section_contexts)
        draft_article = runner.run_article_generation_module(
            outline=article_outline,
            information_table=information_table,
            callback_handler=BaseCallbackHandler(),
        )
        final_article = draft_article
        if args.do_polish_article:
            final_article = runner.run_article_polishing_module(
                draft_article=draft_article,
                remove_duplicate=False,
            )
        runner.post_run()

        markdown_with_citations = format_textbook_markdown(
            chapter_title=chapter.get("chapter_title", chapter_id),
            raw_article=final_article.to_string(),
        )
        (raw_dir / "chapter_with_citations.md").write_text(
            markdown_with_citations, encoding="utf-8"
        )
        citation_count_removed = len(
            NUMERIC_CITATION_PATTERN.findall(markdown_with_citations)
        )
        markdown = strip_numeric_citations(markdown_with_citations)
        if NUMERIC_CITATION_PATTERN.search(markdown):
            raise RuntimeError("Generated markdown still contains numeric citation markers.")
        actual_word_count = count_words(markdown)
        if actual_word_count < 50:
            raise RuntimeError(
                f"Generated markdown is too short ({actual_word_count} words)."
            )
        structure_metrics = output_structure_metrics(
            chapter, markdown, ignore_summary=args.do_polish_article
        )
        if structure_metrics["max_heading_depth"] > 4:
            raise RuntimeError(
                "Generated markdown has headings deeper than level 4 "
                f"(max depth {structure_metrics['max_heading_depth']})."
            )
        if not structure_metrics["section_count_matches"]:
            raise RuntimeError(
                "Generated section count does not match dataset structure: "
                f"expected {structure_metrics['expected_section_count']} H2 sections, "
                f"got {structure_metrics['actual_section_count']} H2 sections "
                f"({structure_metrics['section_headings']})."
            )
        md_path.write_text(markdown, encoding="utf-8")

        elapsed = time.time() - start_time
        search_metrics = rm.budget_log()
        warnings = []
        if search_metrics["accepted_source_count"] == 0:
            warnings.append(
                "No accepted search sources were available; article generation used benchmark context only."
            )
        log.update(
            {
                "status": "success",
                "elapsed_seconds": elapsed,
                "actual_word_count": actual_word_count,
                "max_heading_depth": structure_metrics["max_heading_depth"],
                "structure": structure_metrics,
                "stage_runtime_seconds": runner.time,
                "token_usage": runner.lm_cost,
                "retriever_usage": runner.rm_cost,
                "search_metrics": search_metrics,
                "section_context_count": len(section_contexts),
                "polishing": {
                    "enabled": args.do_polish_article,
                    "summary_only": args.do_polish_article,
                    "remove_duplicate": False,
                },
                "citation_stripping": {
                    "removed_numeric_citation_markers": citation_count_removed,
                    "with_citations_markdown": str(raw_dir / "chapter_with_citations.md"),
                },
                "warnings": warnings,
                "markdown": markdown,
            }
        )
    except Exception as exc:
        elapsed = time.time() - start_time
        log.update(
            {
                "status": "failed",
                "elapsed_seconds": elapsed,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    write_json(log_path, {k: v for k, v in log.items() if k != "markdown"})
    return log


def dry_run_chapter(chapter: dict, args) -> dict:
    use_storm_outline = chapter_needs_generated_headings(chapter)
    outline = "" if use_storm_outline else build_fixed_outline(chapter)
    budgets = compute_budgets(chapter)
    section_contexts = {} if use_storm_outline else build_section_contexts(chapter, outline)
    weak_model, strong_model = resolve_model_pair(args)
    return {
        "chapter_id": chapter["chapter_id"],
        "dataset_id": chapter.get("dataset_id"),
        "output_id": chapter_output_id(chapter),
        "benchmark_index": chapter.get("_benchmark_index"),
        "chapter_title": chapter.get("chapter_title"),
        "status": "dry_run",
        "article_model": strong_model,
        "aux_model": weak_model,
        "model_assignments": {
            "article_generation": strong_model,
            "knowledge_curation": weak_model,
            "question_asking": weak_model,
            "outline_generation": weak_model,
            "article_polishing": weak_model,
        },
        "outline_source": "storm_generated" if use_storm_outline else "benchmark_fixed",
        "length_budget": length_budget(chapter),
        "budgets": budgets,
        "structure": {
            "expected_section_count": len(ordered_sections(chapter)),
            "expected_final_heading_shape": "# Chapter, ## Section, optional ### Subsection, optional #### Subsubsection",
        },
        "outline_line_count": len(outline.splitlines()),
        "section_context_count": len(section_contexts),
        "polishing": {
            "enabled": args.do_polish_article,
            "summary_only": args.do_polish_article,
            "remove_duplicate": False,
        },
    }


def select_chapters(chapters: List[dict], args) -> List[dict]:
    selected = chapters
    if args.chapter_id:
        wanted = set(args.chapter_id)
        selected = [chapter for chapter in selected if chapter.get("chapter_id") in wanted]
    if args.start_index:
        selected = selected[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def summarize(results: List[dict]) -> dict:
    statuses = {}
    for result in results:
        statuses[result.get("status", "unknown")] = statuses.get(result.get("status", "unknown"), 0) + 1
    successful = [result for result in results if result.get("status") == "success"]
    return {
        "chapter_count": len(results),
        "statuses": statuses,
        "skipped_existing_count": sum(1 for result in results if result.get("skipped_existing")),
        "total_elapsed_seconds": sum(result.get("elapsed_seconds", 0) for result in results),
        "total_actual_words": sum(result.get("actual_word_count", 0) for result in successful),
        "total_queries": sum(
            result.get("search_metrics", {}).get("actual_query_count", 0)
            for result in successful
        ),
        "total_accepted_sources": sum(
            result.get("search_metrics", {}).get("accepted_source_count", 0)
            for result in successful
        ),
        "total_leakage_blocked": sum(
            result.get("search_metrics", {}).get("leakage_blocked_count", 0)
            for result in successful
        ),
        "total_leakage_query_blocked": sum(
            result.get("search_metrics", {}).get("leakage_query_blocked_count", 0)
            for result in successful
        ),
    }


def validate_keys_for_run(args) -> None:
    weak_model, strong_model = resolve_model_pair(args)
    provider = resolve_provider(args, weak_model, strong_model)
    required_model_key = "DASHSCOPE_API_KEY" if provider == "qwen" else "OPENAI_API_KEY"
    missing = [
        name for name in [required_model_key, "SERPER_API_KEY"] if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def save_progress(output_dir: Path, ordered_results: List[Optional[dict]]) -> List[dict]:
    available_results = [result for result in ordered_results if result is not None]
    write_json(output_dir / "all_results.json", available_results)
    write_json(output_dir / "run_summary.json", summarize(available_results))
    return available_results


def progress_enabled(args) -> bool:
    return tqdm is not None and not args.no_progress


def progress_write(args, message: str) -> None:
    if progress_enabled(args):
        tqdm.write(message)
    else:
        print(message)


def run_selected_chapters(chapters: List[dict], args, output_dir: Path) -> List[dict]:
    ordered_results: List[Optional[dict]] = [None] * len(chapters)

    def execute_one(chapter: dict) -> dict:
        if args.dry_run:
            return dry_run_chapter(chapter, args)
        return run_chapter(chapter, args, output_dir)

    if args.parallel_chapters <= 1:
        iterator = enumerate(chapters, start=1)
        if progress_enabled(args):
            iterator = tqdm(
                iterator,
                total=len(chapters),
                desc="Textbook chapters",
                unit="chapter",
            )
        for index, chapter in iterator:
            progress_write(args, f"[{index}/{len(chapters)}] start {chapter_output_id(chapter)}")
            result = execute_one(chapter)
            ordered_results[index - 1] = result
            progress_write(
                args,
                f"[{index}/{len(chapters)}] done {result.get('output_id', chapter_output_id(chapter))}: {result.get('status')}"
            )
            save_progress(output_dir, ordered_results)
            if result.get("status") == "failed" and args.stop_on_error:
                break
        return [result for result in ordered_results if result is not None]

    progress_write(
        args,
        f"Running {len(chapters)} chapters with {args.parallel_chapters} parallel chapter workers."
    )
    executor = ThreadPoolExecutor(max_workers=args.parallel_chapters)
    future_to_index = {
        executor.submit(execute_one, chapter): index
        for index, chapter in enumerate(chapters)
    }
    completed = 0
    progress_bar = (
        tqdm(total=len(chapters), desc="Textbook chapters", unit="chapter")
        if progress_enabled(args)
        else None
    )
    try:
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            chapter = chapters[index]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "chapter_id": chapter["chapter_id"],
                    "dataset_id": chapter.get("dataset_id"),
                    "output_id": chapter_output_id(chapter),
                    "benchmark_index": chapter.get("_benchmark_index"),
                    "chapter_title": chapter.get("chapter_title"),
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            ordered_results[index] = result
            progress_write(
                args,
                f"[{completed}/{len(chapters)}] done {result.get('output_id', chapter_output_id(chapter))}: {result.get('status')}"
            )
            if progress_bar is not None:
                progress_bar.update(1)
            save_progress(output_dir, ordered_results)
            if result.get("status") == "failed" and args.stop_on_error:
                for pending in future_to_index:
                    pending.cancel()
                break
    finally:
        if progress_bar is not None:
            progress_bar.close()
        executor.shutdown(wait=True, cancel_futures=True)

    return [result for result in ordered_results if result is not None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chapter-id", action="append")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--smoke-then-full", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--do-polish-article",
        action="store_true",
        help="If set, add the default STORM summary polishing step. Disabled by default.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "qwen"],
        default="auto",
        help="Model provider. auto selects qwen when model names start with qwen.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Compatibility alias for benchmark runners. First model is used for "
            "article generation, second model for research/question tasks."
        ),
    )
    parser.add_argument("--baselines", nargs="+", default=["storm"])
    parser.add_argument("--weak-model", default="gpt-5-mini")
    parser.add_argument("--strong-model", default="gpt-5")
    parser.add_argument("--weak-max-tokens", type=int, default=800)
    parser.add_argument(
        "--article-max-tokens",
        "--max-model-output-tokens",
        type=int,
        default=25000,
        dest="article_max_tokens",
        help="Maximum output tokens for article generation.",
    )
    parser.add_argument(
        "--max-conv-turns-per-chapter",
        type=int,
        help="Cap STORM research conversation turns per chapter.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9, dest="top_p")
    parser.add_argument("--openai-reasoning-effort", default="low")
    parser.add_argument(
        "--cache-mode",
        choices=["implicit", "explicit", "off"],
        default="implicit",
        help="Qwen cache mode. implicit keeps local cache and provider automatic cache; explicit adds cache_control markers.",
    )
    parser.add_argument(
        "--explicit-cache-prefix-chars",
        type=int,
        default=0,
        help="With --cache-mode explicit, mark only the first N prompt characters as cacheable. 0 marks the full prompt.",
    )
    parser.add_argument(
        "--qwen-weak-thinking",
        choices=["default", "on", "off", "disable"],
        default="disable",
        help="Thinking mode for the weaker Qwen model used during research/question asking.",
    )
    parser.add_argument(
        "--qwen-strong-thinking",
        choices=["default", "on", "off", "disable"],
        default="disable",
        help="Thinking mode for the stronger Qwen model used for textbook article generation.",
    )
    parser.add_argument("--qwen-weak-thinking-budget", type=int, default=None)
    parser.add_argument("--qwen-strong-thinking-budget", type=int, default=4096)
    parser.add_argument("--max-search-queries-per-turn", type=int, default=3)
    parser.add_argument("--search-top-k", type=int, default=3)
    parser.add_argument("--retrieve-top-k", type=int, default=3)
    parser.add_argument("--max-thread-num", type=int, default=1)
    parser.add_argument(
        "--parallel-chapters",
        type=int,
        default=1,
        help="Number of chapters to run concurrently. Resume skips completed successful chapters unless --overwrite is set.",
    )
    parser.add_argument("--extra-snippet-extraction", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    unsupported_baselines = [baseline for baseline in args.baselines if baseline != "storm"]
    if unsupported_baselines:
        raise RuntimeError(
            f"Unsupported baselines: {', '.join(unsupported_baselines)}. Only 'storm' is implemented."
        )
    chapters = attach_output_keys(list(iter_chapters(args.benchmark_path)))
    selected = select_chapters(chapters, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.parallel_chapters < 1:
        raise RuntimeError("--parallel-chapters must be at least 1.")

    if not args.dry_run:
        validate_keys_for_run(args)

    if args.smoke_then_full and not args.dry_run:
        full_selection = selected
        if not full_selection:
            raise RuntimeError("No chapters selected.")
        smoke_result = run_selected_chapters(full_selection[:1], args, args.output_dir)
        if smoke_result and smoke_result[0].get("status") != "success":
            return 1
        remaining = full_selection[1:]
        if remaining:
            remaining_results = run_selected_chapters(remaining, args, args.output_dir)
            combined = smoke_result + remaining_results
            write_json(args.output_dir / "all_results.json", combined)
            write_json(args.output_dir / "run_summary.json", summarize(combined))
        return 0

    run_selected_chapters(selected, args, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
