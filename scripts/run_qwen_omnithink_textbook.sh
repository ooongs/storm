#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BENCHMARK_PATH="${BENCHMARK_PATH:-chapter_benchmark_final_outline_blind.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qwen_omnithink_textbook_benchmark}"
OMNITHINK_DIR="${OMNITHINK_DIR:-external/OmniThink}"
MODEL="${MODEL:-qwen3.7-plus}"
AUX_MODEL="${AUX_MODEL:-${WEAK_MODEL:-qwen3.6-flash}}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"

PARALLEL_CHAPTERS="${PARALLEL_CHAPTERS:-1}"
MAX_THREAD_NUM="${MAX_THREAD_NUM:-3}"
OUTLINE_MODE="${OUTLINE_MODE:-auto}"
MAX_SEARCH_QUERIES="${MAX_SEARCH_QUERIES:-10}"
SEARCH_TOP_K="${SEARCH_TOP_K:-5}"
RETRIEVE_TOP_K="${RETRIEVE_TOP_K:-5}"
MAX_SOURCES="${MAX_SOURCES:-30}"

OUTLINE_MAX_TOKENS="${OUTLINE_MAX_TOKENS:-2400}"
SECTION_MAX_TOKENS="${SECTION_MAX_TOKENS:-9000}"
MAX_MODEL_OUTPUT_TOKENS="${MAX_MODEL_OUTPUT_TOKENS:-24000}"

CACHE_MODE="${CACHE_MODE:-implicit}"
QWEN_THINKING="${QWEN_THINKING:-disable}"
QWEN_THINKING_BUDGET="${QWEN_THINKING_BUDGET:-4096}"
POLISH_ARTICLE="${POLISH_ARTICLE:-0}"

args=(
  --benchmark-path "$BENCHMARK_PATH"
  --output-dir "$OUTPUT_DIR"
  --omnithink-dir "$OMNITHINK_DIR"
  --model "$MODEL"
  --aux-model "$AUX_MODEL"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --provider qwen
  --baselines omnithink
  --parallel-chapters "$PARALLEL_CHAPTERS"
  --max-thread-num "$MAX_THREAD_NUM"
  --outline-mode "$OUTLINE_MODE"
  --max-search-queries "$MAX_SEARCH_QUERIES"
  --search-top-k "$SEARCH_TOP_K"
  --retrieve-top-k "$RETRIEVE_TOP_K"
  --max-sources "$MAX_SOURCES"
  --outline-max-tokens "$OUTLINE_MAX_TOKENS"
  --section-max-tokens "$SECTION_MAX_TOKENS"
  --max-model-output-tokens "$MAX_MODEL_OUTPUT_TOKENS"
  --cache-mode "$CACHE_MODE"
  --qwen-thinking "$QWEN_THINKING"
  --qwen-thinking-budget "$QWEN_THINKING_BUDGET"
)

if [[ -n "${START_INDEX:-}" ]]; then
  args+=(--start-index "$START_INDEX")
fi

if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "$LIMIT")
fi

if [[ -n "${CHAPTER_ID:-}" ]]; then
  args+=(--chapter-id "$CHAPTER_ID")
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

if [[ "${STOP_ON_ERROR:-0}" == "1" ]]; then
  args+=(--stop-on-error)
fi

if [[ "${QUIET:-0}" == "1" ]]; then
  args+=(--quiet)
fi

if [[ "$POLISH_ARTICLE" == "1" ]]; then
  args+=(--do-polish-article)
fi

echo "Running Qwen OmniThink textbook benchmark"
echo "  benchmark: $BENCHMARK_PATH"
echo "  output: $OUTPUT_DIR"
echo "  omnithink: $OMNITHINK_DIR"
echo "  article_model: $MODEL"
echo "  aux_model: $AUX_MODEL"
echo "  temperature: $TEMPERATURE"
echo "  top_p: $TOP_P"
echo "  outline_mode: $OUTLINE_MODE"
echo "  parallel_chapters: $PARALLEL_CHAPTERS"
echo "  max_thread_num: $MAX_THREAD_NUM"
echo "  cache_mode: $CACHE_MODE"
echo "  polish_article: $POLISH_ARTICLE"

python examples/textbook_benchmark/run_textbook_baselines.py "${args[@]}"
