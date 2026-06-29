#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BENCHMARK_PATH="${BENCHMARK_PATH:-chapter_benchmark_final_outline_blind.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qwen_textbook_direct_baselines}"
MODEL="${MODEL:-qwen3.7-plus}"
BASELINES="${BASELINES:-llm_only rag_serper}"

PARALLEL_CHAPTERS="${PARALLEL_CHAPTERS:-8}"
MAX_MODEL_OUTPUT_TOKENS="${MAX_MODEL_OUTPUT_TOKENS:-24000}"
MAX_SEARCH_QUERIES="${MAX_SEARCH_QUERIES:-8}"
SEARCH_TOP_K="${SEARCH_TOP_K:-5}"
MAX_SOURCES="${MAX_SOURCES:-24}"

CACHE_MODE="${CACHE_MODE:-implicit}"
QWEN_THINKING="${QWEN_THINKING:-disable}"
QWEN_THINKING_BUDGET="${QWEN_THINKING_BUDGET:-4096}"
QUERY_GENERATION="${QUERY_GENERATION:-llm}"

read -r -a BASELINE_ARGS <<< "$BASELINES"

args=(
  --benchmark-path "$BENCHMARK_PATH"
  --output-dir "$OUTPUT_DIR"
  --model "$MODEL"
  --baselines "${BASELINE_ARGS[@]}"
  --parallel-chapters "$PARALLEL_CHAPTERS"
  --max-model-output-tokens "$MAX_MODEL_OUTPUT_TOKENS"
  --max-search-queries "$MAX_SEARCH_QUERIES"
  --search-top-k "$SEARCH_TOP_K"
  --max-sources "$MAX_SOURCES"
  --cache-mode "$CACHE_MODE"
  --qwen-thinking "$QWEN_THINKING"
  --qwen-thinking-budget "$QWEN_THINKING_BUDGET"
  --query-generation "$QUERY_GENERATION"
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

echo "Running direct Qwen textbook baselines"
echo "  benchmark: $BENCHMARK_PATH"
echo "  output: $OUTPUT_DIR"
echo "  model: $MODEL"
echo "  baselines: $BASELINES"
echo "  parallel_chapters: $PARALLEL_CHAPTERS"
echo "  query_generation: $QUERY_GENERATION"

python examples/textbook_benchmark/run_qwen_direct_baselines.py "${args[@]}"
