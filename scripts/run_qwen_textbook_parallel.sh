#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BENCHMARK_PATH="${BENCHMARK_PATH:-chapter_benchmark_final_outline_blind.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/storm_qwen_final}"
STRONG_MODEL="${STRONG_MODEL:-qwen3.7-plus}"
WEAK_MODEL="${WEAK_MODEL:-qwen3.6-flash}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"

PARALLEL_CHAPTERS="${PARALLEL_CHAPTERS:-4}"
MAX_THREAD_NUM="${MAX_THREAD_NUM:-1}"
MAX_CONV_TURNS_PER_CHAPTER="${MAX_CONV_TURNS_PER_CHAPTER:-3}"
MAX_MODEL_OUTPUT_TOKENS="${MAX_MODEL_OUTPUT_TOKENS:-32000}"

CACHE_MODE="${CACHE_MODE:-implicit}"
QWEN_STRONG_THINKING="${QWEN_STRONG_THINKING:-disable}"
QWEN_STRONG_THINKING_BUDGET="${QWEN_STRONG_THINKING_BUDGET:-4096}"
QWEN_WEAK_THINKING="${QWEN_WEAK_THINKING:-disable}"
POLISH_ARTICLE="${POLISH_ARTICLE:-0}"

args=(
  --benchmark-path "$BENCHMARK_PATH"
  --output-dir "$OUTPUT_DIR"
  --models "$STRONG_MODEL" "$WEAK_MODEL"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --provider qwen
  --baselines storm
  --parallel-chapters "$PARALLEL_CHAPTERS"
  --max-thread-num "$MAX_THREAD_NUM"
  --max-conv-turns-per-chapter "$MAX_CONV_TURNS_PER_CHAPTER"
  --max-model-output-tokens "$MAX_MODEL_OUTPUT_TOKENS"
  --cache-mode "$CACHE_MODE"
  --qwen-strong-thinking "$QWEN_STRONG_THINKING"
  --qwen-strong-thinking-budget "$QWEN_STRONG_THINKING_BUDGET"
  --qwen-weak-thinking "$QWEN_WEAK_THINKING"
)

if [[ -n "${START_INDEX:-}" ]]; then
  args+=(--start-index "$START_INDEX")
fi

if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "$LIMIT")
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

if [[ "$POLISH_ARTICLE" == "1" ]]; then
  args+=(--do-polish-article)
fi

echo "Running Qwen textbook benchmark"
echo "  benchmark: $BENCHMARK_PATH"
echo "  output: $OUTPUT_DIR"
echo "  article_model: $STRONG_MODEL"
echo "  aux_model: $WEAK_MODEL"
echo "  temperature: $TEMPERATURE"
echo "  top_p: $TOP_P"
echo "  parallel_chapters: $PARALLEL_CHAPTERS"
echo "  max_thread_num: $MAX_THREAD_NUM"
echo "  cache_mode: $CACHE_MODE"
echo "  polish_article: $POLISH_ARTICLE"

python examples/textbook_benchmark/run_textbook_baselines.py "${args[@]}"
