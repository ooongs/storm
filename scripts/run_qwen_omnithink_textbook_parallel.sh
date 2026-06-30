#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export OUTPUT_DIR="${OUTPUT_DIR:-results/qwen_omnithink_textbook_parallel}"
export MODEL="${MODEL:-qwen3.7-plus}"
if [[ -z "${AUX_MODEL:-}" ]]; then
  export AUX_MODEL="${WEAK_MODEL:-qwen3.6-flash}"
fi
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.9}"

export PARALLEL_CHAPTERS="${PARALLEL_CHAPTERS:-8}"
export MAX_THREAD_NUM="${MAX_THREAD_NUM:-1}"
export QWEN_THINKING="${QWEN_THINKING:-disable}"

exec scripts/run_qwen_omnithink_textbook.sh
