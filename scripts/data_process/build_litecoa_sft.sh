#!/usr/bin/env bash
set -e

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY in your shell before running this script.}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
export PYTHONUNBUFFERED=1

TARGET_COUNT="${1:-20}"
MAX_CANDIDATES="${2:-$((TARGET_COUNT * 3))}"
OUTPUT_PREFIX="${3:-data/litecoa_sft/litecoa_${TARGET_COUNT}/litecoa_sft_${TARGET_COUNT}}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
API_TIMEOUT="${API_TIMEOUT:-60}"
API_RETRIES="${API_RETRIES:-1}"

INPUT_ARGS=(--input_jsonl "")
if [[ "$TARGET_COUNT" == "20" && -f trajectory/phase2_smoke/phase2_smoke_samples.jsonl ]]; then
  INPUT_ARGS=(--input_jsonl trajectory/phase2_smoke/phase2_smoke_samples.jsonl)
elif [[ -f data/nq_search/train.parquet ]]; then
  INPUT_ARGS=(--input_jsonl "" --input_parquet data/nq_search/train.parquet)
fi

python scripts/data_process/build_litecoa_sft.py \
  "${INPUT_ARGS[@]}" \
  --data_source nq \
  --split train \
  --output "${OUTPUT_PREFIX}.jsonl" \
  --rejected_output "${OUTPUT_PREFIX}_rejected.jsonl" \
  --report_output "${OUTPUT_PREFIX}_report.json" \
  --retriever_url "$RETRIEVER_URL" \
  --target_count "$TARGET_COUNT" \
  --max_candidates "$MAX_CANDIDATES" \
  --max_turns 2 \
  --max_queries_per_turn 3 \
  --topk 3 \
  --api_timeout "$API_TIMEOUT" \
  --api_retries "$API_RETRIES"
