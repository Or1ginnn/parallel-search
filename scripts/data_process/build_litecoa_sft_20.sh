#!/usr/bin/env bash
set -e

export DEEPSEEK_API_KEY="PUT_YOUR_DEEPSEEK_KEY_HERE"
export DEEPSEEK_MODEL="deepseek-v4-pro"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"

RETRIEVER_URL="http://127.0.0.1:8000/retrieve"

python scripts/data_process/build_litecoa_sft.py \
  --input_jsonl docs/phase2/phase2_smoke_samples.jsonl \
  --output data/litecoa_sft/litecoa_sft_20.jsonl \
  --rejected_output data/litecoa_sft/litecoa_sft_20_rejected.jsonl \
  --report_output data/litecoa_sft/litecoa_sft_20_report.json \
  --retriever_url "$RETRIEVER_URL" \
  --target_count 20 \
  --max_candidates 50 \
  --max_turns 2 \
  --max_queries_per_turn 3 \
  --topk 3
