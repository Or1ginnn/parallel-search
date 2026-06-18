#!/usr/bin/env bash
set -e

# Edit training settings in:
#   configs/sft/litecoa_lora_qwen25_3b_smoke.yaml
#   configs/sft/litecoa_lora_qwen25_3b_full.yaml

CONFIG=configs/sft/litecoa_lora_qwen25_3b_smoke.yaml
# CONFIG=configs/sft/litecoa_lora_qwen25_3b_full.yaml

llamafactory-cli train "$CONFIG"
