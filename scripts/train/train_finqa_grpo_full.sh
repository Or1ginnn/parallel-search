#!/usr/bin/env bash
set -euo pipefail

# Full FinQA LiteCoA-GRPO run. The shared script holds the actual veRL command.
export EXPERIMENT_NAME="finqa-litecoa-grpo-qwen2.5-3b-full"
export TRAJECTORY_LOG_DIR="trajectory/finance_finqa_grpo"
export RAY_TMPDIR="/mnt/data/suiqiuyi/rf"
export RAY_SPILL_DIR="/mnt/data/suiqiuyi/rs"

export TRAIN_DATA_NUM=null
export VAL_DATA_NUM=null
export TRAIN_BATCH_SIZE=4
export VAL_BATCH_SIZE=16
export MAX_RESPONSE_LENGTH=384
export PPO_MINI_BATCH_SIZE=4
export PPO_MICRO_BATCH_SIZE=4
export LOGPROB_MICRO_BATCH_SIZE=4
export TOTAL_EPOCHS=10
export TOTAL_TRAINING_STEPS=1005
export TEST_FREQ=50
export SAVE_FREQ=100

exec bash scripts/train/train_finqa_grpo_smoke.sh
