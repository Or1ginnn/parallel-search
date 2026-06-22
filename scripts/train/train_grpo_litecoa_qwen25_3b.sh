#!/usr/bin/env bash
set -euo pipefail

# LiteCoA-GRPO full run.
# This is intended for long training; stop manually after W&B reward/validation stabilizes.

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_memory_usage_threshold=0.99

DATA_DIR="data/nq_search_litecoa"
BASE_MODEL="outputs/sft/litecoa_lora_qwen25_3b_full_merged"
EXPERIMENT_NAME="nq-litecoa-grpo-qwen2.5-3b"
WAND_PROJECT="Search-R1"
TRAJECTORY_LOG_DIR="trajectory/litecoa_grpo"

RAY_TMPDIR="ray_tmp/litecoa_grpo"
RAY_SPILL_DIR="ray_spill/litecoa_grpo"

TRAIN_DATA_NUM=null
VAL_DATA_NUM=null
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=32

MAX_START_LENGTH=2048
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=500
MAX_OBS_LENGTH=1000
MAX_TURNS=3

TOPK=3
MAX_QUERIES_PER_TURN=3
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"

TOTAL_TRAINING_STEPS=1005
TEST_FREQ=50
SAVE_FREQ=100

mkdir -p "$RAY_TMPDIR" "$RAY_SPILL_DIR" "$TRAJECTORY_LOG_DIR" "verl_checkpoints/$EXPERIMENT_NAME"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_data_num=$TRAIN_DATA_NUM \
    data.val_data_num=$VAL_DATA_NUM \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.max_start_length=$MAX_START_LENGTH \
    data.max_obs_length=$MAX_OBS_LENGTH \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.project_name="$WAND_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.log_best_trajectory=true \
    trainer.trajectory_log_dir="$TRAJECTORY_LOG_DIR" \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
    +ray_kwargs.ray_init.object_spilling_directory="$RAY_SPILL_DIR" \
    max_turns=$MAX_TURNS \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=$TOPK \
    retriever.max_queries_per_turn=$MAX_QUERIES_PER_TURN \
    2>&1 | tee "$EXPERIMENT_NAME.log"
