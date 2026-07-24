#!/usr/bin/env bash
set -euo pipefail

# Phase 5 smoke: 64 FinQA train questions, 4 candidates each, 10 GRPO updates.
# Run the all-corpus E5 retriever on GPU 3 before launching this script.

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_memory_usage_threshold=0.99

DATA_DIR="data/finance_finqa/grpo"
BASE_MODEL="models/finance_finqa_qwen25_3b_sft_merged"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-finqa-litecoa-grpo-qwen2.5-3b-smoke}"
WAND_PROJECT="Finance_Agent"
TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-trajectory/finance_finqa_grpo}"

RAY_TMPDIR="${RAY_TMPDIR:-ray_tmp/finqa_grpo_smoke}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-ray_spill/finqa_grpo_smoke}"
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"

TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-64}"
VAL_DATA_NUM="${VAL_DATA_NUM:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"
LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-20}"

mkdir -p "$RAY_TMPDIR" "$RAY_SPILL_DIR" "$TRAJECTORY_LOG_DIR" "verl_checkpoints/$EXPERIMENT_NAME"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/dev.parquet" \
    data.train_data_num="$TRAIN_DATA_NUM" \
    data.val_data_num="$VAL_DATA_NUM" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_start_length=2048 \
    data.max_prompt_length=8192 \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.max_obs_length=1500 \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size="$PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="$LOGPROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size="$LOGPROB_MICRO_BATCH_SIZE" \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    reward_model.litecoa_reward=true \
    reward_model.litecoa_plan_once_bonus=0.0 \
    reward_model.litecoa_answer_present_bonus=0.0 \
    reward_model.litecoa_no_generated_information_bonus=0.0 \
    reward_model.litecoa_evidence_hit_bonus=0.0 \
    reward_model.litecoa_valid_search_bonus=0.0 \
    reward_model.litecoa_parallel_evidence_bonus=0.05 \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=4 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.train_num_examine=1 \
    trainer.project_name="$WAND_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.log_best_trajectory=true \
    trainer.trajectory_log_dir="$TRAJECTORY_LOG_DIR" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
    +ray_kwargs.ray_init.object_spilling_directory="$RAY_SPILL_DIR" \
    max_turns=3 \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=3 \
    retriever.max_queries_per_turn=3 \
    +retriever.use_report_scope=true \
    2>&1 | tee "$EXPERIMENT_NAME.log"
