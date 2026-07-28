#!/usr/bin/env bash
set -euo pipefail

# FinQA Phase 5 V2: start a new experiment from the best Phase 4 Step200 policy.
# RESUME_FROM_CHECKPOINT is only for resuming a checkpoint created by this V2 run.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"

DATA_DIR="${DATA_DIR:-/mnt/data1/zar/finance/data/finance_finqa/grpo_nocalc}"
BASE_MODEL="${BASE_MODEL:-/mnt/data1/zar/search-1/Search-R1/verl_checkpoints/finqa-phase4-grpo-v3-stable/actor/global_step_200}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-null}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-finqa-phase5-v2-step200}"
WAND_PROJECT="${WAND_PROJECT:-Finance_Agent}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-verl_checkpoints/$EXPERIMENT_NAME}"
TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-/mnt/data1/zar/finance/trajectory/finqa_phase5_v2}"
RAY_TMPDIR="${RAY_TMPDIR:-/mnt/data1/zar/finance/ray_tmp/finqa_phase5_v2}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-/mnt/data1/zar/finance/ray_spill/finqa_phase5_v2}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
VAL_DATA_NUM="${VAL_DATA_NUM:-320}"
ROLLOUT_N_AGENT="${ROLLOUT_N_AGENT:-5}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
TEST_FREQ="${TEST_FREQ:-25}"
SAVE_FREQ="${SAVE_FREQ:-25}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"

mkdir -p "$RAY_TMPDIR" "$RAY_SPILL_DIR" "$TRAJECTORY_LOG_DIR" "$CHECKPOINT_DIR"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/dev.parquet" \
    data.seed=42 \
    data.train_data_num=null \
    data.val_data_num="$VAL_DATA_NUM" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_start_length=2048 \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.max_obs_length=1500 \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.resume_path="$RESUME_FROM_CHECKPOINT" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.max_ppo_kl=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n_agent="$ROLLOUT_N_AGENT" \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.state_masking=true \
    reward_model.litecoa_reward=true \
    reward_model.litecoa_answer_present_bonus=0.05 \
    reward_model.litecoa_no_generated_information_bonus=0.05 \
    reward_model.litecoa_valid_search_bonus=0.05 \
    reward_model.finqa_v2_reward=true \
    reward_model.finqa_retrieval_coverage_bonus=0.05 \
    reward_model.finqa_parallel_retrieval_gain_bonus=0.05 \
    reward_model.finqa_near_miss_bonus=0.05 \
    reward_model.finqa_near_miss_max_relative_error=0.05 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.train_num_examine=1 \
    trainer.project_name="$WAND_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.log_best_trajectory=true \
    trainer.trajectory_log_dir="$TRAJECTORY_LOG_DIR" \
    trainer.resume_from_checkpoint="$RESUME_FROM_CHECKPOINT" \
    trainer.total_epochs=10 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
    +ray_kwargs.ray_init.object_spilling_directory="$RAY_SPILL_DIR" \
    max_turns=3 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    retriever.max_queries_per_turn=3 \
    +retriever.use_report_scope=true \
    2>&1 | tee "$EXPERIMENT_NAME.log"
