#!/usr/bin/env bash
set -euo pipefail

# FinQA Phase 5 V2 smoke: Step200 initialization, 64 questions, 10 updates.
# The report-aware E5 retriever must already be serving on RETRIEVER_URL.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_memory_usage_threshold=0.99

DATA_DIR="${DATA_DIR:-/mnt/data1/zar/finance/data/finance_finqa/grpo_nocalc}"
BASE_MODEL="${BASE_MODEL:-/mnt/data1/zar/search-1/Search-R1/verl_checkpoints/finqa-phase4-grpo-v3-stable/actor/global_step_200}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-null}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-finqa-phase5-v2-smoke}"
WAND_PROJECT="${WAND_PROJECT:-Finance_Agent}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-verl_checkpoints/$EXPERIMENT_NAME}"
TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-/mnt/data1/zar/finance/trajectory/finqa_phase5_v2_smoke}"

RAY_TMPDIR="${RAY_TMPDIR:-/mnt/data1/zar/finance/ray_tmp/finqa_phase5_v2_smoke}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-/mnt/data1/zar/finance/ray_spill/finqa_phase5_v2_smoke}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
NUM_GPUS="${NUM_GPUS:-2}"
ROLLOUT_N_AGENT="${ROLLOUT_N_AGENT:-5}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
TRAIN_DATA_NUM="${TRAIN_DATA_NUM:-64}"
VAL_DATA_NUM="${VAL_DATA_NUM:-64}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-1500}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-16}"
LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-32}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.285}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.005}"
PPO_CLIP_RATIO="${PPO_CLIP_RATIO:-0.2}"
MAX_PPO_KL="${MAX_PPO_KL:-0.1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:-20}"

mkdir -p "$RAY_TMPDIR" "$RAY_SPILL_DIR" "$TRAJECTORY_LOG_DIR" "$CHECKPOINT_DIR"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/dev.parquet" \
    data.seed=42 \
    data.train_data_num="$TRAIN_DATA_NUM" \
    data.val_data_num="$VAL_DATA_NUM" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_start_length=2048 \
    data.max_prompt_length=8192 \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.max_obs_length="$MAX_OBS_LENGTH" \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.resume_path="$RESUME_FROM_CHECKPOINT" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.clip_ratio="$PPO_CLIP_RATIO" \
    actor_rollout_ref.actor.max_ppo_kl="$MAX_PPO_KL" \
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
    actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    reward_model.litecoa_reward=true \
    reward_model.litecoa_answer_present_bonus=0.05 \
    reward_model.litecoa_no_generated_information_bonus=0.05 \
    reward_model.litecoa_valid_search_bonus=0.05 \
    reward_model.finqa_v2_reward=true \
    reward_model.finqa_retrieval_coverage_bonus=0.05 \
    reward_model.finqa_parallel_retrieval_gain_bonus=0.05 \
    reward_model.finqa_near_miss_bonus=0.05 \
    reward_model.finqa_near_miss_max_relative_error=0.05 \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent="$ROLLOUT_N_AGENT" \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.train_num_examine=1 \
    trainer.project_name="$WAND_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.log_best_trajectory=true \
    trainer.trajectory_log_dir="$TRAJECTORY_LOG_DIR" \
    trainer.resume_from_checkpoint="$RESUME_FROM_CHECKPOINT" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
    +ray_kwargs.ray_init.object_spilling_directory="$RAY_SPILL_DIR" \
    max_turns=3 \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk="$RETRIEVER_TOPK" \
    retriever.max_queries_per_turn=3 \
    +retriever.use_report_scope=true \
    2>&1 | tee "$EXPERIMENT_NAME.log"
