#!/usr/bin/env bash
set -euo pipefail

# Edit these values on the target server.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"

NUM_GPUS="${NUM_GPUS:-2}"
DATA_DIR="${DATA_DIR:-data/nq_search}"
BASE_MODEL="${BASE_MODEL:-models/parallel_search_qwen25_3b_step900}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
EITR_ENABLED="${EITR_ENABLED:-true}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eitr-nq-gate-c-smoke}"
WANDB_PROJECT="${WANDB_PROJECT:-EITR-Search-Agent}"

RAY_TMPDIR="${RAY_TMPDIR:-ray_tmp/eitr_gate_c_smoke}"
RAY_SPILL_DIR="${RAY_SPILL_DIR:-ray_spill/eitr_gate_c_smoke}"
mkdir -p "$RAY_TMPDIR" "$RAY_SPILL_DIR" "trajectory/eitr_gate_c"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_data_num=32 \
    data.val_data_num=64 \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_start_length=2048 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=false \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.eitr.enabled="$EITR_ENABLED" \
    actor_rollout_ref.actor.eitr.probe_source=online_same_state \
    actor_rollout_ref.actor.eitr.probe_count=4 \
    actor_rollout_ref.actor.eitr.probe_oversample=2 \
    actor_rollout_ref.actor.eitr.max_query_tokens=96 \
    actor_rollout_ref.actor.eitr.probe_seed=20260805 \
    actor_rollout_ref.actor.eitr.max_action_tokens=128 \
    actor_rollout_ref.actor.eitr.max_probe_prompt_tokens=2304 \
    actor_rollout_ref.actor.eitr.retrieval_score_temperature=0.1 \
    actor_rollout_ref.actor.eitr.min_state_coverage=0.5 \
    actor_rollout_ref.actor.eitr.target_js=0.01 \
    actor_rollout_ref.actor.eitr.initial_beta=0.1 \
    actor_rollout_ref.actor.eitr.dual_lr=0.05 \
    actor_rollout_ref.actor.eitr.beta_max=10.0 \
    actor_rollout_ref.actor.eitr.log_ratio_clip=10.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    trainer.logger="['wandb']" \
    +trainer.val_before_train=false \
    +trainer.val_only=false \
    trainer.n_gpus_per_node="$NUM_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=11 \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    trainer.log_best_trajectory=true \
    trainer.trajectory_log_dir=trajectory/eitr_gate_c \
    +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
    +ray_kwargs.ray_init.object_spilling_directory="$RAY_SPILL_DIR" \
    max_turns=4 \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=3 \
    retriever.max_queries_per_turn=1 \
    2>&1 | tee "$EXPERIMENT_NAME.log"
