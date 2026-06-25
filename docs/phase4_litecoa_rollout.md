# Phase 4：LiteCoA-GRPO Rollout 与训练闭环报告

## 1. 阶段目标

Phase 4 的目标是把 Search-R1 / VeRL 的训练链路改造成能支持 LiteCoA 格式：

```text
LiteCoA prompt
-> model rollout
-> <plan> / <search> / <answer>
-> multi-query retriever
-> <information>
-> rule reward
-> GRPO actor update
-> validation
```

本阶段关注工程闭环，不追求最终准确率最优。准确率和稳定性改进进入 Phase 5 reward 设计。

## 2. 改造范围

本阶段修改：

```text
search_r1/llm_agent/generation.py
verl/trainer/ppo/ray_trainer.py
verl/trainer/config/ppo_trainer.yaml
scripts/data_process/nq_search.py
scripts/data_process/qa_search_train_merge.py
scripts/data_process/qa_search_test_merge.py
verl/utils/reward_score/qa_em.py
scripts/train/train_grpo_litecoa_smoke.sh
scripts/train/train_grpo_litecoa_qwen25_3b.sh
```

本阶段不改：

```text
GRPO loss
advantage estimator
KL loss 公式
VeRL 原生 rollout worker
tensor_helper padding/mask 逻辑
```

Search-R1 的 agent loop 在 `LLMGenerationManager` 中实现。VeRL 的 `RayPPOTrainer` 负责调用它，但不理解 `<search>` / `<information>` 的语义。

## 3. 原 Search-R1 交互

原 Search-R1 实际环境只处理两个 action：

```text
<search>query</search>
<answer>answer</answer>
```

`<think>` 在 prompt 中被要求，但 parser 和 reward 不强制检查。原单 query search 流程是：

```text
model output
-> postprocess_predictions()
-> extract <search>...</search>
-> 把整个 search content 当成一个 query
-> batch_search([query])
-> 拼成 <information>retrieved docs</information>
-> 继续生成
```

如果旧代码遇到：

```text
<search>q1 || q2 || q3</search>
```

它会把整段 `"q1 || q2 || q3"` 当成一个 query，不符合 LiteCoA 的并行检索目标。

## 4. LiteCoA Rollout 改造

LiteCoA 支持一次 search action 发出多个 query：

```text
<search>q1 || q2 || q3</search>
```

解析规则：

```text
1. 用 "||" 分割 query。
2. 去掉首尾空格并压缩内部空白。
3. 丢弃空 query。
4. 丢弃包含 "<" 或 ">" 的 query。
5. 同一轮内 query 去重。
6. 每轮最多保留 max_queries_per_turn 个 query，默认 3。
```

batch 检索方式：

```text
sample A: [q1, q2]
sample B: [q3]
sample C: [q4, q5, q6]

flat_queries = [q1, q2, q3, q4, q5, q6]
retriever.batch_search(flat_queries)
再按每条样本 query 数切回去
```

observation 拼回格式：

```text
<information>
[Query] q1
Doc 1(Title: ...)
Doc 2(Title: ...)

[Query] q2
Doc 1(Title: ...)
Doc 2(Title: ...)
</information>
```

这样 Phase 4 rollout 与 Phase 2 SFT 数据、Phase 3 vLLM eval 轨迹格式保持一致。

## 5. 数据与 Prompt

`scripts/data_process/nq_search.py` 新增：

```text
template_type=litecoa
```

生成：

```text
data/nq_search_litecoa/train.parquet
data/nq_search_litecoa/test.parquet
```

数据检查结果：

```text
train.parquet: 79168 rows
test.parquet: 3610 rows
prompt role: user only
system role: false
<answer> Beijing </answer>: false
prompt prefix: You are a search-augmented reasoning agent.
reward_model: rule + golden_answers
```

LiteCoA prompt 与 Phase 3 SFT 最终训练格式对齐：单 user prompt，无 system message，无 Beijing answer 示例。

`qa_search_train_merge.py` 和 `qa_search_test_merge.py` 也加入 `template_type=litecoa`，用于后续多 QA 数据集构造。

## 6. Reward 兼容

原 Search-R1 reward 曾依赖 prompt 中的：

```text
<answer> Beijing </answer>
```

因此旧逻辑需要至少两个 `<answer>` 才取最后一个。LiteCoA prompt 不再放示例 answer，所以 `verl/utils/reward_score/qa_em.py` 改为：

```text
没有 <answer> -> None
有一个或多个 <answer> -> 取最后一个
```

该改动既支持 LiteCoA 单 answer 输出，也兼容原 Search-R1 prompt。

## 7. Smoke 实验

### 7.1 目标

Smoke 只验证训练链路是否闭环：

```text
LiteCoA prompt
-> SFT merged model
-> LiteCoA rollout
-> multi-query retriever
-> <information>
-> <answer>
-> rule reward
-> actor update
-> validation
```

### 7.2 配置

```text
data.train_data_num=64
data.val_data_num=64
data.train_batch_size=64
data.val_batch_size=32
data.max_start_length=2048
data.max_prompt_length=6144
data.max_response_length=500
data.max_obs_length=500
max_turns=3
retriever.topk=3
retriever.max_queries_per_turn=3
actor_rollout_ref.rollout.n_agent=5
actor_rollout_ref.rollout.temperature=1
trainer.total_training_steps=10
trainer.test_freq=5
trainer.save_freq=20
```

第一次 smoke 因 Ray 主机内存超过阈值被杀；后续仅调整：

```text
actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
Ray temp/spill directory moved to project data disk
```

没有降低 `n_agent=5`，没有减少 smoke 数据量。

### 7.3 结果

最终 smoke 跑完 10-step GRPO，并完成 validation。

本地日志最终结果：

```text
Final validation metrics: {'val/test_score/nq': 0.265625}
```

W&B run：

```text
project: Search-R1
run id: 4ih8xv0y
```

W&B 状态显示 `crashed`，但本地日志显示训练和最终 validation 已完成；因此本次 smoke 以本地日志为准。

### 7.4 行为验证

smoke log 中出现真实 multi-query 搜索：

```text
<search>date of first space rendezvous gemini 6 and gemini 7 || gemini 6 and gemini 7 first rendezvous date</search>
```

对应 observation 按 query 分块：

```text
<information>
[Query] date of first space rendezvous gemini 6 and gemini 7
Doc 1(...)
Doc 2(...)
Doc 3(...)

[Query] gemini 6 and gemini 7 first rendezvous date
Doc 1(...)
Doc 2(...)
Doc 3(...)
</information>
```

这证明 Phase 4 不只是改 prompt，而是真正把多 query 送入 retriever 并回填 `<information>`。

### 7.5 Smoke 发现的问题

Smoke 使用：

```text
data.max_obs_length=500
```

日志中多次出现：

```text
[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG
```

LiteCoA 一次 search 可能包含多个 query，`<information>` 比原 Search-R1 单 query 更长。正式训练建议使用：

```text
data.max_prompt_length=8192
data.max_obs_length=1000
data.max_start_length=2048
data.max_response_length=500
max_turns=3
```

Smoke 还发现 invalid action 和 answer 后异常续写，提示 Phase 5 需要更强 reward 约束。

## 8. Full Run 实验

### 8.1 目标

Full run 验证：

```text
SFT cold-start LiteCoA model
-> NQ LiteCoA GRPO 数据
-> answer-only GRPO reward
-> 长训练过程中的稳定性和准确率变化
```

本次 full run 仍使用原 answer EM reward，不使用 Phase 5 reward。

### 8.2 归档位置

```text
output/phase4_full_grpo/
```

主要文件：

```text
output/phase4_full_grpo/nq-litecoa-grpo-qwen2.5-3b-bs64-warmup005-fullval.log
output/phase4_full_grpo/nq-litecoa-grpo-qwen2.5-3b-bs64-warmup005-fullval.best_trajectory.jsonl
output/phase4_full_grpo/nq-litecoa-grpo-qwen2.5-3b-bs64-warmup005-fullval.archive_summary.md
output/phase4_full_grpo/nq-litecoa-grpo-qwen2.5-3b-bs64-warmup005-fullval/train_grpo_litecoa_qwen25_3b_bs64_warmup005_fullval.sh
```

W&B run：

```text
https://wandb.ai/sun19150956991-beijing-university-of-posts-and-telecommu/Search-R1/runs/z8ihenbu
```

### 8.3 配置

```text
BASE_MODEL=outputs/sft/litecoa_lora_qwen25_3b_full_merged
DATA_DIR=data/nq_search_litecoa
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=32
VAL_DATA_NUM=null
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=500
MAX_OBS_LENGTH=1000
MAX_TURNS=3
TOPK=3
MAX_QUERIES_PER_TURN=3
TOTAL_TRAINING_STEPS=1005
TEST_FREQ=50
SAVE_FREQ=100
actor lr=1e-6
lr_warmup_steps_ratio=0.05
kl_loss_coef=0.001
n_agent=5
temperature=1
```

`VAL_DATA_NUM=null` 表示使用完整 NQ test set validation。

### 8.4 结果

归档 summary：

```text
archived_at: 2026-06-24 00:02:53 CST
last_seen_step: 119
best_trajectory_lines: 119
error_matches_in_log: 0
W&B state: crashed
```

W&B summary 末尾状态：

```text
_step: 118
critic/score/mean: 0
critic/rewards/mean: 0
val/test_score/nq: 0.24860491071428573
env/finish_ratio: 0
env/ratio_of_valid_action: 0
env/number_of_valid_search: 0
response_length/clip_ratio: 1
actor/kl_loss: NaN
actor/ppo_kl: NaN
actor/pg_loss: NaN
actor/grad_norm: NaN
actor/lr: 1e-6
actor/kl_coef: 0.001
```

最后 5 条 best trajectory 均为：

```text
score: 0.0
turns: 4
valid_actions: 0
valid_searches: 0
response_length: 411
score_mean: 0.0
reward_mean: 0.0
```

日志尾部反复出现：

```text
ACTIVE_TRAJ_NUM: [320, 320, 320, 320, 320]
My previous action is invalid...
```

早期 best trajectory 仍有正常高分样本：

```text
step 1: score=1.0, valid_actions=2, valid_searches=1
step 60: score=1.0, valid_actions=2, valid_searches=1
```

但后期完全坍塌：

```text
step 115-119: score=0.0, valid_actions=0, valid_searches=0
```

## 9. Phase 4 结论

Phase 4 工程目标完成：

```text
LiteCoA prompt 可被 GRPO 数据加载
SFT merged model 可作为 GRPO actor 初始模型
rollout 可生成 <plan>
rollout 可解析 <search>q1 || q2</search>
retriever 可被多 query 调用
<information> 可按 [Query] 分块返回
reward 可从单个 <answer> 抽取答案
GRPO 可以完成 actor update
validation 可以跑完
```

但 Phase 4 full run 未达预期：

```text
仅靠 answer-only reward 不足以稳定约束 LiteCoA。
模型后期会进入无效 action / max-length 输出状态。
最终 score_mean=0, finish_ratio=0, valid_action_ratio=0, response_length/clip_ratio=1, actor metrics=NaN。
```

因此 Phase 4 的结论是：

```text
工程闭环成功；
answer-only LiteCoA-GRPO 训练失败；
必须进入 Phase 5 reward 设计。
```

## 10. 下一步

Phase 5 需要重点解决：

```text
1. 无效 action / max-length 输出不能只靠 answer EM 间接惩罚。
2. soft bonus 可能提高 reward 密度，但不一定阻止格式坍塌。
3. LiteCoA 需要更明确的 hard reward 或更简化的 prompt/action space。
```

后续文档：

```text
docs/phase5_litecoa_reward.md
```
