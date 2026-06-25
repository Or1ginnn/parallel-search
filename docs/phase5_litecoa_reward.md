# Phase 5：LiteCoA Reward 实验总结

## 目标

Phase 4 已经证明 LiteCoA rollout 链路能跑通，但只用最终 answer EM 奖励时，3B 模型容易出现格式漂移、无效 action 增多、并行 query 不稳定等问题。

Phase 5 的目标是验证 LiteCoA 是否需要额外 reward 约束，并比较不同 reward 版本对训练稳定性的影响。

本阶段已经跑过三类设置：

```text
Phase 4 answer-only reward:
  工程链路能跑通，但 full run 后期格式坍塌。

Phase 5 soft reward:
  加小额正向 shaping，但没有 hard zero，后期仍格式坍塌。

Phase 5 hard reward noSFT:
  从 Qwen2.5-3B base 直接 GRPO，加入 hard zero 和 query/evidence bonus，
  训练保持稳定，validation 最好达到 0.4875。
```

## Reward 实现

训练 reward：

```text
reward = answer_em
       + plan_once_bonus * plan_once
       + answer_present_bonus * answer_present
       + no_generated_information_bonus * no_generated_information
       + evidence_hit_bonus * evidence_hit
       + valid_search_bonus * valid_search
       + parallel_evidence_bonus * parallel_evidence_hit
```

验证 reward 仍保持原 answer EM，不加 shaping bonus，保证 `val/test_score/nq` 可以和 Phase 4 及 Search-R1 baseline 对比。

hard zero 规则在 `RewardManager` 里执行：

```text
response 打满 max_response_length -> reward = 0
没有 valid action -> reward = 0
没有 <answer> -> reward = 0
模型自己生成 <information> -> reward = 0
```

## 关键边界

格式奖励只看模型生成的 response，不看 prompt。因为 LiteCoA prompt 本身包含 `<plan>`、`<search>`、`<information>`、`<answer>` 这些规则说明，不能把 prompt 里的 tag 计入 reward。

实现里使用 rollout 返回的 `info_mask`：

- `attention_mask` response 段：完整 response，包括模型生成内容和 retriever 插入的 `<information>`。
- `info_mask` response 段为 1：模型生成 token，用于判断 `<plan>`、`<search>`、`<answer>` 和是否生成了 `<information>`。
- `info_mask` response 段为 0：retriever observation token，用于判断 evidence 是否命中 gold answer。

这样可以避免两类污染：

1. prompt 里的标签不会让 `plan_once`、`answer_present` 等格式奖励虚高。
2. 模型自己生成的 `<information>` 不会让 `evidence_hit` 虚高。

## 当前实现文件

- `verl/utils/reward_score/litecoa_qa.py`
- `verl/trainer/main_ppo.py`
- `verl/trainer/config/ppo_trainer.yaml`
- `scripts/train/train_grpo_litecoa_qwen25_3b.sh`
- `scripts/train/train_grpo_litecoa_smoke.sh`

LiteCoA 启动脚本会设置：

```text
reward_model.litecoa_reward=true
```

默认配置中该开关为 `False`，因此原 Search-R1 训练入口不会自动使用 LiteCoA shaping reward。

当前仓库里的正式脚本使用更简化的正向 reward：

```text
PLAN_ONCE_BONUS=0.0
ANSWER_PRESENT_BONUS=0.0
NO_GENERATED_INFORMATION_BONUS=0.0
EVIDENCE_HIT_BONUS=0.0
VALID_SEARCH_BONUS=0.0
PARALLEL_EVIDENCE_BONUS=0.05
```

也就是：

```text
reward = answer_em + 0.05 * parallel_evidence_hit
```

并继续保留 hard zero 规则。

## Soft Reward 实验归档

本节记录 Phase 5 第一版 soft reward 的 SFT-init 训练结果。该实验使用的是正向 shaping reward，没有 hard zero 约束。

归档目录：

```text
output/phase5_sftbased_reward/nq-litecoa-grpo-qwen2.5-3b/
```

主要文件：

```text
phase5_litecoa_grpo_kl005_val320.log
nq-litecoa-grpo-qwen2.5-3b.log
nq-litecoa-grpo-qwen2.5-3b.best_trajectory.jsonl
train_grpo_litecoa_qwen25_3b.sh
```

W&B run：

```text
https://wandb.ai/sun19150956991-beijing-university-of-posts-and-telecommu/Search-R1/runs/6x6vz8at
```

### 配置

归档脚本关键配置：

```text
BASE_MODEL=outputs/sft/litecoa_lora_qwen25_3b_full_merged
DATA_DIR=data/nq_search_litecoa
TRAIN_BATCH_SIZE=64
VAL_DATA_NUM=320
VAL_BATCH_SIZE=32
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
lr_warmup_steps_ratio=0.285
kl_loss_coef=0.005
n_agent=5
temperature=1
```

Reward 系数：

```text
PLAN_ONCE_BONUS=0.05
ANSWER_PRESENT_BONUS=0.05
NO_GENERATED_INFORMATION_BONUS=0.05
EVIDENCE_HIT_BONUS=0.05
VALID_SEARCH_BONUS=0.03
PARALLEL_EVIDENCE_BONUS=0.03
```

最大训练 reward 为：

```text
1.26
```

validation 仍只计算原 answer EM。

### 运行结果

归档 summary：

```text
archived_at: 2026-06-24 13:41:04 CST
last_seen_step: epoch 0, step 220
best_trajectory_lines: 255
error_matches_in_tee_log: 0
W&B state: crashed
```

W&B summary 末尾状态：

```text
_step: 218
critic/score/mean: 0.05000000447034836
critic/rewards/mean: 0.05000000447034836
train/answer_em_mean: 0
val/test_score/nq: 0
env/finish_ratio: 0
env/ratio_of_valid_action: 0
env/number_of_valid_search: 0
response_length/clip_ratio: 1
actor/kl_loss: NaN
actor/ppo_kl: NaN
actor/pg_loss: NaN
actor/grad_norm: NaN
actor/lr: 7.622377622377621e-07
actor/kl_coef: 0.005
```

best trajectory 早期和中期仍可出现高分：

```text
step 1: score=1.0, valid_actions=2, valid_searches=1
step 92: score=1.26, valid_actions=2, valid_searches=1
```

但最后 5 条 best trajectory 均退化为：

```text
score: 0.05
turns: 4
valid_actions: 0
valid_searches: 0
response_length: 411
score_mean: 0.05000000447034836
reward_mean: 0.05000000447034836
```

日志尾部同样出现：

```text
ACTIVE_TRAJ_NUM: [320, 320, 320, 320, 320]
My previous action is invalid...
```

说明模型后期依然进入所有 trajectory 均 active、无有效 action 的状态。

### 结论

Soft reward 第一版确实提高了早期 reward 密度，并能在中期产生 `score=1.26` 的完整高分轨迹；但它仍不能阻止后期格式坍塌。

关键问题是：

```text
小额正向 bonus 只能奖励好行为，不能强力排除坏轨迹。
```

在后期无效 action 状态下，模型仍能通过 `no_generated_information_bonus=0.05` 获得非零 reward，因此最后的 `score_mean` 固定在约 `0.05`，而不是回到 0。这会让完全无效的 trajectory 仍保留一点正向信号，不利于稳定训练。

因此后续 Phase 5.2 需要 hard reward：

```text
response 打满 max_response_length -> reward = 0
没有 valid action -> reward = 0
没有 <answer> -> reward = 0
模型生成 <information> -> reward = 0
```

同时应简化正向 reward，把训练目标收敛到：

```text
answer_em + parallel_evidence_bonus
```

这也是后续 hard reward 版本的设计来源。

## Hard Reward noSFT 实验归档

本节记录 Phase 5 hard reward 版本的完整训练结果。该实验不从 Phase 3 SFT 模型开始，而是直接使用 Qwen2.5-3B base model 做 GRPO。

归档目录：

```text
output/phase5_hard_reward_noSFT/
```

主要文件：

```text
phase5_litecoa_grpo_base_hard_topk2_lr5e7.log
nq-litecoa-grpo-qwen2.5-3b-base-hard.log
nq-litecoa-grpo-qwen2.5-3b-base-hard.best_trajectory.jsonl
train_grpo_litecoa_qwen25_3b.sh
tail/phase5_litecoa_grpo_base_hard_topk2_lr5e7.last300.log
tail/nq-litecoa-grpo-qwen2.5-3b-base-hard.last10_trajectory.jsonl
```

W&B run：

```text
https://wandb.ai/sun19150956991-beijing-university-of-posts-and-telecommu/Search-R1/runs/5xg7t8nj
```

### 配置

归档脚本关键配置：

```text
BASE_MODEL=/mnt/data1/zar/search-1/Search-R1/hf_cache/Qwen2.5-3B
DATA_DIR=data/nq_search_litecoa
TRAIN_BATCH_SIZE=64
VAL_DATA_NUM=320
VAL_BATCH_SIZE=32
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=500
MAX_OBS_LENGTH=1000
MAX_TURNS=3
TOPK=2
MAX_QUERIES_PER_TURN=3
TOTAL_TRAINING_STEPS=1005
TEST_FREQ=50
SAVE_FREQ=100
actor lr=5e-7
lr_warmup_steps_ratio=0.285
kl_loss_coef=0.005
n_agent=5
temperature=1
```

归档脚本使用的正向 reward 系数：

```text
PLAN_ONCE_BONUS=0.0
ANSWER_PRESENT_BONUS=0.05
NO_GENERATED_INFORMATION_BONUS=0.05
EVIDENCE_HIT_BONUS=0.05
VALID_SEARCH_BONUS=0.05
PARALLEL_EVIDENCE_BONUS=0.05
```

最大训练 reward 为：

```text
1.25
```

validation 仍只计算原 answer EM。

### 运行结果

归档 summary：

```text
archived_at: 2026-06-25 22:30:58 CST
last_seen_step: epoch 0, step 987
best_trajectory_lines: 986
W&B state: crashed
```

W&B 最终 summary：

```text
_step: 985
val/test_score/nq: 0.471875
critic/score/mean: 0.6618749499320984
critic/rewards/mean: 0.6618749499320984
train/answer_em_mean: 0.44999998807907104
train/hard_zero_rate: 0.0062500000931322575
env/finish_ratio: 0.996875
env/ratio_of_valid_action: 0.99765625
env/number_of_valid_search: 1
response_length/clip_ratio: 0.0031250000465661287
actor/kl_loss: 0.6694269180297852
actor/ppo_kl: 0.003107007022481412
actor/pg_loss: 0.005565020069479942
actor/grad_norm: 1.8418385982513428
actor/lr: 5e-7
actor/kl_coef: 0.005
```

validation 曲线：

```text
step 50:  0.046875
step 100: 0.206250
step 150: 0.325000
step 200: 0.387500
step 250: 0.390625
step 300: 0.418750
step 350: 0.431250
step 400: 0.446875
step 450: 0.443750
step 500: 0.421875
step 550: 0.475000
step 600: 0.475000
step 650: 0.459375
step 700: 0.478125
step 750: 0.462500
step 800: 0.459375
step 850: 0.487500
step 900: 0.481250
step 950: 0.471875
```

最佳 validation：

```text
best step: 850
best val/test_score/nq: 0.4875
```

### Trajectory 观察

best trajectory 文件共 986 条。分窗口统计：

```text
step 1-100:
  score_mean=1.225
  valid_actions_mean=2.08
  valid_searches_mean=1.08

step 101-300:
  score_mean=1.25
  valid_actions_mean=2.00
  valid_searches_mean=1.00

step 301-600:
  score_mean=1.25
  valid_actions_mean=2.00
  valid_searches_mean=1.00

step 601-850:
  score_mean=1.25
  valid_actions_mean=2.00
  valid_searches_mean=1.00

step 851-986:
  score_mean=1.25
  valid_actions_mean=2.00
  valid_searches_mean=1.00
```

最后 10 条 best trajectory 全部保持正常：

```text
score=1.25
turns=2
valid_actions=2
valid_searches=1
```

并且大多使用并行 query：

```text
<search> q1 || q2 </search>
<information>
[Query] q1
...
[Query] q2
...
</information>
<answer> ... </answer>
```

在 986 条 best trajectory 中：

```text
multi_query: 976
answer: 986
information: 986
plan: 381
```

说明 hard zero 版本没有复现 Phase 4 / soft reward 中的后期格式坍塌。模型倾向于稳定地产生一次 search、一次 information、一次 answer，其中 search 大多包含两个并行 query。

### 结论

Hard reward noSFT 版本是目前最有效的一版：

```text
1. 不再依赖 Phase 3 SFT cold-start。
2. 没有出现后期 valid_action=0 / clip_ratio=1 的整体坍塌。
3. validation 从 0.046875 稳定提升到最高 0.4875。
4. 训练后期 trajectory 仍保持有效 search 和 answer。
5. 并行 query 行为明显稳定，best trajectory 中 976/986 条包含 multi-query。
```

但它仍有两个问题：

```text
1. validation 在 step 850 后连续回落，最终 step 950 为 0.471875。
2. best trajectory 很少保留 <plan>，说明模型主要学会了 search-answer 路径，
   而不是完整 <think>/<plan>/<search>/<answer> 格式。
```

因此当前建议：

```text
1. 选择 step 850 作为本轮 best checkpoint。
2. 用 step 850 做完整 benchmark eval。
3. 下一版训练可以继续简化 reward：answer_em + parallel_evidence_bonus，并保留 hard zero。
4. 如果仍需要 <plan> 格式，应单独设计格式约束；如果只看检索效率和答案准确率，可以接受 prompt 中有 plan 说明但模型不强制输出 plan。
```

## Step 900 Full NQ Eval

由于服务器归档时可直接使用的 checkpoint 为 `global_step_900`，因此额外对 step 900 做完整 NQ test eval。

Greedy eval：

```text
output/phase5_step900_validate/eval_step900_nq_full/
```

配置：

```text
checkpoint: verl_checkpoints/nq-litecoa-grpo-qwen2.5-3b-base-hard/actor/global_step_900
adapter: none
input_parquet: data/nq_search_litecoa/test.parquet
total: 3610
batch_size: 128
topk: 2
max_turns: 3
max_queries_per_turn: 3
do_sample: false
temperature: 0.0
tensor_parallel_size: 1
```

结果：

```text
answer_em: 1674 / 3610 = 46.37%
answer_subem: 1786 / 3610 = 49.47%
answer_count: 3605 / 3610 = 99.86%
generated_information_count: 0
parser_warning_count: 0
agent_warning_count: 5
max_turns_exceeded: 0
```

检索行为：

```text
search_turn_distribution:
  0 turn: 1
  1 turn: 3609

query_count_distribution:
  0 query: 1
  2 query: 3609
```

Temp=1 eval：

```text
output/phase5_step900_validate_temp/eval_step900_nq_full_temp1/
```

配置：

```text
checkpoint: verl_checkpoints/nq-litecoa-grpo-qwen2.5-3b-base-hard/actor/global_step_900
adapter: none
input_parquet: data/nq_search_litecoa/test.parquet
total: 3610
batch_size: 128
topk: 2
max_turns: 3
max_queries_per_turn: 3
do_sample: true
temperature: 1.0
tensor_parallel_size: 1
```

结果：

```text
answer_em: 1569 / 3610 = 43.46%
answer_subem: 1678 / 3610 = 46.48%
answer_count: 3551 / 3610 = 98.37%
generated_information_count: 0
parser_warning_count: 6
agent_warning_count: 59
max_turns_exceeded: 0
```

检索行为：

```text
search_turn_distribution:
  0 turn: 19
  1 turn: 3591

query_count_distribution:
  0 query: 19
  1 query: 32
  2 query: 3558
  3 query: 1
```

结论：

```text
step 900 greedy 明显优于 temp=1。
模型已经学成高确定性的 parallel-search 策略，采样会破坏 query 格式和答案稳定性。
```

## Search-R1 Baseline 对比

原 Search-R1 W&B run：

```text
https://wandb.ai/sun19150956991-beijing-university-of-posts-and-telecommu/Search-R1/runs/zhzflg3k
```

配置：

```text
experiment: nq-search-r1-grpo-qwen2.5-3b-em
base_model: Qwen2.5-3B
train_files: data/nq_search/train.parquet
val_files: data/nq_search/test.parquet
val_data_num: null
train_batch_size: 64
val_batch_size: 32
max_prompt_length: 4096
max_response_length: 500
max_obs_length: 500
max_turns: 2
retriever.topk: 3
actor lr: 1e-6
lr_warmup_steps_ratio: 0.285
kl_loss_coef: 0.001
```

W&B validation：

```text
step 50:  0.135882
step 100: 0.309710
step 150: 0.395647
step 200: 0.415737
step 250: 0.426897
step 300: 0.000000
```

最佳 validation：

```text
best step: 250
best val/test_score/nq: 0.42689732142857145
```

step 300 后出现坍塌：

```text
critic/score/mean: 0
critic/rewards/mean: 0
env/finish_ratio: 0
env/ratio_of_valid_action: 0
env/number_of_valid_search: 0
response_length/clip_ratio: 1
actor metrics: NaN
```

与 Phase 5 hard reward step900 greedy 对比：

| 实验 | Eval | NQ EM | NQ SubEM | Search 行为 | 备注 |
|---|---|---:|---:|---|---|
| Search-R1 baseline | W&B full val best step250 | 42.69% | - | 单 query，单 search | step300 后坍塌 |
| Phase 5 hard reward noSFT | full eval step900 greedy | 46.37% | 49.47% | 3609/3610 为 2-query parallel search | 稳定，无 generated information |
| Phase 5 hard reward noSFT | full eval step900 temp=1 | 43.46% | 46.48% | 3558/3610 为 2-query parallel search | 采样导致 query 质量下降 |

## Phase 5 收官结论

Phase 5 达成了核心目标：

```text
1. hard zero 解决了 Phase 4 answer-only 和 Phase 5 soft reward 的格式坍塌问题。
2. 从 Qwen2.5-3B base 直接 GRPO 比 SFT-init 更稳定。
3. 模型学会了稳定的 parallel search：
   greedy eval 中 3609/3610 样本使用 2-query search。
4. NQ full eval greedy EM 达到 46.37%，高于 Search-R1 baseline best 42.69%。
5. 模型不生成 <information>，不发生 max_turns_exceeded。
```

同时，Phase 5 也暴露了 LiteCoA-3B 的边界：

```text
1. 模型没有保留 <plan>：
   greedy/temp=1 full eval 中 has_plan=0。
2. format_valid=0，因为模型跳过了 <think>/<plan>，直接输出 <search>。
3. 3B 在 GRPO 中会自然收敛到最短有效路径：
   <search> q1 || q2 </search> -> <information> -> <answer>
```

因此，Phase 5 的准确表述是：

```text
LiteCoA 的 parallel search 目标已经完成；
完整 plan-first LiteCoA 格式没有在 3B GRPO 中保留；
当前有效形态更接近 Parallel-Search-R1。
```

后续 Phase 6 应进入实验对比与固化：

```text
1. 固定 step900 greedy 作为当前 Phase 5 可复现主结果。
2. 如有 checkpoint step850，可补测 step850 full eval；否则记录 W&B best step850、可用 checkpoint step900。
3. 继续跑 TriviaQA / PopQA / HotpotQA / 2Wiki / Musique / Bamboogle 等 benchmark。
4. 整理 Phase 0 / Phase 3 / Phase 4 / Phase 5 的统一对比表。
5. 决定最终命名：LiteCoA-Search-R1 或 Parallel-Search-R1。
```
