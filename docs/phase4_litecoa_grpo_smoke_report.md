# Phase 4 LiteCoA-GRPO Smoke Report

## 1. 目标

本次 smoke 的目标不是追求最终效果，而是验证 Phase 4 的完整训练链路是否闭环：

```text
LiteCoA prompt
-> SFT merged model
-> VeRL / Search-R1 GRPO
-> LiteCoA rollout
-> multi-query retriever
-> <information>
-> <answer>
-> rule reward
-> actor update
-> validation
```

结论：

```text
Phase 4 smoke 通过。
```

## 2. 代码与模型

代码版本：

```text
a58ab6a Add LiteCoA GRPO rollout support
```

远程服务器没有直接 merge 全仓库，而是同步 Phase 4 相关文件，避免覆盖服务器已有改动。

起始模型：

```text
Qwen2.5-3B base model + Phase 3 LiteCoA LoRA
```

远程服务器将 LoRA 合并为完整模型：

```text
outputs/sft/litecoa_lora_qwen25_3b_full_merged
```

合并后修复过 `tokenizer_config.json` 与 Transformers 4.47 的兼容问题。该修复是 smoke 前置环境修复，不属于 LiteCoA rollout 逻辑改动。

## 3. 数据

生成 LiteCoA 版 NQ GRPO 数据：

```text
data/nq_search_litecoa/train.parquet
data/nq_search_litecoa/test.parquet
```

本地读取回传 parquet 后确认：

```text
train.parquet: 79168 rows
test.parquet: 3610 rows
prompt role: user only
system role: false
<answer> Beijing </answer>: false
prompt prefix: You are a search-augmented reasoning agent.
reward_model: rule + golden_answers
```

这说明 Phase 4 GRPO prompt 已与 Phase 3 SFT 最终训练格式对齐：单 user prompt，无 system message，无 Beijing answer 示例。

## 4. Smoke 配置

本次 smoke 使用临时脚本：

```text
train_grpo_litecoa_smoke.sh
```

关键配置：

```text
data.train_files=data/nq_search_litecoa/train.parquet
data.val_files=data/nq_search_litecoa/test.parquet
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

远程第一次运行因为 Ray 主机内存超过阈值被杀，随后仅调整 smoke 脚本：

```text
actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
Ray temp/spill directory moved to project data disk
```

没有降低 `n_agent=5`，没有减少 smoke 数据量。

## 5. 运行结果

最终 smoke 跑完 10-step GRPO，并完成 validation。

日志最终结果：

```text
Final validation metrics: {'val/test_score/nq': 0.265625}
```

W&B run：

```text
project: Search-R1
run: nq-litecoa-grpo-qwen2.5-3b-smoke
id: 4ih8xv0y
```

W&B 状态显示 `crashed`，但本地日志显示训练和最终 validation 已完成。W&B summary 只同步到 `_step=8`，其中 `val/test_score/nq=0.140625`，因此本次最终 validation 以本地日志为准。

W&B step 8 summary：

```text
critic/rewards/mean: 0.246875
critic/score/mean: 0.246875
critic/returns/mean: -0.081078
actor/pg_loss: 0.167209
actor/kl_loss: 0.003966
actor/ppo_kl: 0.000299
actor/grad_norm: 1.356807
actor/lr: 1e-6
env/finish_ratio: 0.86875
env/ratio_of_valid_action: 0.897656
env/number_of_valid_search: 1.284375
env/number_of_actions/mean: 2.55
response_length/mean: 1151.356
response_length/max: 2432
prompt_length/mean: 199.3125
state_tokens/coverage: 0.451972
timing_s/step: 204.49
timing_s/gen: 131.78
timing_s/testing: 134.68
```

## 6. LiteCoA 行为验证

从 smoke log 统计：

```text
<plan>: 179
<search>: 275
multi-query <search>: 103
[Query] blocks: 109
<information>: 230
<answer>: 175
invalid action: 22
OBSERVATION TOO LONG: 37
Traceback in final successful run: 0
```

日志中出现真实多 query 搜索：

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

这证明 Phase 4 的 multi-query rollout 不是只改了 prompt，而是真正进入了 retriever 并按 LiteCoA 格式返回 `<information>`。

## 7. 发现的问题

### 7.1 max_obs_length 偏小

本次 smoke 使用：

```text
data.max_obs_length=500
```

日志中出现 37 次：

```text
[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, xxxx & 500
```

这是启动脚本配置问题，不是 rollout 代码问题。LiteCoA 一次 search 可能包含多个 query，`<information>` 天然比原 Search-R1 单 query 更长。正式训练应与 Phase 3 SFT 实际上下文设置对齐。

服务器实际 SFT 使用：

```text
cutoff_len=8192
```

因此正式 GRPO 建议：

```text
data.max_prompt_length=8192
data.max_obs_length=1000
data.max_start_length=2048
data.max_response_length=500
max_turns=3
```

按 Search-R1 原公式：

```text
max_prompt_length =
max_start_length
+ max_response_length * (max_turns - 1)
+ max_obs_length * max_turns

2048 + 500 * 2 + 1000 * 3 = 6048
```

所以 `max_prompt_length=8192` 足够覆盖该设置，并与 SFT 实际 `cutoff_len=8192` 一致。

### 7.2 invalid action 仍存在

Smoke 中出现 22 次 invalid action 修正提示：

```text
My previous action is invalid...
```

这说明模型仍会偶发生成不完整 tag、错误 action 或 answer 后继续续写。该问题不阻塞 Phase 4 链路，但 Phase 5 reward 需要考虑 format/action penalty。

### 7.3 answer 后偶发异常续写

stdout 中可见部分 `<answer>...</answer>` 后继续出现奇怪字符或 role-like 文本。由于当前返回的是混合 stdout log，无法完全区分是真实模型输出还是多样本日志交错。

已增加单独 trajectory jsonl 保存。LiteCoA 启动脚本会开启：

```text
trainer.log_best_trajectory=true
trainer.trajectory_log_dir=trajectory/litecoa_grpo
```

每个训练 batch 在 reward 计算完成后取 `score` 最高的一条样本，追加到：

```text
trajectory/litecoa_grpo/${EXPERIMENT_NAME}.jsonl
```

字段与原 Search-R1 trajectory 文件基本一致，包括 `step`、`epoch`、`sample_idx`、`data_source`、`uid`、`index`、`ground_truth`、`score`、`prompt_length`、`response_length`、`turns`、`valid_actions`、`valid_searches`、`prompt`、`trajectory`、`metrics`。LiteCoA 版额外增加 `batch_size`、`candidate_count`、`question_count`，方便确认每条记录来自多大的 GRPO batch。

### 7.4 W&B 状态不干净

W&B run 状态显示 `crashed`，但本地日志完成 10-step 和 final validation。推测原因是前序失败 run、Ray 退出、或 W&B 没有优雅 finish。

正式脚本应尽量保证：

```text
每次 smoke/full 使用独立 experiment_name
失败 run 不复用同一个 W&B run id
脚本结束时让主进程正常退出
```

## 8. 结论

Phase 4 smoke 已验证：

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

因此：

```text
Phase 4 主链路通过 smoke。
```

下一步进入：

```text
1. 固化 LiteCoA-GRPO smoke/full 启动脚本。
2. 增加结构化 trajectory 保存或更干净的 rollout 日志。
3. 进入 Phase 5 path-aware reward 设计。
```

正式脚本建议基线配置：

```text
data.train_files=data/nq_search_litecoa/train.parquet
data.val_files=data/nq_search_litecoa/test.parquet
actor_rollout_ref.model.path=outputs/sft/litecoa_lora_qwen25_3b_full_merged
data.max_prompt_length=8192
data.max_obs_length=1000
data.max_start_length=2048
data.max_response_length=500
max_turns=3
retriever.topk=3
retriever.max_queries_per_turn=3
actor_rollout_ref.rollout.n_agent=5
actor_rollout_ref.rollout.temperature=1
actor_rollout_ref.actor.optim.lr=1e-6
```
