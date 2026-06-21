# Phase 3 LiteCoA LoRA SFT Summary

## 1. 阶段目标

Phase 3 的目标是用 Phase 2 固化的 LiteCoA SFT 数据，对 Qwen2.5-3B 做 LoRA SFT 冷启动。

这一步不追求最终高准确率，核心目标是让模型先学会 LiteCoA 的基本行为：

```text
1. 首轮生成一次 <plan>。
2. 每次 action 前先生成 <think>。
3. 用 <search> 发起检索，支持 q1 || q2 || q3 并行 query。
4. 不自己生成 <information>，只消费 retriever 返回的 evidence。
5. 证据充分后用短答案 <answer> 结束。
```

最终期望轨迹：

```text
<think>...</think>
<plan>...</plan>
<think>...</think>
<search>q1 || q2</search>
<information>真实检索结果</information>
<think>...</think>
<answer>短答案</answer>
```

## 2. 使用数据

SFT 数据来自 Phase 2 固化版本：

```text
data/litecoa_sft/litecoa_1000/litecoa_sft_1000_v1_llamafactory.jsonl
```

数据格式是 LLaMA-Factory 可读的多轮 messages 格式：

```text
user: LiteCoA prompt + question
assistant: <think><plan><think><search>
user: <information>真实 retriever evidence</information>
assistant: <think><answer>
```

如果是多轮补搜，则继续追加：

```text
assistant: <think><search>
user: <information>...</information>
assistant: <think><answer>
```

关键设计：

```text
<information> 不作为 assistant loss target。
SFT 只学习 assistant 侧的 plan/search/answer 行为。
```

## 3. 文件入口

Phase 3 相关文件集中如下：

```text
data/litecoa_sft/litecoa_1000/litecoa_sft_1000_v1_llamafactory.jsonl
data/litecoa_sft/litecoa_1000/dataset_info.json
configs/sft/litecoa_lora_qwen25_3b_smoke.yaml
configs/sft/litecoa_lora_qwen25_3b_full.yaml
scripts/sft/train_litecoa_lora_qwen25_3b.sh
scripts/eval/eval_litecoa_sft.py
scripts/eval/eval_litecoa_sft_vllm.py
```

训练脚本本身保持简单，只负责选择 YAML 配置并调用：

```text
llamafactory-cli train "$CONFIG"
```

训练参数写在 `configs/sft/` 下，避免 shell 脚本过长。

## 4. 训练设置

训练脚本：

```text
scripts/sft/train_litecoa_lora_qwen25_3b.sh
```

full 训练配置：

```text
configs/sft/litecoa_lora_qwen25_3b_full.yaml
```

核心配置：

```text
base_model: Qwen2.5-3B
dataset: litecoa_sft_1000_v1
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
cutoff_len: 16384
num_train_epochs: 2.0
learning_rate: 5e-5
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 2
GPU: 2 x A800 80G
effective_batch_size: 8
```

训练步数：

```text
1000 samples x 2 epochs / effective_batch_size 8 = 250 optimizer steps
```

## 5. Smoke 训练

Smoke 配置：

```text
configs/sft/litecoa_lora_qwen25_3b_smoke.yaml
```

Smoke 目标是确认：

```text
1. LLaMA-Factory 能读取 dataset_info.json。
2. messages/sharegpt 格式能被正确解析。
3. Qwen2.5-3B 本地路径可加载。
4. LoRA 配置可训练。
5. W&B 日志链路正常。
```

Smoke 结果：

```text
samples: 32
steps: 2
train_loss: 1.5655
runtime: 10.72 sec
status: success
```

Smoke 通过后，切换到 full 配置继续训练。

## 6. SFT 训练结果

远程 full SFT 训练成功完成：

```text
epoch: 2.0
optimization_steps: 250
train_runtime: 509.8563 sec
train_loss: 0.8226485109
train_samples_per_second: 3.923
train_steps_per_second: 0.49
trainable_params: 119,734,272
```

loss 走势：

```text
step 5:   1.5498
step 10:  1.3062
step 20:  0.9715
step 50:  0.8685
step 160: 0.7298
step 200: 0.6942
step 250: 0.7255
```

训练判断：

```text
1. loss 从约 1.55 下降并稳定到 0.7 左右。
2. 没有 OOM、CUDA error、Traceback。
3. 学习率 warmup 后按 cosine 衰减。
4. LoRA SFT 冷启动训练成功。
```

## 7. Phase 3.5 评估设置

SFT 后评估使用真实 retriever 和 vLLM 推理，目的是检查 SFT 后模型是否能在真实检索交互中保持 LiteCoA 格式。

评估脚本：

```text
scripts/eval/eval_litecoa_sft_vllm.py
```

评估数据：

```text
data/nq_search/test.parquet
total: 3610
```

模型设置：

```text
base_model: Qwen2.5-3B
adapter: litecoa_lora_qwen25_3b_full
backend: vLLM + runtime LoRA
retriever_url: http://127.0.0.1:8000/retrieve
topk: 3
max_turns: 3
max_queries_per_turn: 3
```

`max_turns` 的语义：

```text
最多允许 3 轮 search。
如果第 3 轮 search 后拿到 <information>，仍允许模型额外生成一次 <answer>。
只有最后仍没有 <answer>，才记为 max_turns_exceeded。
```

评估方式：

```text
greedy: do_sample=false, temperature=0
temp=1: do_sample=true, temperature=1.0
```

## 8. Greedy 全验证集结果

```json
{
  "total": 3610,
  "errors": 0,
  "format_valid": 3388,
  "has_plan": 3609,
  "plan_once": 3609,
  "answer_count": 3388,
  "generated_information_count": 0,
  "parser_warning_count": 8,
  "agent_warning_count": 222,
  "max_turns_exceeded": 220,
  "answer_em": 1358,
  "answer_subem": 1691,
  "gold_count": 3610
}
```

比例：

```text
answer_em: 1358 / 3610 = 37.62%
answer_subem: 1691 / 3610 = 46.84%
format_valid: 3388 / 3610 = 93.85%
answer_count: 3388 / 3610 = 93.85%
generated_information_count: 0 / 3610 = 0.00%
max_turns_exceeded: 220 / 3610 = 6.09%
```

搜索轮数分布：

```text
0 turns: 1
1 turn: 3117
2 turns: 255
3 turns: 237
```

query 数分布：

```text
0 query: 1
1 query: 129
2 query: 2983
3 query: 43
4 query: 222
5 query: 27
6 query: 168
7 query: 18
8 query: 19
```

## 9. Temp=1 全验证集结果

```json
{
  "total": 3610,
  "errors": 0,
  "format_valid": 3480,
  "has_plan": 3604,
  "plan_once": 3600,
  "answer_count": 3489,
  "generated_information_count": 3,
  "parser_warning_count": 74,
  "agent_warning_count": 121,
  "max_turns_exceeded": 104,
  "answer_em": 1080,
  "answer_subem": 1391,
  "gold_count": 3610
}
```

比例：

```text
answer_em: 1080 / 3610 = 29.92%
answer_subem: 1391 / 3610 = 38.53%
format_valid: 3480 / 3610 = 96.40%
has_plan: 3604 / 3610 = 99.83%
plan_once: 3600 / 3610 = 99.72%
answer_count: 3489 / 3610 = 96.65%
generated_information_count: 3 / 3610 = 0.08%
max_turns_exceeded: 104 / 3610 = 2.88%
```

搜索轮数分布：

```text
0 turns: 7
1 turn: 2923
2 turns: 465
3 turns: 215
```

query 数分布：

```text
0 query: 7
1 query: 174
2 query: 2547
3 query: 296
4 query: 299
5 query: 113
6 query: 109
7 query: 37
8 query: 21
9 query: 7
```

## 10. Greedy vs Temp=1 对比

| 指标 | Greedy | Temp=1 | 说明 |
| --- | ---: | ---: | --- |
| answer_em | 37.62% | 29.92% | 采样下答案更容易漂移 |
| answer_subem | 46.84% | 38.53% | temp=1 仍有一定答案召回 |
| format_valid | 93.85% | 96.40% | 两种模式格式都稳定 |
| has_plan | 99.97% | 99.83% | plan 学习成功 |
| plan_once | 99.97% | 99.72% | 首轮一次 plan 基本稳定 |
| answer_count | 93.85% | 96.65% | temp=1 更愿意结束并 answer |
| generated_information_count | 0 | 3 | 自编 information 基本不存在 |
| max_turns_exceeded | 6.09% | 2.88% | temp=1 下不回答的情况更少 |

## 11. 结论

Phase 3 / Phase 3.5 可以判定通过。

通过依据：

```text
1. LoRA SFT 训练稳定完成。
2. vLLM + runtime LoRA + retriever 全验证集评估跑通，errors = 0。
3. 模型几乎总能生成 <plan>，且基本只生成一次。
4. 模型基本不会自编 <information>。
5. Greedy 和 temp=1 下 LiteCoA 格式都稳定。
6. 模型已经具备进入 GRPO 训练的冷启动行为基础。
```

需要注意：

```text
1. 当前准确率不是最终目标。
2. Greedy EM 37.62%，temp=1 EM 29.92%，说明 SFT 主要学会了格式和交互，搜索策略和答案选择仍需 RL 改善。
3. temp=1 是后续 GRPO rollout 的采样环境参考，准确率下降是预期现象。
4. parser warning 和少量 generated_information 需要在 Phase 4/5 中继续监控。
```

## 12. 下一步

进入 Phase 4：VeRL / Search-R1 LiteCoA GRPO 改造。

第一版 Phase 4 的目标应保持克制：

```text
1. 改 prompt，让 GRPO rollout 使用 LiteCoA 格式。
2. 改 parser，支持 <plan> 和多 query <search>q1 || q2 || q3</search>。
3. 改 search 交互，把多 query 一次性送到 retriever。
4. validation 和 rollout 使用同一套 LiteCoA loop。
5. reward 第一版仍使用原 answer EM，不先加路径感知 reward。
```

路径感知 reward 放到 Phase 5。
