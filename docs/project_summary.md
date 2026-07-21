# Parallel Search 项目总结

## 1. 项目一句话

本项目基于 Search-R1 / veRL 改造了一个支持并行检索的 RL search agent，使 Qwen2.5-3B 在一次 `<search>` action 中稳定生成两个 query，并通过真实 retriever 返回 evidence 后完成回答。

最终有效形态：

```text
<search> query1 || query2 </search>
<information> ... </information>
<answer> ... </answer>
```

核心结果：

```text
Search-R1 baseline best NQ EM: 42.69%
Parallel Search step900 greedy NQ EM: 46.37%
完整 NQ test: 3610 samples
3609 / 3610 samples use 2-query parallel search
generated_information_count: 0
max_turns_exceeded: 0
```

## 2. 背景与动机

原 Search-R1 已经能训练模型使用搜索工具，但 Qwen2.5-3B 在稳定窗口里主要收敛到单一路径：

```text
one query search -> information -> answer
```

Phase 0 baseline 观察显示：

```text
稳定窗口 step 100-250:
平均 score: 0.4305
平均 valid searches: 1.0
<search> 内包含 || 的 action: 0
<plan> 出现次数: 0
```

这说明原 Search-R1 能学会工具调用，但没有自然学会：

```text
1. query decomposition
2. parallel search
3. plan-guided search
4. 多 query evidence 聚合
```

因此项目目标从完整 LiteCoA 逐步收敛为更务实的目标：

```text
让 3B 模型稳定学会 parallel query search。
```

## 3. 阶段回顾

### Phase 1：LiteCoA 推理原型

完成 `infer_litecoa.py`，验证推理阶段可以支持：

```text
first-turn plan
multi-query search
retriever information injection
follow-up answer
```

结论：

```text
client-side LiteCoA 推理闭环可行。
```

### Phase 2：LiteCoA 数据构造

使用 teacher API 和真实 Search-R1 retriever 构造 SFT 数据，保证 `<information>` 来自真实检索器，而不是 teacher 编造。

关键点：

```text
teacher 只生成 plan/search/answer
retriever 负责填充 information
先做 20 条闭环，再扩展到 1000 条
```

结论：

```text
LiteCoA SFT 数据可构造，格式可控。
```

### Phase 3：LoRA SFT 冷启动

用 1000 条 LiteCoA SFT 数据训练 Qwen2.5-3B LoRA，让模型先学格式。

full NQ eval：

```text
SFT greedy:
answer_em: 1358 / 3610 = 37.62%

SFT temp=1:
answer_em: 1080 / 3610 = 29.92%
```

结论：

```text
SFT 可以学会 <think>/<plan>/<search>/<answer> 格式，
但准确率有限，且后续 GRPO 中格式不稳定。
```

### Phase 4：LiteCoA Rollout 改造

改造 Search-R1 / veRL 训练链路，让 GRPO rollout 支持：

```text
<search> q1 || q2 || q3 </search>
```

主要工作：

```text
1. search parser 支持 || 分隔多个 query。
2. retriever batch_search 支持 flatten queries 后再按 sample 回填。
3. <information> 按 [Query] 分块返回。
4. data_process 支持 litecoa prompt。
5. reward 兼容单个 <answer>。
6. validation 和训练日志支持 LiteCoA trajectory。
```

Phase 4 smoke 跑通，但 answer-only full run 后期坍塌：

```text
finish_ratio -> 0
valid_action_ratio -> 0
response_length/clip_ratio -> 1
actor metrics -> NaN
```

结论：

```text
工程闭环完成，但 answer-only reward 不足以稳定训练 LiteCoA。
```

### Phase 5：Hard Reward 与 Parallel Search 收官

Phase 5 比较了三类 reward：

```text
1. answer-only reward:
   能跑通，但 full run 后期坍塌。

2. soft reward:
   加小额正向 shaping，但没有 hard zero，后期仍坍塌。

3. hard reward noSFT:
   从 Qwen2.5-3B base 直接 GRPO，加入 hard zero 和 parallel evidence bonus，
   训练稳定，parallel search 成功。
```

hard zero 规则：

```text
response 打满 max_response_length -> reward = 0
没有 valid action -> reward = 0
没有 <answer> -> reward = 0
模型自己生成 <information> -> reward = 0
```

当前有效 reward 形态：

```text
reward = answer_em + 0.05 * parallel_evidence_hit
```

并保留 hard zero。

## 4. 最终实验结果

### Search-R1 Baseline

W&B run：

```text
nq-search-r1-grpo-qwen2.5-3b-em
```

full validation best：

```text
best step: 250
best val/test_score/nq: 0.426897
```

step 300 后坍塌：

```text
val/test_score/nq: 0
finish_ratio: 0
valid_action_ratio: 0
clip_ratio: 1
```

### Parallel Search Hard Reward

W&B run：

```text
nq-litecoa-grpo-qwen2.5-3b-base-hard
```

训练阶段 best：

```text
best step: 850
best val/test_score/nq: 0.4875
```

可复现 checkpoint 使用 step900 full eval：

```text
checkpoint: global_step_900
eval: full NQ test, 3610 samples
```

Greedy：

```text
answer_em: 1674 / 3610 = 46.37%
answer_subem: 1786 / 3610 = 49.47%
answer_count: 3605 / 3610 = 99.86%
generated_information_count: 0
max_turns_exceeded: 0
```

Temp=1：

```text
answer_em: 1569 / 3610 = 43.46%
answer_subem: 1678 / 3610 = 46.48%
answer_count: 3551 / 3610 = 98.37%
```

检索行为：

```text
greedy:
  3609 / 3610 samples use 1 search turn
  3609 / 3610 samples use exactly 2 queries

temp=1:
  3591 / 3610 samples use 1 search turn
  3558 / 3610 samples use exactly 2 queries
```

### 对比表

| 实验 | Eval | NQ EM | NQ SubEM | Search 行为 |
|---|---|---:|---:|---|
| Search-R1 baseline | W&B best step250 | 42.69% | - | 单 query search |
| Phase 3 SFT | full NQ greedy | 37.62% | - | 格式较完整但准确率低 |
| Phase 5 hard reward | full NQ step900 greedy | 46.37% | 49.47% | 3609/3610 为 2-query search |
| Phase 5 hard reward | full NQ step900 temp=1 | 43.46% | 46.48% | 3558/3610 为 2-query search |

## 5. 技术贡献

### 5.1 Rollout 改造

把原 Search-R1 的单 query search：

```text
<search> query </search>
```

扩展为：

```text
<search> q1 || q2 || q3 </search>
```

并实现：

```text
query 去重
query 数量截断
batch flatten retrieval
按样本回填 information
[Query] 分块 observation
```

### 5.2 数据链路

构造了 LiteCoA SFT 数据和 LiteCoA GRPO parquet 数据：

```text
teacher 生成 search/answer
retriever 生成 information
模型不允许自己生成 information
```

### 5.3 Reward 设计

实现了基于 response / information 分离的 reward 计算：

```text
model_response_str: 只包含模型生成 token
retrieved_information_str: 只包含 retriever 插入 observation
```

这样避免 prompt tag 或 generated information 污染 reward。

### 5.4 训练稳定性

发现并验证：

```text
soft shaping 不足以阻止格式坍塌
hard zero 是稳定 agentic GRPO 的关键
3B 模型更适合 compact parallel-search policy
```

### 5.5 评测工具

实现 batched vLLM eval，支持：

```text
greedy / temp=1
真实 retriever
trajectory jsonl 输出
format/search/query 分布统计
full NQ evaluation
```

## 6. 关键结论

本项目最终证明：

```text
在 Qwen2.5-3B 上，完整 plan-first LiteCoA 格式不稳定；
但通过 hard reward 和 rollout 改造，可以稳定训练出 2-query parallel search agent。
```

更准确地说：

```text
LiteCoA 的 parallel search 目标完成；
完整 <think>/<plan> 格式没有在 3B GRPO 中保留；
最终有效形态是 Parallel-Search-R1 / compact LiteCoA agent。
```

这个结论很重要，因为它说明：

```text
1. 小模型 RL 会自然压缩掉非必要中间格式。
2. 对 3B 模型而言，显式 plan 不是最有效的 credit assignment 载体。
3. parallel query 本身是主要收益来源。
4. reward 设计比 SFT 格式模仿更关键。
```

## 7. 当前局限

```text
1. 只完成 NQ full eval，多数据集 benchmark 还没系统跑完。
2. step850 是 W&B best，但本地完整 eval 使用的是可复现 step900 checkpoint。
3. 模型没有保留 <plan>，不适合声称完整 LiteCoA plan-first agent。
4. 目前结果集中在 Qwen2.5-3B，更大模型可能能保留 plan 格式。
5. 当前 parallel search 几乎固定为 2-query，一定程度上是 reward 和 topk 设置共同塑造的策略。
```

## 8. 下一步

Phase 6 建议：

```text
1. 跑 TriviaQA / PopQA / HotpotQA / 2Wiki / Musique / Bamboogle benchmark。
2. 如果有 step850 checkpoint，补跑 step850 full eval。
3. 继续比较 greedy 和 temp=1。
4. 决定最终命名：Parallel-Search-R1 或 LiteCoA-Search-R1。
5. 如果论文或简历要突出 plan，应在 7B/14B 上单独做 plan retention 实验。
```

## 9. 简历包装

### 中文简历短版

```text
基于 Search-R1 / veRL 构建并行检索式 RL Agent，改造 rollout、retriever 调用和 reward 逻辑，使 Qwen2.5-3B 在 GRPO 训练中稳定学会一次生成两个 search query 并并行检索；在 NQ full test 上将 EM 从 Search-R1 baseline 的 42.69% 提升到 46.37%，且 3609/3610 样本稳定使用 2-query parallel search。
```

### 中文简历要点版

```text
- 基于 Search-R1 / veRL 改造 agentic RL 训练框架，支持 `<search>q1 || q2</search>` 并行检索与按 query 分块 evidence 回填。
- 设计 LiteCoA/Parallel Search 数据构造、rollout parser、vLLM batch eval 和 trajectory 记录流程，完成从 SFT 数据构造到 GRPO 训练评测的端到端闭环。
- 设计 hard-zero reward，约束无效 action、超长输出、缺失 answer 和模型伪造 information 等失败模式，显著提升 GRPO 训练稳定性。
- 在 Qwen2.5-3B + NQ full test 上，相比 Search-R1 baseline best EM 42.69%，Parallel Search step900 greedy 达到 46.37% EM；3609/3610 样本稳定使用 2-query parallel search。
```

### 英文简历版本

```text
Built a Search-R1/veRL based parallel-search RL agent that extends single-query tool use to multi-query retrieval within one search action. Modified rollout parsing, retriever integration, reward computation, and batched vLLM evaluation for Qwen2.5-3B GRPO training. Designed hard-zero rewards to suppress invalid actions, clipped responses, missing answers, and generated fake information. Achieved 46.37% EM on full NQ test vs. 42.69% Search-R1 baseline, with 3609/3610 samples using stable two-query parallel search.
```

### 面试讲法

可以按这个顺序讲：

```text
1. 我先复现和分析 Search-R1 baseline，发现 3B 模型基本只学会单 query search。
2. 然后我设计了 LiteCoA/Parallel Search，让模型一次 search 生成多个 query，用真实 retriever 并行检索。
3. 工程上我改了 rollout parser、retriever batch 调用、information 回填、reward 解析和 vLLM eval。
4. 早期 answer-only 和 soft reward 都会后期坍塌，所以我设计了 hard-zero reward 来过滤无效 action。
5. 最后模型稳定学会了 2-query parallel search，在 NQ full test 上超过原 Search-R1 baseline。
6. 一个重要发现是：3B 模型不适合强保留 plan-first 格式，它会自然收敛到更短的 parallel-search -> answer 策略。
```

### 项目亮点关键词

```text
Agentic RL
Search-augmented LLM
GRPO
veRL
Search-R1
Parallel retrieval
Reward shaping
Hard-zero reward
vLLM evaluation
Trajectory analysis
Qwen2.5-3B
```
