# Phase 0：Search-R1 Baseline 稳定窗口行为分析

## 1. 分析范围

本阶段分析本地已有 trajectory 文件：

```text
trajectory/nq-search-r1-grpo-qwen2.5-3b-em.jsonl
```

该文件共有 389 条记录。按照项目判断，前期模型尚未稳定，后期又出现一定 collapse / invalid action 噪声，因此本报告将主分析窗口限定为：

```text
step 100-250
```

该稳定窗口共包含 151 条样本。

## 2. 稳定窗口总体统计

| 指标 | 数值 |
|---|---:|
| 样本数 | 151 |
| step 范围 | 100-250 |
| 平均 score | 0.4305 |
| 正确样本数 | 65 |
| 错误样本数 | 86 |
| 平均 turns | 2.0000 |
| 平均 valid actions | 2.0000 |
| 平均 valid searches | 1.0000 |
| response length 中位数 | 524 |
| invalid-action 样本数 | 0 |

## 3. Action 分布

| 字段 | 分布 |
|---|---|
| `turns` | `{2: 151}` |
| `valid_actions` | `{2: 151}` |
| `valid_searches` | `{1: 151}` |
| `score` | `{0.0: 86, 1.0: 65}` |

这个结果非常清楚：在稳定训练窗口内，所有样本都表现为：

```text
2 个 turn
2 个 valid action
1 次 valid search
```

也就是说，模型已经学会了稳定、合法的 Search-R1 工具调用格式，但主要策略收敛为：

```text
search -> information -> answer
```

而不是更复杂的多轮检索：

```text
search -> information -> search -> information -> answer
```

## 4. Tag 统计

| Tag | 分布 | 平均次数 |
|---|---|---:|
| `<search>...</search>` | `{0: 2, 1: 149}` | 0.9868 |
| `<information>...</information>` | `{0: 2, 1: 148, 2: 1}` | 0.9934 |
| `<answer>...</answer>` | `{1: 151}` | 1.0000 |
| `<think>...</think>` | `{0: 4, 1: 143, 2: 1, 3: 2, 7: 1}` | 1.0464 |
| `<plan>...</plan>` | `{0: 151}` | 0.0000 |

额外检查结果：

| 行为 | 数量 |
|---|---:|
| 出现 invalid-action 提示的样本 | 0 |
| valid search 大于等于 2 的样本 | 0 |
| regex 检测到 2 个以上 `<search>` 的样本 | 0 |
| `<search>` 内包含 `||` 的 action | 0 |
| 包含 `<plan>` 的样本 | 0 |

## 5. 关键结论

稳定窗口内，Search-R1 baseline 的行为可以概括为：

```text
模型会合法 search
模型会合法 answer
但模型基本只 search 一次
```

这说明当前 Qwen2.5-3B 在 Search-R1 训练后，已经具备基本工具调用能力，但没有自然形成 LiteCoA 计划中的能力：

```text
1. 没有 plan-guided decomposition
2. 没有每轮多个 query 的并行检索
3. 没有多轮补搜行为
4. 没有 search decision 的显式规划
```

因此，LiteCoA-Search-R1 的设计动机是成立的：后续需要通过轻量 CoA 轨迹格式和少量高质量 SFT 数据，把模型从单一的 `search -> answer` 行为，引导到更结构化的：

```text
think -> plan -> search -> information -> think -> answer
```

以及在证据不足时：

```text
think -> search -> information -> think -> search -> information -> answer
```

## 6. 代表样本

### 6.1 正确的单轮搜索样本

元信息：

```text
step=101
uid=37176
score=1.0
turns=2
valid_actions=2
valid_searches=1
```

问题：

```text
who in the bible almost sacrificed his son?
```

行为模式：

```text
<think> The question is asking about a biblical event in which someone almost
sacrificed their son. </think>
<search> Who in the Bible almost sacrificed their son? </search>
<information>...</information>
<answer> Abraham </answer>
```

这个样本说明：对于简单单跳问题，baseline 的单次搜索可以成功找到答案。

### 6.2 错误的单轮搜索样本

元信息：

```text
step=100
uid=37005
score=0.0
turns=2
valid_actions=2
valid_searches=1
```

问题：

```text
who plays young pt barnum in greatest showman?
```

行为摘要：

```text
<think>...</think>
<search> Who plays young Peter Bumman in the movie "Greatest Showman?" </search>
<information>...</information>
<answer>...</answer>
```

该样本中，模型把 `P. T. Barnum` 写成了 `Peter Bumman`，导致 query 质量较差。这类错误说明后续 LiteCoA 可以重点优化：

```text
1. plan 与 query 的对齐
2. query 中关键实体的保真
3. query quality reward
4. 重复或错误 query 的惩罚
```

## 7. 全量数据背景

虽然主结论基于稳定窗口，但全量 389 条数据也提供了背景参考：

| 指标 | 全量数值 |
|---|---:|
| 平均 score | 0.2648 |
| 平均 turns | 2.2519 |
| 平均 valid actions | 1.5296 |
| 平均 valid searches | 0.7995 |
| invalid-action 样本数 | 115 |
| valid search 大于等于 2 的样本 | 16 |
| 包含 `<plan>` 的样本 | 0 |
| `<search>` 内包含 `||` 的 action | 0 |

全量 valid-search 分布：

```text
{0: 95, 1: 278, 2: 15, 3: 1}
```

全量数据中，不同 valid search 次数对应的平均 score：

| valid searches | 样本数 | 平均 score |
|---:|---:|---:|
| 0 | 95 | 0.0211 |
| 1 | 278 | 0.3633 |
| 2 | 15 | 0.0000 |
| 3 | 1 | 0.0000 |

这说明全量数据中确实存在少量多 search 样本，但这些样本并没有带来更高 score，而且很多噪声来自训练早期或后期不稳定行为。因此，使用 `step 100-250` 作为主分析窗口更合理。

## 8. Phase 0 结论

在稳定训练窗口 `step 100-250` 中，Search-R1 baseline 已经学会了可靠但很窄的策略：

```text
一次有效搜索 + 一次最终回答
```

它没有自然展现 LiteCoA 需要的三类关键行为：

```text
1. plan-guided decomposition
2. 每轮多个 query 的并行检索
3. 证据不足时的多轮补搜
```

因此，下一步不应该直接大改 GRPO，而应该先实现轻量 LiteCoA parser 和 client-side agent loop，验证以下闭环：

```text
<plan> 生成
  -> 解析 parallel queries
  -> batched retrieval
  -> 格式化 <information>
  -> 继续生成 answer 或 follow-up search
```

这个闭环跑通后，再把 LiteCoA 逻辑接入现有 Search-R1 / verl rollout manager。
