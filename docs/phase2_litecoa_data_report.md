# Phase 2 LiteCoA Data Construction Report

## 1. Phase 2 目标

Phase 2 的目标是构造 LiteCoA SFT 冷启动数据。数据构造遵循：

```text
NQ question
-> DeepSeek V4 Pro teacher 生成 <think><plan><think><search>
-> 真实 Search-R1 retriever 填充 <information>
-> DeepSeek V4 Pro teacher 基于 evidence 生成 <think><answer>
-> 规则过滤
-> 固化 SFT JSONL 数据集
```

关键原则：

```text
1. <information> 必须来自真实 retriever，不能由 teacher 编造。
2. 每轮 <search> 最多 3 个并行 query，用 "||" 分隔。
3. 最多 2 轮 search，用于表达第一轮证据不足后的补搜。
4. <answer> 必须是短答案。
5. 最终 SFT 训练主字段是 messages。
```

最终 assistant 轨迹格式：

```text
<think>...</think>
<plan>...</plan>
<think>...</think>
<search>q1 || q2</search>
<information>真实检索结果</information>
<think>...</think>
<answer>短答案</answer>
```

多轮补搜格式：

```text
<think>...</think>
<plan>...</plan>
<think>...</think>
<search>q1 || q2</search>
<information>第一轮真实检索结果</information>
<think>证据不足，需要补搜</think>
<search>q3 || q4</search>
<information>第二轮真实检索结果</information>
<think>证据充分</think>
<answer>短答案</answer>
```

## 2. Phase 2 Smoke Test

原始 smoke test artifact 保存在：

```text
trajectory/phase2_smoke/
```

### 实验设置

```text
时间：2026-06-14
样本：从 data/nq_search/train.parquet 固定随机种子 20260614 抽取 50 条
Retriever：POST http://127.0.0.1:8000/retrieve，预检返回 HTTP 200
LiteCoA 原型模型：Qwen2.5-3B base
Search-R1 baseline：nq-search-r1-grpo-qwen2.5-3b-em global_step_200
最大搜索轮数：3
LiteCoA 每轮最多 query 数：3
```

### 链路稳定性

| 模式 | 样本数 | 报错数 | 出现 `<answer>` | parser warning | agent warning | 平均 search_turns | 平均 total_queries | 平均耗时/条 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LiteCoA | 50 | 0 | 50 | 4 | 0 | 1.22 | 1.90 | 5.69s |
| Search-R1 baseline | 50 | 0 | 50 | 0 | 0 | 1.00 | 1.00 | 2.31s |

### 准确率

| 模式 | 官方 qa_em EM 正确数 | 官方 qa_em EM 准确率 | 参考 subEM 正确数 | 参考 subEM 准确率 |
| --- | ---: | ---: | ---: | ---: |
| LiteCoA | 1/50 | 2.00% | 12/50 | 24.00% |
| Search-R1 baseline | 28/50 | 56.00% | 28/50 | 56.00% |

### Smoke Test 结论

LiteCoA 推理链路可以稳定跑通：

```text
模型生成 <think><plan><search>
-> parser 解析 query
-> retriever 返回 <information>
-> 模型继续生成 <answer>
```

但 base 模型不能直接作为高质量 teacher：

```text
1. 容易生成解释性长答案。
2. 会出现自造 <information>。
3. 多轮时更容易 tag 混乱。
4. query 可能偏题或混入 prompt/tag。
```

因此 Phase 2 数据构造必须采用强 teacher、真实 retriever、分段生成和硬过滤。

## 3. 数据构造脚本

主脚本：

```text
scripts/data_process/build_litecoa_sft.py
scripts/data_process/build_litecoa_sft.sh
```

通用运行方式：

```bash
bash scripts/data_process/build_litecoa_sft.sh 100
bash scripts/data_process/build_litecoa_sft.sh 1000
```

参数含义：

```text
第 1 个参数：目标 accepted 样本数
第 2 个参数：最多尝试候选数，默认 target_count * 3
第 3 个参数：输出文件前缀，默认 data/litecoa_sft/litecoa_${TARGET_COUNT}/litecoa_sft_${TARGET_COUNT}
```

脚本默认优先使用本地：

```text
data/nq_search/train.parquet
```

避免构造时依赖 Hugging Face 网络下载。

## 4. 1000 条数据集固化

### 数据版本

```text
Version: litecoa_sft_1000_v1
Final dataset: data/litecoa_sft/litecoa_1000/litecoa_sft_1000_v1.jsonl
Rejected samples: data/litecoa_sft/litecoa_1000/litecoa_sft_1000_v1_rejected.jsonl
Build report: data/litecoa_sft/litecoa_1000/litecoa_sft_1000_v1_report.json
```

### 校验 Hash

```text
6ab1876a3fdfe46cde7119f55df8160e05d8046468721ebdff285dadd8378f50  data/litecoa_sft/litecoa_1000/litecoa_sft_1000_v1.jsonl
9c2e5c63c09ab16abf0130be46e70213a87438cab155d521934ba7dc8ae4a4eb  data/litecoa_sft/litecoa_1000/litecoa_sft_1000_v1_rejected.jsonl
29a2ea8acb454e3549bfb7ab3d5301aaa7d450f9579883b6cd641c8caf3c7d72  data/litecoa_sft/litecoa_1000/litecoa_sft_1000_v1_report.json
```

### 构造报告

```text
target_count: 1000
accepted: 1000
rejected: 412
processed: 1412
accepted_rate: 70.8%
elapsed_sec: 7918.285
model: deepseek-v4-pro
retriever_url: http://127.0.0.1:8000/retrieve
max_turns: 2
max_queries_per_turn: 3
topk: 3
```

## 5. Accepted 数据质量

```text
samples: 1000
unique_ids: 1000
source: nq

format_valid: 1000 / 1000
answer_match: 1000 / 1000
answer_subem: 1000 / 1000
answer_em: 907 / 1000
evidence_hit: 1000 / 1000
warnings: 0
bad_queries: 0
tag_structure_issues: 0
```

轨迹结构检查：

```text
每条恰好 1 个 <plan>
每条恰好 1 个 <answer>
<information> 数量与 search turn 数一致
query 中无 XML tag
query 中无 query1/query2 占位符
```

答案长度：

```text
average_answer_words: 2.42
p95_answer_words: 5
max_answer_words: 12
```

## 6. 搜索行为

搜索轮数分布：

```text
1 search turn: 871
2 search turns: 129
multi_turn_rate: 12.9%
```

累计 query 数分布：

```text
1 query: 76
2 queries: 774
3 queries: 42
4 queries: 66
5 queries: 39
6 queries: 3
```

说明：

```text
query 数是整条 trajectory 的累计值。
单轮 <search> 仍然最多 3 个并行 query。
例如 5 query = 第一轮 2 query + 第二轮 3 query。
例如 6 query = 第一轮 3 query + 第二轮 3 query。
```

## 7. 多轮样本示例

### 示例：`train_76248`

问题：

```text
who hosted the tv show *jeena isi ka naam hain*?
```

第一轮搜索：

```text
<search>"Jeena Isi Ka Naam Hain" host || "Jeena Isi Ka Naam Hain" TV show</search>
```

第一轮检索结果主要命中电影页面和干扰主持人，证据不足。

第二轮搜索：

```text
<search>"Jeena Isi Ka Naam Hai" host TV show || "Jeena Isi Ka Naam" host || "Jeena Isi Ka Naam Hai" Suresh Oberoi</search>
```

第二轮检索命中：

```text
Doc 1(Title: "Suresh Oberoi")
In addition to acting and singing, he was also the host of the show "Jeena Isi Ka Naam Hai" on Zee TV...
```

最终答案：

```text
<answer>Suresh Oberoi</answer>
```

### 示例：`train_44063`

问题：

```text
when does bts go on the james corden show?
```

第一轮搜索：

```text
<search>BTS James Corden show date || BTS Late Late Show James Corden || BTS on James Corden</search>
```

第一轮主要命中 James Corden / Late Late Show 泛信息。

第二轮搜索：

```text
<search>BTS first appearance Late Late Show James Corden date || BTS November 2017 James Corden || BTS Late Late Show 2017</search>
```

第二轮检索命中：

```text
Doc 3(Title: "DNA (BTS song)")
BTS performed their single "DNA" ... on November 30 at "The Late Late Show with James Corden"...
```

最终答案：

```text
<answer>November 30, 2017</answer>
```

## 8. Rejected 样本分析

Rejected 总数：

```text
412
```

主要原因：

```text
missing_answer_after_max_turns: 171
answer_match_false: 109
evidence_hit_false: 118
missing_search: 32
answer_too_long: 16
retriever timeout: 5
```

这些 rejected 样本未进入最终 SFT 数据集。整体看，过滤器有效拦截了不收尾、答案不匹配、证据不足、格式异常和过长答案。

## 9. 数据格式

每条样本为 JSONL 一行：

```json
{
  "id": "train_xxx",
  "source": "nq",
  "question": "...",
  "gold_answer": ["..."],
  "messages": [
    {"role": "system", "content": "...LiteCoA system prompt..."},
    {"role": "user", "content": "Question: ..."},
    {"role": "assistant", "content": "<think>...</think>...<answer>...</answer>"}
  ],
  "queries_by_turn": [["q1", "q2"], ["q3"]],
  "quality_flags": {
    "format_valid": true,
    "answer_em": true,
    "answer_subem": true,
    "answer_match": true,
    "evidence_hit": true,
    "num_search_turns": 1,
    "num_queries": 2,
    "multi_turn": false,
    "warnings": []
  }
}
```

SFT 主训练字段：

```text
messages
```

其他字段用于分析、过滤和复查。

## 10. Phase 2.3 结论

`litecoa_sft_1000_v1` 满足 Phase 2.3：

```text
1. 数据量达到 1000 条 accepted。
2. 全部样本格式合法。
3. 全部样本 answer_match。
4. 全部样本 evidence_hit。
5. 无 warning、无坏 query、无 tag 结构问题。
6. 包含 129 条两轮补搜样本，能体现 LiteCoA 多轮补搜行为。
7. 答案整体短，适合 SFT 冷启动。
```

下一阶段：

```text
Phase 3: LoRA SFT
Training input: data/litecoa_sft/litecoa_1000/litecoa_sft_1000_v1.jsonl
```
