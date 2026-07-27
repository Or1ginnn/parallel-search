# FinQA Retriever 开发集 Smoke 报告

## 目标

验证 Finance Research Agent 的金融语料能否被现有 Search-R1 本地 dense retriever 正确加载和检索。该实验只测 retriever，不使用 Qwen Agent，不进行 SFT 或 GRPO。

## 配置

- 数据：FinQA dev split。
- 语料：`corpus_dev.jsonl`，3,194 个报告正文/表格 chunk。
- Encoder：`intfloat/e5-base-v2`。
- 索引：FAISS `IndexFlatIP`，CPU 检索；E5 编码使用单张 GPU。
- 服务：`search_r1/search/retrieval_server.py`，接口为 `POST /retrieve`。
- 请求：直接将 FinQA 原始问题作为单个 query，`topk=3`。
- 评测：`smoke_dev_50.jsonl`，固定随机种子的 50 条开发样本。

## 链路验证

```text
FinQA 问题
-> E5 query embedding
-> FAISS top-3 检索
-> 返回 title/text/score
-> 与人工 gold evidence 做词元覆盖匹配
```

服务成功返回 HTTP 200。以“what is the average payment volume per transaction for american express?”为例，top-1 正确召回 American Express 的 payments volume、transactions 和 cards 表格行。

## 指标

证据命中使用 gold evidence 与 retrieved document 的词元覆盖率，阈值为 0.8。该指标衡量“相关人工证据是否出现在 top-3 中”，不等同于最终 QA 正确率。

| 指标 | 结果 |
| --- | ---: |
| 样本数 | 50 |
| 请求错误 | 0 |
| 任一 gold evidence 命中 | 27 / 50 (54.0%) |
| 所有 gold evidence 命中 | 10 / 50 (20.0%) |
| 逐证据召回率 | 35.79% |

## 结论

金融 retriever 已形成可用闭环，但直接使用原问题做单 query 的证据召回仍有明显提升空间。该结果属于当前阶段定义中的 Phase 2 检索基线：模型若能将金融问题拆为互补的指标、时间、同比或原因 query，理论上应提升多证据问题的 evidence recall。

## 产物路径

- 运行时索引：`data/finance_finqa/index/dev/e5_Flat.index`
- 运行时服务日志：`data/finance_finqa/index/dev/retriever.log`
- 归档逐样本结果：[`retrieval_smoke_dev_50.jsonl`](./finqa_eval/20260727/phase2_retriever/retrieval_smoke_dev_50.jsonl)
- 归档指标报告：[`retrieval_smoke_dev_50_report.json`](./finqa_eval/20260727/phase2_retriever/retrieval_smoke_dev_50_report.json)
- 固定 smoke 输入：[`smoke_dev_50.jsonl`](./finqa_eval/20260727/phase2_retriever/smoke_dev_50.jsonl)
