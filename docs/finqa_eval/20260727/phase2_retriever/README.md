# Phase 2 FinQA Retriever 归档

本目录保存金融检索器 50 条开发集 smoke 的原始证据：

- `audit_report.json`：FinQA 分割、报告和 corpus chunk 统计；
- `smoke_dev_50.jsonl`：固定随机种子的 50 条输入；
- `retrieval_smoke_dev_50.jsonl`：每条问题的 Top-3 文档与 gold evidence 命中；
- `retrieval_smoke_dev_50_report.json`：54.0% any-evidence hit 与 35.79% evidence recall 汇总。

检索语料仅包含财报正文和表格，不包含答案、program 或 gold evidence 标签。
