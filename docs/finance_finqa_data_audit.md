# FinQA 数据与语料审计

## 数据来源

- 官方仓库：`https://github.com/czyssrs/FinQA`
- 原始数据位置：`data/finance_finqa/raw/FinQA/dataset/`
- 许可证：CC BY 4.0。

## 原始分割

| Split | QA 数量 | 独立报告数 |
| --- | ---: | ---: |
| train | 6,251 | 2,110 |
| dev | 883 | 299 |
| test | 1,147 | 380 |

三个 split 的报告文件名没有重叠，因此可以分别构建训练、开发与测试检索语料，不存在跨 split 的报告泄漏。

## 生成产物

预处理脚本：`scripts/data_process/prepare_finqa.py`。

- `train.jsonl`、`dev.jsonl`、`test.jsonl`：金融问答、标准答案、执行答案、程序与 gold evidence。
- `corpus_train.jsonl`、`corpus_dev.jsonl`、`corpus_test.jsonl`：分别用于训练、开发和测试的报告检索 chunk。
- `corpus_all.jsonl`：全部报告 chunk 的汇总，仅供分析或开放语料实验。
- `smoke_dev_50.jsonl`：固定随机种子的 50 条开发集 smoke 样本。
- `audit_report.json`：机器可读的样本、报告与 chunk 统计。

## Chunk 规则

- 正文：将 `pre_text` 与 `post_text` 按最多 5 段、最多 1,200 字符切块。
- 表格：首行作为表头，每个数据行独立序列化为 `列名: 数值` 的表格 chunk。
- 元信息：每个 chunk 保留 `id`、`report_id`、`split`、`source_type` 和 `position`。

最终语料规模：

| Split | 总 chunk | 正文 chunk | 表格 chunk |
| --- | ---: | ---: | ---: |
| train | 23,029 | 11,428 | 11,601 |
| dev | 3,194 | 1,592 | 1,602 |
| test | 4,217 | 2,049 | 2,168 |
| all | 30,440 | 15,069 | 15,371 |

## 防泄漏策略

检索语料仅由原始报告的 `pre_text`、`post_text` 和 `table` 构建。`qa.answer`、`qa.exe_ans`、`qa.program`、`qa.gold_inds` 及任何 QA 专属检索字段不会进入 corpus。

对每个 QA 的人工 gold evidence 和同报告 corpus 做词元覆盖审计，覆盖率仅用于验证原始证据仍可由报告语料表达，不是 retriever 指标：

| Split | 平均 gold-evidence 词元覆盖率 | 完整覆盖样本 |
| --- | ---: | ---: |
| train | 99.27% | 5,440 / 6,251 |
| dev | 99.33% | 761 / 883 |
| test | 99.20% | 975 / 1,147 |

下一阶段将使用 `corpus_dev.jsonl` 和 `smoke_dev_50.jsonl` 构建金融 FAISS 索引，并测量真实 retrieval evidence hit，而不是只做词面覆盖检查。
