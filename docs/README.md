# 文档索引

本目录同时保存通用 Parallel Search（NQ）和金融迁移（FinQA）两条实验线。
阶段编号以各自报告为准，不将 NQ Phase 3 与 Finance Phase 3 混用。

## Finance / FinQA

| 文档 | 内容 | 状态 |
| --- | --- | --- |
| [`finance_agent_plan.md`](./finance_agent_plan.md) | 金融迁移阶段定义与当前状态 | Phase 0-4 完成，Phase 5 待完成 |
| [`finance_phase0_3_summary.md`](./finance_phase0_3_summary.md) | 数据、检索器、teacher 数据构造、LoRA SFT 与评测 | 完成 |
| [`finance_phase4_grpo_report.md`](./finance_phase4_grpo_report.md) | veRL GRPO、reward、稳定性与 Step200 全量 dev 结果 | 完成 |
| [`finance_reproduction.md`](./finance_reproduction.md) | 环境、数据、索引、SFT、GRPO 与最终评测命令 | 完成 |
| [`finance_finqa_data_audit.md`](./finance_finqa_data_audit.md) | FinQA split、chunk 与防泄漏审计 | 完成 |
| [`finance_finqa_retriever_smoke.md`](./finance_finqa_retriever_smoke.md) | E5 + FAISS 50 条检索基线 | 完成 |
| [`finqa_eval/20260727/README.md`](./finqa_eval/20260727/README.md) | 评测 JSON、trajectory、W&B 配置与训练曲线索引 | 完成 |

当前主要结果均为 FinQA dev：

| 模型 | Numeric EM |
| --- | ---: |
| 原始 Step900 并行检索策略 | 11.83% |
| FinQA LoRA SFT，Top-3 / info1500 | 45.01% |
| FinQA GRPO Step200 | 54.88% |

尚未完成的最终实验：repaired merged v3 严格 pre-GRPO A/B、单 query
对照、FinQA test 评测、393 条错误归因和多随机种子重复。

## Parallel Search / NQ

| 文档 | 内容 |
| --- | --- |
| [`LiteCoA-Search-R1-Plan.md`](./LiteCoA-Search-R1-Plan.md) | 初始 LiteCoA 总体设计 |
| [`phase0_baseline_stable_window.md`](./phase0_baseline_stable_window.md) | NQ 原始基线 |
| [`phase1_litecoa_infer.md`](./phase1_litecoa_infer.md) | LiteCoA 推理原型 |
| [`phase2_litecoa_data_report.md`](./phase2_litecoa_data_report.md) | NQ SFT 数据构造 |
| [`phase3_litecoa_sft_report.md`](./phase3_litecoa_sft_report.md) | NQ LoRA SFT |
| [`phase4_litecoa_rollout.md`](./phase4_litecoa_rollout.md) | 并行 rollout 与 veRL 接入 |
| [`phase5_litecoa_reward.md`](./phase5_litecoa_reward.md) | hard reward 与最终并行搜索训练 |
| [`project_summary.md`](./project_summary.md) | 通用 Parallel Search 项目总结 |

## 归档约定

- 报告正文、聚合 JSON、少量公开数据样本和曲线可以进入 Git。
- API key、`.env`、模型权重、checkpoint、完整训练数据、retrieval index 和缓存禁止进入 Git。
- `trajectories.jsonl` 仅归档公开 benchmark 的评测轨迹；新增前仍需执行敏感信息扫描。
- 所有正式结果必须写明 split、denominator、解码温度、Top-K、information 上限和模型加载方式。
