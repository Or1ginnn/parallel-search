# Phase 3 FinQA SFT 归档

本目录保存 `finance_phase0_3_summary.md` 使用的原始证据文件：

- `data_merge_report.json`：4 个 teacher 数据构造 worker 的 accepted/rejected 与耗时；
- `dataset_info.json`：LLaMA-Factory ShareGPT 数据注册；
- `sft_20_report.json`：20 条严格闭环报告；
- `sft_smoke_train_results.json`：SFT smoke 训练结果；
- `sft_train_results.json`：500 条正式 SFT 汇总；
- `sft_trainer_state.json`：126-step loss、学习率与梯度历史；
- `sft_raw_example.json`：teacher 构造阶段的真实三消息样本；
- `sft_llamafactory_example.json`：实际送入 LLaMA-Factory 的多轮 ShareGPT 样本；
- `llamafactory_training_loss.png`：LLaMA-Factory 原始 loss 图；
- `finqa_sft_training_curves.png`：根据 trainer state 重绘的报告曲线。

这些文件不包含模型权重、API key 或完整训练数据正文。
