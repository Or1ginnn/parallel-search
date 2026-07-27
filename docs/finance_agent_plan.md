# Finance Research Agent 阶段计划与状态

## 项目目标

面向金融年报数值问答，将通用并行检索 Agent 迁移为金融投研问答 Agent。模型一次生成两个互补 query，在指定报告范围内检索真实文本与表格证据，再完成数值计算或事实回答。

```text
金融问题
-> 识别所需变量
-> 并行生成互补 query
-> report-aware 金融检索
-> 按 query 回填真实 evidence
-> 生成简短答案
-> SFT 冷启动与 GRPO 对齐
```

## Phase 0：环境与初始模型

状态：完成。

- 建立 `finance-agent` 开发分支和服务器环境。
- 复用 Qwen2.5-3B 并行检索 step900 模型。
- 验证 vLLM、retriever、并行 search action、information 回填与 trajectory 落盘。

验收：原有并行检索链路可在金融环境中复现。

## Phase 1：FinQA 数据与金融语料

状态：完成。

- 处理官方 FinQA train/dev/test：6,251 / 883 / 1,147 个问题。
- 从财报 `pre_text`、`post_text` 和 table 构建 30,440 个检索 chunks。
- 检索语料排除 answer、program 和 gold-evidence 标注字段，防止标签泄漏；
  证据对应的原始年报文本与表格仍正常保留。
- 保留 page-level `report_id`、gold answer、program 与 evidence 供训练和评测使用。

验收：三组报告无交集，平均 gold-evidence token 覆盖率均超过 99%；完整覆盖
样本比例约为 85%-87%，两类指标不混用。

## Phase 2：金融 Retriever 与零样本基线

状态：完成。

- 使用 `intfloat/e5-base-v2` 与 FAISS 构建金融索引。
- 保持批量 `/retrieve` 接口与分 query `<information>` 格式。
- 加入 page-level report-aware retrieval，减少跨公司、跨年份干扰。
- 完成 50 条检索 smoke、50 条 Agent smoke 与 883 条 dev 全量零样本评测。

验收：原 step900 模型无运行错误、无伪造 information，FinQA Numeric EM 为 11.83%。

## Phase 3：金融 SFT 冷启动

状态：完成。

- 使用 `deepseek-v4-pro` teacher 生成 `<think>/<search>/<answer>`。
- 使用真实 E5 retriever 回填 `<information>`，teacher 不生成 evidence。
- 先完成 20 条闭环，再由 4 worker 构造并合并 500 条唯一轨迹。
- 转换为多轮 ShareGPT messages，将 information 放在 user observation 侧。
- 使用 LLaMA-Factory + PEFT LoRA 对 step900 模型训练 2 epochs。
- LoRA rank 64、alpha 128、effective batch 8、BF16、cosine LR。

验收：FinQA dev greedy Numeric EM 从 11.83% 提升至 45.01%。完整过程见 `finance_phase0_3_summary.md`。

## Phase 4：veRL GRPO 对齐

状态：完成。

- 从已验证等价的 SFT merged v3 模型开始训练。
- 使用 veRL + GRPO 与并行 vLLM rollout。
- 保留答案 reward、通用搜索行为 shaping、hard-zero 约束和 KL 稳定项。当前
  `evidence_hit` 只检查最终答案字符串是否出现在 information 中，不等价于
  gold-evidence 路径奖励。
- 增加 checkpoint 恢复、best trajectory、W&B 训练正确率与流式 validation 汇总。

验收：step200 FinQA dev greedy Numeric EM 达到 54.88%，格式有效率和 answer coverage 均为 100%，无 parser/agent warning。详见 `finance_phase4_grpo_report.md`。

## Phase 5：错误分析与业务化评测

状态：待完成。

- 按 retrieval miss、证据命中但计算错误、交互格式错误拆分失败样本。
- 对 1-step 与多步 program 分层统计准确率。
- 与单 query、原始并行模型、金融 SFT 和金融 GRPO 做统一对比。
- 输出营收计算、比例变化、财务指标同比等典型案例。
- 固化可复现命令、模型版本、数据版本和报告范围检索约束。
- 在 FinQA test 上进行一次性最终评测，并补充 merged v3 严格 A/B 与单 query
  对照。

验收：形成完整误差报告、对照实验表和可展示的金融投研问答案例。
