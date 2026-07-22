# 金融投研并行检索 Agent：Phase 0-3 阶段总结

## 当前目标

将已完成的通用并行检索 Agent 迁移到 FinQA 金融数值问答，先建立真实的金融检索与零样本基线，再决定领域 SFT 和强化学习优化方向。

## Phase 0：环境与资产准备

- 远程项目：`/mnt/data/suiqiuyi/Finance_Agent`，分支 `finance-agent`。
- 复用并验证原有检索环境，并建立 `finance-agent-train` 推理/训练环境。
- 同步可直接 HuggingFace 推理的并行检索模型：`parallel_search_qwen25_3b_step900`。
- 模型加载与 GPU 单条生成 smoke 通过。

## Phase 1：FinQA 数据与金融语料

- 采用 FinQA 官方 train/dev/test 划分：train 6,251、dev 883、test 1,147 个问题。
- 构建报告文本与表格行级语料，严格排除 answer、program、gold evidence，避免检索泄漏。
- 语料规模：train 23,029 chunks、dev 3,194 chunks、test 4,217 chunks。
- 处理后的 QA 记录保留 `report_id`、`gold_answer`、`program` 和 `gold_evidence`，用于后续检索范围控制、SFT 与评测。

## Phase 2：金融检索器

- 使用 `intfloat/e5-base-v2` 编码器和 FAISS Flat 索引，服务接口与原并行检索 Agent 保持一致：`POST /retrieve`。
- dev 索引包含 3,194 个金融报告 chunk；检索服务已稳定启动。
- 50 条 direct-question retrieval smoke：Top-3 any-evidence hit 54.0%，evidence recall 35.79%。
- 结论：基础检索链路可用，但直接用问题在全局报告库检索存在明显召回空间。

## Phase 3：零样本并行检索 Agent

### 50 条 dev smoke

- 设置：Qwen 并行检索模型 `step-900`，贪心解码，最多 3 轮，每轮最多 3 query，E5 Top-3 全局检索。
- 运行结果：50/50 无错误；50/50 有 `<answer>`；无伪造 `<information>`；无最大轮次超限。
- 行动行为：50/50 为单轮双 query 并行检索，说明通用并行检索格式和 action 能力已迁移。
- 结果指标：49 条有有效 gold answer，其中 EM 2/49（4.08%），SubEM 6/49（12.24%）。

### 失败归因样本

问题：`what percentage of total commercial mortgages were at fair value?`

- 正确计算为 `(1050 + 1401) / (1301 + 2148) = 71.1%`。
- 模型两个 query 都围绕 `fair value`，召回了分子表 `commercial mortgages at fair value`，但未召回分母表 `total commercial mortgages`，最终猜测 `74%`。
- 分母表确实位于同一报告，使用定向 query `total commercial mortgages PNC 2009 2008` 时 E5 Top-1 可命中。
- 原 query 下正确分母表不在 Top-10，说明主要瓶颈是全局跨报告检索与缺乏变量互补 query，而不是 observation 长度。该条 `<information>` 实际仅 672 tokens，未触发 1000 token 截断。

## 优化前基线与下一步

- 正在运行 FinQA dev 全量 883 条零样本验证，配置与 50 条 smoke 一致，作为优化前全局检索 baseline。
- 优化重点不是盲目扩大 information 长度，而是：基于 `report_id` 的报告范围控制、金融计算题的变量级 query 分解，以及后续金融 SFT 数据构造。
- 全量基线完成后进入 Phase 4：先实现 report-aware retrieval，再评估其对证据召回和 FinQA EM 的增益；之后再决定金融 SFT 的具体轨迹格式。
