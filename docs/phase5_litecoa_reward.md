# Phase 5：LiteCoA Reward 第一版

## 目标

Phase 4 已经证明 LiteCoA rollout 链路能跑通，但只用最终 answer EM 奖励时，3B 模型容易出现格式漂移、无效 action 增多、并行 query 不稳定等问题。

Phase 5 第一版 reward 不做惩罚，只在原 answer EM 外加入小额正向 shaping，先让模型明确知道哪些 LiteCoA 行为是有收益的。

## Reward 组成

训练 reward：

```text
reward = answer_em
       + 0.05 * plan_once
       + 0.05 * answer_present
       + 0.05 * no_generated_information
       + 0.05 * evidence_hit
       + 0.03 * valid_search
       + 0.03 * parallel_evidence_hit
```

最大值为 `1.26`。

验证 reward 仍保持原 answer EM，不加 shaping bonus，保证 `val/test_score/nq` 可以和 Phase 4 及 Search-R1 baseline 对比。

## 关键边界

格式奖励只看模型生成的 response，不看 prompt。因为 LiteCoA prompt 本身包含 `<plan>`、`<search>`、`<information>`、`<answer>` 这些规则说明，不能把 prompt 里的 tag 计入 reward。

实现里使用 rollout 返回的 `info_mask`：

- `attention_mask` response 段：完整 response，包括模型生成内容和 retriever 插入的 `<information>`。
- `info_mask` response 段为 1：模型生成 token，用于判断 `<plan>`、`<search>`、`<answer>` 和是否生成了 `<information>`。
- `info_mask` response 段为 0：retriever observation token，用于判断 evidence 是否命中 gold answer。

这样可以避免两类污染：

1. prompt 里的标签不会让 `plan_once`、`answer_present` 等格式奖励虚高。
2. 模型自己生成的 `<information>` 不会让 `evidence_hit` 虚高。

## 当前实现文件

- `verl/utils/reward_score/litecoa_qa.py`
- `verl/trainer/main_ppo.py`
- `verl/trainer/config/ppo_trainer.yaml`
- `scripts/train/train_grpo_litecoa_qwen25_3b.sh`
- `scripts/train/train_grpo_litecoa_smoke.sh`

LiteCoA 启动脚本会设置：

```text
reward_model.litecoa_reward=true
```

默认配置中该开关为 `False`，因此原 Search-R1 训练入口不会自动使用 LiteCoA shaping reward。
