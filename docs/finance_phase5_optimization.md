# Finance Phase 5：错误分析与稳定性优化记录

状态：进行中。本文先记录 checkpoint 选择、continuation 数值故障与本地修复，后续与失败归因、对照实验和最终 test 结果统一整合。

## 1. 当前目标

Phase 4 已得到可稳定运行的 FinQA GRPO 模型。Phase 5 不再单纯增加训练步数，而是完成：

1. 冻结当前最佳 checkpoint；
2. 修复长程 GRPO 中的非有限梯度；
3. 对全量 dev 错误进行 retrieval / reasoning 分层归因；
4. 依据主要失败类型设计单一 V2 实验；
5. 补齐单 query 对照和最终 test 评测。

## 2. Checkpoint 选择

统一评测协议：FinQA dev 全量 883 条，871 条具有可评分 gold answer；greedy 解码，report-aware retrieval，Top-3，information 最多 1500 tokens，无 calculator action。

| Checkpoint | Numeric EM | 正确题数 | answer / format | warnings / errors |
| --- | ---: | ---: | ---: | ---: |
| Step200 | 54.88% | 478 / 871 | 883 / 883 | 0 |
| Step300 | 54.31% | 473 / 871 | 883 / 883 | 0 |

Step300 比 Step200 少答对 5 题，下降 0.57 个百分点。Step300 在固定 320 条在线验证子集上的 55% 没有转化为全量 dev 提升，因此当前正式最佳模型仍为 Step200。Step300 作为完整可恢复断点保留，不作为最终模型。

Step300 全量评测产物位于服务器：

```text
/mnt/data1/zar/finance/eval/finqa_phase4_step300_nocalc_topk3_info1500_dev_full/
├── summary.json
├── trajectories.jsonl  # 883 lines
└── eval.log
```

continuation 实验清理了 `actor/global_step_325` 和 `actor/global_step_350`，保留 Step225、250、275、300 及 Step300 的模型、optimizer、scheduler 和 RNG 状态。checkpoint 根目录由 214 GB 降至 143 GB，释放约 71 GB。

## 3. Step332 非有限梯度现象

W&B continuation run：`pmxm122b`，实验名 `finqa-phase4-grpo-v3-step200-400`。

稳定的 Step50-200 运行共有 150 条训练记录，没有非有限梯度。continuation 从 Step201 运行到 Step360，首次异常出现在 Step332：

```text
actor/pg_loss                 = 0.131578
actor/ppo_kl                  = 0.000112
actor/kl_loss                 = 0.226622
critic/advantages/max         = 1.788850
actor/grad_norm               = NaN
actor/optimizer_step_skipped  = 0.2
actor/skip_nonfinite          = 0.2
```

前向 loss、PPO KL、reference KL 和 advantage 都是有限值，但 5 个 PPO mini-batch 中有 1 个在反向传播后产生非有限梯度。之后跳步比例快速扩大：Step334 为 0.4，Step335 为 0.8，Step338 为 1.0。Step332-360 中除 Step333 外均出现非有限梯度，共 28 个异常 step。

该现象不等价于“某个 batch 太长”。例如 Step341 的最大 response length 仅为 1607 tokens，仍然跳过全部 optimizer update。长序列会增加极端 token 出现概率，但不能单独解释持续 NaN。

## 4. 根因

原 `low_var_kl` 实现为：

```python
kl = ref_logprob - logprob
ratio = torch.exp(kl)
kld = ratio - kl - 1
return torch.clamp(kld, min=-10, max=10)
```

当少数 token 的 `ref_logprob - logprob` 很大时，`exp(kl)` 会先溢出为 `Inf`。最终 clamp 会把前向结果显示为 10，因此平均 `actor/kl_loss` 仍是正常有限值；但反向传播经过 `exp` 时可能计算 `0 * Inf`，产生 NaN 梯度。平均 PPO KL 和裁剪后的 KL loss 都会隐藏这种单 token 极值。

本地最小复现结果：

```text
FP32: old_forward=10.0, old_grad=NaN
BF16: old_forward=10.0, old_grad=NaN
```

这与 Step332 的“前向指标有限、反向 grad norm 为 NaN”完全一致。当前将其视为高置信度根因；最终验收仍需 A800 训练 smoke 证明修复后不再触发非有限梯度。

## 5. 数值稳定性修复

修复位置：

```text
verl/trainer/ppo/core_algos.py
verl/workers/actor/dp_actor.py
tests/test_core_algos.py
```

新实现：

```python
kl = (ref_logprob.float() - logprob.float()).clamp(-20.0, 20.0)
kld = torch.expm1(kl) - kl
return torch.clamp(kld, min=-10, max=10)
```

修复原则：

- log-prob 差值强制使用 FP32；
- 在指数运算前限制到 `[-20, 20]`；
- 使用 `expm1` 改善接近 0 时的数值精度；
- 保留最终 `[-10, 10]` penalty 范围；
- 正常差值范围内与原公式一致；
- 极端差值的前向 penalty 仍为 10，但反向梯度保持有限。

新增 W&B 指标：

- `actor/kl_log_ratio_abs_max`：当前更新中最大的 `|ref_logprob-logprob|`；
- `actor/kl_log_ratio_clipfrac`：超过稳定阈值 20 的 token 比例。

新增回归测试覆盖正常公式、FP32 极端差值和 BF16 极端差值。本地已通过 `py_compile`、`git diff --check` 以及实际函数的 FP32/BF16 前向与反向检查。完整训练环境测试和恢复 smoke 尚待 A800 执行。

本次修复没有修改任务 reward、学习率、`KL coef=0.005`、PPO clip 或数据协议，只消除 `low_var_kl` 的明确溢出路径。

## 6. 下一步验收

1. 将修复同步到 A800；
2. 从 Step300 或独立复制的稳定断点运行 10-step smoke；
3. 验收 `grad_norm`、loss、PPO KL 全部有限，`optimizer_step_skipped=0`；
4. 观察新增的 KL 极值与 clip fraction；
5. 不将该 smoke 视为精度实验，也不直接继续长程训练；
6. 对 Step200 的 393 条 scored errors 做 retrieval miss、evidence-hit-but-reasoning-wrong、单位/百分比和多步计算分层统计；
7. 根据最大失败类型确定唯一的 V2 优化实验。

## 7. 当前结论

- Step200 是当前正式最佳 checkpoint，FinQA dev Numeric EM 为 54.88%；
- Step300 没有超过 Step200，继续堆训练步数已经出现平台期；
- Step332 后的训练记录受到 KL 数值溢出和大量 optimizer skip 影响，不可用于判断正常学习趋势；
- 修复数值稳定性是训练安全前置条件，但它本身不保证准确率超过 55%；
- 冲击 60% 需要建立在 Step200 全量错误归因和针对性 V2 设计之上。

## 8. Step200 全量错误归因

已将 GitHub `finance-eval-results-20260727` 分支的 Step200 评测产物刷新到最新状态。本地 `summary.json` 和 883 行 `trajectories.jsonl` 与远端 SHA256 完全一致；分析同时对齐 FinQA 官方 commit `0f16e2867befa6840783e58be38c9efb9229d742` 的 gold evidence、program 和 steps。

871 条可评分样本中有 393 条错误：

| 类型 | 数量 | 错误占比 |
| --- | ---: | ---: |
| 关键操作数基本齐全，但选数/推理/计算错误 | 250 | 63.61% |
| 只检索到部分关键数值 | 104 | 26.46% |
| 基本未命中 gold evidence / program 操作数 | 39 | 9.92% |

重要发现：

- program 操作数全部出现时准确率为 64.16%，部分出现时只有 7.22%；
- 2-step 及以上 program 准确率为 47.74%，比 1-step 的 59.77% 低 12.03 个百分点；
- 69 条错误与正确值相差 1%-5%，22 条存在单位量级错误，10 条符号错误，6 条百分比 100 倍尺度错误；
- 虽然 882 / 883 条轨迹固定生成两个 query，但第二个 query 带来额外 gold evidence 的严格代理只覆盖 58 / 871 条，即 6.66%；
- 当前最大瓶颈是证据出现后的变量选择和数值推理，其次是只检索到部分关键数值，而不是纯 retrieval miss。

完整方法、分层统计和典型样本见 [`finqa_eval/20260728/phase5_step200_error_analysis/report.md`](./finqa_eval/20260728/phase5_step200_error_analysis/report.md)。

## 9. V2 定向 Reward 实现

V2 不增加 `<calculate>` action，不修改 prompt 和评测标准，也不把 gold program
暴露给模型。它建立在 Phase 4 已验证稳定的 hard-zero reward 上，只替换证据 shaping
并增加一个很小的数值近似信号。

训练奖励为：

```text
R_raw = 1.00 * answer_numeric_em
      + 0.05 * answer_present
      + 0.05 * no_generated_information
      + 0.05 * valid_search
      + 0.05 * retrieval_coverage
      + 0.05 * parallel_retrieval_gain
      + 0.05 * numeric_near_miss_quality
```

其中：

- `answer_numeric_em`：继续使用正式 FinQA Numeric EM；这是唯一的 1.0 主奖励；
- `retrieval_coverage`：从 gold program 提取需从报告中取得的数值操作数，忽略
  `const_*` 和 `#0/#1` 中间结果，计算真实 retriever information 中的覆盖比例；
  对无法提取数值操作数的 table program，回退到 gold-evidence 覆盖率；
- `[Query] ...` 行会从证据计算中剔除，模型不能通过在 query 里复述 gold 数值骗取奖励；
- `parallel_retrieval_gain`：第一轮至少有两个不同 query，且多个 query 的证据并集
  比最佳单个 query 覆盖更多操作数时，按新增覆盖比例奖励；重复或同质 query 得 0；
- `numeric_near_miss_quality`：仅对 Numeric EM 仍为 0、但相对误差位于 1%-5%
  的数值答案给线性递减奖励；5% 处为 0，接近 1% 时最多接近 0.05；
- 百分数/小数转换与单位处理复用正式 FinQA 数值解析，例如 `0.935` 与 `93.5%`
  等价；符号错误、100 倍百分比错误和明显单位量级错误不会获得 near-miss 奖励。

hard-zero 规则保持不变：response 截断、无合法 action、出现非法 action、超过最大
轮次、缺少 `<answer>` 或模型伪造 `<information>`，最终奖励直接归零。validation
仍只计算 Numeric EM，不包含任何 shaping bonus。

新增 W&B 指标：

```text
train/retrieval_coverage_mean
train/parallel_retrieval_gain_mean
train/parallel_retrieval_gain_rate
train/numeric_near_miss_quality_mean
train/numeric_near_miss_rate
train/grpo_group_task_score_std_mean
train/grpo_zero_task_score_variance_group_rate
train/grpo_group_reward_std_mean
train/grpo_zero_reward_variance_group_rate
```

在 Step200 的 883 条真实轨迹上做离线重算，得到：

| V2 信号 | 均值 | 正样本数 |
| --- | ---: | ---: |
| retrieval coverage | 0.8865 | 831 / 883 |
| parallel retrieval gain | 0.0438 | 81 / 883 |
| numeric near-miss quality | 0.0152 | 21 / 883 |

这说明新增信号既不是全 0，也不是所有样本固定同分。正式启用位置为：

```text
verl/utils/reward_score/finqa_metrics.py
verl/utils/reward_score/litecoa_qa.py
verl/trainer/main_ppo.py
verl/trainer/ppo/ray_trainer.py
verl/trainer/config/ppo_trainer.yaml
scripts/train/train_finqa_grpo_smoke.sh
scripts/train/train_finqa_grpo_full.sh
tests/test_finqa_v2_reward.py
```

本地已通过 8 项 reward 单测、Python 编译、shell 语法和 `git diff --check`。下一步
先运行 10-step A800 smoke，验收新增指标有变化、hard-zero 正常、KL 与梯度有限；
smoke 通过后从 Step200 启动首轮 200-step V2 实验，由人工根据每 50 steps 的验证
趋势决定是否提前停止。

两个训练入口已同时纠正初始化口径：`BASE_MODEL` 默认指向 Phase 4 最佳
`global_step_200`，并以新实验重新初始化 optimizer、scheduler 和 step 计数；不会再
错误地从 merged-v3 或历史 Step50 开始。checkpoint 恢复为 LiteCoA 的轻量保存方式，
只保存 Actor 模型权重、配置和 tokenizer，不保存 optimizer、scheduler、RNG 或数据
游标，也不支持原地续训。smoke 使用与正式实验一致的 2 卡、batch 32、
每题 5 条采样、temperature 1.0；首轮 V2 上限为 200 steps，Step 0 先验证一次，
之后保持每 50 steps 验证和保存。

首轮 V2 在 Step200 被固定上限终止。W&B `0bokv2n0` 显示验证分数从 Step100 的
50.94% 回升到 Step150 的 52.81% 和 Step200 的 54.38%，最后 25 步训练 Numeric
EM 均值升至 59.88%，因此 200-step 上限过早。后续入口默认改为 2000-step 安全上限
并由人工停止，warmup 调整为 10 步（ratio 0.005）。`ACTOR_INIT_MODEL` 只加载 V2
Step200 的 Actor 权重，`BASE_MODEL` 仍指向 Phase 4 Step200 参考策略；optimizer、
scheduler 和步数从零初始化，checkpoint 继续只保存模型与 tokenizer。

## 10. V2 动态题组采样

V2 正式训练中约一半题组的 5 条 trajectory 获得完全相同的 reward。这类题组经
GRPO 组内归一化后 advantage 全为 0，继续送入 Actor 只消耗 rollout 和反向计算，
不提供有效策略梯度。因此下一轮只加入一项 DAPO 风格的动态题组采样，不叠加
GDPO、IGPO、过程奖励或新的 action。

具体流程：

1. 每个生成批次采样 64 个 question，每题保持 5 条 trajectory；
2. 继续使用现有 FinQA V2 reward 和 hard-zero 规则计算每条 trajectory 的标量分数；
3. 以 `uid` 按题分组，仅保留组内 reward 标准差大于 `1e-8` 的完整题组；
4. 凑齐 32 个有效题组后，才计算 old/ref log-prob、GRPO advantage 和 Actor 更新；
5. 第一批不足时最多再生成一批，两个生成批次后仍不足则明确报错，不用零方差题组
   静默填充训练 batch。

保持不变的部分包括：Step200 初始化模型、V2 reward 系数、hard-zero、无
`<calculate>` prompt、`n_agent=5`、Top-K 3、information 1500 tokens、KL loss
系数 0.005 和 Numeric EM 验证。Actor 学习率默认设为 `7.5e-7`，有效训练 batch
仍为 32 个 question；64 只是 rollout 候选生成 batch，不是优化 batch。

新增 W&B 指标：

```text
train/dynamic_sampling/generated_batch_count
train/dynamic_sampling/candidate_group_count
train/dynamic_sampling/effective_group_count
train/dynamic_sampling/informative_group_rate
train/dynamic_sampling/zero_variance_group_rate
train/dynamic_sampling/filtered_group_rate
train/dynamic_sampling/all_correct_group_rate
train/dynamic_sampling/all_wrong_group_rate
```

该版本的验收重点不是 10-step smoke 的 validation 高低，而是每次更新稳定获得
32 个完整有效题组、过滤前后的统计正确、reward 小数项未丢失，并且 KL、梯度和
optimizer step 保持有限且无跳过。
