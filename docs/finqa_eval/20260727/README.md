# FinQA Recent Evaluation Archive

This archive collects the recent FinQA dev evaluation records used for the
LiteCoA experiments. All full evaluations use 883 dev examples, of which 871
have scored gold answers. Numeric EM is `answer_numeric_em / gold_count`.

## Full Dev Results

| Run | Model / configuration | Numeric EM | Raw EM | SubEM | Answers | Errors | Warnings | Max turns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `step900_raw_policy` | Step900 base policy, report-aware top2 | 103/871 (11.83%) | 75 | 120 | 881 | 0 | 2 | 0 |
| `step900_sft_lora_top2` | Step900 + FinQA SFT LoRA, report-aware top2 | 374/871 (42.94%) | 101 | 290 | 835 | 0 | 54 | 45 |
| `step900_sft_lora_topk3_info1500` | Step900 + FinQA SFT LoRA, report-aware top3, information capped at 1500 tokens | 392/871 (45.01%) | 105 | 298 | 856 | 0 | 31 | 26 |
| `sft_merged_nocalc_topk3_info1500` | Earlier merged SFT model, no-plan prompt, top3/info1500 | 228/871 (26.18%) | 64 | 194 | 771 | 0 | 114 | 17 |
| `sft_merged_plan_topk3_info1500` | Earlier merged SFT model, plan prompt, top3/info1500 | 223/871 (25.60%) | 72 | 192 | 741 | 0 | 144 | 23 |
| `phase4_step200_nocalc_topk3_info1500` | FinQA Phase 4 GRPO Step200, merged FP32 v3, no-plan, top3/info1500 | 478/871 (54.88%) | 192 | 335 | 883 | 0 | 0 | 0 |

The Step200 result is +9.87 percentage points over the Step900 + SFT LoRA
top3/info1500 reference (45.01%). It has no parser or agent warnings and no
generated-information events.

## Phase 4 Training Record

The complete Chinese Phase 4 report is stored at
[`docs/finance_phase4_grpo_report.md`](../../finance_phase4_grpo_report.md).
The W&B run `q51bp5pw` was re-exported as raw JSON and an aligned CSV covering
all 150 updates from Step51 through Step200. Reconstructed plots are available
under `phase4_training/`:

- `phase4_task_performance.png`
- `phase4_optimization_stability.png`
- `wandb_q51bp5pw_raw_history.json`
- `wandb_q51bp5pw_history.csv`
- `wandb_q51bp5pw_metadata.json`

## Merge Equivalence Checks

The FP32 safe merge diagnostic recorded:

- Dynamic LoRA vs in-memory merged logits: max absolute difference
  `4.673004150390625e-05`, mean `5.717841304431204e-06`, `allclose=true` at
  `atol=rtol=1e-4`.
- Dynamic LoRA vs reloaded merged logits: identical values and `allclose=true`.

The repaired v3 merged model retained the Step900 legacy RoPE setting
(`rope_theta=1000000`). In the controlled 100-example BF16 evaluation, dynamic
LoRA and v3 merged both reached Numeric EM 44/100. Predictions matched on 48
examples; 52 differed, with 7 dynamic-only and 7 merged-only numeric-correct
examples. The detailed comparison, including one divergent trajectory, is
stored in `merge_equivalence_100_bf16/comparison_dynamic_vs_merged_v3.json`.

## Contents

Each evaluation folder contains its original `summary.json` and
`trajectories.jsonl`. The archive also includes the merge-logit diagnostic and
the controlled 100-example comparison artifact. Large model files, checkpoints,
raw FinQA data, and training logs are intentionally excluded.
