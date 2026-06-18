# Phase 3: LiteCoA LoRA SFT

## Goal

Use `litecoa_sft_1000_v1_llamafactory.jsonl` to cold-start Qwen2.5-3B with LoRA SFT so the model learns the LiteCoA output pattern:

```text
<think> -> <plan> -> <think> -> <search> -> information observation -> <think> -> <answer>
```

The SFT dataset uses a multi-turn chat format:

```json
{
  "messages": [
    {"role": "user", "content": "LiteCoA prompt + Question"},
    {"role": "assistant", "content": "<think>...</think>\n<plan>...</plan>\n<think>...</think>\n<search>...</search>"},
    {"role": "user", "content": "<information>retriever results</information>"},
    {"role": "assistant", "content": "<think>...</think>\n<answer>...</answer>"}
  ]
}
```

This keeps `<information>` as an observation, not an assistant target. With assistant-only SFT loss, the model learns to generate plan/search/answer tokens but does not learn to generate retriever content.

## Files

- `data/litecoa_sft/litecoa_1000/litecoa_sft_1000_v1_llamafactory.jsonl`
- `data/litecoa_sft/litecoa_1000/dataset_info.json`
- `configs/sft/litecoa_lora_qwen25_3b_smoke.yaml`
- `configs/sft/litecoa_lora_qwen25_3b_full.yaml`
- `scripts/sft/train_litecoa_lora_qwen25_3b.sh`

## Smoke Run

Edit the smoke config first:

```text
configs/sft/litecoa_lora_qwen25_3b_smoke.yaml
```

Set `model_name_or_path` to the local Qwen2.5-3B path.

Then run:

```bash
wandb login
bash scripts/sft/train_litecoa_lora_qwen25_3b.sh
```

The smoke run uses `max_samples: 32` and verifies that LLaMA-Factory can load the model, dataset, chat template, and LoRA config.
Training metrics are reported to Weights & Biases with:

- `WANDB_PROJECT=litecoa-search-r1`
- `WANDB_NAME=litecoa_lora_qwen25_3b_smoke`

## Full Run

After smoke passes, switch `CONFIG` in `scripts/sft/train_litecoa_lora_qwen25_3b.sh` to:

```text
configs/sft/litecoa_lora_qwen25_3b_full.yaml
```

Then run:

```bash
bash scripts/sft/train_litecoa_lora_qwen25_3b.sh
```

Default full-run settings:

- `cutoff_len: 16384`
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 8`
- `learning_rate: 5.0e-5`
- `num_train_epochs: 2.0`
- `lora_rank: 16`
- `lora_alpha: 32`

If GPU memory is tight, keep `cutoff_len: 16384` and reduce batch pressure through gradient checkpointing or lower LoRA rank before reducing the cutoff length.

The launch script only selects a YAML config and calls `llamafactory-cli train`. Training settings live in `configs/sft/`.
