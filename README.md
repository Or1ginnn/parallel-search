# Parallel Search

Parallel Search is an experimental reinforcement-learning codebase for
search-augmented LLM agents. It starts from the open-source Search-R1 / veRL
implementation and adds LiteCoA-style parallel query search experiments.

The current best result is a compact parallel-search agent trained with GRPO
from Qwen2.5-3B base. It learns the stable pattern:

```text
<search> query1 || query2 </search>
<information> ... </information>
<answer> ... </answer>
```

On full NQ test with the Phase 5 hard-reward checkpoint at step 900:

| Model / Run | Eval | NQ EM | NQ SubEM | Search Behavior |
|---|---|---:|---:|---|
| Search-R1 baseline Qwen2.5-3B | W&B best step 250 | 42.69% | - | single-query search |
| Parallel Search Qwen2.5-3B | step 900 greedy | 46.37% | 49.47% | 3609/3610 samples use 2-query search |
| Parallel Search Qwen2.5-3B | step 900 temp=1 | 43.46% | 46.48% | 3558/3610 samples use 2-query search |

The Phase 5 result should be read as a successful **parallel search** result,
not as a full preservation of the original plan-first LiteCoA format. With a 3B
model, GRPO converges to the shorter effective action path above rather than
keeping explicit `<think>` / `<plan>` tags.

## Status

This repository is not an official Search-R1 repository and is not presented as
the original project. It is a derivative development workspace.

Current project status:

```text
Phase 1: LiteCoA inference prototype completed.
Phase 2: LiteCoA SFT data construction completed.
Phase 3: LoRA SFT cold start completed and evaluated.
Phase 4: LiteCoA rollout / GRPO training loop completed.
Phase 5: hard reward parallel-search GRPO completed.
Phase 6: benchmark comparison and result consolidation next.
```

Large local artifacts are intentionally excluded from git, including model
weights, checkpoints, datasets, retrieval indexes, trajectory logs, cache
directories, and `.jsonl` / parquet-style data files.

## Attribution

This project contains code derived from:

- Search-R1: https://github.com/PeterGriffinJin/Search-R1
- veRL: https://github.com/volcengine/verl

The upstream project is licensed under Apache License 2.0. This repository keeps
the upstream `LICENSE` file and `Notice.txt` copyright notice. Modified files
and this README identify this repository as a derivative project.

If you use this repository in a paper or public release, cite the upstream
Search-R1 work where appropriate:

```bibtex
@article{jin2025searchr1,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Wang, Guoyin and Xie, Jiawei and Han, Junda and Liu, Jifan and Xiong, Chenyan and Wang, Xiaozhi and Yang, Jinyuan and Du, Yifei and others},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```

## Installation

```bash
conda create -n parallel-search python=3.9
conda activate parallel-search
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e .
pip install wandb
```

For local retrieval experiments, use a separate retriever environment:

```bash
conda create -n parallel-retriever python=3.10
conda activate parallel-retriever
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini uvicorn fastapi
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## Usage

The inherited scripts expect training data and retrieval resources to live
outside git:

```bash
bash retrieval_launch.sh
bash train_grpo.sh
```

LiteCoA / Parallel Search entry points:

```bash
# Generate LiteCoA NQ GRPO data.
python scripts/data_process/nq_search.py \
  --template_type litecoa \
  --local_dir data/nq_search_litecoa

# Train the current Phase 5 GRPO setup.
bash scripts/train/train_grpo_litecoa_qwen25_3b.sh

# Batched vLLM eval for a trained actor checkpoint.
python scripts/eval/eval_litecoa_sft_vllm.py \
  --base_model verl_checkpoints/nq-litecoa-grpo-qwen2.5-3b-base-hard/actor/global_step_900 \
  --adapter "" \
  --input_parquet data/nq_search_litecoa/test.parquet \
  --output_dir output/phase5_step900_validate/eval_step900_nq_full \
  --num_samples -1 \
  --batch_size 128 \
  --topk 2 \
  --max_turns 3 \
  --max_queries_per_turn 3 \
  --temperature 0
```

For the current Phase 5 setup, the intended base model path on the training
server is Qwen2.5-3B. The script keeps local server paths as editable variables
near the top of the shell file.

## Documentation

Main project reports:

- `docs/phase1_litecoa_infer.md`
- `docs/phase2_litecoa_data_report.md`
- `docs/phase3_litecoa_sft_report.md`
- `docs/phase4_litecoa_rollout.md`
- `docs/phase5_litecoa_reward.md`

The Phase 5 report contains the final reward design, Search-R1 baseline
comparison, greedy/temp=1 full NQ eval, and the conclusion that the 3B model
successfully learns parallel search while dropping explicit plan tags.

## Notes

- Package and directory names such as `search_r1` are still inherited from the
  upstream code for compatibility.
- Model and dataset files must be downloaded or generated locally; they are not
  included in this repository.
- Original upstream documentation can be consulted at
  https://github.com/PeterGriffinJin/Search-R1.
