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

## 中文说明

这个仓库是基于 Search-R1 / veRL 改造的并行检索实验项目，目标是让
LLM agent 不只生成一个 search query，而是在一次 search action 中生成多个
query，并行检索后再回答。

当前已经完成的核心结果是：

```text
Qwen2.5-3B base
-> LiteCoA / Parallel Search rollout
-> hard reward GRPO
-> 稳定学会 2-query parallel search
```

最终有效轨迹更接近下面这种紧凑形式：

```text
<search> query1 || query2 </search>
<information> ... </information>
<answer> ... </answer>
```

在完整 NQ test set 上，Phase 5 hard reward 的 step 900 greedy 结果为：

```text
EM: 46.37%
SubEM: 49.47%
3609 / 3610 条样本使用 2-query parallel search
没有 generated information
没有 max_turns_exceeded
```

对比原 Search-R1 baseline：

```text
Search-R1 baseline best NQ EM: 42.69%
Parallel Search step900 greedy NQ EM: 46.37%
```

需要注意的是，这个结果说明 **parallel search 目标已经完成**，但 3B 模型没有
保留完整的 plan-first LiteCoA 格式。也就是说，模型没有稳定输出：

```text
<think> -> <plan> -> <think> -> <search> -> ...
```

而是自然收敛到了更短、更容易获得 reward 的：

```text
parallel search -> information -> answer
```

因此当前阶段更准确的定位是：**Parallel-Search-R1 / compact LiteCoA agent**。
如果后续要强制保留 `<plan>`，需要更强模型或单独的格式约束实验。

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
