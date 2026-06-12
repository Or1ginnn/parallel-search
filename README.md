# Parallel Search

Parallel Search is an experimental reinforcement-learning codebase for
reasoning-and-search workflows with LLMs. The current code starts from the
open-source Search-R1 / veRL implementation and will be modified for parallel
query decomposition and parallel retrieval experiments.

## Status

This repository is not an official Search-R1 repository and is not presented as
the original project. It is a derivative development workspace.

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
bash train_ppo.sh
bash train_grpo.sh
```

The default local training scripts currently target:

```bash
Qwen/Qwen3-4B-Instruct-2507
```

## Notes

- Package and directory names such as `search_r1` are still inherited from the
  upstream code for compatibility.
- Model and dataset files must be downloaded or generated locally; they are not
  included in this repository.
- Original upstream documentation can be consulted at
  https://github.com/PeterGriffinJin/Search-R1.
