# Attribution

This repository contains derivative work based on open-source projects:

- Search-R1: https://github.com/PeterGriffinJin/Search-R1
- veRL: https://github.com/volcengine/verl

The repository keeps the upstream Apache License 2.0 text in `LICENSE` and the
upstream copyright notice in `Notice.txt`.

Local modifications include:

- Repackaging the repository as `parallel-search`.
- Excluding datasets, model weights, indexes, cache directories, and trajectory
  logs from git.
- Changing the top-level PPO and GRPO training scripts to use
  `Qwen/Qwen3-4B-Instruct-2507` by default.
