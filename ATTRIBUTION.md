# Attribution

This repository contains derivative work based on the following open-source
projects:

- Search-R1: https://github.com/PeterGriffinJin/Search-R1
- veRL: https://github.com/volcengine/verl

The repository keeps the upstream Apache License 2.0 text in `LICENSE` and the
upstream veRL copyright notice in `Notice.txt`.

Local modifications include:

- Repackaging the repository as `parallel-search`.
- Excluding datasets, model weights, indexes, cache directories, and trajectory
  logs from git.
- Adding LiteCoA / Parallel Search experiments on top of Search-R1.
- Extending the rollout loop to parse multi-query search actions such as
  `<search>q1 || q2</search>`.
- Adding LiteCoA data-processing, SFT-data construction, GRPO training scripts,
  and batched vLLM evaluation utilities.
- Adding LiteCoA / Parallel Search reward logic, including hard-zero checks for
  clipped responses, missing answers, invalid actions, and generated
  `<information>`.
- Documenting the Phase 1-5 experiment process and final Parallel Search result
  for Qwen2.5-3B.

The current experimental result should be understood as a derivative research
extension, not as an official Search-R1 release.
