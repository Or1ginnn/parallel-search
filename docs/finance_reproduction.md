# 金融并行检索 Agent：复现与最终评测协议

本文给出 Finance / FinQA 实验线从数据处理到最终评测的最短可复现路径。已有结果、历史问题和结论分别见：

- [`finance_phase0_3_summary.md`](./finance_phase0_3_summary.md)
- [`finance_phase4_grpo_report.md`](./finance_phase4_grpo_report.md)
- [`finance_agent_plan.md`](./finance_agent_plan.md)

## 1. 复现边界

- 代码基线：`finance-agent` 分支。
- 基础策略：NQ 并行检索 Step900，具体权重由运行环境提供，不进入 Git。
- FinQA LoRA：rank 64、alpha 128；历史训练数据使用旧 plan prompt，监督轨迹实际为 no-plan。
- GRPO 初始化：修复 RoPE 配置后的 `finqa_litecoa_sft_merged_fp32_v3`。
- 主要已报告结果来自 FinQA dev，不是 test。
- 模型、LoRA、FAISS index、完整训练数据和 checkpoint 均不进入 Git。

历史服务器曾通过手工同步文件运行实验，W&B 自动记录的 Git commit 指向上游仓库，不能代表 Finance 代码版本。因此，代码以 `finance-agent` 分支为准，训练超参数以归档的 [`wandb_q51bp5pw_config.json`](./finqa_eval/20260727/phase4_training/wandb_q51bp5pw_config.json) 为准。

## 2. 环境

SFT 已核对环境：

```text
Python 3.11.15
PyTorch 2.11.0+cu130
Transformers 5.6.0
PEFT 0.18.1
Datasets 4.0.0
Accelerate 1.11.0
W&B 0.28.0
```

GRPO 的 W&B 元数据为 Python 3.9.25、CUDA 13.0、2 x A800 80GB。SFT 与 GRPO 使用不同 Transformers 版本；合并 LoRA 后必须保留 Qwen2.5-3B 的 `rope_theta=1000000`，并进行 logits 或小规模 rollout 等价验证。

## 3. 数据和检索器

```bash
# 官方 FinQA -> processed QA + 无监督字段泄漏的财报 corpus
python scripts/data_process/prepare_finqa.py \
  --raw_dir data/finance_finqa/raw/FinQA/dataset \
  --output_dir data/finance_finqa/processed

# 构建 E5 Flat index
CUDA_VISIBLE_DEVICES=1 python search_r1/search/index_builder.py \
  --retrieval_method e5 \
  --model_path intfloat/e5-base-v2 \
  --corpus_path data/finance_finqa/processed/corpus_all.jsonl \
  --save_dir data/finance_finqa/index \
  --faiss_type Flat \
  --max_length 256 \
  --batch_size 256 \
  --use_fp16

# 启动 report-aware retriever；不要添加 --faiss_gpu，CPU Flat index 才支持按报告重排
CUDA_VISIBLE_DEVICES=1 python search_r1/search/retrieval_server.py \
  --index_path data/finance_finqa/index/e5_Flat.index \
  --corpus_path data/finance_finqa/processed/corpus_all.jsonl \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --topk 3
```

`report_ids` 必须使用完整 page 级 ID，例如 `CDNS/2012/page_31.pdf`，不能传 `CDNS/2012`。

## 4. SFT 数据与训练

Teacher 构造器只生成 `<think>`、`<search>` 和 `<answer>`；`<information>` 必须由上述 `/retrieve` 服务返回。API key 只通过环境变量传入。

```bash
export DEEPSEEK_API_KEY='...'
python scripts/data_process/build_finqa_sft.py \
  --input_jsonl data/finance_finqa/processed/train.jsonl \
  --output data/finance_finqa/sft/raw.jsonl \
  --rejected_output data/finance_finqa/sft/rejected.jsonl \
  --report_output data/finance_finqa/sft/report.json \
  --retriever_url http://127.0.0.1:8000/retrieve \
  --target_count 500

python scripts/data_process/convert_litecoa_sft_for_llamafactory.py \
  --input data/finance_finqa/sft/raw.jsonl \
  --output data/finance_finqa/sft/finqa_litecoa_sft_500.jsonl \
  --keep_metadata \
  --max_queries_per_turn 2 \
  --prompt_style finance_no_plan

CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train \
  configs/sft/finqa_litecoa_lora_qwen25_3b_full.yaml
```

修复后的 `finance_no_plan` 用于未来数据。复现历史 LoRA 时必须使用历史归档训练 JSONL，因为历史 prompt 要求 plan，而监督 target 没有 plan。

## 5. GRPO 数据与训练

```bash
# 默认生成 train/dev；追加 test 仅用于最终离线评测
python scripts/data_process/finqa_search.py \
  --input_dir data/finance_finqa/processed \
  --output_dir data/finance_finqa/grpo_nocalc \
  --splits train dev test

# 历史 Step50 -> Step200 的稳定配置
bash scripts/train/train_finqa_grpo_full.sh

# 从 merged v3 独立启动 200-step 实验
RESUME_FROM_CHECKPOINT=null \
EXPERIMENT_NAME=finqa-phase4-grpo-v3-fresh \
bash scripts/train/train_finqa_grpo_full.sh
```

默认脚本与 W&B `q51bp5pw` 对齐：2 卡、batch 32、每题 5 条、Top-3、information 1500、temperature 1.0、LR `5e-7`、KL coefficient `0.005`、每 25 step 保存和验证。

## 6. 统一评测命令

以下命令评测已经合并的模型，因此 `--adapter` 传空字符串。`--num_samples -1` 表示完整 split；不传 `--do_sample` 即 greedy。

```bash
MODEL=/path/to/merged_model_or_step200_actor
DATA=data/finance_finqa/grpo_nocalc

CUDA_VISIBLE_DEVICES=0,1 python scripts/eval/eval_litecoa_sft_vllm.py \
  --base_model "$MODEL" \
  --adapter '' \
  --input_parquet "$DATA/dev.parquet" \
  --output_dir outputs/eval/finqa_dev_full \
  --num_samples -1 \
  --batch_size 32 \
  --topk 3 \
  --max_information_tokens 1500 \
  --max_turns 3 \
  --max_queries_per_turn 3 \
  --use_report_scope \
  --tensor_parallel_size 2 \
  --dtype bfloat16

# test 只在模型和协议冻结后运行一次
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval/eval_litecoa_sft_vllm.py \
  --base_model "$MODEL" \
  --adapter '' \
  --input_parquet "$DATA/test.parquet" \
  --output_dir outputs/eval/finqa_test_final \
  --num_samples -1 \
  --batch_size 32 \
  --topk 3 \
  --max_information_tokens 1500 \
  --max_turns 3 \
  --max_queries_per_turn 3 \
  --use_report_scope \
  --tensor_parallel_size 2 \
  --dtype bfloat16
```

## 7. Phase 5 严格对照矩阵

在发表或简历中使用“并行检索带来提升”前，至少补齐：

| 实验 | 模型 | Query 设置 | 目的 |
| --- | --- | --- | --- |
| Pre-GRPO | merged v3 | 2-query、Top-3 | 与 Step200 严格同加载形态 A/B |
| Parallel | Step200 | 最多 3 query、Top-3 | 当前主结果 |
| Single-query | Step200 | 最多 1 query、Top-3 | 固定每 query Top-K 的动作消融 |
| Single-query fixed budget | Step200 | 最多 1 query、Top-6 | 与双 query x Top-3 对齐最大文档预算 |
| Final test | 最终冻结 checkpoint | 主协议 | 一次性 test 结果 |

所有实验必须保存 `summary.json`、`trajectories.jsonl`、`eval.log`，并记录模型路径、代码 commit、split、seed、解码温度、Top-K、query 上限、information 上限、report scope 和 denominator。多随机种子只针对随机解码或重新训练；greedy 同 checkpoint 不需要伪造“多次重复”。

## 8. 尚未固化的资产

当前仓库没有模型权重、LoRA、完整训练数据、FAISS index 和 checkpoint 的内容哈希。它们必须由运行服务器单独生成 manifest（路径、文件大小、SHA256、生成时间），否则只能复现实验流程，不能证明二进制产物完全一致。
