#!/usr/bin/env python3
"""Convert processed FinQA QA records into report-aware GRPO parquet files."""

import argparse
import json
from pathlib import Path

import pandas as pd


PROMPT_TEMPLATE = """You are a search-augmented reasoning agent. \
You can only use the following tags: <think>, <search>, <information>, <answer>. \
You must conduct reasoning inside <think> and </think> before every search or answer. \
If you lack knowledge, call a search engine by <search> query </search>. \
You can put multiple independent queries in one search action with "||", for example <search> query1 || query2 </search>. \
Each search action can contain at most 3 queries. \
The search engine will return results between <information> and </information>. \
Do not generate <information> yourself. \
If the evidence is sufficient, provide the answer inside <answer> and </answer>, without detailed illustrations. \
For financial calculation questions, search for each required quantity with complementary queries (for example, numerator and denominator). Use the retrieved evidence to reason through any arithmetic inside <think>; never guess a missing value. \
Question: {question}
"""


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def convert_row(row, index):
    question = " ".join(row["question"].strip().split())
    if question and not question.endswith("?"):
        question += "?"
    return {
        "id": row["id"],
        "report_id": row["report_id"],
        "data_source": "finqa",
        "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
        "ability": "financial-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target": [row["gold_answer"]],
                "answer_type": "finqa",
                "gold_evidence": row.get("gold_evidence", []),
                "program": row.get("program", ""),
                "executable_answer": row.get("executable_answer", ""),
            },
        },
        "extra_info": {"split": row["split"], "index": index},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/finance_finqa/processed")
    parser.add_argument("--output_dir", default="data/finance_finqa/grpo")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "dev", "test"),
        default=("train", "dev"),
        help="Processed FinQA splits to convert. Defaults preserve the training workflow.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        records = load_jsonl(input_dir / f"{split}.jsonl")
        rows = [convert_row(record, index) for index, record in enumerate(records)]
        pd.DataFrame(rows).to_parquet(output_dir / f"{split}.parquet", index=False)
        print(f"{split}: {len(rows)} -> {output_dir / f'{split}.parquet'}")


if __name__ == "__main__":
    main()
