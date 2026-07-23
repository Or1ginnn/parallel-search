#!/usr/bin/env python3
"""Merge the FinQA LoRA adapter into its Qwen base for full-parameter GRPO."""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter).merge_and_unload()
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"merged model written to {args.output}")


if __name__ == "__main__":
    main()
