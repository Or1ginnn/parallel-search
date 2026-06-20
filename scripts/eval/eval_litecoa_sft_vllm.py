#!/usr/bin/env python3
import argparse
import inspect
import json
import random
import re
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
import transformers
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


PROMPT_TEMPLATE = """You are a search-augmented reasoning agent. \
You can only use the following tags: <think>, <plan>, <search>, <information>, <answer>. \
You must conduct reasoning inside <think> and </think> before every plan, search, or answer. \
Use <plan> to decompose the question into searchable sub-questions. \
If you lack knowledge, call a search engine by <search> query </search>. \
You can put multiple independent queries in one search action with "||", for example <search> query1 || query2 </search>. \
Each search action can contain at most {max_queries_per_turn} queries. \
The search engine will return results between <information> and </information>. \
Do not generate <information> yourself. \
If the evidence is sufficient, provide the answer inside <answer> and </answer>, without detailed illustrations. \
Question: {question}
"""

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
LOG_F = None


def log(message):
    print(message, flush=True)
    if LOG_F is not None:
        LOG_F.write(message + "\n")
        LOG_F.flush()


def normalize_question(question):
    question = " ".join(str(question).strip().split())
    if question and question[-1] != "?":
        question += "?"
    return question


def normalize_answer(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def em_check(prediction, gold_answers):
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(gold) for gold in gold_answers)


def subem_check(prediction, gold_answers):
    pred = normalize_answer(prediction)
    return any(normalize_answer(gold) and normalize_answer(gold) in pred for gold in gold_answers)


def extract_answer(text):
    matches = ANSWER_RE.findall(text)
    if not matches:
        return ""
    return " ".join(matches[-1].strip().split())


def truncate_at_first_action(text):
    candidates = []
    for tag in ("search", "answer"):
        end = text.find(f"</{tag}>")
        if end >= 0:
            candidates.append(end + len(f"</{tag}>"))
    return text[: min(candidates)] if candidates else text


def parse_queries(text, max_queries_per_turn):
    warnings = []
    matches = SEARCH_RE.findall(text)
    if not matches:
        return [], warnings

    queries = []
    seen = set()
    for item in matches[-1].split("||"):
        query = " ".join(item.strip().split())
        if not query:
            continue
        if "<" in query or ">" in query:
            warnings.append(f"query contains tag chars: {query}")
            continue
        key = query.lower()
        if key in seen:
            warnings.append(f"duplicate query dropped: {query}")
            continue
        seen.add(key)
        queries.append(query)

    if len(queries) > max_queries_per_turn:
        warnings.append(
            f"query count {len(queries)} exceeds max {max_queries_per_turn}; truncating"
        )
        queries = queries[:max_queries_per_turn]
    return queries, warnings


def retrieve(args, queries):
    payload = {"queries": queries, "topk": args.topk, "return_scores": True}
    response = requests.post(args.retriever_url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    return response.json()["result"]


def format_information(queries, results):
    information = "<information>"
    for query, retrieval_result in zip(queries, results):
        information += f"\n[Query] {query}\n"
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            body = "\n".join(content.split("\n")[1:])
            information += f"Doc {idx + 1}(Title: {title}) {body}\n"
    information += "</information>"
    return information


def load_samples(args):
    if args.input_jsonl:
        rows = []
        with open(args.input_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                rows.append(
                    {
                        "id": item.get("id") or f"sample_{len(rows)}",
                        "question": item["question"],
                        "gold_answer": item.get("gold_answer")
                        or item.get("golden_answers")
                        or [],
                    }
                )
        return rows

    frame = pd.read_parquet(args.input_parquet)
    rows = []
    for idx, item in frame.iterrows():
        rows.append(
            {
                "id": item.get("id") or f"sample_{idx}",
                "question": item["question"],
                "gold_answer": item.get("golden_answers")
                or item.get("gold_answer")
                or [],
            }
        )
    return rows


def filter_supported_kwargs(cls, kwargs):
    supported = inspect.signature(cls).parameters
    return {key: value for key, value in kwargs.items() if key in supported}


def load_backend(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    llm_kwargs = filter_supported_kwargs(
        LLM,
        {
            "model": args.base_model,
            "tokenizer": args.base_model,
            "trust_remote_code": True,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "enable_lora": bool(args.adapter),
            "max_lora_rank": args.max_lora_rank,
        },
    )
    llm = LLM(**llm_kwargs)
    lora_request = None
    if args.adapter:
        lora_request = LoRARequest("litecoa_sft", 1, args.adapter)
    return tokenizer, llm, lora_request


def build_prompt(tokenizer, question, max_queries_per_turn):
    prompt = PROMPT_TEMPLATE.format(
        question=normalize_question(question),
        max_queries_per_turn=max_queries_per_turn,
    )
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def build_sampling_params(args):
    stop_sequences = [
        "</search>",
        " </search>",
        "</answer>",
        " </answer>",
    ]
    temperature = args.temperature if args.do_sample else 0.0
    params = filter_supported_kwargs(
        SamplingParams,
        {
            "max_tokens": args.max_new_tokens,
            "temperature": temperature,
            "top_p": args.top_p,
            "stop": stop_sequences,
            "include_stop_str_in_output": True,
        },
    )
    return SamplingParams(**params)


def generate_once(args, llm, lora_request, prompt):
    sampling_params = build_sampling_params(args)
    outputs = llm.generate(
        [prompt],
        sampling_params,
        lora_request=lora_request,
        use_tqdm=False,
    )
    text = outputs[0].outputs[0].text
    text = truncate_at_first_action(text)
    return text


def run_one(args, tokenizer, llm, lora_request, sample):
    question = normalize_question(sample["question"])
    gold_answers = sample.get("gold_answer") or []
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    prompt = build_prompt(tokenizer, question, args.max_queries_per_turn)
    trajectory_parts = []
    queries_by_turn = []
    information_blocks = []
    parser_warnings = []
    agent_warnings = []
    error = None

    try:
        for _ in range(args.max_turns):
            output_text = generate_once(args, llm, lora_request, prompt)
            trajectory_parts.append(output_text)

            if "<information>" in output_text or "</information>" in output_text:
                parser_warnings.append("model generated information tag")

            if ANSWER_RE.search(output_text):
                break

            queries, warnings = parse_queries(output_text, args.max_queries_per_turn)
            parser_warnings.extend(warnings)
            if not queries:
                agent_warnings.append("no valid query found")
                break

            results = retrieve(args, queries)
            information = format_information(queries, results)
            queries_by_turn.append(queries)
            information_blocks.append(information)
            trajectory_parts.append(information)
            prompt += f"\n\n{output_text}{information}\n\n"
        else:
            agent_warnings.append("max_turns_exceeded")
    except Exception:
        error = traceback.format_exc()

    generated_text = "\n\n".join(trajectory_parts)
    final_answer = extract_answer(generated_text)
    return {
        "id": sample.get("id"),
        "question": question,
        "gold_answer": gold_answers,
        "final_answer": final_answer,
        "answer_em": em_check(final_answer, gold_answers) if gold_answers else None,
        "answer_subem": subem_check(final_answer, gold_answers) if gold_answers else None,
        "has_answer": bool(ANSWER_RE.search(generated_text)),
        "has_plan": bool(PLAN_RE.search(generated_text)),
        "plan_count": len(PLAN_RE.findall(generated_text)),
        "search_turns": len(queries_by_turn),
        "query_count": sum(len(qs) for qs in queries_by_turn),
        "queries_by_turn": queries_by_turn,
        "generated_information": any(
            "model generated information tag" == item for item in parser_warnings
        ),
        "parser_warnings": parser_warnings,
        "agent_warnings": agent_warnings,
        "error": error,
        "trajectory": generated_text,
    }


def summarize(records):
    n = len(records)
    search_turns = Counter(record["search_turns"] for record in records)
    query_counts = Counter(record["query_count"] for record in records)
    with_gold = [record for record in records if record["answer_em"] is not None]
    return {
        "total": n,
        "errors": sum(bool(record["error"]) for record in records),
        "format_valid": sum(
            record["has_plan"] and record["plan_count"] == 1 and record["has_answer"]
            for record in records
        ),
        "has_plan": sum(record["has_plan"] for record in records),
        "plan_once": sum(record["plan_count"] == 1 for record in records),
        "answer_count": sum(record["has_answer"] for record in records),
        "generated_information_count": sum(record["generated_information"] for record in records),
        "parser_warning_count": sum(bool(record["parser_warnings"]) for record in records),
        "agent_warning_count": sum(bool(record["agent_warnings"]) for record in records),
        "max_turns_exceeded": sum(
            "max_turns_exceeded" in record["agent_warnings"] for record in records
        ),
        "search_turn_distribution": dict(sorted(search_turns.items())),
        "query_count_distribution": dict(sorted(query_counts.items())),
        "answer_em": sum(record["answer_em"] for record in with_gold),
        "answer_subem": sum(record["answer_subem"] for record in with_gold),
        "gold_count": len(with_gold),
    }


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    global LOG_F
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        default="/mnt/data1/zar/search-1/Search-R1/hf_cache/Qwen2.5-3B",
    )
    parser.add_argument(
        "--adapter",
        default="/mnt/data1/zar/search-1/Search-R1/outputs/sft/litecoa_lora_qwen25_3b_full",
    )
    parser.add_argument("--input_parquet", default="data/nq_search/test.parquet")
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--output_dir", default="outputs/sft/litecoa_sft_eval_vllm")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=2)
    parser.add_argument("--max_queries_per_turn", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--max_lora_rank", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_F = open(output_dir / "eval.log", "w", encoding="utf-8")
    log(f"[config] backend=vllm")
    log(f"[config] base_model={args.base_model}")
    log(f"[config] adapter={args.adapter}")
    log(f"[config] retriever_url={args.retriever_url}")
    log(f"[config] num_samples={args.num_samples}")
    log(f"[config] max_turns={args.max_turns}")
    log(f"[config] do_sample={args.do_sample}")
    log(f"[config] temperature={args.temperature if args.do_sample else 0.0}")
    log(f"[config] tensor_parallel_size={args.tensor_parallel_size}")

    samples = load_samples(args)
    random.Random(args.seed).shuffle(samples)
    samples = samples[: args.num_samples]
    log(f"[data] loaded_eval_samples={len(samples)}")

    tokenizer, llm, lora_request = load_backend(args)
    records = []
    for idx, sample in enumerate(samples, start=1):
        log(f"[eval] {idx}/{len(samples)} id={sample.get('id')}")
        record = run_one(args, tokenizer, llm, lora_request, sample)
        records.append(record)
        log(
            "[eval] result "
            f"answer={record['has_answer']} plan_count={record['plan_count']} "
            f"turns={record['search_turns']} queries={record['query_count']} "
            f"warnings={len(record['parser_warnings']) + len(record['agent_warnings'])} "
            f"error={bool(record['error'])}"
        )

    summary = summarize(records)
    write_jsonl(output_dir / "trajectories.jsonl", records)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(json.dumps(summary, ensure_ascii=False, indent=2))
    LOG_F.close()
    LOG_F = None


if __name__ == "__main__":
    main()
