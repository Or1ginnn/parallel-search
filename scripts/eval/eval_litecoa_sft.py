#!/usr/bin/env python3
import argparse
import json
import random
import re
import traceback
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
import torch
import transformers
from peft import PeftModel


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

CURR_EOS = {151645, 151643}
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
LOG_F = None


def log(message):
    print(message, flush=True)
    if LOG_F is not None:
        LOG_F.write(message + "\n")
        LOG_F.flush()


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False)
            for target_sequence in target_sequences
        ]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for target_ids, target_len in zip(self.target_ids, self.target_lengths):
            target = torch.as_tensor(target_ids, device=input_ids.device)
            if torch.equal(input_ids[0, -target_len:], target):
                return True
        return False


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


def load_model(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    target_sequences = [
        "</search>",
        " </search>",
        "</search>\n",
        " </search>\n",
        "</search>\n\n",
        " </search>\n\n",
        "</answer>",
        " </answer>",
        "</answer>\n",
        " </answer>\n",
        "</answer>\n\n",
        " </answer>\n\n",
    ]
    stopping = transformers.StoppingCriteriaList(
        [StopOnSequence(target_sequences, tokenizer)]
    )
    return tokenizer, model, stopping


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


def generate_once(args, tokenizer, model, stopping, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "stopping_criteria": stopping,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p

    with torch.no_grad():
        outputs = model.generate(**kwargs)
    generated = outputs[0][input_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    text = truncate_at_first_action(text)
    eos_hit = outputs[0][-1].item() in CURR_EOS
    return text, eos_hit


def run_one(args, tokenizer, model, stopping, sample):
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
            output_text, eos_hit = generate_once(args, tokenizer, model, stopping, prompt)
            trajectory_parts.append(output_text)

            if "<information>" in output_text or "</information>" in output_text:
                parser_warnings.append("model generated information tag")

            if ANSWER_RE.search(output_text) or eos_hit:
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
    parser.add_argument("--output_dir", default="outputs/sft/litecoa_sft_eval")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=2)
    parser.add_argument("--max_queries_per_turn", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG_F = open(output_dir / "eval.log", "w", encoding="utf-8")
    log(f"[config] base_model={args.base_model}")
    log(f"[config] adapter={args.adapter}")
    log(f"[config] retriever_url={args.retriever_url}")
    log(f"[config] num_samples={args.num_samples}")
    log(f"[config] max_turns={args.max_turns}")

    samples = load_samples(args)
    random.Random(args.seed).shuffle(samples)
    samples = samples[: args.num_samples]
    log(f"[data] loaded_eval_samples={len(samples)}")

    tokenizer, model, stopping = load_model(args)
    records = []
    for idx, sample in enumerate(samples, start=1):
        log(f"[eval] {idx}/{len(samples)} id={sample.get('id')}")
        record = run_one(args, tokenizer, model, stopping, sample)
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
