#!/usr/bin/env python3
"""
Build a small LiteCoA SFT dataset with a real Search-R1 retriever.

The teacher model only generates <think>, <plan>, <search>, and <answer>.
The <information> blocks are always filled from the retriever response.
"""

import argparse
import json
import os
import random
import re
import string
import time
from pathlib import Path

import requests


SYSTEM_PROMPT = """You are a search-augmented reasoning agent.
You can only use the following tags:
<think>...</think>
<plan>...</plan>
<search>...</search>
<information>...</information>
<answer>...</answer>

Rules:
1. Use <think> before <plan>, <search>, or <answer>.
2. Use <plan> exactly once at the beginning to decompose the question.
3. Use <search> to issue one or multiple search queries.
4. Multiple independent queries must be separated by "||".
5. Each <search> can contain at most 3 queries.
6. Never generate <information>; it is returned by the retriever.
7. If evidence is sufficient, finish with <answer>...</answer>.
8. The final answer must be a short answer, not an explanation."""


FIRST_TURN_PROMPT = """Generate the first LiteCoA search action for the question.

Output exactly this structure:
<think>brief reasoning about what must be found</think>
<plan>numbered searchable sub-questions</plan>
<think>briefly choose the first search queries</think>
<search>query1 || query2</search>

Do not answer the question.
Do not generate <information>.
Do not use placeholder queries such as query1 or query2.
Do not include XML tags inside a query.
Question: {question}"""


NEXT_TURN_PROMPT = """Continue the LiteCoA trajectory using only the retrieved information below.

Question: {question}
Gold answers for quality control: {gold_answers}

Current trajectory:
{trajectory}

If the retrieved evidence is sufficient, output exactly:
<think>brief evidence check</think>
<answer>short answer only</answer>

If the evidence is not sufficient, output exactly:
<think>what is missing</think>
<search>one or more new queries separated by ||</search>

Do not generate <information>.
Do not include explanations inside <answer>.
Do not include XML tags inside a query."""


TAG_RE = re.compile(r"<[^>]+>")
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
INFORMATION_RE = re.compile(r"</?information>", re.IGNORECASE)


def normalize_answer(text):
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s):
        return " ".join(s.split())

    def remove_punc(s):
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def em_check(prediction, gold_answers):
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(gold) for gold in gold_answers)


def subem_check(prediction, gold_answers):
    pred = normalize_answer(prediction)
    return any(normalize_answer(gold) in pred for gold in gold_answers)


def evidence_hit(information, gold_answers):
    info = normalize_answer(information)
    return any(normalize_answer(gold) and normalize_answer(gold) in info for gold in gold_answers)


def ensure_question_mark(question):
    question = " ".join(str(question).strip().split())
    if question and question[-1] != "?":
        question += "?"
    return question


def load_examples(args):
    if args.input_jsonl:
        rows = []
        with open(args.input_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    rows.append(
                        {
                            "id": item.get("id") or f"{item.get('data_source', 'sample')}_{len(rows)}",
                            "source": item.get("data_source", args.data_source),
                            "question": item["question"],
                            "gold_answer": item.get("golden_answers")
                            or item.get("gold_answer")
                            or item.get("reward_model", {})
                            .get("ground_truth", {})
                            .get("target", []),
                        }
                    )
        return rows

    if args.input_parquet:
        import datasets

        ds = datasets.load_dataset("parquet", data_files=args.input_parquet, split="train")
    else:
        import datasets

        ds = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", args.data_source, split=args.split)

    rows = []
    for idx, item in enumerate(ds):
        rows.append(
            {
                "id": item.get("id") or f"{args.split}_{idx}",
                "source": args.data_source,
                "question": item["question"],
                "gold_answer": item.get("golden_answers") or item.get("gold_answer") or [],
            }
        )
    return rows


def chat_completion(args, messages):
    load_env_file(args.env_file)
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY or pass --api_key.")

    if args.api_url:
        url = args.api_url
    else:
        base_url = (args.base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        url = f"{base_url}/chat/completions"

    payload = {
        "model": args.model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4",
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    for attempt in range(args.api_retries + 1):
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=args.api_timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt >= args.api_retries:
                raise
            time.sleep(args.retry_sleep * (attempt + 1))
    raise RuntimeError("unreachable")


def load_env_file(path):
    if not path:
        return
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_queries(text, max_queries):
    matches = SEARCH_RE.findall(text)
    if not matches:
        return [], ["missing_search"]

    raw_items = matches[-1].split("||")
    queries = []
    warnings = []
    seen = set()
    for raw in raw_items:
        query = " ".join(raw.strip().split())
        if not query:
            continue
        low = query.lower()
        if TAG_RE.search(query):
            warnings.append(f"tag_in_query:{query}")
            continue
        if re.fullmatch(r"query\s*\d*", low) or low in {"q1", "q2", "q3"}:
            warnings.append(f"placeholder_query:{query}")
            continue
        if low in seen:
            warnings.append(f"duplicate_query:{query}")
            continue
        seen.add(low)
        queries.append(query)

    if len(queries) > max_queries:
        warnings.append(f"too_many_queries:{len(queries)}")
        queries = queries[:max_queries]
    if not queries:
        warnings.append("no_valid_query")
    return queries, warnings


def validate_first_turn(text, max_queries):
    warnings = []
    if INFORMATION_RE.search(text):
        warnings.append("teacher_generated_information")
    if len(PLAN_RE.findall(text)) != 1:
        warnings.append("plan_count_not_one")
    if not re.search(r"<think>.*?</think>", text, re.DOTALL):
        warnings.append("missing_think")
    queries, query_warnings = parse_queries(text, max_queries)
    warnings.extend(query_warnings)
    return queries, warnings


def retrieve(args, queries):
    payload = {"queries": queries, "topk": args.topk, "return_scores": True}
    response = requests.post(args.retriever_url, json=payload, timeout=args.retriever_timeout)
    response.raise_for_status()
    data = response.json()
    return data["result"]


def format_information(queries, results):
    information = "<information>"
    for query, retrieval_result in zip(queries, results):
        information += f"\n[Query] {query}\n"
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            information += f"Doc {idx + 1}(Title: {title}) {text}\n"
    information += "</information>"
    return information


def extract_answer(text):
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return " ".join(matches[-1].strip().split())


def make_record(example, assistant_content, queries_by_turn, information_blocks, final_answer, warnings):
    gold_answers = example["gold_answer"]
    answer_em = em_check(final_answer, gold_answers)
    answer_subem = subem_check(final_answer, gold_answers)
    info_text = "\n".join(information_blocks)
    hit = evidence_hit(info_text, gold_answers)
    num_queries = sum(len(qs) for qs in queries_by_turn)
    return {
        "id": example["id"],
        "source": example["source"],
        "question": example["question"],
        "gold_answer": gold_answers,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {example['question']}"},
            {"role": "assistant", "content": assistant_content},
        ],
        "queries_by_turn": queries_by_turn,
        "quality_flags": {
            "format_valid": not warnings,
            "answer_em": answer_em,
            "answer_subem": answer_subem,
            "answer_match": answer_em or answer_subem,
            "evidence_hit": hit,
            "num_search_turns": len(queries_by_turn),
            "num_queries": num_queries,
            "multi_turn": len(queries_by_turn) > 1,
            "warnings": warnings,
        },
    }


def build_one(args, example):
    warnings = []
    question = ensure_question_mark(example["question"])
    gold_answers = example["gold_answer"]
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    example = {**example, "question": question, "gold_answer": gold_answers}

    first_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FIRST_TURN_PROMPT.format(question=question)},
    ]
    first_text = chat_completion(args, first_messages)
    queries, first_warnings = validate_first_turn(first_text, args.max_queries_per_turn)
    warnings.extend(first_warnings)
    if first_warnings and args.strict:
        raise ValueError(";".join(first_warnings))

    assistant_parts = [first_text]
    queries_by_turn = []
    information_blocks = []

    for turn_idx in range(args.max_turns):
        if turn_idx == 0:
            turn_queries = queries
        else:
            turn_queries, query_warnings = parse_queries(next_text, args.max_queries_per_turn)
            warnings.extend(query_warnings)
            if query_warnings and args.strict:
                raise ValueError(";".join(query_warnings))

        queries_by_turn.append(turn_queries)
        results = retrieve(args, turn_queries)
        information = format_information(turn_queries, results)
        information_blocks.append(information)
        assistant_parts.append(information)

        trajectory = "\n".join(assistant_parts)
        next_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": NEXT_TURN_PROMPT.format(
                    question=question,
                    gold_answers=json.dumps(gold_answers, ensure_ascii=False),
                    trajectory=trajectory,
                ),
            },
        ]
        next_text = chat_completion(args, next_messages)
        assistant_parts.append(next_text)

        if INFORMATION_RE.search(next_text):
            warnings.append("teacher_generated_information")
            if args.strict:
                raise ValueError("teacher_generated_information")

        final_answer = extract_answer(next_text)
        if final_answer is not None:
            if len(final_answer.split()) > args.max_answer_words:
                warnings.append(f"answer_too_long:{len(final_answer.split())}")
            record = make_record(
                example,
                "\n".join(assistant_parts),
                queries_by_turn,
                information_blocks,
                final_answer,
                warnings,
            )
            return record

        if turn_idx == args.max_turns - 1:
            raise ValueError("missing_answer_after_max_turns")

    raise RuntimeError("unreachable")


def should_accept(record, args):
    flags = record["quality_flags"]
    if args.keep_rejected:
        return True
    if flags["warnings"]:
        return False
    if args.require_answer_match and not flags["answer_match"]:
        return False
    if args.require_evidence_hit and not flags["evidence_hit"]:
        return False
    return True


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default="docs/phase2/phase2_smoke_samples.jsonl")
    parser.add_argument("--input_parquet", default=None)
    parser.add_argument("--data_source", default="nq")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/litecoa_sft/litecoa_sft_20.jsonl")
    parser.add_argument("--rejected_output", default="data/litecoa_sft/litecoa_sft_20_rejected.jsonl")
    parser.add_argument("--report_output", default="data/litecoa_sft/litecoa_sft_20_report.json")
    parser.add_argument("--target_count", type=int, default=20)
    parser.add_argument("--max_candidates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260616)

    parser.add_argument("--api_key", default=None)
    parser.add_argument("--env_file", default=".env")
    parser.add_argument("--api_url", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--api_timeout", type=int, default=120)
    parser.add_argument("--api_retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)

    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--retriever_timeout", type=int, default=60)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=2)
    parser.add_argument("--max_queries_per_turn", type=int, default=3)
    parser.add_argument("--max_answer_words", type=int, default=12)
    parser.add_argument("--require_answer_match", action="store_true", default=True)
    parser.add_argument("--no_require_answer_match", action="store_false", dest="require_answer_match")
    parser.add_argument("--require_evidence_hit", action="store_true", default=True)
    parser.add_argument("--no_require_evidence_hit", action="store_false", dest="require_evidence_hit")
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no_strict", action="store_false", dest="strict")
    parser.add_argument("--keep_rejected", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    examples = load_examples(args)
    random.shuffle(examples)
    examples = examples[: args.max_candidates]

    accepted = []
    rejected = []
    started = time.time()
    for idx, example in enumerate(examples, start=1):
        if len(accepted) >= args.target_count:
            break
        try:
            record = build_one(args, example)
            if should_accept(record, args):
                accepted.append(record)
                status = "accepted"
            else:
                rejected.append(record)
                status = "rejected"
        except Exception as exc:
            rejected.append(
                {
                    "id": example.get("id"),
                    "source": example.get("source"),
                    "question": example.get("question"),
                    "gold_answer": example.get("gold_answer"),
                    "error": str(exc),
                }
            )
            status = "error"
        print(f"[{idx}/{len(examples)}] {status}: {example.get('id')} accepted={len(accepted)}")

    output = Path(args.output)
    rejected_output = Path(args.rejected_output)
    report_output = Path(args.report_output)
    write_jsonl(output, accepted)
    write_jsonl(rejected_output, rejected)

    report = {
        "target_count": args.target_count,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "elapsed_sec": round(time.time() - started, 3),
        "model": args.model or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4",
        "retriever_url": args.retriever_url,
        "output": str(output),
        "rejected_output": str(rejected_output),
        "settings": {
            "max_turns": args.max_turns,
            "max_queries_per_turn": args.max_queries_per_turn,
            "topk": args.topk,
            "require_answer_match": args.require_answer_match,
            "require_evidence_hit": args.require_evidence_hit,
            "strict": args.strict,
        },
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
