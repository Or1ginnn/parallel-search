import argparse
import json
import os
import random
import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import transformers


PROJECT_DIR = Path("/mnt/data1/zar/search-1/Search-R1")
DATA_PATH = PROJECT_DIR / "data/nq_search/train.parquet"
OUTPUT_DIR = PROJECT_DIR / "trajectory/phase2_smoke"
RETRIEVER_URL = "http://127.0.0.1:8000/retrieve"
LITECOA_MODEL = PROJECT_DIR / "hf_cache/Qwen2.5-3B"
BASELINE_MODEL = PROJECT_DIR / "verl_checkpoints/nq-search-r1-grpo-qwen2.5-3b-em/actor/global_step_200"

TOPK = 3
MAX_TURNS = 3
MAX_QUERIES_PER_TURN = 3
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TIMEOUT = 60
CURR_EOS = {151645, 151643}


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


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def normalize_question(question):
    question = str(question).strip()
    if question and question[-1] != "?":
        question += "?"
    return question


def extract_final_answer(text):
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if not matches:
        return ""
    return " ".join(matches[-1].strip().split())


def has_answer(text):
    return re.search(r"<answer>.*?</answer>", text, re.DOTALL) is not None


def truncate_at_first_action(text):
    candidates = []
    for tag in ("search", "answer"):
        end = text.find(f"</{tag}>")
        if end >= 0:
            candidates.append(end + len(f"</{tag}>"))
    return text[: min(candidates)] if candidates else text


def get_baseline_query(text):
    matches = re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL)
    return " ".join(matches[-1].strip().split()) if matches else None


def get_litecoa_queries(text):
    warnings = []
    matches = re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL)
    if not matches:
        return [], warnings

    queries = []
    seen = set()
    for item in matches[-1].split("||"):
        query = " ".join(item.strip().split())
        if not query:
            continue
        key = query.lower()
        if key in seen:
            warnings.append(f'duplicate query dropped: "{query}"')
            continue
        seen.add(key)
        queries.append(query)

    if len(queries) > MAX_QUERIES_PER_TURN:
        warnings.append(
            f"query count {len(queries)} exceeds max {MAX_QUERIES_PER_TURN}; truncating"
        )
        queries = queries[:MAX_QUERIES_PER_TURN]
    return queries, warnings


def retrieve(queries):
    payload = {"queries": queries, "topk": TOPK, "return_scores": True}
    response = requests.post(RETRIEVER_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()["result"]


def format_baseline_information(retrieval_result):
    text = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        body = "\n".join(content.split("\n")[1:])
        text += f"Doc {idx + 1}(Title: {title}) {body}\n"
    return text


def format_litecoa_information(queries, results):
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


def baseline_prompt(question):
    return (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
        "and it will return the top searched results between <information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
        f"without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
    )


def litecoa_prompt(question):
    return (
        "You are a search-augmented reasoning agent. "
        "You can only use the following tags: <think>, <plan>, <search>, <information>, <answer>. "
        "You must conduct reasoning inside <think> and </think> before every plan, search, or answer. "
        "Use <plan> to decompose the question into searchable sub-questions. "
        "If you lack knowledge, call a search engine by <search> query </search>. "
        f"You can put multiple independent queries in one search action with \"||\", for example <search> query1 || query2 </search>. "
        f"Each search action can contain at most {MAX_QUERIES_PER_TURN} queries. "
        "The search engine will return results between <information> and </information>. "
        "Do not generate <information> yourself. "
        "If the evidence is sufficient, provide the answer inside <answer> and </answer>, without detailed illustrations. "
        f"Question: {question}\n"
    )


def load_model(model_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, device_map="auto"
    )
    target_sequences = [
        "</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n",
        "</answer>", " </answer>", "</answer>\n", " </answer>\n", "</answer>\n\n", " </answer>\n\n",
    ]
    stopping = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    return tokenizer, model, stopping


def run_one(sample, mode, tokenizer, model, stopping):
    question = normalize_question(sample["question"])
    prompt = litecoa_prompt(question) if mode == "litecoa" else baseline_prompt(question)
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

    full_text = prompt
    search_turns = 0
    total_queries = 0
    queries_by_turn = []
    parser_warnings = []
    agent_warnings = []
    error = None

    try:
        for _ in range(MAX_TURNS):
            input_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)
            attention_mask = torch.ones_like(input_ids)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                stopping_criteria=stopping,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=TEMPERATURE,
            )
            output_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            output_text = truncate_at_first_action(output_text)
            full_text += output_text

            if "<information>" in output_text or "</information>" in output_text:
                parser_warnings.append("model generated information tag")

            if has_answer(output_text) or outputs[0][-1].item() in CURR_EOS:
                break

            if mode == "litecoa":
                queries, warnings = get_litecoa_queries(output_text)
                parser_warnings.extend(warnings)
                if queries:
                    results = retrieve(queries)
                    information = format_litecoa_information(queries, results)
                    full_text += f"\n\n{information}\n\n"
                else:
                    agent_warnings.append("no valid query found")
                    break
            else:
                query = get_baseline_query(output_text)
                if query:
                    queries = [query]
                    results = retrieve(queries)
                    information = format_baseline_information(results[0])
                    full_text += f"<information>{information}</information>\n\n"
                else:
                    queries = []
                    agent_warnings.append("no valid query found")
                    break

            search_turns += 1
            total_queries += len(queries)
            queries_by_turn.append(queries)
    except Exception:
        error = traceback.format_exc()

    return {
        "id": sample.get("id"),
        "question": question,
        "gold_answer": json_safe(sample.get("golden_answers")),
        "final_answer": extract_final_answer(full_text),
        "has_answer_tag": has_answer(full_text),
        "search_turns": search_turns,
        "total_queries": total_queries,
        "queries_by_turn": queries_by_turn,
        "parser_warnings": parser_warnings,
        "agent_warnings": agent_warnings,
        "error": error,
        "output_text": full_text,
    }


def summarize(records):
    n = len(records)
    return {
        "n": n,
        "errors": sum(bool(r["error"]) for r in records),
        "has_answer": sum(bool(r["has_answer_tag"]) for r in records),
        "parser_warnings": sum(bool(r["parser_warnings"]) for r in records),
        "agent_warnings": sum(bool(r["agent_warnings"]) for r in records),
        "avg_search_turns": sum(r["search_turns"] for r in records) / max(n, 1),
        "avg_total_queries": sum(r["total_queries"] for r in records) / max(n, 1),
    }


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    sample_df = df.sample(n=args.num_samples, random_state=args.seed)
    samples = [json_safe(row.to_dict()) for _, row in sample_df.iterrows()]
    write_jsonl(OUTPUT_DIR / "phase2_smoke_samples.jsonl", samples)

    all_summaries = {}
    for mode, model_path in [("litecoa", LITECOA_MODEL), ("baseline", BASELINE_MODEL)]:
        print(f"[phase2] loading {mode}: {model_path}", flush=True)
        tokenizer, model, stopping = load_model(model_path)
        records = []
        for idx, sample in enumerate(samples, start=1):
            start = time.time()
            print(f"[phase2] {mode} sample {idx}/{len(samples)} id={sample.get('id')}", flush=True)
            record = run_one(sample, mode, tokenizer, model, stopping)
            record["mode"] = mode
            record["elapsed_sec"] = round(time.time() - start, 3)
            records.append(record)
            write_jsonl(OUTPUT_DIR / f"phase2_{mode}_results.jsonl", records)
        all_summaries[mode] = summarize(records)
        del model
        torch.cuda.empty_cache()

    report = render_report(all_summaries)
    (OUTPUT_DIR / "phase2_smoke_test_report.md").write_text(report, encoding="utf-8")
    print("[phase2] done", flush=True)


def render_report(summaries):
    lite = summaries.get("litecoa", {})
    base = summaries.get("baseline", {})
    lines = [
        "# Phase 2 Smoke Test Report",
        "",
        "## 实验设置",
        "",
        f"- 样本：训练集随机 50 条，固定随机种子 `20260614`",
        f"- LiteCoA 模型：`{LITECOA_MODEL}`",
        f"- Search-R1 baseline 模型：`{BASELINE_MODEL}`",
        f"- Retriever：`{RETRIEVER_URL}`，`topk={TOPK}`",
        f"- 最大搜索轮数：`{MAX_TURNS}`",
        f"- LiteCoA 每轮最大 query 数：`{MAX_QUERIES_PER_TURN}`",
        "",
        "## 汇总结果",
        "",
        "| 模式 | 样本数 | 报错数 | 出现 `<answer>` | parser warning | agent warning | 平均 search_turns | 平均 total_queries |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in [("LiteCoA", lite), ("Search-R1 baseline", base)]:
        lines.append(
            f"| {name} | {summary.get('n', 0)} | {summary.get('errors', 0)} | "
            f"{summary.get('has_answer', 0)} | {summary.get('parser_warnings', 0)} | "
            f"{summary.get('agent_warnings', 0)} | {summary.get('avg_search_turns', 0):.2f} | "
            f"{summary.get('avg_total_queries', 0):.2f} |"
        )
    lines += [
        "",
        "## LiteCoA 是否能稳定完成推理",
        "",
        "见上表。逐条结果保存在 `trajectory/phase2_smoke/phase2_litecoa_results.jsonl`，包含 question、gold answer、final answer、是否出现 `<answer>`、search_turns、total_queries、每轮 query、parser warning、agent warning 和 error。",
        "",
        "## 常见失败类型",
        "",
        "- `parser_warnings`：模型生成了不应自己生成的 `<information>`，或单轮 query 数超过限制，或重复 query 被丢弃。",
        "- `agent_warnings`：模型没有产生可解析的 `<search>`，且也没有给出 `<answer>`。",
        "- `error`：retriever 请求、模型生成或解析过程中的异常。",
        "",
        "## 和 Search-R1 baseline 的初步差异",
        "",
        "LiteCoA 支持单轮多 query，并在结果中记录每轮 query 列表；baseline 按原始 Search-R1 风格每轮解析一个 `<search>` query。初步差异请结合上表和两份逐条 JSONL 查看。",
        "",
        "## 输出文件",
        "",
        "- `trajectory/phase2_smoke/phase2_smoke_samples.jsonl`",
        "- `trajectory/phase2_smoke/phase2_litecoa_results.jsonl`",
        "- `trajectory/phase2_smoke/phase2_baseline_results.jsonl`",
        "- `trajectory/phase2_smoke/phase2_smoke_test_report.md`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
