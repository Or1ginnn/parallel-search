#!/usr/bin/env python3
"""Prepare FinQA reports for Finance Research Agent retrieval experiments.

The script keeps QA supervision and retrieval corpus separate. Corpus chunks are
created only from each report's text and table; answers, programs, and gold
evidence are emitted exclusively in the QA files for evaluation and training.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


SPLITS = ("train", "dev", "test")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def clean_text(value: Any) -> str:
    return " ".join(str(value).replace("\n", " ").split())


def chunk_lines(lines: List[str], max_lines: int, max_chars: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_chars = 0
    for line in lines:
        line = clean_text(line)
        if not line:
            continue
        should_flush = current and (
            len(current) >= max_lines or current_chars + len(line) + 1 > max_chars
        )
        if should_flush:
            chunks.append(" ".join(current))
            current, current_chars = [], 0
        current.append(line)
        current_chars += len(line) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def table_row_text(table: List[List[Any]], row_index: int) -> str:
    header = [clean_text(cell) for cell in table[0]] if table else []
    row = [clean_text(cell) for cell in table[row_index]]
    width = max(len(header), len(row))
    pairs = []
    for index in range(width):
        column = header[index] if index < len(header) else f"column_{index}"
        value = row[index] if index < len(row) else ""
        pairs.append(f"{column}: {value}")
    return " | ".join(pairs)


def report_chunks(
    row: Dict[str, Any], split: str, max_text_lines: int, max_chars: int
) -> List[Dict[str, Any]]:
    report_id = str(row["filename"])
    chunks: List[Dict[str, Any]] = []
    text_lines = list(row.get("pre_text", [])) + list(row.get("post_text", []))
    for index, text in enumerate(chunk_lines(text_lines, max_text_lines, max_chars)):
        title = f"FinQA | {report_id} | text {index}"
        chunks.append(
            {
                "id": f"{split}::{report_id}::text::{index}",
                "report_id": report_id,
                "split": split,
                "source_type": "text",
                "position": index,
                "title": title,
                "text": text,
                "contents": f"{title}\n{text}",
            }
        )

    table = row.get("table", [])
    for row_index in range(1, len(table)):
        text = table_row_text(table, row_index)
        title = f"FinQA | {report_id} | table {row_index}"
        chunks.append(
            {
                "id": f"{split}::{report_id}::table::{row_index}",
                "report_id": report_id,
                "split": split,
                "source_type": "table",
                "position": row_index,
                "title": title,
                "text": text,
                "contents": f"{title}\n{text}",
            }
        )
    return chunks


def qa_record(row: Dict[str, Any], split: str) -> Dict[str, Any]:
    qa = row["qa"]
    gold = qa.get("gold_inds", {})
    return {
        "id": str(row["id"]),
        "split": split,
        "report_id": str(row["filename"]),
        "question": clean_text(qa["question"]),
        "gold_answer": clean_text(qa.get("answer", "")),
        "executable_answer": clean_text(qa.get("exe_ans", "")),
        "program": qa.get("program", []),
        "gold_evidence": [
            {"source_id": str(source_id), "text": clean_text(text)}
            for source_id, text in gold.items()
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        default="data/finance_finqa/raw/FinQA/dataset",
        help="Directory containing FinQA train/dev/test JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/finance_finqa/processed",
        help="Destination for QA JSONL files, corpus chunks, and audit report.",
    )
    parser.add_argument("--smoke_count", type=int, default=50)
    parser.add_argument("--smoke_split", choices=SPLITS, default="dev")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max_text_lines", type=int, default=5)
    parser.add_argument("--max_chars", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    report_sets: Dict[str, set] = {}
    audit: Dict[str, Any] = {"source": str(raw_dir), "splits": {}}

    for split in SPLITS:
        path = raw_dir / f"{split}.json"
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        split_rows[split] = rows
        report_sets[split] = {str(row["filename"]) for row in rows}

        qas = [qa_record(row, split) for row in rows]
        report_rows: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            report_rows.setdefault(str(row["filename"]), row)
        corpus = []
        for report in report_rows.values():
            corpus.extend(
                report_chunks(report, split, args.max_text_lines, args.max_chars)
            )

        write_jsonl(output_dir / f"{split}.jsonl", qas)
        write_jsonl(output_dir / f"corpus_{split}.jsonl", corpus)
        source_counts = Counter(chunk["source_type"] for chunk in corpus)
        audit["splits"][split] = {
            "qa_count": len(qas),
            "report_count": len(report_rows),
            "corpus_chunk_count": len(corpus),
            "text_chunk_count": source_counts["text"],
            "table_chunk_count": source_counts["table"],
        }

    all_corpus = []
    for split in SPLITS:
        with (output_dir / f"corpus_{split}.jsonl").open(encoding="utf-8") as handle:
            all_corpus.extend(json.loads(line) for line in handle if line.strip())
    write_jsonl(output_dir / "corpus_all.jsonl", all_corpus)

    overlaps = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlaps[f"{left}_{right}"] = len(report_sets[left] & report_sets[right])
    audit["report_overlap_count"] = overlaps
    audit["chunking"] = {
        "max_text_lines": args.max_text_lines,
        "max_chars": args.max_chars,
        "corpus_policy": "Uses report text and table only; excludes QA answers, programs, and gold evidence.",
    }

    smoke_pool = [qa_record(row, args.smoke_split) for row in split_rows[args.smoke_split]]
    random.Random(args.seed).shuffle(smoke_pool)
    smoke = smoke_pool[: min(args.smoke_count, len(smoke_pool))]
    write_jsonl(output_dir / f"smoke_{args.smoke_split}_{len(smoke)}.jsonl", smoke)
    audit["smoke"] = {
        "split": args.smoke_split,
        "count": len(smoke),
        "seed": args.seed,
    }

    with (output_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
