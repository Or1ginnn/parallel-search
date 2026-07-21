#!/usr/bin/env python3
"""Evaluate FinQA evidence retrieval through the Search-R1 /retrieve API."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests


TOKEN_RE = re.compile(r"[a-z0-9.%-]+")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def tokens(text: str) -> set:
    return set(TOKEN_RE.findall(text.lower()))


def evidence_coverage(evidence: str, document: str) -> float:
    evidence_tokens = tokens(evidence)
    if not evidence_tokens:
        return 0.0
    return len(evidence_tokens & tokens(document)) / len(evidence_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa_path", default="data/finance_finqa/processed/smoke_dev_50.jsonl"
    )
    parser.add_argument("--retriever_url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--evidence_threshold", type=float, default=0.8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--output",
        default="outputs/finance_finqa/retrieval_smoke_dev_50.jsonl",
    )
    parser.add_argument(
        "--report_output",
        default="outputs/finance_finqa/retrieval_smoke_dev_50_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(Path(args.qa_path)))
    output_path = Path(args.output)
    report_path = Path(args.report_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total_evidence = 0
    matched_evidence = 0
    any_hit = 0
    all_hit = 0
    errors = 0

    with output_path.open("w", encoding="utf-8") as output_handle:
        for qa in rows:
            result: Dict[str, Any] = {
                "id": qa["id"],
                "question": qa["question"],
                "gold_answer": qa["gold_answer"],
                "gold_evidence": qa["gold_evidence"],
            }
            try:
                response = requests.post(
                    args.retriever_url,
                    json={
                        "queries": [qa["question"]],
                        "topk": args.topk,
                        "return_scores": True,
                    },
                    timeout=args.timeout,
                )
                response.raise_for_status()
                retrieved = response.json()["result"][0]
                documents = [item["document"] for item in retrieved]
                scores = [item["score"] for item in retrieved]
                evidence_scores = []
                for evidence in qa["gold_evidence"]:
                    per_doc = [
                        evidence_coverage(evidence["text"], doc.get("text", ""))
                        for doc in documents
                    ]
                    evidence_scores.append(max(per_doc, default=0.0))
                evidence_hits = [
                    score >= args.evidence_threshold for score in evidence_scores
                ]
                total_evidence += len(evidence_hits)
                matched_evidence += sum(evidence_hits)
                any_hit += bool(evidence_hits and any(evidence_hits))
                all_hit += bool(evidence_hits and all(evidence_hits))
                result.update(
                    {
                        "retrieved": documents,
                        "scores": scores,
                        "evidence_coverage": evidence_scores,
                        "evidence_hits": evidence_hits,
                    }
                )
            except Exception as exc:  # Keep the complete smoke record for debugging.
                errors += 1
                result["error"] = f"{type(exc).__name__}: {exc}"
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    total = len(rows)
    report = {
        "total": total,
        "errors": errors,
        "topk": args.topk,
        "evidence_threshold": args.evidence_threshold,
        "any_evidence_hit": any_hit,
        "any_evidence_hit_rate": any_hit / total if total else 0.0,
        "all_evidence_hit": all_hit,
        "all_evidence_hit_rate": all_hit / total if total else 0.0,
        "evidence_recall": matched_evidence / total_evidence if total_evidence else 0.0,
        "output": str(output_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
