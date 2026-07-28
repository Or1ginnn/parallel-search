# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import re
from typing import Dict, Optional

from verl.utils.reward_score.finqa_metrics import (
    finqa_answer_match,
    minimum_numeric_relative_error,
)
from verl.utils.reward_score.qa_em import em_check, extract_solution, normalize_answer


def _extract_blocks(text: str, tag: str) -> list[str]:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    return [match.strip() for match in re.findall(pattern, text, re.DOTALL)]


def _split_queries(search_text: str) -> list[str]:
    seen = set()
    queries = []
    for query in search_text.split("||"):
        query = " ".join(query.strip().split())
        if not query or "<" in query or ">" in query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries


def _has_gold_in_information(retrieved_information_str: str, golden_answers) -> bool:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    retrieved_blocks = _extract_blocks(retrieved_information_str, "information")
    normalized_information = normalize_answer("\n".join(retrieved_blocks))
    if not normalized_information:
        return False
    return any(normalize_answer(answer) in normalized_information for answer in golden_answers)


_PROGRAM_CALL_RE = re.compile(r"[A-Za-z_]+\((.*?)\)")
_PROGRAM_NUMBER_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)%?$")
_TEXT_NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)%?")
_EVIDENCE_TOKEN_RE = re.compile(r"[a-z0-9.%-]+")


def _numeric_value(text: str):
    text = str(text).strip().replace(",", "")
    if not _PROGRAM_NUMBER_RE.fullmatch(text):
        return None
    is_percent = text.endswith("%")
    try:
        value = float(text.rstrip("%"))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value / 100.0 if is_percent else value


def _program_operands(program) -> list[float]:
    """Extract source numeric operands, excluding constants and step refs."""

    if hasattr(program, "tolist"):
        program = program.tolist()
    if isinstance(program, (list, tuple)):
        program = ", ".join(str(step) for step in program)
    operands = []
    for arguments in _PROGRAM_CALL_RE.findall(str(program or "")):
        for argument in arguments.split(","):
            argument = argument.strip()
            lowered = argument.lower()
            if lowered == "none" or lowered.startswith("#") or lowered.startswith("const_"):
                continue
            value = _numeric_value(argument)
            if value is not None and not any(math.isclose(value, seen, rel_tol=1e-9, abs_tol=1e-9) for seen in operands):
                operands.append(value)
    return operands


def _text_numeric_values(text: str) -> list[float]:
    values = []
    for token in _TEXT_NUMBER_RE.findall(str(text or "")):
        value = _numeric_value(token.replace(",", ""))
        if value is not None:
            values.append(value)
    return values


def _operand_coverage(text: str, program) -> Optional[float]:
    operands = _program_operands(program)
    if not operands:
        return None
    observed_values = _text_numeric_values(text)
    matched = sum(
        any(math.isclose(operand, observed, rel_tol=1e-9, abs_tol=1e-9) for observed in observed_values)
        for operand in operands
    )
    return matched / len(operands)


def _gold_evidence_coverage(text: str, gold_evidence) -> Optional[float]:
    information_tokens = set(_EVIDENCE_TOKEN_RE.findall(str(text or "").lower()))
    evidence_items = []
    if hasattr(gold_evidence, "tolist"):
        gold_evidence = gold_evidence.tolist()
    for item in gold_evidence if gold_evidence is not None else []:
        evidence_text = item.get("text", "") if isinstance(item, dict) else str(item)
        tokens = set(_EVIDENCE_TOKEN_RE.findall(evidence_text.lower()))
        if len(tokens) >= 3:
            evidence_items.append(tokens)
    if not evidence_items:
        return None
    hits = sum(
        len(tokens & information_tokens) / len(tokens) >= 0.8
        for tokens in evidence_items
    )
    return hits / len(evidence_items)


def _without_query_lines(text: str) -> str:
    """Exclude model-authored query text from retrieval-evidence scoring."""

    return re.sub(r"(?m)^\s*\[Query\].*$", "", str(text or ""))


def _retrieval_coverage(text: str, ground_truth) -> float:
    """Prefer program-operand coverage; fall back to gold-evidence coverage."""

    text = _without_query_lines(text)
    operand_coverage = _operand_coverage(text, ground_truth.get("program"))
    if operand_coverage is not None:
        return operand_coverage
    evidence_coverage = _gold_evidence_coverage(text, ground_truth.get("gold_evidence"))
    return evidence_coverage if evidence_coverage is not None else 0.0


def _query_information_blocks(retrieved_information_str: str) -> Dict[str, str]:
    query_blocks = {}
    for information in _extract_blocks(retrieved_information_str, "information"):
        matches = list(re.finditer(r"(?:^|\n)\[Query\]\s*(.*?)\n", information))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(information)
            query = " ".join(match.group(1).strip().split()).lower()
            evidence = information[match.end():end].strip()
            if query and evidence:
                query_blocks[query] = "\n".join(filter(None, (query_blocks.get(query), evidence)))
    return query_blocks


def _parallel_retrieval_gain(model_response_str: str, retrieved_information_str: str, ground_truth) -> float:
    search_blocks = _extract_blocks(model_response_str, "search")
    first_queries = _split_queries(search_blocks[0]) if search_blocks else []
    if len(first_queries) < 2:
        return 0.0

    query_blocks = _query_information_blocks(retrieved_information_str)
    first_query_evidence = [query_blocks.get(query.lower(), "") for query in first_queries]
    nonempty_evidence = [evidence for evidence in first_query_evidence if evidence]
    if len(nonempty_evidence) < 2:
        return 0.0

    union_coverage = _retrieval_coverage("\n".join(nonempty_evidence), ground_truth)
    best_single_coverage = max(
        _retrieval_coverage(evidence, ground_truth) for evidence in nonempty_evidence
    )
    return max(0.0, union_coverage - best_single_coverage)


def _numeric_near_miss_quality(
    model_response_str: str,
    ground_truth,
    max_relative_error: float,
    strict_relative_error: float = 0.01,
) -> float:
    if ground_truth.get("answer_type") != "finqa" or compute_answer_em(model_response_str, ground_truth):
        return 0.0
    answer = extract_solution(solution_str=model_response_str)
    if answer is None:
        return 0.0
    relative_error = minimum_numeric_relative_error(
        answer,
        ground_truth.get("target"),
        ground_truth.get("executable_answer"),
    )
    if relative_error is None or relative_error <= strict_relative_error or relative_error > max_relative_error:
        return 0.0
    width = max_relative_error - strict_relative_error
    return max(0.0, (max_relative_error - relative_error) / width) if width > 0 else 0.0


def compute_answer_em(model_response_str: str, ground_truth) -> float:
    answer = extract_solution(solution_str=model_response_str)
    if answer is None:
        return 0.0
    if ground_truth.get("answer_type") == "finqa":
        return float(
            finqa_answer_match(
                answer,
                ground_truth["target"],
                ground_truth.get("executable_answer"),
            )
        )
    if em_check(answer, ground_truth["target"]):
        return 1.0
    return 0.0


def has_answer(model_response_str: str) -> bool:
    return bool(_extract_blocks(model_response_str, "answer"))


def has_generated_information(model_response_str: str) -> bool:
    return "<information>" in model_response_str or "</information>" in model_response_str


def compute_score_em_litecoa_components(
    model_response_str: str,
    retrieved_information_str: str,
    ground_truth,
    score: float = 1.0,
    answer_present_bonus: float = 0.05,
    no_generated_information_bonus: float = 0.05,
    evidence_hit_bonus: float = 0.05,
    valid_search_bonus: float = 0.03,
    parallel_evidence_bonus: float = 0.03,
    finqa_v2_reward: bool = False,
    finqa_retrieval_coverage_bonus: float = 0.05,
    finqa_parallel_retrieval_gain_bonus: float = 0.05,
    finqa_near_miss_bonus: float = 0.05,
    finqa_near_miss_max_relative_error: float = 0.05,
) -> Dict[str, float]:
    """LiteCoA reward with answer EM plus small positive shaping bonuses.

    model_response_str contains only model-generated tokens.
    retrieved_information_str contains only retriever-inserted observation
    tokens, so evidence bonuses cannot be earned by generated information.
    """

    golden_answers = ground_truth["target"]
    answer_em = compute_answer_em(model_response_str, ground_truth)
    total = score * answer_em

    answer_blocks = _extract_blocks(model_response_str, "answer")
    search_blocks = _extract_blocks(model_response_str, "search")
    first_search_queries = _split_queries(search_blocks[0]) if search_blocks else []
    has_valid_search = any(_split_queries(search) for search in search_blocks)
    evidence_hit = _has_gold_in_information(retrieved_information_str, golden_answers)
    retrieval_coverage = 0.0
    parallel_retrieval_gain = 0.0
    numeric_near_miss_quality = 0.0
    if answer_blocks:
        total += answer_present_bonus
    if not has_generated_information(model_response_str):
        total += no_generated_information_bonus
    if has_valid_search:
        total += valid_search_bonus
    if finqa_v2_reward and ground_truth.get("answer_type") == "finqa":
        retrieval_coverage = _retrieval_coverage(retrieved_information_str, ground_truth)
        parallel_retrieval_gain = _parallel_retrieval_gain(
            model_response_str,
            retrieved_information_str,
            ground_truth,
        )
        numeric_near_miss_quality = _numeric_near_miss_quality(
            model_response_str,
            ground_truth,
            max_relative_error=finqa_near_miss_max_relative_error,
        )
        total += finqa_retrieval_coverage_bonus * retrieval_coverage
        total += finqa_parallel_retrieval_gain_bonus * parallel_retrieval_gain
        total += finqa_near_miss_bonus * numeric_near_miss_quality
    else:
        if evidence_hit:
            total += evidence_hit_bonus
        if len(first_search_queries) >= 2 and evidence_hit:
            total += parallel_evidence_bonus

    return {
        "score": total,
        "answer_em": answer_em,
        "retrieval_coverage": retrieval_coverage,
        "parallel_retrieval_gain": parallel_retrieval_gain,
        "numeric_near_miss_quality": numeric_near_miss_quality,
    }


def compute_score_em_litecoa(*args, **kwargs) -> float:
    """Return the scalar reward while preserving the original public API."""

    return compute_score_em_litecoa_components(*args, **kwargs)["score"]
