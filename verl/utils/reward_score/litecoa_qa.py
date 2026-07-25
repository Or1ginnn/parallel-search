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

import re

from search_r1.llm_agent.calculator import evaluate_program, value_matches
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


def compute_answer_em(model_response_str: str, ground_truth) -> float:
    answer = extract_solution(solution_str=model_response_str)
    if answer is None:
        return 0.0
    if em_check(answer, ground_truth["target"]):
        return 1.0
    if ground_truth.get("answer_type") == "finqa" and _numeric_match(answer, ground_truth["target"]):
        return 1.0
    return 0.0


def _extract_single_number(text: str):
    normalized = text.strip().lower().replace(",", "")
    normalized = normalized.replace("$", "").replace("%", "")
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = "-" + normalized[1:-1]
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", normalized)
    if len(matches) != 1:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def _numeric_match(answer: str, golden_answers, relative_tolerance: float = 0.01) -> bool:
    prediction = _extract_single_number(answer)
    if prediction is None:
        return False
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    for golden_answer in golden_answers:
        target = _extract_single_number(str(golden_answer))
        if target is None:
            continue
        if abs(prediction - target) <= max(1e-2, relative_tolerance * max(1.0, abs(target))):
            return True
    return False


def has_answer(model_response_str: str) -> bool:
    return bool(_extract_blocks(model_response_str, "answer"))


def has_generated_information(model_response_str: str) -> bool:
    return "<information>" in model_response_str or "</information>" in model_response_str


def has_generated_calculation(model_response_str: str) -> bool:
    return "<calculation>" in model_response_str or "</calculation>" in model_response_str


def calculation_matches(tool_observation_str: str, ground_truth) -> tuple[bool, bool]:
    """Check calculator observations against executable FinQA program values."""

    calculator_values = [
        value for value in _extract_blocks(tool_observation_str, "calculation")
        if value and not value.startswith("ERROR:")
    ]
    if not calculator_values:
        return False, False

    program_values = evaluate_program(ground_truth.get("program", ""))
    intermediate_values = program_values[:-1] if len(program_values) > 1 else []
    intermediate_hit = any(
        value_matches(calculated, target)
        for calculated in calculator_values
        for target in intermediate_values
    )

    final_target = ground_truth.get("executable_answer")
    if not final_target and program_values:
        final_target = program_values[-1]
    final_hit = bool(final_target) and any(
        value_matches(calculated, final_target) for calculated in calculator_values
    )
    return intermediate_hit, final_hit


def compute_score_em_litecoa(
    model_response_str: str,
    retrieved_information_str: str,
    ground_truth,
    score: float = 1.0,
    plan_once_bonus: float = 0.05,
    answer_present_bonus: float = 0.05,
    no_generated_information_bonus: float = 0.05,
    evidence_hit_bonus: float = 0.05,
    valid_search_bonus: float = 0.03,
    parallel_evidence_bonus: float = 0.03,
    calculation_intermediate_bonus: float = 0.0,
    calculation_final_bonus: float = 0.0,
) -> float:
    """LiteCoA reward with answer EM plus small positive shaping bonuses.

    model_response_str contains only model-generated tokens.
    retrieved_information_str contains only retriever-inserted observation
    tokens, so evidence bonuses cannot be earned by generated information.
    """

    golden_answers = ground_truth["target"]
    total = 0.0
    total += score * compute_answer_em(model_response_str, ground_truth)

    answer_blocks = _extract_blocks(model_response_str, "answer")
    search_blocks = _extract_blocks(model_response_str, "search")
    first_search_queries = _split_queries(search_blocks[0]) if search_blocks else []
    has_valid_search = any(_split_queries(search) for search in search_blocks)
    evidence_hit = _has_gold_in_information(retrieved_information_str, golden_answers)
    calculation_intermediate_hit, calculation_final_hit = calculation_matches(
        retrieved_information_str, ground_truth
    )

    if plan_once_bonus > 0 and len(_extract_blocks(model_response_str, "plan")) == 1:
        total += plan_once_bonus
    if answer_blocks:
        total += answer_present_bonus
    if not has_generated_information(model_response_str):
        total += no_generated_information_bonus
    if evidence_hit:
        total += evidence_hit_bonus
    if has_valid_search:
        total += valid_search_bonus
    if len(first_search_queries) >= 2 and evidence_hit:
        total += parallel_evidence_bonus
    if calculation_intermediate_hit:
        total += calculation_intermediate_bonus
    if calculation_final_hit:
        total += calculation_final_bonus

    return total
