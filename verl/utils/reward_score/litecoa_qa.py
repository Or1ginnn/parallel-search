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
    normalized_information = normalize_answer(retrieved_information_str)
    if not normalized_information:
        return False
    return any(normalize_answer(answer) in normalized_information for answer in golden_answers)


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
) -> float:
    """LiteCoA reward with answer EM plus small positive shaping bonuses.

    model_response_str contains only model-generated tokens.
    retrieved_information_str contains only retriever-inserted observation
    tokens, so evidence bonuses cannot be earned by generated information.
    """

    golden_answers = ground_truth["target"]
    answer = extract_solution(solution_str=model_response_str)
    total = 0.0
    if answer is not None and em_check(answer, golden_answers):
        total += score

    plan_blocks = _extract_blocks(model_response_str, "plan")
    answer_blocks = _extract_blocks(model_response_str, "answer")
    search_blocks = _extract_blocks(model_response_str, "search")
    first_search_queries = _split_queries(search_blocks[0]) if search_blocks else []
    has_valid_search = any(_split_queries(search) for search in search_blocks)
    evidence_hit = _has_gold_in_information(retrieved_information_str, golden_answers)

    if len(plan_blocks) == 1:
        total += plan_once_bonus
    if answer_blocks:
        total += answer_present_bonus
    if "<information>" not in model_response_str and "</information>" not in model_response_str:
        total += no_generated_information_bonus
    if evidence_hit:
        total += evidence_hit_bonus
    if has_valid_search:
        total += valid_search_bonus
    if len(first_search_queries) >= 2 and evidence_hit:
        total += parallel_evidence_bonus

    return total
