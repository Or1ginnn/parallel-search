"""Shared numeric-answer matching for FinQA training and evaluation."""

import math
import re

from verl.utils.reward_score.qa_em import normalize_answer


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?%?", re.IGNORECASE)
_UNIT_RE = re.compile(r"\b(thousand|million|billion|trillion)\b", re.IGNORECASE)
_UNIT_SCALE = {
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _parse_numeric_answer(text):
    """Return ``(value, explicit_unit)`` for a single financial number."""

    normalized = " ".join(str(text).strip().lower().replace(",", "").split())
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = "-" + normalized[1:-1]
    numbers = _NUMBER_RE.findall(normalized)
    if len(numbers) != 1:
        return None

    raw_number = numbers[0]
    is_percent = raw_number.endswith("%") or bool(
        re.search(rf"{re.escape(raw_number.rstrip('%'))}\s*(?:percent|percentage)\b", normalized)
    )
    try:
        value = float(raw_number.rstrip("%"))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if is_percent:
        value /= 100.0

    units = _UNIT_RE.findall(normalized)
    unit = units[-1].lower() if units else None
    return value, unit


def numeric_equivalent(prediction, target, relative_tolerance=0.01):
    """Compare FinQA answers with percentage and optional unit awareness."""

    parsed_prediction = _parse_numeric_answer(prediction)
    parsed_target = _parse_numeric_answer(target)
    if parsed_prediction is None or parsed_target is None:
        return False

    prediction_value, prediction_unit = parsed_prediction
    target_value, target_unit = parsed_target
    if prediction_unit and target_unit:
        prediction_value *= _UNIT_SCALE[prediction_unit]
        target_value *= _UNIT_SCALE[target_unit]

    return abs(prediction_value - target_value) <= max(
        1e-4, relative_tolerance * abs(target_value)
    )


def finqa_answer_match(prediction, gold_answers, executable_answer=None, relative_tolerance=0.01):
    """Accept exact gold strings or numeric equivalents of gold/program outputs."""

    prediction = str(prediction or "").strip()
    if not prediction:
        return False
    gold_answers = _as_list(gold_answers)
    executable_answers = _as_list(executable_answer)
    if any(normalize_answer(prediction) == normalize_answer(answer) for answer in gold_answers):
        return True

    targets = gold_answers + executable_answers
    return any(
        numeric_equivalent(prediction, target, relative_tolerance)
        for target in targets
    )
