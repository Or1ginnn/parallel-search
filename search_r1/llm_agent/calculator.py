"""Restricted calculator for FinQA-style numeric reasoning."""

import math
import re


class CalculationError(ValueError):
    pass


_CONSTANTS = {
    "const_m1": -1.0,
    "const_1": 1.0,
    "const_2": 2.0,
    "const_3": 3.0,
    "const_4": 4.0,
    "const_5": 5.0,
    "const_6": 6.0,
    "const_7": 7.0,
    "const_8": 8.0,
    "const_9": 9.0,
    "const_10": 10.0,
    "const_100": 100.0,
    "const_1000": 1000.0,
    "const_10000": 10000.0,
    "const_100000": 100000.0,
    "const_1000000": 1000000.0,
    "const_1000000000": 1000000000.0,
}


def _split_top_level(text):
    parts, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise CalculationError("unbalanced parentheses")
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    if depth:
        raise CalculationError("unbalanced parentheses")
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _function_call(text):
    match = re.fullmatch(r"([a-z_]+)\((.*)\)", text.strip(), re.DOTALL)
    if not match:
        return None
    return match.group(1), _split_top_level(match.group(2))


def _number(text):
    value = text.strip().lower().replace(",", "").replace("$", "")
    if value.endswith("%"):
        value = value[:-1].strip()
        try:
            return float(value) / 100.0
        except ValueError as error:
            raise CalculationError("invalid percentage") from error
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    try:
        return float(value)
    except ValueError as error:
        raise CalculationError("invalid numeric operand") from error


def _finite(value):
    if not math.isfinite(value):
        raise CalculationError("non-finite result")
    return value


def evaluate_expression(expression, references=None):
    """Evaluate a restricted nested FinQA arithmetic expression.

    The expression language intentionally has no variables, attributes, Python
    operators, or executable code. ``references`` is used only for evaluating
    dataset gold programs with ``#0``-style intermediate references.
    """

    references = references or []
    text = " ".join(str(expression).strip().split())
    if not text:
        raise CalculationError("empty expression")
    if text.startswith("#"):
        try:
            return references[int(text[1:])]
        except (ValueError, IndexError) as error:
            raise CalculationError("unknown intermediate reference") from error
    if text in _CONSTANTS:
        return _CONSTANTS[text]

    call = _function_call(text)
    if call is None:
        return _number(text)
    operation, raw_args = call
    values = [evaluate_expression(arg, references) for arg in raw_args]

    if operation == "add" and len(values) == 2:
        return _finite(values[0] + values[1])
    if operation == "subtract" and len(values) == 2:
        return _finite(values[0] - values[1])
    if operation == "multiply" and len(values) == 2:
        return _finite(values[0] * values[1])
    if operation == "divide" and len(values) == 2:
        if values[1] == 0:
            raise CalculationError("division by zero")
        return _finite(values[0] / values[1])
    if operation == "exp" and len(values) == 2:
        return _finite(values[0] ** values[1])
    if operation == "greater" and len(values) == 2:
        return values[0] > values[1]
    if operation in {"sum", "table_sum"} and values:
        return _finite(sum(values))
    if operation in {"average", "table_average"} and values:
        return _finite(sum(values) / len(values))
    if operation in {"max", "table_max"} and values:
        return _finite(max(values))
    if operation in {"min", "table_min"} and values:
        return _finite(min(values))
    raise CalculationError("unsupported calculator operation")


def format_value(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    return format(float(value), ".12g")


def evaluate_program(program):
    """Return successfully executable intermediate values from a FinQA program."""

    values = []
    for expression in _split_top_level(program or ""):
        try:
            values.append(evaluate_expression(expression, values))
        except CalculationError:
            break
    return values


def value_matches(left, right, relative_tolerance=0.01):
    left_text = str(left).strip().lower()
    right_text = str(right).strip().lower()
    if left_text in {"yes", "no"} or right_text in {"yes", "no"}:
        return left_text == right_text
    try:
        left_value = _number(left_text)
        right_value = _number(right_text)
    except CalculationError:
        return False
    return abs(left_value - right_value) <= max(
        1e-4, relative_tolerance * max(1.0, abs(right_value))
    )
