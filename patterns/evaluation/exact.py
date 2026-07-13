"""Exact / programmatic evaluators: deterministic, cheap, no model call.

These are the baseline every evaluation loop should include before falling
back to an LLM judge: string/regex match, JSON-schema validity, and similar
functional checks. They never call a provider, so they are the fastest and
most reliable scorers in the loop, at the cost of only working when the
task has a verifiable answer.

Both evaluators here read `EvalCase.expected_property`, a string of the form
`"regex:<pattern>"` or `"json_schema:<schema_name>"`, and score a single
candidate output against it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from patterns.evaluation.eval_set import EvalCase

# A tiny fixed registry of named schemas, standing in for a real schema
# store. Each schema lists required keys and, optionally, allowed values.
_SCHEMAS: dict[str, dict[str, Any]] = {
    "order_status": {
        "required": ["order_id", "status"],
        "allowed_status": {"placed", "shipped", "delivered", "cancelled"},
    }
}


@dataclass
class Score:
    """The result of scoring one candidate output against one case.

    Attributes:
        case_id: The `EvalCase.id` this score belongs to.
        evaluator: Name of the evaluator that produced this score, e.g.
            "regex", "json_schema", "semantic_similarity".
        passed: Whether the output cleared the evaluator's bar.
        detail: A short human-readable explanation, useful for triage.
    """

    case_id: str
    evaluator: str
    passed: bool
    detail: str


def regex_match_evaluator(case: EvalCase, output: str) -> Score:
    """Score `output` by checking it matches the case's `regex:` property.

    Args:
        case: The eval case. `case.expected_property` must start with
            "regex:"; the rest of the string is the pattern.
        output: The candidate output to check.

    Raises:
        ValueError: If `case.expected_property` is not a "regex:" property.
    """
    if case.expected_property is None or not case.expected_property.startswith("regex:"):
        raise ValueError(f"Case {case.id!r} has no regex expected_property to check against")
    pattern = case.expected_property.removeprefix("regex:")
    matched = re.search(pattern, output) is not None
    detail = f"pattern {pattern!r} {'matched' if matched else 'not found'} in output"
    return Score(case_id=case.id, evaluator="regex", passed=matched, detail=detail)


def json_schema_evaluator(case: EvalCase, output: str) -> Score:
    """Score `output` by parsing it as JSON and checking it against a named schema.

    Fails closed: invalid JSON, missing required keys, or a `status` value
    outside the allowed set all count as a failure with an explanatory
    detail rather than raising.

    Args:
        case: The eval case. `case.expected_property` must start with
            "json_schema:"; the rest of the string names the schema in
            the local `_SCHEMAS` registry.
        output: The candidate output, expected to be a JSON object.

    Raises:
        ValueError: If `case.expected_property` is not a "json_schema:"
            property, or names a schema this module does not know.
    """
    if case.expected_property is None or not case.expected_property.startswith("json_schema:"):
        raise ValueError(f"Case {case.id!r} has no json_schema expected_property to check against")
    schema_name = case.expected_property.removeprefix("json_schema:")
    if schema_name not in _SCHEMAS:
        raise ValueError(f"Unknown schema {schema_name!r}. Known schemas: {', '.join(_SCHEMAS)}")
    schema = _SCHEMAS[schema_name]

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError as exc:
        return Score(case.id, "json_schema", passed=False, detail=f"invalid JSON: {exc}")

    if not isinstance(parsed, dict):
        return Score(case.id, "json_schema", passed=False, detail="parsed JSON is not an object")

    missing = [key for key in schema["required"] if key not in parsed]
    if missing:
        return Score(case.id, "json_schema", passed=False, detail=f"missing required keys: {missing}")

    allowed_status = schema.get("allowed_status")
    if allowed_status is not None and parsed.get("status") not in allowed_status:
        return Score(
            case.id,
            "json_schema",
            passed=False,
            detail=f"status {parsed.get('status')!r} not in {sorted(allowed_status)}",
        )

    return Score(case.id, "json_schema", passed=True, detail="schema satisfied")
