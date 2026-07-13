"""Parses model-generated plan text into structured `Step` objects.

Planners in this pattern are prompted (and scripted) to emit a JSON array of
step objects. Parsing model output is a trust boundary: malformed JSON or a
step missing a required field raises `PlanParseError` immediately rather than
producing a half-built plan that fails confusingly later during execution.
"""

from __future__ import annotations

import json

from patterns.planning.plan import Plan, Step


class PlanParseError(ValueError):
    """Raised when a planner's output cannot be parsed into a `Plan`."""


def parse_plan(goal: str, raw_text: str) -> Plan:
    """Parse a planner's raw text output into a `Plan`.

    Args:
        goal: The goal this plan was generated for, stored on the `Plan`.
        raw_text: The planner completion's content, expected to be a JSON
            array of objects each with "id", "tool", "args", and optionally
            "depends_on".

    Returns:
        The parsed `Plan`, not yet validated against a tool registry.

    Raises:
        PlanParseError: If `raw_text` is not valid JSON, is not a JSON
            array, or any element is not an object with the required fields.
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"Planner output is not valid JSON: {exc}") from None

    if not isinstance(data, list):
        raise PlanParseError(
            f"Planner output must be a JSON array of steps, got {type(data).__name__}"
        )

    steps: list[Step] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise PlanParseError(f"Step {i} is not a JSON object: {item!r}")
        missing = [f for f in ("id", "tool", "args") if f not in item]
        if missing:
            raise PlanParseError(f"Step {i} is missing required field(s): {', '.join(missing)}")
        steps.append(
            Step(
                id=str(item["id"]),
                tool=str(item["tool"]),
                args=dict(item["args"]),
                depends_on=[str(d) for d in item.get("depends_on", [])],
            )
        )
    return Plan(goal=goal, steps=steps)
