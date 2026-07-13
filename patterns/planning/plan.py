"""Shared plan representation used by every planning variant in this pattern.

A `Step` is one unit of work: a tool to call, arguments for it (which may
reference an earlier step's output by id), and the ids of steps it depends
on. A `Plan` is the ordered collection of steps a planner produced for one
goal. A `StepResult` records what happened when a step ran, so downstream
steps and a final solver can use it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    """One node in a plan.

    Attributes:
        id: Unique step identifier within its plan, e.g. "step1" or "E1".
        tool: Name of the tool this step calls.
        args: Arguments for the tool call. A string value may contain a
            placeholder referencing an earlier step's output; see
            `substitute_args`.
        depends_on: Ids of steps that must complete before this one runs.
    """

    id: str
    tool: str
    args: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """An ordered set of steps produced by a planner for one goal."""

    goal: str
    steps: list[Step]

    def step_ids(self) -> set[str]:
        """Return the set of every step id in this plan."""
        return {s.id for s in self.steps}

    def get(self, step_id: str) -> Step:
        """Look up a step by id.

        Raises:
            KeyError: If no step with that id exists in this plan.
        """
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"Unknown step id {step_id!r}")


@dataclass
class StepResult:
    """The recorded outcome of running one step.

    Attributes:
        step_id: Id of the step this is the result of.
        output: The tool's return value, as a string (tool results in this
            repo are always strings; see `ToolRegistry.execute`).
        ok: False when the tool raised and `output` holds an "ERROR: ..."
            observation instead of a real result.
    """

    step_id: str
    output: str
    ok: bool = True


def is_error_observation(output: str) -> bool:
    """Return True if `output` is the "ERROR: ..." string a failed tool call produces.

    `ToolRegistry.execute` catches exceptions raised by a tool and returns
    `f"ERROR: {exc}"` instead of propagating them, so this string prefix is
    the only signal an executor has that a step failed rather than
    succeeded.

    Args:
        output: A tool call's raw string result.
    """
    return output.startswith("ERROR:")


def substitute_args(
    args: dict[str, Any], results: dict[str, StepResult], prefix: str = "$"
) -> dict[str, Any]:
    """Replace placeholders in string arguments with prior step outputs.

    Walks `args` recursively through nested dicts and lists. Any string
    value containing `f"{prefix}{step_id}"` for a step already in `results`
    has that placeholder replaced with the step's output text. Placeholders
    are matched on exact boundaries: longer ids are tried before their
    prefixes (so "$step10" is not partially consumed by a "$step1"
    replacement) and a match is only accepted when it is not immediately
    followed by another id character, so "$step1" inside "$step10" never
    matches on its own.

    Args:
        args: A step's raw arguments, possibly containing placeholders.
        results: Completed steps' results, keyed by step id.
        prefix: The placeholder marker. Most variants use "$" (e.g.
            "$step1"); ReWOO's own notation is "#" (e.g. "#E1").
    """
    if not results:
        pattern = None
    else:
        # Longest id first so "$step10" wins over "$step1" at the same
        # position; the alternation tries branches in listed order.
        ids_by_length = sorted(results, key=len, reverse=True)
        alternation = "|".join(re.escape(f"{prefix}{step_id}") for step_id in ids_by_length)
        pattern = re.compile(f"(?:{alternation})(?![A-Za-z0-9_])")

    def replace(match: re.Match[str]) -> str:
        step_id = match.group(0)[len(prefix) :]
        return results[step_id].output

    def sub_value(value: Any) -> Any:
        if isinstance(value, str):
            return value if pattern is None else pattern.sub(replace, value)
        if isinstance(value, dict):
            return {k: sub_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [sub_value(v) for v in value]
        return value

    return {k: sub_value(v) for k, v in args.items()}


def topological_waves(steps: list[Step]) -> list[list[Step]]:
    """Group steps into dependency-ordered waves.

    Every step in a wave has all of its dependencies satisfied by steps in
    earlier waves, so an executor may run all steps of one wave concurrently.

    Returns:
        A list of waves. If the input has a dependency cycle or a dangling
        reference to a step id that does not exist, the steps involved never
        become ready and are silently omitted; callers that need to detect
        that compare `sum(len(w) for w in waves)` against `len(steps)`.
    """
    remaining = {s.id: s for s in steps}
    done: set[str] = set()
    waves: list[list[Step]] = []
    while remaining:
        ready = [s for s in remaining.values() if all(d in done for d in s.depends_on)]
        if not ready:
            break
        waves.append(ready)
        for s in ready:
            done.add(s.id)
            del remaining[s.id]
    return waves
