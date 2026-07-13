"""Shared plan representation used by every planning variant in this pattern.

A `Step` is one unit of work: a tool to call, arguments for it (which may
reference an earlier step's output by id), and the ids of steps it depends
on. A `Plan` is the ordered collection of steps a planner produced for one
goal. A `StepResult` records what happened when a step ran, so downstream
steps and a final solver can use it.
"""

from __future__ import annotations

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


def substitute_args(
    args: dict[str, Any], results: dict[str, StepResult], prefix: str = "$"
) -> dict[str, Any]:
    """Replace placeholders in string arguments with prior step outputs.

    Walks `args` recursively through nested dicts and lists. Any string
    value containing `f"{prefix}{step_id}"` for a step already in `results`
    has that placeholder replaced with the step's output text.

    Args:
        args: A step's raw arguments, possibly containing placeholders.
        results: Completed steps' results, keyed by step id.
        prefix: The placeholder marker. Most variants use "$" (e.g.
            "$step1"); ReWOO's own notation is "#" (e.g. "#E1").
    """

    def sub_value(value: Any) -> Any:
        if isinstance(value, str):
            out = value
            for step_id, result in results.items():
                out = out.replace(f"{prefix}{step_id}", result.output)
            return out
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
