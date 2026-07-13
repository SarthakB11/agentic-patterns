"""Localized plan repair: fix a step's blast radius, not the whole remaining plan.

`replanning.py` is the replan-from-scratch baseline this module upgrades: on
any failure it drops the entire flat list of steps still queued, whether or
not a given queued step ever depended on the failure, so an unrelated branch
listed after the failing step gets swept up and regenerated too. This module
treats the plan as a dependency graph instead. When a step fails, it walks
forward through `depends_on` edges and through `$id` placeholders inside a
step's `args` (a step can consume an earlier output without formally
declaring the dependency) to compute the failing step's blast radius: itself
plus every step that would need its output. Everything outside that radius
keeps its already-recorded `StepResult`, identity and all, and never runs
again. Only the blast radius goes to a repairer for a scoped replacement,
spliced back in, and re-executed. See this module's tests for a direct,
counted comparison against `replanning.py` on the same failure.

The blast-radius walk and the preserved/affected partition are pure
functions over the plan, so this is deterministic end to end under
`MockProvider`: graph arithmetic plus one scripted repair call per round.
Like every wave-based executor in this folder, `validate_plan` runs before
`topological_waves` on every candidate plan, including each repaired splice,
per the precondition `plan.topological_waves` documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry, get_provider

from patterns.planning.parser import parse_plan
from patterns.planning.plan import Plan, Step, StepResult, is_error_observation, substitute_args, topological_waves
from patterns.planning.validator import validate_plan

PLANNER_SYSTEM = (
    "You are a trip-planning agent. Respond with ONLY a JSON array of steps: "
    "id, tool, args, depends_on."
)

REPAIRER_SYSTEM = (
    "The plan hit a problem in a bounded region. You are given the goal, the "
    "results of steps outside that region (already fine, do not touch), and "
    "what went wrong. Respond with ONLY a JSON array of replacement steps for "
    "EXACTLY the step ids named as needing repair. Reuse those same ids so "
    "any step outside the region that references them still resolves."
)


class RepairBudgetExceeded(RuntimeError):
    """Raised when the repair cap is hit and a step in the blast radius still fails."""


class RepairScopeError(ValueError):
    """Raised when a repairer's response does not cover exactly the affected step ids."""


@dataclass
class RepairRun:
    """The outcome of a plan-then-execute run that may have needed localized repair.

    Attributes:
        plan: The plan in effect when the run finished: the original plan
            spliced with every repaired blast radius, if any.
        results: Every step's result, keyed by step id.
        repairs: How many repair rounds were run.
        preserved_ids: Step ids recorded once and never touched again.
        repaired_ids: Step ids ever part of a blast radius, cleared and
            re-executed at least once.
    """

    plan: Plan
    results: dict[str, StepResult]
    repairs: int
    preserved_ids: set[str] = field(default_factory=set)
    repaired_ids: set[str] = field(default_factory=set)


def _referenced_step_ids(args: dict, known_ids: set[str], prefix: str = "$") -> set[str]:
    """Return the ids referenced by placeholders inside `args`, from `known_ids`."""
    if not known_ids:
        return set()
    ids_by_length = sorted(known_ids, key=len, reverse=True)
    alternation = "|".join(re.escape(f"{prefix}{i}") for i in ids_by_length)
    pattern = re.compile(f"(?:{alternation})(?![A-Za-z0-9_])")
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, str):
            for match in pattern.finditer(value):
                found.add(match.group(0)[len(prefix) :])
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(args)
    return found


def _direct_dependents(steps: list[Step], target_id: str) -> set[str]:
    """Return the ids of steps that consume `target_id`'s output, declared or not."""
    known_ids = {s.id for s in steps}
    direct: set[str] = set()
    for step in steps:
        if target_id in step.depends_on or target_id in _referenced_step_ids(step.args, known_ids):
            direct.add(step.id)
    return direct


def compute_blast_radius(plan: Plan, failing_id: str) -> set[str]:
    """Return `failing_id` plus every step transitively dependent on it.

    Walks forward through `depends_on` edges and `$id` argument references
    until no further dependent is found. A step outside the returned set has
    no path, declared or implicit, back to `failing_id`.
    """
    radius = {failing_id}
    frontier = [failing_id]
    while frontier:
        current = frontier.pop()
        for dependent in _direct_dependents(plan.steps, current):
            if dependent not in radius:
                radius.add(dependent)
                frontier.append(dependent)
    return radius


def _run_pass(plan: Plan, registry: ToolRegistry, results: dict[str, StepResult]) -> list[str]:
    """Execute every step not yet in `results`, wave by wave.

    Stops at the first wave containing a failure rather than continuing into
    later waves whose steps may depend on the failure. Steps outside that
    wave keep whatever result they already had, whether from an earlier pass
    or this one.

    Returns:
        The ids that failed in the stopping wave, or an empty list if every
        step ran (or was already recorded) successfully.
    """
    for wave in topological_waves(plan.steps):
        pending = [s for s in wave if s.id not in results]
        if not pending:
            continue
        newly_failed: list[str] = []
        for step in pending:
            args = substitute_args(step.args, results)
            output = registry.execute(ToolCall(id=step.id, name=step.tool, arguments=args))
            result = StepResult(step_id=step.id, output=output, ok=not is_error_observation(output))
            results[step.id] = result
            if not result.ok:
                newly_failed.append(step.id)
        if newly_failed:
            return newly_failed
    return []


def run_plan_repair(provider: Provider, goal: str, registry: ToolRegistry, max_repairs: int = 2) -> RepairRun:
    """Execute a plan, repairing only a failing step's blast radius on failure.

    Args:
        provider: Supplies the planner call and one repairer call per round.
        goal: The user's goal, sent to the planner and every repair call.
        registry: Tools available to the plan; also the validator's allowlist.
        max_repairs: Maximum number of repair rounds before giving up.

    Raises:
        RepairBudgetExceeded: If a step is still failing after `max_repairs` rounds.
        RepairScopeError: If a repairer's response covers different step ids
            than the blast radius it was asked to fix.
    """
    plan_completion = provider.complete([Message.user(goal)], system=PLANNER_SYSTEM)
    current_plan = parse_plan(goal, plan_completion.content)
    validate_plan(current_plan, registry)

    results: dict[str, StepResult] = {}
    repaired_ids: set[str] = set()
    repairs = 0
    while True:
        failed_ids = _run_pass(current_plan, registry, results)
        if not failed_ids:
            break
        if repairs >= max_repairs:
            raise RepairBudgetExceeded(
                f"Step(s) {', '.join(sorted(failed_ids))} still failing after {repairs} repair(s)"
            )
        repairs += 1
        blast_radius: set[str] = set()
        for failed_id in failed_ids:
            blast_radius |= compute_blast_radius(current_plan, failed_id)
        repaired_ids |= blast_radius

        preserved_steps = [s for s in current_plan.steps if s.id not in blast_radius]
        preserved_summary = (
            "\n".join(f"- {sid}: {results[sid].output}" for sid in sorted(results) if sid not in blast_radius)
            or "(none)"
        )
        failure_summary = "\n".join(f"- {fid}: {results[fid].output}" for fid in sorted(failed_ids))
        prompt = (
            f"Goal: {goal}\nPreserved results (do not touch):\n{preserved_summary}\n"
            f"Failure:\n{failure_summary}\nRepair exactly these step ids: {', '.join(sorted(blast_radius))}"
        )
        repair_completion = provider.complete([Message.user(prompt)], system=REPAIRER_SYSTEM)
        repair_plan = parse_plan(goal, repair_completion.content)
        if repair_plan.step_ids() != blast_radius:
            raise RepairScopeError(f"Repairer must cover exactly {sorted(blast_radius)}, got {sorted(repair_plan.step_ids())}")

        spliced_plan = Plan(goal=goal, steps=preserved_steps + repair_plan.steps)
        validate_plan(spliced_plan, registry)
        current_plan = spliced_plan
        for step_id in blast_radius:
            results.pop(step_id, None)

    preserved_ids = current_plan.step_ids() - repaired_ids
    return RepairRun(plan=current_plan, results=results, repairs=repairs, preserved_ids=preserved_ids, repaired_ids=repaired_ids)


def demo() -> None:
    """Repair only a failed hotel booking and its downstream itinerary, leaving weather untouched."""
    from patterns.planning.tools import build_travel_registry

    goal = "Plan a Paris trip: check weather, book a 2-night hotel, then draft an itinerary mentioning the booking."
    plan_json = (
        '[{"id": "A", "tool": "get_weather", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "B", "tool": "book_hotel", "args": {"city": "Paris", "nights": 2}, "depends_on": []},'
        ' {"id": "C", "tool": "draft_itinerary",'
        '  "args": {"weather": "$A", "attractions": "Hotel booked: $B"}, "depends_on": ["B"]}]'
    )
    repair_json = (
        '[{"id": "B", "tool": "book_hotel", "args": {"city": "Lyon", "nights": 2}, "depends_on": []},'
        ' {"id": "C", "tool": "draft_itinerary",'
        '  "args": {"weather": "$A", "attractions": "Hotel booked: $B"}, "depends_on": ["B"]}]'
    )
    provider = get_provider(script=[plan_json, repair_json])
    registry = build_travel_registry()

    print("=== Localized plan repair (blast radius, not replan-from-scratch) ===")
    print(f"Goal: {goal}")
    run = run_plan_repair(provider, goal, registry)
    print(f"Repairs: {run.repairs}, preserved: {sorted(run.preserved_ids)}, repaired: {sorted(run.repaired_ids)}")
    for step_id in ("A", "B", "C"):
        print(f"  {step_id} -> {run.results[step_id].output}")
    print("Note: A never re-ran; only B (the failure) and C (its dependent) were repaired.")


if __name__ == "__main__":
    demo()
