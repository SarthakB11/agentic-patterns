"""Plan-then-execute with replanning: revise the remaining steps when a step
fails or an observation invalidates the rest of the plan.

Two distinct triggers are supported, because they come from different
places in practice: a step can raise (a tool call fails, e.g. a sold-out
hotel) or a step can succeed but return an observation that makes the rest
of the plan a bad idea (Plan-and-Act's motivation for regenerating the plan
after every executor step, not only after outright failures). Either way the
replanner sees the goal, what has completed so far, and the problem, then
returns a JSON plan for the remaining work. Replanning is capped: after
`max_replans` revisions the run stops and raises rather than looping forever
against a step that will never succeed.

This is the replan-from-scratch baseline: `remaining = list(revision.steps)`
below throws away every step still queued, in flat list order, whether or
not a given queued step ever depended on the failure. `plan_repair.py` is
the localized upgrade: instead of discarding the whole remaining list, it
computes the failing step's blast radius in the dependency graph and
repairs only that, leaving an independent queued step untouched. See that
module's tests for a direct, counted comparison on the same failure.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry, get_provider
from patterns.planning.parser import parse_plan
from patterns.planning.plan import Plan, Step, StepResult, is_error_observation, substitute_args
from patterns.planning.validator import validate_plan

PLANNER_SYSTEM = (
    "You are a trip-planning agent. Respond with ONLY a JSON array of steps: "
    "id, tool, args, depends_on."
)

REPLANNER_SYSTEM = (
    "The plan hit a problem partway through. You are given the goal, the "
    "steps already completed with their outputs, and what went wrong. "
    "Respond with ONLY a JSON array of the remaining steps, revised to work "
    "around the problem. Do not repeat steps that already completed."
)


class ReplanBudgetExceeded(RuntimeError):
    """Raised when the replan cap is hit and the plan still cannot finish."""


@dataclass
class ReplanRun:
    """The outcome of a plan-then-execute run that may have replanned.

    Attributes:
        plan: The plan actually in effect when the run finished (the
            original plan if no replan was needed, otherwise the last
            revision).
        results: One `StepResult` per step that ran, in execution order.
        replans: How many times the replanner was invoked.
    """

    plan: Plan
    results: list[StepResult]
    replans: int


def _invalidates(output: str) -> bool:
    """Flag an observation that invalidates the rest of the plan.

    A real system would run a model or rule check over the observation; this
    stands in with one explicit signal so the trigger is easy to follow and
    to test: a storm warning invalidates an outdoor-attraction itinerary.
    `premortem.py` is the principled version of this same signal: it predicts
    the observation and catches the storm in simulation, before the real
    tool call and its side effect, instead of matching a substring on a real
    observation after the fact.
    """
    return "storm warning" in output.lower()


def run_with_replanning(
    provider: Provider, goal: str, registry: ToolRegistry, max_replans: int = 2
) -> ReplanRun:
    """Execute a plan step by step, replanning on failure or invalidation.

    Args:
        provider: Supplies the planner, replanner, and (implicitly) no
            solver call; this variant returns raw step results.
        goal: The user's goal, sent to the planner and every replan call.
        registry: Tools available to the plan; also the validator's allowlist.
        max_replans: Maximum number of replanner invocations before giving up.

    Raises:
        ReplanBudgetExceeded: If a step still fails after `max_replans`
            revisions have been attempted.
    """
    plan_completion = provider.complete([Message.user(goal)], system=PLANNER_SYSTEM)
    plan = parse_plan(goal, plan_completion.content)
    validate_plan(plan, registry)
    current_plan = plan

    results: dict[str, StepResult] = {}
    ordered: list[StepResult] = []
    remaining: list[Step] = list(plan.steps)
    replans = 0

    while remaining:
        step = remaining.pop(0)
        args = substitute_args(step.args, results)
        output = registry.execute(ToolCall(id=step.id, name=step.tool, arguments=args))
        failed = is_error_observation(output)
        invalidated = not failed and _invalidates(output)

        if failed or invalidated:
            if replans >= max_replans:
                raise ReplanBudgetExceeded(
                    f"Step {step.id!r} still blocked after {replans} replan(s): {output}"
                )
            replans += 1
            if invalidated:
                result = StepResult(step_id=step.id, output=output, ok=not is_error_observation(output))
                results[step.id] = result
                ordered.append(result)
            completed_summary = "\n".join(f"- {r.step_id}: {r.output}" for r in ordered) or "(none yet)"
            reason = output if failed else f"observation invalidates the rest of the plan: {output}"
            replan_completion = provider.complete(
                [Message.user(f"Goal: {goal}\nCompleted:\n{completed_summary}\nProblem: {reason}")],
                system=REPLANNER_SYSTEM,
            )
            revision = parse_plan(goal, replan_completion.content)
            validate_plan(revision, registry)
            current_plan = revision
            remaining = list(revision.steps)
            continue

        result = StepResult(step_id=step.id, output=output, ok=not is_error_observation(output))
        results[step.id] = result
        ordered.append(result)

    return ReplanRun(plan=current_plan, results=ordered, replans=replans)


def demo() -> None:
    """Run replanning after a scripted hotel-booking failure and print the recovery."""
    from patterns.planning.tools import build_travel_registry

    goal = "Book a 2-night hotel stay in Paris and confirm the booking."
    plan_json = '[{"id": "step1", "tool": "book_hotel", "args": {"city": "Paris", "nights": 2}, "depends_on": []}]'
    revised_json = '[{"id": "step2", "tool": "book_hotel", "args": {"city": "Lyon", "nights": 2}, "depends_on": []}]'
    provider = get_provider(script=[plan_json, revised_json])
    registry = build_travel_registry()

    print("=== Plan-then-execute with replanning ===")
    print(f"Goal: {goal}")
    run = run_with_replanning(provider, goal, registry)
    print(f"Replans used: {run.replans}")
    for result in run.results:
        print(f"  {result.step_id} -> {result.output}")
    print(
        "Note: Paris had no rooms, so book_hotel raised, the executor caught it "
        "as an ERROR: observation, and the replanner substituted Lyon."
    )


if __name__ == "__main__":
    demo()
