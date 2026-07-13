"""LLM-Modulo: generate a plan, check it against sound verifiers, back-prompt, regenerate.

`validator.py` is single-shot and structural only: it confirms every tool
name is registered, every dependency resolves, and the graph is acyclic,
then either passes or raises. It cannot see that a structurally valid plan
books a hotel before pricing it, drafts an itinerary before checking the
weather, or blows a budget, because none of those are structural facts.
This module adds a second layer: a small suite of sound, deterministic
semantic verifiers, each a pure function over a `Plan` that returns `None`
or a concrete critique, and an iterative loop that feeds every failing
verifier's critique back to the planner as one back-prompt and asks for a
revision, capped at a fixed number of rounds. The two layers stay distinct
on purpose: `validate_plan` runs first and would accept a plan that books
before it prices, which is exactly the gap this module closes.

Because the verifiers are real Python, not another model call grading
itself, this is the most faithful-to-source module in the folder to run
under `MockProvider`: the paper's whole argument is that the verifier must
be external and sound, and a hand-rolled deterministic checker already is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry, get_provider

from patterns.planning.parser import parse_plan
from patterns.planning.plan import Plan, StepResult, is_error_observation, substitute_args
from patterns.planning.tools import hotel_rate_per_night
from patterns.planning.validator import validate_plan

PLANNER_SYSTEM = (
    "You are a trip-planning agent. Respond with ONLY a JSON array of steps: "
    "id, tool, args, depends_on."
)

BUDGET_CAP = 500

Verifier = Callable[[Plan], "str | None"]


def verify_budget_cap(plan: Plan, cap: int = BUDGET_CAP) -> str | None:
    """Reject a plan whose `book_hotel` steps would exceed a fixed budget cap.

    Prices every booking with the same rate table `tools.py`'s `book_hotel`
    tool uses, so this is a sound check computed before any tool runs.
    """
    total = 0
    for step in plan.steps:
        if step.tool != "book_hotel":
            continue
        city, nights = step.args.get("city"), step.args.get("nights")
        if isinstance(city, str) and isinstance(nights, (int, float)):
            total += hotel_rate_per_night(city) * int(nights)
    if total > cap:
        return f"Estimated hotel cost ${total} exceeds the ${cap} budget cap"
    return None


def verify_temporal_order(plan: Plan) -> str | None:
    """Reject a plan that drafts the itinerary without depending on a weather check."""
    weather_ids = {s.id for s in plan.steps if s.tool == "get_weather"}
    for step in plan.steps:
        if step.tool == "draft_itinerary" and not weather_ids & set(step.depends_on):
            return f"Step {step.id!r} drafts the itinerary without depending on a get_weather step"
    return None


def verify_hotel_precondition(plan: Plan) -> str | None:
    """Reject a plan that books a hotel without depending on a cost estimate first."""
    estimate_ids = {s.id for s in plan.steps if s.tool == "estimate_hotel_cost"}
    for step in plan.steps:
        if step.tool == "book_hotel" and not estimate_ids & set(step.depends_on):
            return f"Step {step.id!r} books a hotel without depending on an estimate_hotel_cost step"
    return None


def verify_no_duplicate_work(plan: Plan) -> str | None:
    """Reject a plan that calls the same tool with the same arguments more than once."""
    seen: dict[tuple, str] = {}
    for step in plan.steps:
        key = (step.tool, tuple(sorted(step.args.items())))
        if key in seen:
            return f"Step {step.id!r} duplicates step {seen[key]!r}: same tool and args"
        seen[key] = step.id
    return None


DEFAULT_VERIFIERS: tuple[Verifier, ...] = (
    verify_budget_cap,
    verify_temporal_order,
    verify_hotel_precondition,
    verify_no_duplicate_work,
)


def run_verifiers(plan: Plan, verifiers: tuple[Verifier, ...] = DEFAULT_VERIFIERS) -> list[str]:
    """Run every verifier against `plan` and return the critiques from any that fail, in order."""
    return [critique for verifier in verifiers if (critique := verifier(plan)) is not None]


@dataclass
class VerifiedPlanRun:
    """The outcome of a generate-verify-back-prompt loop.

    Attributes:
        plan: The last candidate plan produced, verified if `verified` is True.
        rounds: How many back-prompt rounds ran (0 means the first candidate
            passed every verifier).
        verifier_log: The critiques from each round, in round order; an
            empty list for a round means that candidate passed everything.
        verified: True if the final plan passed every verifier.
        results: Step results from executing the verified plan, or None if
            the plan was never verified or `execute` was False.
    """

    plan: Plan
    rounds: int
    verifier_log: list[list[str]] = field(default_factory=list)
    verified: bool = False
    results: list[StepResult] | None = None


def run_modulo_loop(
    provider: Provider,
    goal: str,
    registry: ToolRegistry,
    verifiers: tuple[Verifier, ...] = DEFAULT_VERIFIERS,
    max_rounds: int = 3,
    execute: bool = True,
) -> VerifiedPlanRun:
    """Generate a plan, verify it, and back-prompt on failure until it passes or the cap hits.

    Args:
        provider: Supplies the initial planner call and one revision call
            per back-prompt round.
        goal: The user's goal, sent to the planner and every revision call.
        registry: Tools available to the plan; also `validate_plan`'s
            allowlist for the structural layer that runs before verification.
        verifiers: The sound semantic checkers to run each round.
        max_rounds: Maximum number of back-prompt rounds before giving up.
        execute: If True and the plan is eventually verified, execute it
            with a plain sequential loop and attach the results.
    """
    plan_completion = provider.complete([Message.user(goal)], system=PLANNER_SYSTEM)
    plan = parse_plan(goal, plan_completion.content)
    validate_plan(plan, registry)  # structural layer: distinct from the semantic verifiers below

    verifier_log: list[list[str]] = []
    rounds = 0
    critiques = run_verifiers(plan, verifiers)
    verifier_log.append(critiques)

    while critiques:
        if rounds >= max_rounds:
            return VerifiedPlanRun(plan=plan, rounds=rounds, verifier_log=verifier_log, verified=False)
        rounds += 1
        back_prompt = "\n".join(f"- {c}" for c in critiques)
        revise_completion = provider.complete(
            [
                Message.user(
                    f"Goal: {goal}\nYour plan violates:\n{back_prompt}\n"
                    "Respond with ONLY a corrected JSON array of steps."
                )
            ],
            system=PLANNER_SYSTEM,
        )
        plan = parse_plan(goal, revise_completion.content)
        validate_plan(plan, registry)
        critiques = run_verifiers(plan, verifiers)
        verifier_log.append(critiques)

    results: list[StepResult] | None = None
    if execute:
        results_map: dict[str, StepResult] = {}
        results = []
        for step in plan.steps:
            args = substitute_args(step.args, results_map)
            output = registry.execute(ToolCall(id=step.id, name=step.tool, arguments=args))
            result = StepResult(step_id=step.id, output=output, ok=not is_error_observation(output))
            results_map[step.id] = result
            results.append(result)

    return VerifiedPlanRun(plan=plan, rounds=rounds, verifier_log=verifier_log, verified=True, results=results)


def demo() -> None:
    """Back-prompt a budget-violating hotel plan once, then execute the corrected version."""
    from patterns.planning.tools import build_travel_registry

    goal = "Book a 3-night Paris hotel and price it first, staying under budget."
    over_budget_json = (
        '[{"id": "e1", "tool": "estimate_hotel_cost", "args": {"city": "Paris", "nights": 3}, "depends_on": []},'
        ' {"id": "b1", "tool": "book_hotel", "args": {"city": "Paris", "nights": 3}, "depends_on": ["e1"]}]'
    )
    within_budget_json = (
        '[{"id": "e1", "tool": "estimate_hotel_cost", "args": {"city": "Lyon", "nights": 3}, "depends_on": []},'
        ' {"id": "b1", "tool": "book_hotel", "args": {"city": "Lyon", "nights": 3}, "depends_on": ["e1"]}]'
    )
    provider = get_provider(script=[over_budget_json, within_budget_json])
    registry = build_travel_registry()

    print("=== LLM-Modulo: verify against sound checkers, back-prompt, regenerate ===")
    print(f"Goal: {goal}")
    run = run_modulo_loop(provider, goal, registry)
    for i, critiques in enumerate(run.verifier_log):
        print(f"  round {i}: {'passed' if not critiques else '; '.join(critiques)}")
    if run.results:
        for result in run.results:
            print(f"  {result.step_id} -> {result.output}")
    print("Note: Paris ($630) broke the $500 cap; the back-prompt carried that critique and round 1's Lyon plan passed.")


if __name__ == "__main__":
    demo()
