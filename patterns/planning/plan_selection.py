"""Plan selection: generate several candidate plans, score them, execute only the winner.

Every other variant in this folder commits to the first plan its planner
call produces. This module generates k candidates, drops any that fail
structural validation (an infeasible-plan filter), scores each surviving
candidate with a critic call, and executes only the highest scorer. This is
search over whole plans before any execution, which is a different search
than `patterns/react/tree_search.py`'s search over actions during a live
rollout against real observations, and different again from
`patterns/react/self_consistency.py`'s vote over completed rollouts' final
answers. Here nothing executes until a plan is chosen, so a losing candidate
never has a side effect: its tool calls simply never happen.

A pairwise tournament variant is included alongside the score-and-argmax
default: instead of a rubric number per candidate, each comparison is its
own scripted verdict between two plans, and the tournament winner can differ
from whichever candidate would have scored highest on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry, get_provider
from patterns.planning.parser import PlanParseError, parse_plan
from patterns.planning.plan import Plan, StepResult, is_error_observation, substitute_args
from patterns.planning.validator import PlanValidationError, validate_plan

PLANNER_SYSTEM = (
    "You are a trip-planning agent proposing one candidate plan among several. "
    "Respond with ONLY a JSON array of steps: id, tool, args, depends_on."
)

CRITIC_SYSTEM = (
    "You are a plan critic. Given the goal and a candidate plan, respond with "
    "a single integer from 0 to 10 rating how well it achieves the goal. "
    "Reply with the number only."
)

TOURNAMENT_SYSTEM = (
    "You are a plan critic judging a head-to-head matchup. Given the goal and "
    "two candidate plans, reply with ONLY the winning candidate's number, 1 or 2."
)


@dataclass
class Candidate:
    """One proposed plan and what became of it.

    Attributes:
        index: Position among the k candidates requested, 0-based.
        plan: The parsed, structurally valid plan, or None if it was dropped.
        error: The parse or validation error that dropped it, if any.
        score: The critic's rubric score, set only by the score-based selector.
    """

    index: int
    plan: Plan | None
    error: str | None = None
    score: float | None = None


@dataclass
class SelectionRun:
    """The outcome of a generate-score-select-execute run.

    Attributes:
        candidates: Every proposed candidate, including dropped ones.
        chosen: The candidate that was selected and executed.
        reason: A short explanation of why `chosen` won.
        results: Step results from executing only `chosen.plan`.
    """

    candidates: list[Candidate]
    chosen: Candidate
    reason: str
    results: list[StepResult]


def _render_plan(plan: Plan) -> str:
    return "\n".join(f"{s.id}: {s.tool}({s.args})" for s in plan.steps)


def _propose_candidates(provider: Provider, goal: str, registry: ToolRegistry, k: int) -> list[Candidate]:
    """Ask for k candidate plans, one call each, dropping any that fail structural validation."""
    candidates: list[Candidate] = []
    for i in range(k):
        completion = provider.complete([Message.user(goal)], system=PLANNER_SYSTEM)
        try:
            plan = parse_plan(goal, completion.content)
            validate_plan(plan, registry)
        except (PlanParseError, PlanValidationError) as exc:
            candidates.append(Candidate(index=i, plan=None, error=str(exc)))
        else:
            candidates.append(Candidate(index=i, plan=plan))
    return candidates


def _execute_plan(plan: Plan, registry: ToolRegistry) -> list[StepResult]:
    """Run every step of `plan` in list order; no side effect happens for any other candidate."""
    results_map: dict[str, StepResult] = {}
    ordered: list[StepResult] = []
    for step in plan.steps:
        args = substitute_args(step.args, results_map)
        output = registry.execute(ToolCall(id=step.id, name=step.tool, arguments=args))
        result = StepResult(step_id=step.id, output=output, ok=not is_error_observation(output))
        results_map[step.id] = result
        ordered.append(result)
    return ordered


def run_plan_selection(provider: Provider, goal: str, registry: ToolRegistry, k: int = 3) -> SelectionRun:
    """Generate k candidates, score each survivor with a critic call, execute only the best.

    Args:
        provider: Supplies k proposal calls, one critic call per surviving
            candidate, and no calls for the losers beyond that.
        goal: The user's goal, sent to every proposal and critic call.
        registry: Tools available to a plan; also the structural filter's allowlist.
        k: Number of candidate plans to propose.

    Raises:
        ValueError: If every candidate fails structural validation.
    """
    candidates = _propose_candidates(provider, goal, registry, k)
    survivors = [c for c in candidates if c.plan is not None]
    if not survivors:
        raise ValueError("No candidate plan survived structural validation")

    for candidate in survivors:
        assert candidate.plan is not None
        prompt = f"Goal: {goal}\nCandidate plan:\n{_render_plan(candidate.plan)}"
        completion = provider.complete([Message.user(prompt)], system=CRITIC_SYSTEM)
        candidate.score = float(completion.content.strip())

    chosen = survivors[0]
    for candidate in survivors[1:]:
        if candidate.score > chosen.score:  # type: ignore[operator]
            chosen = candidate

    reason = f"candidate {chosen.index} scored {chosen.score:g}, the highest among {len(survivors)} valid candidate(s)"
    assert chosen.plan is not None
    results = _execute_plan(chosen.plan, registry)
    return SelectionRun(candidates=candidates, chosen=chosen, reason=reason, results=results)


def run_plan_selection_tournament(provider: Provider, goal: str, registry: ToolRegistry, k: int = 3) -> SelectionRun:
    """Generate k candidates, run a pairwise tournament instead of scoring, execute the winner.

    Args:
        provider: Supplies k proposal calls and one tournament-verdict call
            per challenger the running champion faces.
        goal: The user's goal, sent to every proposal and verdict call.
        registry: Tools available to a plan; also the structural filter's allowlist.
        k: Number of candidate plans to propose.

    Raises:
        ValueError: If every candidate fails structural validation.
    """
    candidates = _propose_candidates(provider, goal, registry, k)
    survivors = [c for c in candidates if c.plan is not None]
    if not survivors:
        raise ValueError("No candidate plan survived structural validation")

    champion = survivors[0]
    for challenger in survivors[1:]:
        assert champion.plan is not None and challenger.plan is not None
        prompt = (
            f"Goal: {goal}\nCandidate 1:\n{_render_plan(champion.plan)}\n"
            f"Candidate 2:\n{_render_plan(challenger.plan)}"
        )
        completion = provider.complete([Message.user(prompt)], system=TOURNAMENT_SYSTEM)
        if completion.content.strip() == "2":
            champion = challenger

    reason = f"candidate {champion.index} won the pairwise tournament"
    assert champion.plan is not None
    results = _execute_plan(champion.plan, registry)
    return SelectionRun(candidates=candidates, chosen=champion, reason=reason, results=results)


def demo() -> None:
    """Propose three Lisbon plans, score them, and execute only the winner."""
    from patterns.planning.tools import build_travel_registry

    goal = "Plan a 2-night Lisbon trip within budget."
    cheap = '[{"id": "e1", "tool": "estimate_hotel_cost", "args": {"city": "Lisbon", "nights": 2}, "depends_on": []}]'
    unknown_tool = '[{"id": "b1", "tool": "levitate", "args": {}, "depends_on": []}]'
    lavish = '[{"id": "e2", "tool": "estimate_hotel_cost", "args": {"city": "Paris", "nights": 2}, "depends_on": []}]'
    provider = get_provider(script=[cheap, unknown_tool, lavish, "9", "2"])
    registry = build_travel_registry()

    print("=== Plan selection (generate, score, execute only the winner) ===")
    print(f"Goal: {goal}")
    run = run_plan_selection(provider, goal, registry, k=3)
    for candidate in run.candidates:
        status = f"score {candidate.score:g}" if candidate.plan else "dropped: unknown tool"
        print(f"  candidate {candidate.index}: {status}")
    print(f"Chosen: {run.reason}")
    for result in run.results:
        print(f"  {result.step_id} -> {result.output}")
    print("Note: candidate 1 was dropped before scoring; only candidate 0's tool call ran.")


if __name__ == "__main__":
    demo()
