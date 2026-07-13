"""ReWOO: decoupled planner, workers, and solver.

The planner writes the whole blueprint of tool calls up front, using "#E1",
"#E2", ... placeholders for evidence it has not gathered yet. Workers then
fetch each piece of evidence by running the referenced tool, with no model
call involved. A solver reads only the recorded evidence, not the planner's
reasoning, and composes the final answer. Because the planner commits to the
full blueprint in one call and the solver only reads evidence, this needs
exactly two model calls no matter how many tools the blueprint uses, unlike
ReAct's one call per step.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry, get_provider

from patterns.planning.parser import parse_plan
from patterns.planning.plan import Plan, StepResult, is_error_observation, substitute_args
from patterns.planning.validator import validate_plan

PLANNER_SYSTEM = (
    "You are a trip-planning agent using the ReWOO style. Respond with ONLY "
    "a JSON array of steps: id (use E1, E2, ...), tool, args, depends_on. "
    "Args may reference an earlier step's evidence with the placeholder #Eid."
)

SOLVER_SYSTEM = (
    "You are the ReWOO solver. You are given the goal and a list of evidence "
    "variables with their values, gathered by workers whose planning you did "
    "not see. Compose the final answer from the evidence alone."
)


@dataclass
class RewooRun:
    """The outcome of a ReWOO planner-worker-solver run.

    Attributes:
        plan: The evidence-gathering blueprint the planner produced.
        evidence: One `StepResult` per worker call, in blueprint order.
        final_answer: The solver's synthesis of the evidence.
        model_calls: Total planner + solver calls (always 2 for this variant).
    """

    plan: Plan
    evidence: list[StepResult]
    final_answer: str
    model_calls: int


def run_rewoo(provider: Provider, goal: str, registry: ToolRegistry) -> RewooRun:
    """Run the planner once, gather all evidence with no further model calls, then solve once.

    Args:
        provider: Supplies exactly two calls: the blueprint and the solve.
        goal: The user's goal, sent to both the planner and the solver.
        registry: Tools available to the blueprint; also the validator's allowlist.
    """
    blueprint_completion = provider.complete([Message.user(goal)], system=PLANNER_SYSTEM)
    plan = parse_plan(goal, blueprint_completion.content)
    validate_plan(plan, registry)

    evidence: dict[str, StepResult] = {}
    ordered: list[StepResult] = []
    for step in plan.steps:  # workers: pure tool execution, no model calls
        args = substitute_args(step.args, evidence, prefix="#")
        output = registry.execute(ToolCall(id=step.id, name=step.tool, arguments=args))
        result = StepResult(step_id=step.id, output=output, ok=not is_error_observation(output))
        evidence[step.id] = result
        ordered.append(result)

    evidence_block = "\n".join(f"{r.step_id} = {r.output}" for r in ordered)
    solve_completion = provider.complete(
        [Message.user(f"Goal: {goal}\nEvidence:\n{evidence_block}")], system=SOLVER_SYSTEM
    )
    return RewooRun(plan=plan, evidence=ordered, final_answer=solve_completion.content, model_calls=2)


def demo() -> None:
    """Run ReWOO on a three-tool trip goal and print the blueprint, evidence, and answer."""
    from patterns.planning.tools import build_travel_registry

    goal = "For a trip to Lisbon, gather the weather, the attractions, and a 4-night hotel estimate."
    blueprint_json = (
        '[{"id": "E1", "tool": "get_weather", "args": {"city": "Lisbon"}, "depends_on": []},'
        ' {"id": "E2", "tool": "search_attractions", "args": {"city": "Lisbon"}, "depends_on": []},'
        ' {"id": "E3", "tool": "estimate_hotel_cost", "args": {"city": "Lisbon", "nights": 4}, "depends_on": []}]'
    )
    final_answer = (
        "Lisbon will be sunny and warm with no rain, so plan full days at Belem "
        "Tower and Alfama, and a hotel budget of about $560 for 4 nights."
    )
    provider = get_provider(script=[blueprint_json, final_answer])
    registry = build_travel_registry()

    print("=== ReWOO (planner, workers, solver) ===")
    print(f"Goal: {goal}\n")
    run = run_rewoo(provider, goal, registry)
    print("Blueprint:")
    for step in run.plan.steps:
        print(f"  {step.id}: {step.tool}({step.args})")
    print("\nEvidence gathered by workers (no model calls):")
    for result in run.evidence:
        print(f"  {result.step_id} = {result.output}")
    print(f"\nSolver's final answer: {run.final_answer}")
    print(f"\nTotal model calls: {run.model_calls} (independent of the {len(run.plan.steps)} tool calls)")


if __name__ == "__main__":
    demo()
