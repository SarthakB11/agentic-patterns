"""Classic plan-then-execute: one planner call, then a separate executor.

A dedicated planner call produces the full, ordered list of steps up front.
A separate executor runs them one by one, substituting any earlier step's
output into later steps' arguments, then a final call synthesizes the
recorded outputs into an answer. This is the LangChain / LangGraph
plan-and-execute shape: planning and execution are clearly separated phases,
and the planner and executor could use different models even though this
demo uses one provider for both.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry, get_provider
from patterns.planning.parser import parse_plan
from patterns.planning.plan import Plan, StepResult, is_error_observation, substitute_args
from patterns.planning.validator import validate_plan

PLANNER_SYSTEM = (
    "You are a trip-planning agent. Respond with ONLY a JSON array of steps. "
    "Each step has: id, tool, args, depends_on (a list of step ids). "
    "Reference an earlier step's output in an arg with the placeholder $step_id."
)

SOLVER_SYSTEM = "Synthesize the step outputs into one direct final answer for the goal."


@dataclass
class SequentialRun:
    """The outcome of a full plan-then-execute run.

    Attributes:
        plan: The validated plan that was executed.
        results: One `StepResult` per step, in execution order.
        final_answer: The solver's synthesis of all step outputs.
    """

    plan: Plan
    results: list[StepResult]
    final_answer: str


def run_sequential(provider: Provider, goal: str, registry: ToolRegistry) -> SequentialRun:
    """Plan once, then execute every step of the resulting list in order.

    Args:
        provider: Supplies both the planner and the final solver call.
        goal: The user's goal, sent verbatim to the planner.
        registry: Tools available to the plan; also the validator's allowlist.
    """
    plan_completion = provider.complete([Message.user(goal)], system=PLANNER_SYSTEM)
    plan = parse_plan(goal, plan_completion.content)
    validate_plan(plan, registry)

    results: dict[str, StepResult] = {}
    ordered: list[StepResult] = []
    for step in plan.steps:
        args = substitute_args(step.args, results)
        output = registry.execute(ToolCall(id=step.id, name=step.tool, arguments=args))
        result = StepResult(step_id=step.id, output=output, ok=not is_error_observation(output))
        results[step.id] = result
        ordered.append(result)

    summary = "\n".join(f"- {r.step_id}: {r.output}" for r in ordered)
    solve_completion = provider.complete(
        [Message.user(f"Goal: {goal}\nStep outputs:\n{summary}")],
        system=SOLVER_SYSTEM,
    )
    return SequentialRun(plan=plan, results=ordered, final_answer=solve_completion.content)


def demo() -> None:
    """Run the classic plan-then-execute flow on a two-step trip goal and print it."""
    from patterns.planning.tools import build_travel_registry

    goal = "Plan a 2-day trip to Lisbon: check the weather, list attractions, then draft an itinerary."
    plan_json = (
        '[{"id": "step1", "tool": "get_weather", "args": {"city": "Lisbon"}, "depends_on": []},'
        ' {"id": "step2", "tool": "search_attractions", "args": {"city": "Lisbon"}, "depends_on": []},'
        ' {"id": "step3", "tool": "draft_itinerary",'
        '  "args": {"weather": "$step1", "attractions": "$step2"}, "depends_on": ["step1", "step2"]}]'
    )
    final_answer = (
        "Lisbon will be sunny and warm (26C, no rain), so plan outdoor time at "
        "Belem Tower and the Alfama district, with an evening stop at Time Out "
        "Market for dinner."
    )
    provider = get_provider(script=[plan_json, final_answer])
    registry = build_travel_registry()

    print("=== Classic plan-then-execute (sequential) ===")
    print(f"Goal: {goal}")
    run = run_sequential(provider, goal, registry)
    print(f"Planner produced {len(run.plan.steps)} steps:")
    for step in run.plan.steps:
        print(f"  {step.id}: {step.tool}({step.args}) depends_on={step.depends_on}")
    print("Execution:")
    for result in run.results:
        print(f"  {result.step_id} -> {result.output}")
    print(f"Final answer: {run.final_answer}")


if __name__ == "__main__":
    demo()
