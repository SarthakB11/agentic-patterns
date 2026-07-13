"""DAG executor: runs a dependency-ordered plan wave by wave, dispatching
every step in a wave concurrently.

Unlike the sequential executor, this treats the plan as a graph rather than
a list: a step whose dependencies are all satisfied can run as soon as its
wave starts, at the same time as any other step in that wave. This is the
shape behind LLMCompiler's task-fetching unit, which dispatches ready tasks
and substitutes resolved variables into the ones still waiting.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry, get_provider

from patterns.planning.parser import parse_plan
from patterns.planning.plan import (
    Plan,
    Step,
    StepResult,
    is_error_observation,
    substitute_args,
    topological_waves,
)
from patterns.planning.validator import validate_plan

PLANNER_SYSTEM = (
    "You are a trip-planning agent. Respond with ONLY a JSON array of steps: "
    "id, tool, args, depends_on. Give independent steps an empty depends_on "
    "list so they can run in parallel. Reference an earlier step's output "
    "in an arg with the placeholder $step_id."
)


@dataclass
class DagRun:
    """The outcome of a DAG-ordered, wave-by-wave execution.

    Attributes:
        plan: The validated plan that was executed.
        waves: Step ids grouped by the wave they ran in, in run order.
        results: Every step's result, keyed by step id.
    """

    plan: Plan
    waves: list[list[str]]
    results: dict[str, StepResult]


def _run_step(step: Step, registry: ToolRegistry, results: dict[str, StepResult]) -> StepResult:
    """Substitute upstream outputs into `step.args` and run its tool."""
    args = substitute_args(step.args, results)
    output = registry.execute(ToolCall(id=step.id, name=step.tool, arguments=args))
    return StepResult(step_id=step.id, output=output, ok=not is_error_observation(output))


def run_dag(provider: Provider, goal: str, registry: ToolRegistry, max_workers: int = 4) -> DagRun:
    """Plan once, then execute the resulting DAG wave by wave, in parallel within a wave.

    Args:
        provider: Supplies the single planner call.
        goal: The user's goal, sent verbatim to the planner.
        registry: Tools available to the plan; also the validator's allowlist.
        max_workers: Thread pool size for concurrent step dispatch.
    """
    plan_completion = provider.complete([Message.user(goal)], system=PLANNER_SYSTEM)
    plan = parse_plan(goal, plan_completion.content)
    validate_plan(plan, registry)

    waves = topological_waves(plan.steps)
    results: dict[str, StepResult] = {}
    wave_ids: list[list[str]] = []
    for wave in waves:
        wave_ids.append([s.id for s in wave])
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {s.id: pool.submit(_run_step, s, registry, results) for s in wave}
            for step_id, future in futures.items():
                results[step_id] = future.result()
    return DagRun(plan=plan, waves=wave_ids, results=results)


def demo() -> None:
    """Run the DAG executor on a plan with two independent branches feeding a third step."""
    from patterns.planning.tools import build_travel_registry

    goal = (
        "Plan a trip to Paris: check weather, list attractions, estimate hotel "
        "cost for 3 nights, then draft an itinerary from the weather and attractions."
    )
    plan_json = (
        '[{"id": "weather", "tool": "get_weather", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "attractions", "tool": "search_attractions", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "hotel", "tool": "estimate_hotel_cost", "args": {"city": "Paris", "nights": 3}, "depends_on": []},'
        ' {"id": "itinerary", "tool": "draft_itinerary",'
        '  "args": {"weather": "$weather", "attractions": "$attractions"},'
        '  "depends_on": ["weather", "attractions"]}]'
    )
    provider = get_provider(script=[plan_json])
    registry = build_travel_registry()

    print("=== DAG executor (parallel dispatch within each wave) ===")
    print(f"Goal: {goal}\n")
    run = run_dag(provider, goal, registry)
    for i, wave in enumerate(run.waves, start=1):
        print(f"Wave {i} (dispatched together): {wave}")
    print("\nResults:")
    for step_id, result in run.results.items():
        print(f"  {step_id} -> {result.output}")
    print(
        "\nNote: 'weather', 'attractions', and 'hotel' have no dependencies and "
        "all ran in wave 1 on their own threads; 'itinerary' waited for both of "
        "its dependencies before wave 2 started."
    )


if __name__ == "__main__":
    demo()
