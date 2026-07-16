"""Premortem: simulate a plan against a tracked world state before any real tool runs.

`modulo_loop.py`'s verifiers are sound but static: they judge the plan's
shape (which tool calls what, in what order) and cannot know what a tool
will actually return. `replanning.py`'s `_invalidates` check is the reactive
opposite: it looks at a real observation after the real, possibly
irreversible, side effect already happened. This module sits before both.
For each step in dependency order, it asks a world-model provider to predict
that step's observation given the state so far, folds the prediction into an
explicit tracked state, and checks a small set of deterministic constraints
against that predicted state, reusing the same $500 budget cap
`modulo_loop.verify_budget_cap` uses but applied to predicted text instead
of static plan args, since the input shapes differ. If a prediction dooms
the plan, the doomed step is reported and no real tool call is ever made. If
every step simulates clean, the plan executes for real with the same
sequential loop `sequential_executor.py` uses.

Like every wave-based executor in this folder, `validate_plan` runs before
`topological_waves` on the plan being simulated. Catching a doomed step
here and repairing it is `plan_repair.py`'s job: this module reuses
`plan_repair.compute_blast_radius` to scope which steps a caught failure
touches, and leaves the splice-and-repair call itself to the caller, since
that call is exactly `plan_repair.run_plan_repair` with a simulated trigger
instead of a real one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry, get_provider
from patterns.planning.parser import parse_plan
from patterns.planning.plan import Plan, Step, StepResult, is_error_observation, substitute_args, topological_waves
from patterns.planning.validator import validate_plan

PLANNER_SYSTEM = (
    "You are a trip-planning agent. Respond with ONLY a JSON array of steps: "
    "id, tool, args, depends_on."
)

WORLD_MODEL_SYSTEM = (
    "You are a world model. Given the state predicted so far and the next "
    "planned step, predict the observation that step would produce if it "
    "ran for real. Reply with the predicted observation text only."
)

BUDGET_CAP = 500


@dataclass
class SimulatedStep:
    """One step's predicted outcome during a premortem.

    Attributes:
        step_id: The step this prediction is for.
        predicted_observation: The world model's predicted observation text.
        violation: A constraint critique if this prediction dooms the plan,
            None if it looks fine.
    """

    step_id: str
    predicted_observation: str
    violation: str | None


@dataclass
class PremortemResult:
    """The outcome of simulating a plan, and executing it for real if the simulation was clean.

    Attributes:
        doomed: True if a predicted observation violated a constraint.
        doomed_step_id: The step the doom was caught at, None if not doomed.
        simulated_trajectory: Every simulated step, in dependency order.
        state: The predicted observation text, keyed by step id.
        executed: True if real execution ran (only when not doomed).
        real_results: Real step results, only when `executed` is True.
    """

    doomed: bool
    doomed_step_id: str | None
    simulated_trajectory: list[SimulatedStep]
    state: dict[str, str]
    executed: bool
    real_results: list[StepResult] | None


def _apply_constraints(step: Step, predicted: str) -> str | None:
    """Check one predicted observation against a small, explicit constraint set.

    Deliberately small, not exhaustive: a storm warning dooms the plan
    outright, and a predicted hotel cost over the budget dooms a booking.
    """
    if "storm warning" in predicted.lower():
        return f"Predicted observation for {step.id!r} carries a storm warning"
    if step.tool == "estimate_hotel_cost":
        match = re.search(r"\$(\d+)", predicted)
        if match and int(match.group(1)) > BUDGET_CAP:
            return f"Predicted hotel cost ${match.group(1)} for {step.id!r} exceeds the ${BUDGET_CAP} budget cap"
    return None


def simulate_plan(
    provider: Provider, goal: str, plan: Plan, registry: ToolRegistry
) -> tuple[list[SimulatedStep], str | None]:
    """Predict every step's observation in dependency order, stopping at the first doomed step.

    Args:
        provider: Called once per simulated step to predict its observation.
        goal: The goal the plan serves, given as context to each prediction.
        plan: The plan to simulate; not executed for real here.
        registry: Only used for `validate_plan`'s allowlist, not for execution.

    Returns:
        The simulated trajectory up to (and including) any doomed step, and
        that step's id, or None if nothing was doomed.
    """
    validate_plan(plan, registry)  # precondition before topological_waves, see module docstring
    state: dict[str, str] = {}
    trajectory: list[SimulatedStep] = []
    doomed_id: str | None = None
    for wave in topological_waves(plan.steps):
        for step in wave:
            state_summary = "\n".join(f"{k}: {v}" for k, v in state.items()) or "(none yet)"
            prompt = f"Goal: {goal}\nState so far:\n{state_summary}\nNext step: {step.tool}({step.args})"
            completion = provider.complete([Message.user(prompt)], system=WORLD_MODEL_SYSTEM)
            predicted = completion.content
            state[step.id] = predicted
            violation = _apply_constraints(step, predicted)
            trajectory.append(SimulatedStep(step.id, predicted, violation))
            if violation and doomed_id is None:
                doomed_id = step.id
        if doomed_id is not None:
            break
    return trajectory, doomed_id


def run_premortem(provider: Provider, goal: str, registry: ToolRegistry, plan: Plan | None = None) -> PremortemResult:
    """Simulate a plan, and execute it for real only if the simulation is clean.

    Args:
        provider: Supplies the planner call (if `plan` is None), one
            prediction call per simulated step, and no calls at all for
            real execution, which reuses `registry.execute` directly.
        goal: The user's goal.
        registry: Tools available to the plan; also the validator's allowlist.
        plan: A plan to simulate directly, skipping the planner call. Lets a
            caller re-simulate a repaired plan without re-planning from goal.
    """
    if plan is None:
        plan_completion = provider.complete([Message.user(goal)], system=PLANNER_SYSTEM)
        plan = parse_plan(goal, plan_completion.content)

    trajectory, doomed_id = simulate_plan(provider, goal, plan, registry)
    state = {s.step_id: s.predicted_observation for s in trajectory}
    if doomed_id is not None:
        return PremortemResult(
            doomed=True,
            doomed_step_id=doomed_id,
            simulated_trajectory=trajectory,
            state=state,
            executed=False,
            real_results=None,
        )

    results_map: dict[str, StepResult] = {}
    ordered: list[StepResult] = []
    for step in plan.steps:
        args = substitute_args(step.args, results_map)
        output = registry.execute(ToolCall(id=step.id, name=step.tool, arguments=args))
        result = StepResult(step_id=step.id, output=output, ok=not is_error_observation(output))
        results_map[step.id] = result
        ordered.append(result)
    return PremortemResult(
        doomed=False,
        doomed_step_id=None,
        simulated_trajectory=trajectory,
        state=state,
        executed=True,
        real_results=ordered,
    )


def demo() -> None:
    """Catch a simulated storm before an outdoor tour runs, repair, then execute the fix for real."""
    from patterns.planning.plan_repair import compute_blast_radius
    from patterns.planning.tools import build_travel_registry

    goal = "Spend a day outdoors in Paris, then visit an indoor museum."
    plan = parse_plan(
        goal,
        '[{"id": "w", "tool": "get_weather", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "outdoor", "tool": "search_attractions", "args": {"city": "Paris"}, "depends_on": ["w"]}]',
    )
    registry = build_travel_registry()

    print("=== Premortem: simulate before executing ===")
    print(f"Goal: {goal}")
    provider = get_provider(
        script=["Partly cloudy, no rain expected", "Storm warning: heavy winds expected, outdoor market closed"]
    )
    caught = run_premortem(provider, goal, registry, plan=plan)
    affected = compute_blast_radius(plan, caught.doomed_step_id) if caught.doomed_step_id else set()
    print(f"Doomed: {caught.doomed} at {caught.doomed_step_id!r}, executed for real: {caught.executed}")
    print(f"Blast radius (via plan_repair.compute_blast_radius): {sorted(affected)}")

    print("--- Corrected plan: swap the outdoor step for an indoor one ---")
    fixed_plan = parse_plan(
        goal,
        '[{"id": "w", "tool": "get_weather", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "outdoor", "tool": "draft_itinerary",'
        '  "args": {"weather": "$w", "attractions": "Louvre Museum (indoor)"}, "depends_on": ["w"]}]',
    )
    clean_provider = get_provider(script=["Mild and cloudy, no storm", "Given weather, visit the Louvre"])
    fixed = run_premortem(clean_provider, goal, registry, plan=fixed_plan)
    print(f"Doomed: {fixed.doomed}, executed for real: {fixed.executed}")
    for result in fixed.real_results or []:
        print(f"  {result.step_id} -> {result.output}")
    print("Note: the storm was caught in simulation, before search_attractions ever ran for real.")


if __name__ == "__main__":
    demo()
