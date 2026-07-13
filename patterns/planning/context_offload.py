"""Context offload: persist the plan and step outputs to a file instead of
holding them only in memory, so a run can resume after a restart without
re-planning or re-running steps that already completed successfully.

This is what lets a long agent run survive a process crash, a context-window
compaction, or a deliberate pause: the state that matters is not trapped in
one process's memory. A resumed run with a checkpoint on disk should call
the planner zero times and only execute the steps still missing or whose
checkpointed result failed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry

from patterns.planning.parser import parse_plan
from patterns.planning.plan import Plan, Step, StepResult, is_error_observation, substitute_args
from patterns.planning.validator import validate_plan

PLANNER_SYSTEM = (
    "You are a trip-planning agent. Respond with ONLY a JSON array of steps: "
    "id, tool, args, depends_on."
)


def save_state(path: Path, plan: Plan, results: dict[str, StepResult]) -> None:
    """Write `plan` and the completed `results` to `path` as JSON."""
    data = {
        "goal": plan.goal,
        "steps": [
            {"id": s.id, "tool": s.tool, "args": s.args, "depends_on": s.depends_on} for s in plan.steps
        ],
        "results": {sid: {"output": r.output, "ok": r.ok} for sid, r in results.items()},
    }
    path.write_text(json.dumps(data, indent=2))


def load_state(path: Path) -> tuple[Plan, dict[str, StepResult]]:
    """Read a plan and its completed results back from `path`."""
    data = json.loads(path.read_text())
    steps = [
        Step(id=s["id"], tool=s["tool"], args=s["args"], depends_on=s["depends_on"]) for s in data["steps"]
    ]
    plan = Plan(goal=data["goal"], steps=steps)
    results = {
        sid: StepResult(step_id=sid, output=r["output"], ok=r["ok"]) for sid, r in data["results"].items()
    }
    return plan, results


@dataclass
class OffloadRun:
    """The outcome of a context-offloaded plan-then-execute run.

    Attributes:
        plan: The plan in effect (loaded from disk on a resumed run).
        results: Every step's result, keyed by step id, including any that
            were already complete before this call started.
        resumed: True if this run loaded a prior checkpoint instead of planning.
        planner_calls: 1 for a fresh run, 0 for a resumed one.
    """

    plan: Plan
    results: dict[str, StepResult]
    resumed: bool
    planner_calls: int


def run_with_offload(provider: Provider, goal: str, registry: ToolRegistry, state_path: Path) -> OffloadRun:
    """Run a plan, checkpointing to `state_path` after every step.

    If `state_path` already holds a checkpoint, this resumes from it: the
    planner is never called, and only steps missing from the saved results
    or whose saved result has `ok=False` execute. Otherwise this plans
    fresh, saves immediately, then saves again after every step.

    Args:
        provider: Supplies the planner call on a fresh run; unused on resume.
        goal: The user's goal, sent to the planner on a fresh run only.
        registry: Tools available to the plan; also the validator's allowlist.
        state_path: File to read an existing checkpoint from, or write to.
    """
    planner_calls = 0
    if state_path.exists():
        plan, results = load_state(state_path)
        resumed = True
    else:
        plan_completion = provider.complete([Message.user(goal)], system=PLANNER_SYSTEM)
        planner_calls += 1
        plan = parse_plan(goal, plan_completion.content)
        validate_plan(plan, registry)
        results = {}
        resumed = False
        save_state(state_path, plan, results)

    for step in plan.steps:
        if step.id in results and results[step.id].ok:
            continue
        args = substitute_args(step.args, results)
        output = registry.execute(ToolCall(id=step.id, name=step.tool, arguments=args))
        results[step.id] = StepResult(step_id=step.id, output=output, ok=not is_error_observation(output))
        save_state(state_path, plan, results)

    return OffloadRun(plan=plan, results=results, resumed=resumed, planner_calls=planner_calls)


def demo() -> None:
    """Simulate a crash after step 1, then resume from the checkpoint and finish step 2."""
    import tempfile

    from agentic_patterns import ToolCall as _ToolCall
    from agentic_patterns import get_provider
    from patterns.planning.tools import build_travel_registry

    goal = "Plan a weekend in Lyon: check the weather, then estimate hotel cost for 2 nights."
    registry = build_travel_registry()

    with tempfile.TemporaryDirectory() as tmp:
        state_path = Path(tmp) / "plan_state.json"

        print("=== Context offload (checkpoint, simulated restart, resume) ===")
        print(f"Goal: {goal}")

        print("--- Before the crash: planner runs, step 1 completes and is checkpointed ---")
        plan = Plan(
            goal=goal,
            steps=[
                Step(id="step1", tool="get_weather", args={"city": "Lyon"}, depends_on=[]),
                Step(id="step2", tool="estimate_hotel_cost", args={"city": "Lyon", "nights": 2}, depends_on=[]),
            ],
        )
        weather = registry.execute(_ToolCall(id="step1", name="get_weather", arguments={"city": "Lyon"}))
        save_state(state_path, plan, {"step1": StepResult(step_id="step1", output=weather)})
        print(f"Checkpoint written after step1: {weather}")
        print("(process exits here in a real crash; state_path survives on disk)")

        print("\n--- After the restart: resume from the checkpoint ---")
        provider = get_provider(script=[])  # empty script: proves the planner is never called
        run = run_with_offload(provider, goal, registry, state_path)
        print(f"Resumed: {run.resumed}, planner calls this run: {run.planner_calls}")
        for step_id, result in run.results.items():
            print(f"  {step_id} -> {result.output}")


if __name__ == "__main__":
    demo()
