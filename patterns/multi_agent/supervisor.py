"""Supervisor / orchestrator-worker (star topology): the default multi-agent shape.

A single supervisor sees the whole picture. It decomposes the goal into
scoped subtasks, dispatches them to workers that run in parallel and never
talk to each other, collects their proposals, and performs the one
synthesis write that becomes the final answer. This is the canonical
control flow from the brief:

    receive goal -> decompose -> dispatch (parallel) -> collect -> write
    proposals -> synthesize -> write final answer

The supervisor is the sole holder of `SharedState.WRITER_ROLE`; workers
return `WorkerResult` values and never touch `SharedState` themselves (see
`state.py` for why). Decomposition itself is a scripted tool call rather
than free text, so a subtask's objective, output format, and boundaries
arrive as a structured object instead of prose the supervisor would have to
re-parse.

`run_resume_demo` shows the durable-execution refinement from the July 2026
expansion: given a checkpoint where one subtask already completed, the
supervisor skips dispatching a worker for it and only runs the remaining
work, rather than redoing a finished subtask.
"""

from __future__ import annotations

from agentic_patterns import Message, Provider, get_provider, scripted_tool_call

from patterns.multi_agent import aggregation
from patterns.multi_agent.state import SharedState
from patterns.multi_agent.worker import Subtask, Worker, WorkerResult, dispatch_parallel

DELEGATE_TOOL = {
    "name": "delegate_subtasks",
    "description": "Propose the subtask breakdown for a goal, one entry per specialist worker.",
    "parameters": {
        "type": "object",
        "properties": {
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "role": {"type": "string"},
                        "objective": {"type": "string"},
                        "output_format": {"type": "string"},
                        "boundaries": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "role", "objective", "output_format"],
                },
            }
        },
        "required": ["subtasks"],
    },
}

DECOMPOSE_SYSTEM = (
    "You are a supervisor planning a research task. Break the goal into 2-4 "
    "scoped subtasks for specialist workers, each with a clear objective, "
    "required output format, and boundaries. Call delegate_subtasks."
)

SYNTHESIS_SYSTEM = (
    "You are a supervisor combining specialist findings into one coherent "
    "answer for the goal. Do not just list the findings; connect them."
)

GOAL = "Produce a one-page competitive brief on note-taking apps for the product team."


def decompose(provider: Provider, goal: str, *, system: str = DECOMPOSE_SYSTEM) -> list[Subtask]:
    """Ask the supervisor's provider to break a goal into scoped `Subtask` objects.

    Args:
        provider: The supervisor's provider, scripted (for `MockProvider`)
            to return a `delegate_subtasks` tool call next.
        goal: The top-level goal to decompose.
        system: System prompt guiding the decomposition.

    Raises:
        ValueError: If the model's response is not a `delegate_subtasks`
            tool call, since a supervisor has nothing to dispatch otherwise.
    """
    completion = provider.complete([Message.user(goal)], tools=[DELEGATE_TOOL], system=system)
    if not completion.tool_calls or completion.tool_calls[0].name != "delegate_subtasks":
        raise ValueError("supervisor did not propose a subtask breakdown via delegate_subtasks")
    raw = completion.tool_calls[0].arguments["subtasks"]
    return [
        Subtask(
            id=s["id"],
            role=s["role"],
            objective=s["objective"],
            output_format=s["output_format"],
            boundaries=s.get("boundaries", []),
        )
        for s in raw
    ]


def _default_worker_scripts() -> dict[str, str]:
    """Scripted answers for the note-taking-apps demo, one per worker role."""
    return {
        "market": (
            "Notion prices at $10/user/month targeting knowledge workers, Obsidian is a "
            "one-time $50 purchase targeting power users who want local files, and Evernote "
            "runs $15/month targeting general consumers. All three now bundle AI search as of 2026."
        ),
        "tech": (
            "Obsidian is the only one of the three with offline-first storage as plain "
            "markdown files; Notion and Evernote both require network sync and lock content "
            "into proprietary formats; only Notion ships a public API with write access."
        ),
        "risk": (
            "Obsidian's local-first, no-lock-in model is the single biggest risk to our "
            "roadmap because it removes the switching cost our retention plan depends on."
        ),
    }


def _default_subtask_args() -> list[dict[str, object]]:
    """The exact `delegate_subtasks` arguments the scripted decompose call returns."""
    return [
        {
            "id": "market",
            "role": "market_researcher",
            "objective": "Summarize the top 3 competitors' pricing and target users",
            "output_format": "3 bullet points",
            "boundaries": ["Do not evaluate our own product", "Cite named competitors only"],
        },
        {
            "id": "tech",
            "role": "tech_researcher",
            "objective": "Summarize technical differentiators: offline support, sync, API",
            "output_format": "3 bullet points",
            "boundaries": ["Stick to technical capabilities, not price"],
        },
        {
            "id": "risk",
            "role": "risk_analyst",
            "objective": "Flag the single biggest competitive risk to our roadmap",
            "output_format": "one sentence",
            "boundaries": ["State the risk plainly, no hedging"],
        },
    ]


def _build_assignments(subtasks: list[Subtask], scripts: dict[str, str]) -> list[tuple[Worker, Subtask]]:
    """Build one scripted `Worker` per subtask that has a script entry."""
    assignments = []
    for subtask in subtasks:
        if subtask.id not in scripts:
            continue
        provider = get_provider(script=[scripts[subtask.id]])
        worker = Worker(subtask.role, f"You are a {subtask.role}. Stay within the stated boundaries.", provider)
        assignments.append((worker, subtask))
    return assignments


def run_supervisor_demo() -> tuple[SharedState, list[WorkerResult]]:
    """Run the full star-topology loop: decompose, dispatch, collect, synthesize.

    Returns:
        The `SharedState` the supervisor wrote to (with the final report
        under the "final_report" key) and the raw `WorkerResult` list.
    """
    supervisor_provider = get_provider(
        script=[
            scripted_tool_call("delegate_subtasks", {"subtasks": _default_subtask_args()}),
            (
                "Notion ($10/mo) and Evernote ($15/mo) both require network sync and lock "
                "content into proprietary formats, while Obsidian's one-time $50 price and "
                "local markdown storage remove our biggest lock-in argument, which is also "
                "our top competitive risk. Recommendation: ship a local-export path before "
                "the next renewal cycle."
            ),
        ]
    )
    state = SharedState(goal=GOAL)

    subtasks = decompose(supervisor_provider, GOAL)
    state.record("supervisor", "decompose", f"{len(subtasks)} subtasks proposed: {[s.id for s in subtasks]}")

    assignments = _build_assignments(subtasks, _default_worker_scripts())
    for _, subtask in assignments:
        state.set_status(subtask.id, "in_progress")
    results = dispatch_parallel(assignments)

    for result in results:
        state.set_status(result.subtask_id, "done" if result.status == "ok" else "failed")
        if result.status == "ok":
            state.write_result(SharedState.WRITER_ROLE, result.subtask_id, result.content)

    final_report = aggregation.model_synthesize(
        supervisor_provider, results, goal=GOAL, system=SYNTHESIS_SYSTEM
    )
    state.write_result(SharedState.WRITER_ROLE, "final_report", final_report)
    return state, results


def run_resume_demo() -> tuple[SharedState, list[tuple[Worker, Subtask]]]:
    """Resume a run where one subtask ("market") already completed.

    Builds a checkpoint as if a prior run finished the market-research
    subtask, restores it with `SharedState.resume`, decomposes the goal
    again (the plan itself is cheap and deterministic to recompute), and
    shows that the resumed run only builds worker assignments for subtasks
    still missing from `completed_subtask_ids`. The "market" worker is never
    constructed, so its provider is never called: no completed worker is
    replayed. This checkpoints results, not the plan: unlike LangGraph-style
    durable execution, which also checkpoints the plan so no completed step
    (including planning) re-executes, this demo still re-runs `decompose()`
    on resume. A reader should not assume nothing re-runs here.

    Returns:
        The resumed `SharedState` and the assignments that would actually
        be dispatched (only "tech" and "risk").
    """
    prior = SharedState(goal=GOAL)
    prior.write_result(
        SharedState.WRITER_ROLE,
        "market",
        "Notion $10/mo, Obsidian $50 one-time, Evernote $15/mo; all three bundle AI search.",
    )
    checkpoint = prior.checkpoint()
    resumed = SharedState.resume(checkpoint)

    supervisor_provider = get_provider(
        script=[scripted_tool_call("delegate_subtasks", {"subtasks": _default_subtask_args()})]
    )
    subtasks = decompose(supervisor_provider, GOAL)
    resumed.record("supervisor", "decompose", f"{len(subtasks)} subtasks in plan, resuming remaining work")

    remaining = [s for s in subtasks if s.id not in resumed.completed_subtask_ids]
    scripts = {sid: text for sid, text in _default_worker_scripts().items() if sid != "market"}
    assignments = _build_assignments(remaining, scripts)
    return resumed, assignments
