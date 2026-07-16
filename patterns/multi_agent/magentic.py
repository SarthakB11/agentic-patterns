"""Magentic-style dual-ledger orchestrator: stall detection and replan.

`supervisor.py` decomposes a goal once, dispatches, and never adapts; the
README has said since the first cut of this folder that replanning-on-stall
was skipped for size. This module is that completion, not a permanent gap.
Magentic-One (Fourney et al., arXiv:2411.04468) separates two ledgers: an
outer-loop Task Ledger (facts, guesses, and a plan) and an inner-loop
Progress Ledger (is the task done, did the last step make progress, who
goes next and with what instruction). The inner loop runs until the task is
done or a stall counter trips; a stall triggers an outer-loop reflection
that rewrites the Task Ledger and restarts the inner loop. This shipped into
the Microsoft Agent Framework's Magentic orchestration.

`state.py`'s status ledger is passive bookkeeping a caller updates by hand;
this is the active control loop Magentic-One adds on top of it: it decides,
per step, whether the run is making progress and what to do next, rather
than recording decisions a caller already made.

Every ledger call and every worker dispatch is one `Provider.complete()`
call, scripted like everywhere else in this repo. This is not a faithful
Magentic-One: there is no dedicated web/file/code agent roster (a stand-in
`Worker` from `worker.py` plays every named agent), no cost-aware
model-switching, and the Progress Ledger's `NEXT_INSTRUCTION` field doubles
as the final answer text when `DONE: yes`, since the ledger schema has no
separate answer field. The distinctive mechanism, the stall counter and the
reflect-revise-restart loop, is implemented in full.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Message, MockProvider, Provider
from patterns.multi_agent.worker import Subtask, Worker, run_worker

FALLBACK_MESSAGE = "Stopped: replan budget exhausted without the progress ledger reporting done."

TASK_LEDGER_SYSTEM = (
    "You are the outer-loop planner for a multi-agent orchestrator. Given the goal, write "
    "known FACTS, working GUESSES, and a numbered PLAN toward the goal. Reply with FACTS:, "
    "GUESSES:, and PLAN: sections, each a bulleted list of lines starting with '-'."
)
PROGRESS_LEDGER_SYSTEM = (
    "You are the inner-loop progress judge for a multi-agent orchestrator. Given the task "
    "ledger and the transcript so far, decide whether the task is DONE, whether the last step "
    "made PROGRESS, and who should act NEXT with what INSTRUCTION. Reply with four lines: "
    "'DONE: yes/no', 'PROGRESS: yes/no', 'NEXT_AGENT: <name>', 'NEXT_INSTRUCTION: <text>'. "
    "When DONE is yes, put the final answer in NEXT_INSTRUCTION."
)
REFLECT_SYSTEM = (
    "You are the outer-loop reflector for a stalled multi-agent orchestrator. Given the "
    "current task ledger and the stalled transcript, decide what went wrong and rewrite the "
    "ledger: revised FACTS:, GUESSES:, and PLAN: sections in the same bulleted format."
)


@dataclass
class TaskLedger:
    """The outer loop's shared understanding of the task.

    Attributes:
        goal: The top-level goal.
        facts: Known facts, rebuilt on every replan.
        guesses: Working guesses, rebuilt on every replan.
        plan: An ordered list of step descriptions, rebuilt on every replan.
    """

    goal: str
    facts: list[str] = field(default_factory=list)
    guesses: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)


@dataclass
class ProgressLedger:
    """The inner loop's per-step verdict.

    Attributes:
        done: True if the task is complete.
        progress: True if the last step moved the task forward.
        next_agent: Name of the agent to dispatch next.
        next_instruction: Instruction for that agent, or the final answer
            text when `done` is True.
    """

    done: bool
    progress: bool
    next_agent: str
    next_instruction: str


@dataclass
class MagenticResult:
    """The outcome of a `run_magentic` call.

    Attributes:
        answer: The final answer, or `FALLBACK_MESSAGE` if the replan cap
            was hit without the progress ledger ever reporting done.
        replans: Number of times the outer loop reflected and rewrote the plan.
        stop_reason: "completed" or "replan_cap".
        ledger: The final `TaskLedger` in effect when the run stopped.
        transcript: Every dispatched agent's result, in order, as
            "<agent>: <content>" lines.
    """

    answer: str
    replans: int
    stop_reason: str
    ledger: TaskLedger
    transcript: list[str]


def _sections(text: str, keys: list[str]) -> dict[str, list[str]]:
    """Parse a `KEY:\\n- line\\n- line` formatted completion into key -> bullet lines."""
    result: dict[str, list[str]] = {k: [] for k in keys}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        matched = next((k for k in keys if upper == f"{k}:"), None)
        if matched:
            current = matched
        elif current and stripped.startswith("-"):
            result[current].append(stripped[1:].strip())
    return result


def _field(text: str, key: str) -> str:
    prefix = f"{key.upper()}:"
    for line in text.splitlines():
        if line.strip().upper().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def _build_task_ledger(provider: Provider, goal: str) -> TaskLedger:
    completion = provider.complete([Message.user(f"Goal: {goal}")], system=TASK_LEDGER_SYSTEM)
    parsed = _sections(completion.content, ["FACTS", "GUESSES", "PLAN"])
    return TaskLedger(goal=goal, facts=parsed["FACTS"], guesses=parsed["GUESSES"], plan=parsed["PLAN"])


def _render_ledger(ledger: TaskLedger, transcript: list[str]) -> str:
    lines = [
        f"Goal: {ledger.goal}",
        "Facts:",
        *[f"- {f}" for f in ledger.facts],
        "Plan:",
        *[f"- {p}" for p in ledger.plan],
    ]
    lines.append("Transcript so far:")
    lines.extend(transcript or ["(none yet)"])
    return "\n".join(lines)


def _ask_progress_ledger(provider: Provider, ledger: TaskLedger, transcript: list[str]) -> ProgressLedger:
    completion = provider.complete([Message.user(_render_ledger(ledger, transcript))], system=PROGRESS_LEDGER_SYSTEM)
    text = completion.content
    done, progress = _field(text, "DONE").lower() == "yes", _field(text, "PROGRESS").lower() == "yes"
    next_agent, next_instruction = _field(text, "NEXT_AGENT"), _field(text, "NEXT_INSTRUCTION")
    if not next_agent or not next_instruction:
        raise ValueError(f"malformed progress ledger verdict: {text!r}")
    return ProgressLedger(done=done, progress=progress, next_agent=next_agent, next_instruction=next_instruction)


def _reflect_and_replan(provider: Provider, ledger: TaskLedger, transcript: list[str]) -> TaskLedger:
    prompt = f"Goal: {ledger.goal}\nPrior plan: {ledger.plan}\nStalled transcript:\n" + "\n".join(transcript)
    completion = provider.complete([Message.user(prompt)], system=REFLECT_SYSTEM)
    parsed = _sections(completion.content, ["FACTS", "GUESSES", "PLAN"])
    return TaskLedger(goal=ledger.goal, facts=parsed["FACTS"], guesses=parsed["GUESSES"], plan=parsed["PLAN"])


def run_magentic(
    orchestrator_provider: Provider,
    agents: dict[str, Worker],
    goal: str,
    *,
    stall_threshold: int = 2,
    max_replans: int = 2,
    max_inner_steps: int = 6,
) -> MagenticResult:
    """Run the dual-ledger orchestrator to completion or a replan cap.

    Args:
        orchestrator_provider: Provider for the Task Ledger, Progress
            Ledger, and reflection calls, all made by the orchestrator.
        agents: Agent name mapped to the `Worker` that plays it; the
            Progress Ledger's `NEXT_AGENT` must name one of these.
        goal: The top-level goal.
        stall_threshold: Consecutive no-progress (or repeated-action) steps
            before the inner loop breaks and the outer loop replans.
        max_replans: Hard cap on replans before returning the fallback
            instead of looping forever on a task that never converges.
        max_inner_steps: Hard cap on inner-loop steps per plan, independent
            of the stall counter, so a plan that always reports progress
            without ever finishing cannot loop forever either.
    """
    ledger = _build_task_ledger(orchestrator_provider, goal)
    transcript: list[str] = []
    seen_actions: set[str] = set()
    replans = 0

    while True:
        stall_counter = 0
        for _ in range(max_inner_steps):
            progress = _ask_progress_ledger(orchestrator_provider, ledger, transcript)
            if progress.done:
                return MagenticResult(progress.next_instruction, replans, "completed", ledger, list(transcript))
            if progress.next_agent not in agents:
                raise ValueError(f"progress ledger named an unknown agent: {progress.next_agent!r}")
            subtask = Subtask(
                f"step-{len(transcript) + 1}", progress.next_agent, progress.next_instruction, "free text"
            )
            result = run_worker(agents[progress.next_agent], subtask)
            action_key = f"{progress.next_agent}:{result.content}"
            transcript.append(f"{progress.next_agent}: {result.content}")
            stall_counter = 0 if (progress.progress and action_key not in seen_actions) else stall_counter + 1
            seen_actions.add(action_key)
            if stall_counter >= stall_threshold:
                break
        replans += 1
        if replans > max_replans:
            return MagenticResult(FALLBACK_MESSAGE, replans - 1, "replan_cap", ledger, list(transcript))
        ledger = _reflect_and_replan(orchestrator_provider, ledger, transcript)


# --- demo --------------------------------------------------------------


def run_magentic_demo() -> MagenticResult:
    """A room-lookup task that stalls once, replans, then finishes.

    The first plan sends `room_bot` to check two wrong rooms (two
    no-progress steps trip the stall threshold). The orchestrator reflects,
    guesses the room is in the calendar system instead, and dispatches
    `calendar_bot`, which finds it on the first try, ending the run.
    """
    goal = "Find which conference room is booked for the 3pm design review."
    orchestrator = MockProvider(
        script=[
            "FACTS:\n- The design review is at 3pm today\nGUESSES:\n- The room is one of the usual rooms\n"
            "PLAN:\n- Check Room A\n- Check Room B",
            "DONE: no\nPROGRESS: no\nNEXT_AGENT: room_bot\nNEXT_INSTRUCTION: Check whether Room A is booked at 3pm",
            "DONE: no\nPROGRESS: no\nNEXT_AGENT: room_bot\nNEXT_INSTRUCTION: Check whether Room B is booked at 3pm",
            "FACTS:\n- The design review is at 3pm today\n- Neither Room A nor Room B is booked at 3pm\n"
            "GUESSES:\n- The booking is likely in the calendar system's room field, not a manual room list\n"
            "PLAN:\n- Ask calendar_bot for the room field on the 3pm design review event",
            "DONE: no\nPROGRESS: yes\nNEXT_AGENT: calendar_bot\n"
            "NEXT_INSTRUCTION: Look up the room field for the 3pm design review event",
            "DONE: yes\nPROGRESS: yes\nNEXT_AGENT: calendar_bot\n"
            "NEXT_INSTRUCTION: Room C (Innovation Lab) is booked for the 3pm design review.",
        ]
    )
    room_bot = Worker(
        "room_bot",
        "You check named rooms for a booking.",
        MockProvider(script=["Room A is free at 3pm; not a match.", "Room B is free at 3pm; not a match either."]),
    )
    calendar_bot = Worker(
        "calendar_bot",
        "You look up calendar event fields.",
        MockProvider(script=["The calendar event lists Room C (Innovation Lab) as the booked room."]),
    )
    return run_magentic(orchestrator, {"room_bot": room_bot, "calendar_bot": calendar_bot}, goal)
