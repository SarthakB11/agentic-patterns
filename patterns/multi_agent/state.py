"""Shared state: the single source of truth a multi-agent run reads and writes.

Every variant in this package threads one `SharedState` object through its
run instead of passing results directly between agents. This is the
"blackboard" idea reduced to its essentials: a shared object plus a trace of
who touched it, so a run is inspectable after the fact.

Single-writer rule: workers never see `SharedState` and cannot write to it.
They return a `WorkerResult` (see `worker.py`) to whoever dispatched them.
Only a caller holding `SharedState.WRITER_ROLE` may call `write_result`;
every other role gets `PermissionError`. This follows Cognition's "Don't
Build Multi-Agents" observation that parallel agents making independent
writes to shared state produce conflicting decisions from partial context;
keeping one writer avoids that by construction, not by convention.

Checkpoint and resume: `checkpoint()` serializes which subtasks are already
done, and `SharedState.resume()` rebuilds a state from that snapshot. A
supervisor that consults `completed_subtask_ids` before dispatching a worker
can stop and restart a long run without redoing finished work, mirroring
durable-execution checkpointing (LangGraph 1.0) rather than treating the run
as a single in-memory function call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEntry:
    """One recorded event in a run: who did what, in order.

    Attributes:
        seq: 1-based position in the trace, assigned at record time.
        actor: Name of the agent or role responsible for the event.
        action: Short verb phrase, e.g. "decompose", "dispatch", "write_result".
        detail: Free-text detail about the event.
    """

    seq: int
    actor: str
    action: str
    detail: str = ""

    def __str__(self) -> str:
        suffix = f": {self.detail}" if self.detail else ""
        return f"[{self.seq}] {self.actor} {self.action}{suffix}"


@dataclass
class SharedState:
    """The single source of truth threaded through one multi-agent run.

    Attributes:
        goal: The top-level goal this run is working toward.
        results: Subtask id (or a named key like "final_report") mapped to
            the content written for it. Only ever populated through
            `write_result`.
        statuses: Subtask id mapped to a lifecycle label such as "pending",
            "in_progress", "done", or "failed". A lightweight planner ledger
            a supervisor updates as it dispatches and collects work.
        completed_subtask_ids: Ids that have a finished, written result.
            Consulted by `resume()` callers to skip finished work.
        trace: Ordered log of every recorded event, for inspection.
    """

    WRITER_ROLE = "supervisor"

    goal: str
    results: dict[str, str] = field(default_factory=dict)
    statuses: dict[str, str] = field(default_factory=dict)
    completed_subtask_ids: set[str] = field(default_factory=set)
    trace: list[TraceEntry] = field(default_factory=list)

    def record(self, actor: str, action: str, detail: str = "") -> None:
        """Append an event to the trace. Anyone may record; only the writer may write results."""
        self.trace.append(TraceEntry(seq=len(self.trace) + 1, actor=actor, action=action, detail=detail))

    def set_status(self, subtask_id: str, status: str) -> None:
        """Update a subtask's ledger entry and record the transition."""
        self.statuses[subtask_id] = status
        self.record("ledger", "status", f"{subtask_id} -> {status}")

    def write_result(self, writer_role: str, key: str, content: str) -> None:
        """Write a result into shared state. Only `WRITER_ROLE` may call this.

        Args:
            writer_role: The role of the caller attempting to write.
            key: A subtask id, or a named key such as "final_report".
            content: The content to store.

        Raises:
            PermissionError: If `writer_role` is not `SharedState.WRITER_ROLE`.
                Workers return proposals through their own return value; they
                are never handed this method, so this check is a second,
                defense-in-depth guard against a caller wiring them up wrong.
        """
        if writer_role != self.WRITER_ROLE:
            raise PermissionError(
                f"Only role {self.WRITER_ROLE!r} may write to shared state; "
                f"got a write attempt from {writer_role!r}. Workers must return "
                "proposals to the supervisor instead of writing directly."
            )
        self.results[key] = content
        self.completed_subtask_ids.add(key)
        preview = content if len(content) <= 60 else content[:57] + "..."
        self.record(writer_role, "write_result", f"{key}: {preview}")

    def checkpoint(self) -> dict[str, Any]:
        """Serialize the resumable parts of this state to a plain dict."""
        return {
            "goal": self.goal,
            "results": dict(self.results),
            "statuses": dict(self.statuses),
            "completed_subtask_ids": sorted(self.completed_subtask_ids),
        }

    @classmethod
    def resume(cls, checkpoint: dict[str, Any]) -> SharedState:
        """Rebuild a `SharedState` from a `checkpoint()` snapshot and record the resume.

        Args:
            checkpoint: A dict previously returned by `checkpoint()`.
        """
        state = cls(
            goal=checkpoint["goal"],
            results=dict(checkpoint["results"]),
            statuses=dict(checkpoint["statuses"]),
            completed_subtask_ids=set(checkpoint["completed_subtask_ids"]),
        )
        state.record(
            "supervisor",
            "resume",
            f"resumed with {len(state.completed_subtask_ids)} subtask(s) already done",
        )
        return state

    def format_trace(self) -> str:
        """Render the trace as a readable, newline-joined block."""
        return "\n".join(str(entry) for entry in self.trace)
