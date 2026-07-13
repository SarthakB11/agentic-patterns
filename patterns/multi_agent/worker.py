"""Worker abstraction and parallel fan-out (concurrent / parallel variant).

A `Worker` is a narrow agent: a role, a system prompt, and its own
`Provider`, matching the brief's "each with its own prompt, tools, and often
its own context window." A `Subtask` is the delegation payload a supervisor
hands a worker: an explicit objective, output format, and boundaries, so
vague delegation (the top cause of duplicated or missing work per the
brief) is structurally harder to write.

`run_worker` is a worker's own small agent loop: call the model, execute any
requested tool calls, feed observations back, repeat up to a bounded number
of rounds. It always returns a `WorkerResult`, converting a raised exception
into an "error" result rather than propagating, so one worker's failure is
isolated and does not take down a fan-out of independent workers.

`dispatch_parallel` runs several (worker, subtask) pairs concurrently with a
thread pool and returns results in the caller's original order, not
completion order, so aggregation downstream is deterministic regardless of
which worker happened to finish first. This is the concurrent / parallel
(fan-out) variant from the taxonomy; see `aggregation.py` for the matching
fan-in step.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, ToolRegistry


@dataclass
class Subtask:
    """An explicit delegation payload handed from a supervisor to a worker.

    Attributes:
        id: Unique identifier, used to track completion and order results.
        role: The worker role expected to handle this subtask, e.g.
            "market_researcher".
        objective: What the worker must accomplish. One sentence, specific.
        output_format: The shape the answer must take, e.g. "3 bullet
            points" or "a single yes/no vote".
        boundaries: Explicit limits on scope, kept as a list so they render
            as a checklist in a worker's prompt instead of getting buried in
            prose.
        context: Optional extra background the worker needs but should not
            treat as part of its own objective.
    """

    id: str
    role: str
    objective: str
    output_format: str
    boundaries: list[str] = field(default_factory=list)
    context: str = ""

    def to_prompt(self) -> str:
        """Render this subtask as a worker-facing instruction block."""
        lines = [f"Objective: {self.objective}", f"Required format: {self.output_format}"]
        if self.boundaries:
            lines.append("Boundaries:")
            lines.extend(f"- {b}" for b in self.boundaries)
        if self.context:
            lines.append(f"Context: {self.context}")
        return "\n".join(lines)


@dataclass
class WorkerResult:
    """A worker's structured return for one subtask.

    Attributes:
        subtask_id: The `Subtask.id` this result answers.
        role: The worker role that produced it.
        status: "ok" or "error".
        content: The worker's answer, or an "ERROR: ..." message when
            `status == "error"`.
    """

    subtask_id: str
    role: str
    status: str
    content: str


@dataclass
class Worker:
    """A narrow agent: one role, one system prompt, one provider.

    Attributes:
        role: Name of this worker's role, matched against `Subtask.role`.
        system_prompt: Instructions scoping this worker to its role.
        provider: This worker's own `Provider`, scripted independently of
            every other agent in the run.
        tools: Optional tool registry this worker may call.
    """

    role: str
    system_prompt: str
    provider: Provider
    tools: ToolRegistry | None = None


def run_worker(worker: Worker, subtask: Subtask, *, max_rounds: int = 3) -> WorkerResult:
    """Run one worker's agent loop (reason, call tools, observe) on one subtask.

    Args:
        worker: The worker to run.
        subtask: The subtask it was delegated.
        max_rounds: Maximum reason/act rounds before giving up. Bounds a
            worker that keeps calling tools without ever answering.

    Returns:
        A `WorkerResult`. Exceptions raised while running the worker (a
        provider error, a tool crashing outside `ToolRegistry.execute`'s own
        try/except, or a malformed script) are caught here and turned into
        an "error" result instead of propagating, so a fan-out of workers
        isolates one bad worker from the rest.
    """
    try:
        messages = [Message.user(subtask.to_prompt())]
        specs = worker.tools.specs() if worker.tools else None
        for _ in range(max_rounds):
            completion = worker.provider.complete(messages, tools=specs, system=worker.system_prompt)
            if not completion.tool_calls:
                return WorkerResult(subtask.id, worker.role, "ok", completion.content)
            messages.append(Message.assistant(completion.content, completion.tool_calls))
            if worker.tools is None:
                return WorkerResult(
                    subtask.id, worker.role, "error", "ERROR: worker requested a tool call with no tools registered"
                )
            for call in completion.tool_calls:
                observation = worker.tools.execute(call)
                messages.append(Message.tool(call.id, observation))
        return WorkerResult(
            subtask.id, worker.role, "error", f"ERROR: exceeded {max_rounds} rounds without a final answer"
        )
    except Exception as exc:  # noqa: BLE001 - isolate one worker's failure from the rest
        return WorkerResult(subtask.id, worker.role, "error", f"ERROR: {exc}")


def dispatch_parallel(assignments: list[tuple[Worker, Subtask]], *, max_workers: int = 8) -> list[WorkerResult]:
    """Run independent (worker, subtask) pairs concurrently and fan the results back in.

    Each worker has its own `Provider`, so running them on separate threads
    does not race a shared script index; every worker only ever reads its
    own conversation. Results are returned in the same order as
    `assignments`, not completion order, so downstream aggregation is
    deterministic no matter which thread finishes first.

    Args:
        assignments: (worker, subtask) pairs to run.
        max_workers: Maximum number of threads in the pool.
    """
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(assignments) or 1))) as pool:
        futures = [pool.submit(run_worker, worker, subtask) for worker, subtask in assignments]
        return [future.result() for future in futures]
