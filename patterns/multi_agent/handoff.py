"""Handoff / routing / triage, and the subagent variant that returns control.

Both variants move a task between agents using an A2A-style task object: a
small envelope with an id, a payload, and an explicit lifecycle
(`pending -> in_progress -> completed` or `failed`), matching the Agent2Agent
protocol's task states rather than an ad hoc dict. This is the decentralized
cousin of the supervisor: control moves between agents instead of being
scheduled by one central agent.

The two variants differ in exactly one way, which is the point of this
module:

- **Handoff** (`run_handoff_demo`): a triage agent inspects a request and
  transfers it to a specialist. Control does not return; the specialist's
  answer is the final answer.
- **Subagent** (`run_subagent_demo`): a parent agent dispatches a bounded
  question to a child agent. When the child completes, control returns to
  the parent, which continues its own turn using the child's answer. The
  April 2026 OpenAI Agents SDK release added this as a primitive distinct
  from a handoff for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Message, Provider

_VALID_STATUSES = ("pending", "in_progress", "completed", "failed")


@dataclass
class DelegationTask:
    """An A2A-style delegation payload with an explicit lifecycle.

    Attributes:
        task_id: Unique identifier for this delegation.
        from_agent: Name of the agent that created the task.
        to_agent: Name of the agent the task is delegated to.
        payload: The content being delegated (a request, question, or
            subtask description).
        status: One of "pending", "in_progress", "completed", "failed".
        history: One entry per status transition, for inspection.
    """

    task_id: str
    from_agent: str
    to_agent: str
    payload: str
    status: str = "pending"
    history: list[str] = field(default_factory=list)

    def transition(self, status: str, note: str = "") -> None:
        """Move the task to a new lifecycle status and record it.

        Raises:
            ValueError: If `status` is not one of the recognized A2A states.
        """
        if status not in _VALID_STATUSES:
            raise ValueError(f"Unknown status {status!r}; expected one of {_VALID_STATUSES}")
        self.status = status
        self.history.append(f"{status}" + (f": {note}" if note else ""))


def _decide_route(provider: Provider, request: str, *, system: str) -> tuple[str, str]:
    """Ask an agent which specialist a request should route to.

    Expects a plain-text reply of the form "ROUTE: <agent>\\nREASON: ...".
    """
    completion = provider.complete([Message.user(request)], system=system)
    text = completion.content
    route_line = next((line for line in text.splitlines() if line.upper().startswith("ROUTE:")), "")
    to_agent = route_line.split(":", 1)[1].strip() if route_line else ""
    if not to_agent:
        raise ValueError(f"triage response did not include a ROUTE: line: {text!r}")
    return to_agent, text


def run_handoff_demo(triage_provider: Provider, specialist_provider: Provider) -> DelegationTask:
    """Triage a support request and hand it off permanently to a specialist.

    Args:
        triage_provider: Provider for the triage agent. Scripted to reply
            with a ROUTE decision for the incoming request.
        specialist_provider: Provider for the specialist the request is
            routed to. Scripted with the specialist's resolution.

    Returns:
        A `DelegationTask` whose final status is "completed", with a
        history showing pending -> in_progress -> completed and no
        transition back to the triage agent, since control never returns.
    """
    request = "My invoice charged me twice this month. Can you refund the duplicate charge?"
    task = DelegationTask(task_id="support-1", from_agent="triage", to_agent="", payload=request)

    to_agent, triage_reply = _decide_route(
        triage_provider,
        request,
        system=(
            "You are a support triage agent. Read the request and route it to exactly one "
            "specialist: billing_specialist or technical_specialist. Reply with a ROUTE: line "
            "naming the agent and a REASON: line."
        ),
    )
    task.to_agent = to_agent
    task.transition("in_progress", f"routed by triage: {triage_reply.splitlines()[-1]}")

    resolution = specialist_provider.complete(
        [Message.user(request)],
        system=f"You are the {to_agent}. Resolve the request directly; you are the final agent to see it.",
    ).content
    task.payload = resolution
    task.transition("completed", f"resolved by {to_agent}")
    return task


def run_subagent_demo(parent_provider: Provider, child_provider: Provider) -> tuple[DelegationTask, str]:
    """Dispatch a bounded question to a child agent, then resume as the parent.

    Args:
        parent_provider: Provider for the parent agent. Scripted with two
            turns: the decision to delegate, and a final answer produced
            after the child's result comes back.
        child_provider: Provider for the child agent. Scripted with the
            child's answer to its bounded question.

    Returns:
        The completed `DelegationTask` and the parent's final answer,
        produced after control returned to it.
    """
    goal = "Write a one-line release note for today's deploy."
    task = DelegationTask(task_id="subagent-1", from_agent="parent", to_agent="child", payload="")

    delegate_reply = parent_provider.complete(
        [Message.user(goal)],
        system=(
            "You are the parent agent. If you need a fact you don't have, delegate a narrow "
            "question to a child subagent, then write the final answer yourself once it replies. "
            "Reply with a QUESTION: line for the child."
        ),
    ).content
    question_line = next(line for line in delegate_reply.splitlines() if line.upper().startswith("QUESTION:"))
    question = question_line.split(":", 1)[1].strip()
    task.payload = question
    task.transition("in_progress", "delegated to child subagent")

    child_answer = child_provider.complete([Message.user(question)], system="Answer the question in one clause.").content
    task.transition("completed", "child returned an answer, control returns to parent")

    final_answer = parent_provider.complete(
        [Message.user(f"The child subagent answered: {child_answer}\nNow write the final release note.")],
        system="You are the parent agent, continuing after your subagent's answer.",
    ).content
    return task, final_answer
