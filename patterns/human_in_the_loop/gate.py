"""Reusable engine for the human-in-the-loop approval gate.

This module holds the pattern's mechanics, kept separate from any one
demo scenario: the review request and decision shapes, the append-only
audit log, the pluggable decision-source interface, and `run_gate`, the
function every variant module builds on. Demo scenarios and scripted
conversations live in the sibling modules; this file has no knowledge of
any particular tool, task, or provider.

The gate never executes a proposed action itself except by calling
`ToolRegistry.execute`, so approving versus rejecting is observable
through the fake tool's own state (see `fake_tools.py`).

Decision vocabulary, kept small and explicit per the design brief:

- approve: run the action exactly as proposed.
- edit: run the action with reviewer-supplied replacement arguments.
- reject: do not run the action; return the reviewer's reason as feedback.
- respond: do not run the action; return a reviewer-supplied value as the
  tool result, used when the gate is fetching information rather than
  policing a side effect.

Fail-closed default: a missing, malformed, or unrecognized decision never
executes the action. `run_gate` records an audit entry and raises
`UnauthorizedDecisionError` instead, so a caller cannot accidentally treat
a blocked action as if it had a result.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agentic_patterns import ToolCall, ToolRegistry

_VALID_DECISION_KINDS = frozenset({"approve", "edit", "reject", "respond"})


@dataclass
class ReviewRequest:
    """A proposed action put in front of a decision source for review.

    Attributes:
        id: Identifier for this request, stable across suspend/resume and
            used to map a batch of decisions back to the right action.
        action: The proposed tool call, produced by the model but not yet
            executed.
        context: Human-readable explanation of why review was triggered,
            shown to the reviewer alongside the action.
    """

    id: str
    action: ToolCall
    context: str


@dataclass
class Decision:
    """A reviewer's ruling on one `ReviewRequest`.

    Attributes:
        kind: One of "approve", "edit", "reject", "respond". Any other
            value is treated as malformed by `run_gate`.
        reviewer: Identity of whoever or whatever made the decision, for
            the audit record.
        reason: Free-form explanation. Required in spirit for "reject";
            optional elsewhere.
        arguments: Replacement arguments for "edit". Ignored otherwise.
        value: The value to return as the tool result for "respond".
            Ignored otherwise.
    """

    kind: str
    reviewer: str = "unspecified"
    reason: str = ""
    arguments: dict[str, Any] | None = None
    value: str | None = None


@dataclass
class AuditRecord:
    """One append-only entry in the audit log.

    Attributes:
        request_id: The `ReviewRequest.id` this entry resolves.
        action_name: Name of the proposed tool call.
        proposed_arguments: Arguments the model originally proposed.
        decision_kind: The decision kind that was recorded, or a sentinel
            like "blocked_by_policy" / "invalid" for a failed gate.
        final_arguments: Arguments the action actually ran with, or None
            if it did not run.
        reviewer: Reviewer identity.
        reason: Reviewer's reason or feedback text.
        requested_at: Clock reading when review was requested.
        decided_at: Clock reading when the decision was recorded.
    """

    request_id: str
    action_name: str
    proposed_arguments: dict[str, Any]
    decision_kind: str
    final_arguments: dict[str, Any] | None
    reviewer: str
    reason: str
    requested_at: float
    decided_at: float

    @property
    def latency(self) -> float:
        """Seconds between the review request and the recorded decision."""
        return self.decided_at - self.requested_at


class AuditLog:
    """An append-only log of `AuditRecord` entries.

    Records are never mutated or removed once appended: the log's value is
    compliance and later learning, so history must stay intact even when a
    gate blocks an action.
    """

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def append(self, record: AuditRecord) -> None:
        """Add a record. This is the only mutating operation the log allows."""
        self._records.append(record)

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        """All records in append order, as an immutable view."""
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)


class DecisionSource(Protocol):
    """Interface a human decision comes through.

    A production CLI implements this by prompting a real person with
    `input()`; tests and demos implement it with a scripted queue. Gate
    logic never knows which one it is talking to.
    """

    def decide(self, request: ReviewRequest) -> Decision:
        """Return the decision for a single review request."""
        ...


class DecisionSourceExhausted(RuntimeError):
    """Raised when a `ScriptedDecisionSource` receives more requests than scripted."""


class ScriptedDecisionSource:
    """A `DecisionSource` that replays a fixed script of decisions.

    Accepts either a plain sequence, consumed in call order (for a single
    linear flow), or a mapping from request id to `Decision` (for batched
    review, where requests may not resolve in submission order).
    """

    def __init__(self, decisions: Sequence[Decision] | Mapping[str, Decision]) -> None:
        if isinstance(decisions, Mapping):
            self._by_id: dict[str, Decision] | None = dict(decisions)
            self._queue: list[Decision] | None = None
        else:
            self._by_id = None
            self._queue = list(decisions)
        self.decisions_served: list[Decision] = []

    def decide(self, request: ReviewRequest) -> Decision:
        if self._by_id is not None:
            try:
                decision = self._by_id[request.id]
            except KeyError:
                raise DecisionSourceExhausted(
                    f"no scripted decision for request id {request.id!r}"
                ) from None
        else:
            assert self._queue is not None
            if not self._queue:
                raise DecisionSourceExhausted("scripted decision source exhausted")
            decision = self._queue.pop(0)
        self.decisions_served.append(decision)
        return decision


class UnauthorizedDecisionError(RuntimeError):
    """Raised by `run_gate` when a proposed action must not execute.

    Covers a deterministic policy denial, a missing or unrecognized
    decision kind, and an edit decision with no replacement arguments. In
    every case an audit record is appended before this is raised, so the
    attempt is not lost even though the action never ran.
    """


@dataclass
class GateOutcome:
    """What happened after a decision was applied to a review request.

    Attributes:
        kind: One of "executed" (approve or edit ran the action),
            "rejected" (no side effect, feedback returned), or "responded"
            (no side effect, a supplied value returned as the tool result).
        tool_result: The observation to feed back into the agent loop:
            the tool's return value on "executed", the reviewer's feedback
            on "rejected", or the reviewer's supplied value on "responded".
        final_arguments: Arguments the action ran with, or None if it did
            not run.
    """

    kind: str
    tool_result: str
    final_arguments: dict[str, Any] | None = None


def run_gate(
    request: ReviewRequest,
    registry: ToolRegistry,
    decision_source: DecisionSource,
    audit_log: AuditLog,
    *,
    policy_guard: Callable[[ToolCall], str | None] | None = None,
    clock: Callable[[], float] = time.time,
    requested_at: float | None = None,
) -> GateOutcome:
    """Run one proposed action through the approval gate.

    Mirrors the canonical control flow: a deterministic policy check runs
    first and can block the action without ever asking a reviewer (the
    hook-versus-permission split: an approval prompt is not authorization
    by itself). If policy allows it, the decision source is asked, and the
    decision is applied. No side effect occurs before a decision is
    reached, and none occurs at all if the decision is missing, malformed,
    or of an unrecognized kind.

    Args:
        request: The proposed action and why it is under review.
        registry: Where the action actually executes on approval or edit.
        decision_source: Where the human decision comes from.
        audit_log: Append-only log every outcome is recorded to, including
            blocked attempts.
        policy_guard: Optional deterministic check run before the human
            decision is even requested. Returns None to allow, or a denial
            reason string to block. Models a `PreToolUse`-style hook that
            is not asking anyone's permission, just enforcing a rule.
        clock: Timestamp source, injectable for deterministic tests and
            demos. Defaults to wall-clock time.
        requested_at: Override for when review was first requested. Used by
            `resume.py` to preserve the original suspend time across a
            durable pause, instead of the moment the gate happens to
            resume. Defaults to `clock()` when not given.

    Returns:
        The outcome: what ran (if anything) and what the agent loop should
        see as the tool's observation.

    Raises:
        UnauthorizedDecisionError: If policy denies the action, or the
            decision is missing, unrecognized, or an edit with no
            replacement arguments. No side effect occurs in any of these
            cases.
    """
    action = request.action
    requested_at = requested_at if requested_at is not None else clock()

    def record(decision_kind: str, final_arguments: dict[str, Any] | None, reviewer: str, reason: str) -> None:
        audit_log.append(
            AuditRecord(
                request.id, action.name, action.arguments, decision_kind, final_arguments,
                reviewer, reason, requested_at, clock(),
            )
        )

    if policy_guard is not None:
        denial = policy_guard(action)
        if denial is not None:
            record("blocked_by_policy", None, "policy", denial)
            raise UnauthorizedDecisionError(
                f"policy denied action {action.name!r} for request {request.id!r}: {denial}"
            )

    decision = decision_source.decide(request)

    if decision is None or decision.kind not in _VALID_DECISION_KINDS:
        record("invalid", None, getattr(decision, "reviewer", "unknown"), "missing or unrecognized decision kind")
        raise UnauthorizedDecisionError(
            f"no side effect: decision for request {request.id!r} was missing or unrecognized"
        )

    if decision.kind == "approve":
        result = registry.execute(action)
        record("approve", action.arguments, decision.reviewer, decision.reason)
        return GateOutcome(kind="executed", tool_result=result, final_arguments=action.arguments)

    if decision.kind == "edit":
        if not decision.arguments:
            record("invalid_edit", None, decision.reviewer, "edit decision carried no replacement arguments")
            raise UnauthorizedDecisionError(
                f"no side effect: edit decision for request {request.id!r} carried no arguments"
            )
        edited = ToolCall(id=action.id, name=action.name, arguments=decision.arguments)
        result = registry.execute(edited)
        record("edit", decision.arguments, decision.reviewer, decision.reason)
        return GateOutcome(kind="executed", tool_result=result, final_arguments=decision.arguments)

    if decision.kind == "reject":
        feedback = decision.reason or "rejected with no reason given"
        record("reject", None, decision.reviewer, feedback)
        return GateOutcome(kind="rejected", tool_result=feedback, final_arguments=None)

    # decision.kind == "respond"
    value = decision.value or ""
    record("respond", None, decision.reviewer, decision.reason)
    return GateOutcome(kind="responded", tool_result=value, final_arguments=None)


def counting_clock(start: float = 1000.0, step: float = 1.0) -> Callable[[], float]:
    """Build a deterministic clock that advances by `step` on every call.

    Used by demos and tests so timestamps and latencies are reproducible
    instead of depending on wall-clock time.
    """
    state = {"t": start - step}

    def clock() -> float:
        state["t"] += step
        return state["t"]

    return clock
