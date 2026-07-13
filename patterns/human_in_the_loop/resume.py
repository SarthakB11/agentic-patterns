"""Sub-module: interrupt-and-resume (durable pause).

The run state at the gate is captured in a plain, JSON-serializable
`GateState` so the process can exit entirely and a decision can arrive
through a completely separate call, possibly seconds or days later. This
is what separates a real gate from a blocking `input()` prompt: nothing
about `GateState` depends on a live call stack, a thread, or an open
connection.

A pending decision that never arrives, or arrives after its deadline, must
still fail closed. `resume_gate` checks the deadline before applying any
decision and refuses late ones instead of running a stale approval.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from agentic_patterns import ToolCall, ToolRegistry

from patterns.human_in_the_loop.fake_tools import build_refund_registry
from patterns.human_in_the_loop.gate import (
    AuditLog,
    AuditRecord,
    Decision,
    GateOutcome,
    ReviewRequest,
    ScriptedDecisionSource,
    UnauthorizedDecisionError,
    counting_clock,
    run_gate,
)


@dataclass
class GateState:
    """Serializable snapshot of a suspended gate.

    Attributes:
        request: The review request the gate suspended on.
        requested_at: Clock reading when review was first requested.
        deadline: Clock reading after which a decision is refused, or None
            for no timeout.
    """

    request: ReviewRequest
    requested_at: float
    deadline: float | None

    def to_dict(self) -> dict:
        """Serialize to a plain dict of JSON-safe values."""
        return {
            "request": {
                "id": self.request.id,
                "context": self.request.context,
                "action": {
                    "id": self.request.action.id,
                    "name": self.request.action.name,
                    "arguments": self.request.action.arguments,
                },
            },
            "requested_at": self.requested_at,
            "deadline": self.deadline,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GateState:
        """Reconstruct a `GateState` from `to_dict`'s output.

        This is the "second call" boundary: a real system would persist
        the dict (database row, queue message, file) after `to_dict()` and
        rebuild it here in a process that may not be the one that
        suspended.
        """
        action_data = data["request"]["action"]
        action = ToolCall(id=action_data["id"], name=action_data["name"], arguments=action_data["arguments"])
        request = ReviewRequest(id=data["request"]["id"], action=action, context=data["request"]["context"])
        return cls(request=request, requested_at=data["requested_at"], deadline=data["deadline"])


class GateExpiredError(UnauthorizedDecisionError):
    """Raised when a decision arrives after the gate's deadline has passed."""


def suspend_at_gate(
    request: ReviewRequest,
    *,
    timeout_seconds: float | None = None,
    clock: Callable[[], float] = time.time,
) -> GateState:
    """Checkpoint a review request into a serializable, suspended state.

    Args:
        request: The proposed action to suspend on.
        timeout_seconds: If given, a decision applied more than this many
            seconds after `requested_at` is refused. None means no timeout.
        clock: Timestamp source.
    """
    requested_at = clock()
    deadline = requested_at + timeout_seconds if timeout_seconds is not None else None
    return GateState(request=request, requested_at=requested_at, deadline=deadline)


def resume_gate(
    state: GateState,
    decision: Decision,
    registry: ToolRegistry,
    audit_log: AuditLog,
    *,
    clock: Callable[[], float] = time.time,
) -> GateOutcome:
    """Apply a decision to a previously suspended gate.

    Args:
        state: The suspended state, typically reconstructed via
            `GateState.from_dict` in a process separate from the one that
            called `suspend_at_gate`.
        decision: The decision to apply.
        registry: Where the action executes on approval or edit.
        audit_log: Append-only log to record the outcome to.
        clock: Timestamp source for when the decision is being applied.

    Raises:
        GateExpiredError: If `clock()` is past `state.deadline`. The
            decision is not applied and no side effect occurs, even if the
            decision itself was "approve".
        UnauthorizedDecisionError: Propagated from `run_gate` for a
            missing, malformed, or unrecognized decision.
    """
    now = clock()
    if state.deadline is not None and now > state.deadline:
        action = state.request.action
        late_by = now - state.deadline
        audit_log.append(
            AuditRecord(
                state.request.id, action.name, action.arguments, "expired", None,
                "timeout_policy", f"decision arrived {late_by:.1f}s past the deadline, auto-denied",
                state.requested_at, now,
            )
        )
        raise GateExpiredError(
            f"no side effect: decision for request {state.request.id!r} arrived past its deadline"
        )

    decision_source = ScriptedDecisionSource([decision])
    return run_gate(
        state.request, registry, decision_source, audit_log,
        clock=clock, requested_at=state.requested_at,
    )


@dataclass
class ResumeDemoResult:
    """Outcome of comparing a suspend/resume run against an uninterrupted one.

    Attributes:
        resumed_outcome: Outcome from the suspend, serialize, reconstruct,
            resume path.
        uninterrupted_outcome: Outcome from calling `run_gate` directly on
            the same request and decision, with no suspension at all.
        resumed_ledger: Refund ledger after the resumed path ran.
        uninterrupted_ledger: Refund ledger after the uninterrupted path ran.
    """

    resumed_outcome: GateOutcome
    uninterrupted_outcome: GateOutcome
    resumed_ledger: list
    uninterrupted_ledger: list


def run_resume_demo() -> ResumeDemoResult:
    """Suspend a gate, serialize it, reconstruct it, and resume with a decision.

    Confirms the resumed path reaches the same result as never suspending
    at all: durability changes when the decision is applied, not what it
    does once applied.
    """
    action = ToolCall(
        id="call_1", name="send_refund",
        arguments={"customer_id": "c-3310", "amount_usd": 120.00, "reason": "warranty claim approved"},
    )
    request = ReviewRequest(id="req-warranty", action=action, context="warranty claim, needs sign-off")
    decision = Decision(kind="approve", reviewer="ops-lead-dana", reason="warranty verified against serial number")

    registry_resumed, ledger_resumed = build_refund_registry()
    audit_resumed = AuditLog()
    state = suspend_at_gate(request, timeout_seconds=3600.0, clock=counting_clock(start=1000.0))
    serialized = state.to_dict()  # would cross a process boundary here
    restored = GateState.from_dict(serialized)
    resumed_outcome = resume_gate(
        restored, decision, registry_resumed, audit_resumed, clock=counting_clock(start=1200.0)
    )

    registry_direct, ledger_direct = build_refund_registry()
    audit_direct = AuditLog()
    direct_source = ScriptedDecisionSource([decision])
    uninterrupted_outcome = run_gate(
        request, registry_direct, direct_source, audit_direct, clock=counting_clock(start=1000.0)
    )

    return ResumeDemoResult(
        resumed_outcome=resumed_outcome,
        uninterrupted_outcome=uninterrupted_outcome,
        resumed_ledger=ledger_resumed,
        uninterrupted_ledger=ledger_direct,
    )


def run_expired_demo() -> tuple[list, AuditLog]:
    """A decision that arrives after the deadline is refused, not executed."""
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    action = ToolCall(
        id="call_1", name="send_refund",
        arguments={"customer_id": "c-4410", "amount_usd": 80.00, "reason": "goodwill credit"},
    )
    request = ReviewRequest(id="req-slow", action=action, context="awaiting reviewer, short deadline")

    state = suspend_at_gate(request, timeout_seconds=5.0, clock=counting_clock(start=2000.0))
    decision = Decision(kind="approve", reviewer="ops-lead-dana", reason="approved, just slow to respond")
    try:
        resume_gate(state, decision, registry, audit_log, clock=counting_clock(start=2500.0))
    except GateExpiredError:
        pass
    return ledger, audit_log
