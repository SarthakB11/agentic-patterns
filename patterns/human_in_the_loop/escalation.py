"""Sub-module: escalation on uncertainty, synchronous and asynchronous.

The gate fires dynamically based on a trigger the agent itself reports,
here a confidence score on the proposed action, rather than a static rule
on the action's shape. A high-confidence proposal auto-approves; a
low-confidence one escalates to a reviewer. Only genuine exceptions reach
a human, the way an anomaly detector or an explicit uncertainty flag would
in a production system.

Self-reported confidence is one noisy trigger, not a crisp oracle: Turan,
"Oversight Has a Capacity" (arXiv:2606.08919), finds reviewers themselves
agree on what is risky at only Fleiss kappa 0.52 on 125 adversarially
weighted actions, so there is no single ground-truth risk label to
threshold confidence against in the first place, on top of self-reported
model confidence being a separately known-poorly-calibrated signal. The
module below is still a fine teaching example of a dynamic trigger; treat
the threshold as illustrative, not as a claim that 0.70 is a validated cut.

Two latency shapes are demonstrated. `run_escalation_demo` is synchronous:
the caller waits for the low-confidence request to resolve before moving
on. `run_async_escalation_demo` is asynchronous: several proposals are
processed in one pass, high-confidence ones resolve immediately, and
low-confidence ones are queued without blocking the others; the queued
ones are resolved afterward, in a second pass, once decisions arrive.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentic_patterns import Completion, Message, Provider, ToolCall, get_provider, scripted_tool_call

from patterns.human_in_the_loop.fake_tools import build_refund_registry
from patterns.human_in_the_loop.gate import (
    AuditLog,
    AuditRecord,
    Decision,
    GateOutcome,
    ReviewRequest,
    ScriptedDecisionSource,
    counting_clock,
    run_gate,
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.70


def extract_confidence(completion: Completion) -> float | None:
    """Pull a `confidence` value out of a completion's raw payload, if present."""
    if isinstance(completion.raw, dict):
        value = completion.raw.get("confidence")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def run_confidence_gate(
    completion: Completion,
    request: ReviewRequest,
    registry,
    decision_source,
    audit_log: AuditLog,
    *,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    clock: Callable[[], float] = time.time,
) -> GateOutcome:
    """Auto-approve a high-confidence proposal, or escalate a low-confidence one.

    Args:
        completion: The model's completion that produced `request.action`,
            inspected for a `confidence` value.
        request: The proposed action under consideration.
        registry: Where the action executes.
        decision_source: Where a human decision comes from, if escalated.
        audit_log: Append-only log, written to either way.
        threshold: Minimum confidence that auto-approves without review.
        clock: Timestamp source.
    """
    confidence = extract_confidence(completion)
    if confidence is not None and confidence >= threshold:
        now = clock()
        result = registry.execute(request.action)
        audit_log.append(
            AuditRecord(
                request.id, request.action.name, request.action.arguments,
                "auto_approved_high_confidence", request.action.arguments,
                "confidence_policy", f"confidence {confidence:.2f} >= threshold {threshold:.2f}", now, now,
            )
        )
        return GateOutcome(kind="executed", tool_result=result, final_arguments=request.action.arguments)

    return run_gate(request, registry, decision_source, audit_log, clock=clock)


def run_escalation_demo(provider: Provider | None = None) -> tuple[GateOutcome, GateOutcome, list]:
    """A confident proposal auto-approves; an unsure one escalates for review."""
    high_confidence = scripted_tool_call(
        "send_refund",
        {"customer_id": "c-6600", "amount_usd": 40.00, "reason": "wrong item shipped, clear-cut case"},
        call_id="call_1",
    )
    high_confidence.raw = {"confidence": 0.93}
    low_confidence = scripted_tool_call(
        "send_refund",
        {"customer_id": "c-6601", "amount_usd": 250.00, "reason": "possible duplicate charge, log is ambiguous"},
        call_id="call_2",
    )
    low_confidence.raw = {"confidence": 0.35}

    if provider is None:
        provider = get_provider(script=[high_confidence, low_confidence])

    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    decision_source = ScriptedDecisionSource(
        [Decision(kind="approve", reviewer="ops-lead-dana", reason="verified as a genuine duplicate charge")]
    )
    clock = counting_clock()

    proposal_1 = provider.complete([Message.user("Wrong item was shipped to c-6600; issue a refund.")])
    request_1 = ReviewRequest(
        id="req-1", action=proposal_1.tool_calls[0], context="model-reported extraction confidence"
    )
    outcome_1 = run_confidence_gate(proposal_1, request_1, registry, decision_source, audit_log, clock=clock)

    proposal_2 = provider.complete([Message.user("Possible duplicate charge for c-6601; refund if confirmed.")])
    request_2 = ReviewRequest(
        id="req-2", action=proposal_2.tool_calls[0], context="model-reported extraction confidence"
    )
    outcome_2 = run_confidence_gate(proposal_2, request_2, registry, decision_source, audit_log, clock=clock)

    return outcome_1, outcome_2, ledger


@dataclass
class AsyncEscalationResult:
    """Outcome of the asynchronous-escalation demo.

    Attributes:
        completed_immediately: Task ids that auto-approved and resolved
            without ever waiting on a reviewer.
        queued_for_review: Task ids that escalated and were queued instead
            of blocking the pass.
        resolved_after_wait: Task ids resolved in the follow-up pass, once
            decisions arrived.
        ledger: The refund ledger after both passes.
    """

    completed_immediately: list[str] = field(default_factory=list)
    queued_for_review: list[str] = field(default_factory=list)
    resolved_after_wait: list[str] = field(default_factory=list)
    ledger: list[dict[str, Any]] = field(default_factory=list)


def run_async_escalation_demo() -> AsyncEscalationResult:
    """Process a batch of proposals without blocking on any single escalation.

    High-confidence tasks resolve in the same pass they were proposed in.
    A low-confidence task is queued and the pass moves on immediately,
    modeling an agent that keeps working while one review is outstanding.
    A second pass resolves whatever was queued, once a decision exists.
    """
    tasks: list[tuple[str, ToolCall, float]] = [
        (
            "task-a",
            ToolCall(id="task-a", name="send_refund", arguments={
                "customer_id": "c-7001", "amount_usd": 25.00, "reason": "late delivery credit",
            }),
            0.95,
        ),
        (
            "task-b",
            ToolCall(id="task-b", name="send_refund", arguments={
                "customer_id": "c-7002", "amount_usd": 310.00, "reason": "damaged goods, photo evidence unclear",
            }),
            0.30,
        ),
        (
            "task-c",
            ToolCall(id="task-c", name="send_refund", arguments={
                "customer_id": "c-7003", "amount_usd": 18.00, "reason": "coupon applied twice",
            }),
            0.88,
        ),
    ]

    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    clock = counting_clock()
    result = AsyncEscalationResult(ledger=ledger)
    pending: dict[str, ReviewRequest] = {}

    for task_id, action, confidence in tasks:
        request = ReviewRequest(id=task_id, action=action, context=f"model-reported confidence {confidence:.2f}")
        if confidence >= DEFAULT_CONFIDENCE_THRESHOLD:
            now = clock()
            registry.execute(action)
            audit_log.append(
                AuditRecord(
                    task_id, action.name, action.arguments, "auto_approved_high_confidence", action.arguments,
                    "confidence_policy", f"confidence {confidence:.2f} >= threshold", now, now,
                )
            )
            result.completed_immediately.append(task_id)
        else:
            # Escalate without blocking: the pass continues to the next
            # task instead of waiting here for a reviewer.
            pending[task_id] = request
            result.queued_for_review.append(task_id)

    # A decision for the queued task arrives later; resolve it in a
    # follow-up pass, well after task-a and task-c already completed.
    decision_source = ScriptedDecisionSource(
        {"task-b": Decision(kind="approve", reviewer="ops-lead-dana", reason="photo confirmed after follow-up")}
    )
    for task_id, request in pending.items():
        run_gate(request, registry, decision_source, audit_log, clock=clock)
        result.resolved_after_wait.append(task_id)

    return result
