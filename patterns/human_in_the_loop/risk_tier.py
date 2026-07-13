"""Sub-module: risk-tiered gating (conditional interrupt).

A predicate decides whether a proposed action needs review at all. Most
calls are low-risk and auto-approve with no reviewer ever in the loop;
only calls that match a rule (amount above a threshold, in this scenario)
interrupt. This is what keeps reviewer load proportional to risk instead
of firing on every call.

A second demo here demonstrates the opposite failure mode, per Turan,
"Oversight Has a Capacity" (arXiv:2606.08919): gating everything does not
automatically make a system safer. A reviewer who has to clear a flood of
low-value requests learns to rubber-stamp, and a rubber-stamping reviewer
approves a malicious high-value action along with everything else. A
deterministic policy backstop, evaluated regardless of what the reviewer
decided, is what actually stops it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from agentic_patterns import ToolCall

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

RiskPredicate = Callable[[ToolCall], bool]


def high_value_refund(action: ToolCall, *, threshold: float = 200.0) -> bool:
    """Risk predicate: a refund is high risk once its amount clears `threshold`."""
    return float(action.arguments.get("amount_usd", 0.0)) > threshold


def run_tiered_gate(
    request: ReviewRequest,
    registry,
    decision_source,
    audit_log: AuditLog,
    *,
    is_high_risk: RiskPredicate,
    policy_guard: Callable[[ToolCall], str | None] | None = None,
    clock: Callable[[], float] = time.time,
) -> GateOutcome:
    """Auto-approve a low-risk action, or fall through to a full gate review.

    Args:
        request: The proposed action under consideration.
        registry: Where the action executes.
        decision_source: Where a human decision comes from, if one is needed.
        audit_log: Append-only log, written to either way.
        is_high_risk: Predicate deciding whether review is required.
        policy_guard: Optional deterministic check, applied even to a
            low-risk auto-approval, not only to reviewed actions.
        clock: Timestamp source.
    """
    action = request.action
    now = clock()

    if policy_guard is not None:
        denial = policy_guard(action)
        if denial is not None:
            audit_log.append(
                AuditRecord(
                    request.id, action.name, action.arguments, "blocked_by_policy", None,
                    "policy", denial, now, now,
                )
            )
            raise UnauthorizedDecisionError(f"policy denied action {action.name!r}: {denial}")

    if not is_high_risk(action):
        result = registry.execute(action)
        audit_log.append(
            AuditRecord(
                request.id, action.name, action.arguments, "auto_approved_low_risk", action.arguments,
                "risk_tier_policy", "below risk threshold, no review required", now, now,
            )
        )
        return GateOutcome(kind="executed", tool_result=result, final_arguments=action.arguments)

    return run_gate(request, registry, decision_source, audit_log, clock=clock)


@dataclass
class TierDemoResult:
    """Outcome of the low/high risk-tiering demo.

    Attributes:
        low_risk_outcome: Outcome of the auto-approved low-risk request.
        high_risk_outcome: Outcome of the reviewed high-risk request.
        reviewer_was_asked: Whether the decision source was ever consulted
            for the low-risk request (should be False).
        ledger: The refund ledger after both requests ran.
    """

    low_risk_outcome: GateOutcome
    high_risk_outcome: GateOutcome
    reviewer_was_asked: bool
    ledger: list


def run_risk_tier_demo() -> TierDemoResult:
    """A $15 refund auto-approves; a $900 refund is gated for review."""
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    decision_source = ScriptedDecisionSource(
        [Decision(kind="approve", reviewer="ops-lead-dana", reason="confirmed against the order log")]
    )
    clock = counting_clock()

    low_risk = ReviewRequest(
        id="req-low",
        action=ToolCall(id="call_1", name="send_refund", arguments={
            "customer_id": "c-1001", "amount_usd": 15.00, "reason": "shipping label reprint fee",
        }),
        context="small refund, well under the review threshold",
    )
    low_outcome = run_tiered_gate(
        low_risk, registry, decision_source, audit_log, is_high_risk=high_value_refund, clock=clock
    )
    reviewer_asked_before_high_risk = len(decision_source.decisions_served) > 0

    high_risk = ReviewRequest(
        id="req-high",
        action=ToolCall(id="call_2", name="send_refund", arguments={
            "customer_id": "c-1002", "amount_usd": 900.00, "reason": "damaged item, full replacement value",
        }),
        context="large refund, requires review",
    )
    high_outcome = run_tiered_gate(
        high_risk, registry, decision_source, audit_log, is_high_risk=high_value_refund, clock=clock
    )

    return TierDemoResult(
        low_risk_outcome=low_outcome,
        high_risk_outcome=high_outcome,
        reviewer_was_asked=reviewer_asked_before_high_risk,
        ledger=ledger,
    )


@dataclass
class FloodingDemoResult:
    """Outcome of the escalate-everything flooding demo.

    Attributes:
        rubber_stamp_ledger: Refund ledger after every request, including a
            malicious one, was gated and a fatigued reviewer approved all
            of them without a policy backstop.
        guarded_ledger: Refund ledger from the same flood of requests, this
            time with a deterministic policy cap in front of the gate.
        malicious_amount: The amount of the malicious request, for the
            demo to point at directly.
    """

    rubber_stamp_ledger: list
    guarded_ledger: list
    malicious_amount: float


def _policy_cap(action: ToolCall, *, cap: float = 2000.0) -> str | None:
    """Deterministic backstop: refuse any refund above `cap`, no exceptions."""
    amount = float(action.arguments.get("amount_usd", 0.0))
    if amount > cap:
        return f"amount ${amount:.2f} exceeds the hard policy cap of ${cap:.2f}"
    return None


def _flood_of_requests() -> list[ReviewRequest]:
    """Five routine refunds followed by one malicious, oversized one."""
    routine = [
        (f"req-{i}", f"c-{2000+i}", 30.0 + i, "routine return") for i in range(5)
    ]
    requests = [
        ReviewRequest(
            id=rid,
            action=ToolCall(id=rid, name="send_refund", arguments={
                "customer_id": cust, "amount_usd": amt, "reason": reason,
            }),
            context="routine refund",
        )
        for rid, cust, amt, reason in routine
    ]
    requests.append(
        ReviewRequest(
            id="req-malicious",
            action=ToolCall(id="req-malicious", name="send_refund", arguments={
                "customer_id": "c-9999", "amount_usd": 50000.00, "reason": "account credit adjustment",
            }),
            context="disguised as a routine refund, buried in the flood",
        )
    )
    return requests


def run_flooding_demo() -> FloodingDemoResult:
    """Show that gating everything, without a policy backstop, still fails.

    A reviewer who rubber-stamps six requests in a row (always "approve",
    never reading the amount) lets the malicious one through when every
    request is gated and nothing else stands in the way. Adding a
    deterministic policy cap ahead of the human decision stops it even
    though the same rubber-stamping reviewer is still in the loop.
    """
    always_high_risk: RiskPredicate = lambda _action: True  # noqa: E731 - demo predicate, kept inline for clarity

    # Pass 1: gate everything, no policy backstop, a rubber-stamp reviewer.
    registry_a, ledger_a = build_refund_registry()
    audit_a = AuditLog()
    rubber_stamp = ScriptedDecisionSource(
        [Decision(kind="approve", reviewer="fatigued-reviewer") for _ in range(6)]
    )
    clock_a = counting_clock()
    for request in _flood_of_requests():
        run_tiered_gate(
            request, registry_a, rubber_stamp, audit_a, is_high_risk=always_high_risk, clock=clock_a
        )

    # Pass 2: same flood, same rubber-stamping behavior, but a deterministic
    # policy cap runs ahead of the human decision on every request.
    registry_b, ledger_b = build_refund_registry()
    audit_b = AuditLog()
    rubber_stamp_2 = ScriptedDecisionSource(
        [Decision(kind="approve", reviewer="fatigued-reviewer") for _ in range(5)]
    )
    clock_b = counting_clock()
    for request in _flood_of_requests():
        try:
            run_tiered_gate(
                request, registry_b, rubber_stamp_2, audit_b,
                is_high_risk=always_high_risk, policy_guard=_policy_cap, clock=clock_b,
            )
        except UnauthorizedDecisionError:
            pass  # the malicious request is blocked before it ever reaches the reviewer

    return FloodingDemoResult(rubber_stamp_ledger=ledger_a, guarded_ledger=ledger_b, malicious_amount=50000.00)
