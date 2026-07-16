"""Sub-module: plan review / co-planning.

Rather than gating every step of a multi-step task, the human reviews the
whole plan once, before any step executes. This is cheaper than a gate per
action when a task decomposes into several steps that are only sensible
together, and it lets a reviewer catch a wrong overall approach instead of
approving each step in isolation without seeing where it leads.

`run_plan_review` guarantees no step in the plan executes until the plan
decision is applied: rejecting or a malformed decision leaves every step
unexecuted, and editing replaces the whole step list before anything runs.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

from agentic_patterns import ToolCall, ToolRegistry
from patterns.human_in_the_loop.fake_tools import build_support_ops_registry
from patterns.human_in_the_loop.gate import AuditLog, AuditRecord, GateOutcome, UnauthorizedDecisionError

_VALID_PLAN_DECISION_KINDS = frozenset({"approve", "edit", "reject"})


class PlanDecision:
    """A reviewer's ruling on a whole plan.

    Attributes:
        kind: One of "approve", "edit", "reject".
        reviewer: Identity of the reviewer, for the audit record.
        reason: Explanation for the decision.
        edited_steps: Replacement step list for "edit". Ignored otherwise.
    """

    def __init__(
        self,
        kind: str,
        *,
        reviewer: str = "unspecified",
        reason: str = "",
        edited_steps: list[ToolCall] | None = None,
    ) -> None:
        self.kind = kind
        self.reviewer = reviewer
        self.reason = reason
        self.edited_steps = edited_steps


def run_plan_review(
    steps: list[ToolCall],
    registry: ToolRegistry,
    decision: PlanDecision,
    audit_log: AuditLog,
    *,
    plan_id: str,
    clock: Callable[[], float] = time.time,
) -> list[GateOutcome]:
    """Review a whole plan once, then run every step with no further gate.

    Args:
        steps: The proposed steps, in execution order.
        registry: Where each step executes.
        decision: The reviewer's ruling on the plan as a whole.
        audit_log: Append-only log; one entry is written per executed step,
            plus a single entry for a rejected or malformed plan.
        plan_id: Identifier for this plan, used as the audit record's
            request id for every step.
        clock: Timestamp source.

    Returns:
        One `GateOutcome` per step that ran, in order.

    Raises:
        UnauthorizedDecisionError: If the plan is rejected, or the decision
            is edit with no replacement steps, or an unrecognized kind. No
            step executes in any of these cases.
    """
    requested_at = clock()

    if decision.kind not in _VALID_PLAN_DECISION_KINDS:
        audit_log.append(
            AuditRecord(
                plan_id, "plan", {"step_count": len(steps)}, "invalid", None,
                getattr(decision, "reviewer", "unknown"), "missing or unrecognized plan decision kind",
                requested_at, clock(),
            )
        )
        raise UnauthorizedDecisionError(f"no steps executed: plan {plan_id!r} decision was unrecognized")

    if decision.kind == "reject":
        audit_log.append(
            AuditRecord(
                plan_id, "plan", {"step_count": len(steps)}, "reject", None,
                decision.reviewer, decision.reason or "plan rejected with no reason given",
                requested_at, clock(),
            )
        )
        raise UnauthorizedDecisionError(f"no steps executed: plan {plan_id!r} was rejected")

    if decision.kind == "edit":
        if not decision.edited_steps:
            audit_log.append(
                AuditRecord(
                    plan_id, "plan", {"step_count": len(steps)}, "invalid_edit", None,
                    decision.reviewer, "edit decision carried no replacement steps",
                    requested_at, clock(),
                )
            )
            raise UnauthorizedDecisionError(f"no steps executed: plan {plan_id!r} edit carried no steps")
        final_steps = decision.edited_steps
    else:
        final_steps = steps

    outcomes: list[GateOutcome] = []
    for step in final_steps:
        result = registry.execute(step)
        audit_log.append(
            AuditRecord(
                plan_id, step.name, step.arguments, decision.kind, step.arguments,
                decision.reviewer, decision.reason, requested_at, clock(),
            )
        )
        outcomes.append(GateOutcome(kind="executed", tool_result=result, final_arguments=step.arguments))
    return outcomes


def run_plan_approve_demo() -> tuple[list[GateOutcome], list, AuditLog]:
    """A two-step plan (cancel, then prorated refund) is approved as proposed."""
    registry, ledger = build_support_ops_registry()
    audit_log = AuditLog()
    steps = [
        ToolCall(id="step-1", name="cancel_subscription", arguments={
            "customer_id": "c-8801", "reason": "customer requested cancellation",
        }),
        ToolCall(id="step-2", name="send_refund", arguments={
            "customer_id": "c-8801", "amount_usd": 14.30, "reason": "prorated refund for unused days",
        }),
    ]
    decision = PlanDecision(
        kind="approve", reviewer="ops-lead-dana",
        reason="cancellation and prorated amount both check out against the billing cycle",
    )
    outcomes = run_plan_review(steps, registry, decision, audit_log, plan_id="plan-8801")
    return outcomes, ledger, audit_log


def run_plan_edit_demo() -> tuple[list[GateOutcome], list, AuditLog]:
    """The reviewer corrects a wrong step before any step in the plan executes."""
    registry, ledger = build_support_ops_registry()
    audit_log = AuditLog()
    proposed_steps = [
        ToolCall(id="step-1", name="cancel_subscription", arguments={
            "customer_id": "c-9910", "reason": "customer requested cancellation",
        }),
        ToolCall(id="step-2", name="send_refund", arguments={
            "customer_id": "c-9910", "amount_usd": 40.00, "reason": "prorated refund for unused days",
        }),
    ]
    edited_steps = [
        proposed_steps[0],
        ToolCall(id="step-2", name="send_refund", arguments={
            "customer_id": "c-9910", "amount_usd": 22.50,
            "reason": "prorated refund recalculated from the actual billing cycle",
        }),
    ]
    decision = PlanDecision(
        kind="edit", reviewer="ops-lead-dana",
        reason="the proposed $40.00 assumed a full month; the customer is 12 days into the cycle",
        edited_steps=edited_steps,
    )
    outcomes = run_plan_review(proposed_steps, registry, decision, audit_log, plan_id="plan-9910")
    return outcomes, ledger, audit_log


def run_plan_reject_demo() -> tuple[list, AuditLog]:
    """Rejecting the plan leaves every step, including the cancellation, unexecuted."""
    registry, ledger = build_support_ops_registry()
    audit_log = AuditLog()
    steps = [
        ToolCall(id="step-1", name="cancel_subscription", arguments={
            "customer_id": "c-2231", "reason": "customer requested cancellation",
        }),
        ToolCall(id="step-2", name="send_refund", arguments={
            "customer_id": "c-2231", "amount_usd": 300.00, "reason": "full refund for the current term",
        }),
    ]
    decision = PlanDecision(
        kind="reject", reviewer="ops-lead-dana",
        reason="customer is 2 days from renewal; cancel without a refund per the no-refund-near-renewal policy",
    )
    with contextlib.suppress(UnauthorizedDecisionError):
        run_plan_review(steps, registry, decision, audit_log, plan_id="plan-2231")
    return ledger, audit_log
