"""Sub-module: post-hoc review with override.

The action executes immediately, with no wait, and is queued for
asynchronous review afterward. This trades the immediacy synchronous
gating gives up for throughput, and it only suits actions that are cheap
enough or reversible enough to tolerate a wrong one slipping through
briefly. A reviewer who later inspects the entry can confirm it or issue
an override that rolls the effect back.

This is different from every other variant in this pattern: everywhere
else, no side effect happens before a decision. Here the side effect comes
first and the decision is what happens after, which is exactly the trade
worth naming, not hiding.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from agentic_patterns import ToolCall, ToolRegistry

from patterns.human_in_the_loop.fake_tools import build_refund_registry, build_reversal_registry
from patterns.human_in_the_loop.gate import AuditLog, AuditRecord, UnauthorizedDecisionError


@dataclass
class PostHocRecord:
    """One action that already executed and is awaiting sampled review.

    Attributes:
        id: Identifier for this record, used to apply a later decision.
        action: The action that ran.
        result: What the tool returned when it ran.
        executed_at: When it ran.
    """

    id: str
    action: ToolCall
    result: str
    executed_at: float


def execute_immediately(
    action: ToolCall,
    registry: ToolRegistry,
    audit_log: AuditLog,
    *,
    record_id: str,
    clock: Callable[[], float] = time.time,
) -> PostHocRecord:
    """Run an action right away and log it for review after the fact.

    Args:
        action: The action to run. No review request is built; nothing
            waits for a decision before this executes.
        registry: Where the action executes.
        audit_log: Append-only log, given an entry marking the action as
            executed but not yet reviewed.
        record_id: Identifier for the resulting `PostHocRecord`.
        clock: Timestamp source.
    """
    now = clock()
    result = registry.execute(action)
    audit_log.append(
        AuditRecord(
            record_id, action.name, action.arguments, "auto_executed_pending_review", action.arguments,
            "none_yet", "executed immediately, queued for post-hoc review", now, now,
        )
    )
    return PostHocRecord(id=record_id, action=action, result=result, executed_at=now)


class PostHocDecision:
    """A reviewer's ruling on an already-executed action.

    Attributes:
        kind: One of "confirm" (leave the effect as-is) or "rollback"
            (reverse it).
        reviewer: Reviewer identity, for the audit record.
        reason: Explanation for the decision.
    """

    def __init__(self, kind: str, *, reviewer: str = "unspecified", reason: str = "") -> None:
        self.kind = kind
        self.reviewer = reviewer
        self.reason = reason


def apply_post_hoc_review(
    record: PostHocRecord,
    decision: PostHocDecision,
    ledger: list,
    audit_log: AuditLog,
    *,
    clock: Callable[[], float] = time.time,
) -> str | None:
    """Apply a sampled reviewer's decision to an already-executed action.

    Args:
        record: The executed action under review.
        decision: The reviewer's ruling.
        ledger: The same ledger the original action wrote to, used to
            build the reversal tool for a "rollback".
        audit_log: Append-only log to record the review outcome to.
        clock: Timestamp source.

    Returns:
        The reversal tool's result string on "rollback", None on "confirm".

    Raises:
        UnauthorizedDecisionError: If the decision kind is unrecognized.
            The already-executed action is not silently treated as
            confirmed in that case.
    """
    now = clock()

    if decision.kind == "confirm":
        audit_log.append(
            AuditRecord(
                record.id, record.action.name, record.action.arguments, "post_hoc_confirmed",
                record.action.arguments, decision.reviewer, decision.reason, record.executed_at, now,
            )
        )
        return None

    if decision.kind == "rollback":
        reversal_registry = build_reversal_registry(ledger)
        customer_id = record.action.arguments.get("customer_id", "")
        reversal_call = ToolCall(id=f"{record.id}-reversal", name="reverse_refund", arguments={"customer_id": customer_id})
        reversal_result = reversal_registry.execute(reversal_call)
        audit_log.append(
            AuditRecord(
                record.id, record.action.name, record.action.arguments, "post_hoc_rolled_back", None,
                decision.reviewer, decision.reason, record.executed_at, now,
            )
        )
        return reversal_result

    audit_log.append(
        AuditRecord(
            record.id, record.action.name, record.action.arguments, "invalid", record.action.arguments,
            decision.reviewer, "unrecognized post-hoc decision kind", record.executed_at, now,
        )
    )
    raise UnauthorizedDecisionError(
        f"post-hoc decision for record {record.id!r} was unrecognized; effect left as executed pending manual review"
    )


def run_post_hoc_confirm_demo() -> tuple[PostHocRecord, list, AuditLog]:
    """A refund executes immediately, and sampled review confirms it was correct."""
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    action = ToolCall(
        id="call_1", name="send_refund",
        arguments={"customer_id": "c-1150", "amount_usd": 18.00, "reason": "chatbot-approved refund, under the $25 auto-execute threshold"},
    )
    record = execute_immediately(action, registry, audit_log, record_id="posthoc-1150")
    decision = PostHocDecision(kind="confirm", reviewer="ops-lead-dana", reason="matches the order and the stated reason")
    apply_post_hoc_review(record, decision, ledger, audit_log)
    return record, ledger, audit_log


def run_post_hoc_rollback_demo() -> tuple[PostHocRecord, list, AuditLog]:
    """A refund executes immediately, then sampled review catches it and rolls it back."""
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    action = ToolCall(
        id="call_1", name="send_refund",
        arguments={"customer_id": "c-1151", "amount_usd": 24.00, "reason": "chatbot-approved refund, under the $25 auto-execute threshold"},
    )
    record = execute_immediately(action, registry, audit_log, record_id="posthoc-1151")
    decision = PostHocDecision(
        kind="rollback", reviewer="ops-lead-dana",
        reason="same order was already refunded yesterday under ticket #4410; this is a duplicate",
    )
    apply_post_hoc_review(record, decision, ledger, audit_log)
    return record, ledger, audit_log
