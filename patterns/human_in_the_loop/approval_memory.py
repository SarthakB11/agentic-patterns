"""Sub-module: a learned allow-list that shrinks reviewer load over time.

Every other variant in this pattern re-prompts on every gated action
forever. Learning-to-defer (Mozannar and Sontag, arXiv:2006.01862; DeCCaF,
arXiv:2403.06906) argues for deferring to a human only where the human
adds value, not on every repeat of an already-blessed action. This module
is the in-session structural half of that idea: cache a human's verdict
against a coarsened action signature, and once a signature has been seen
enough times, resolve future matches without asking again. Turan
(arXiv:2606.08919) motivates why this matters beyond convenience: cutting
escalations the reviewer would trivially approve again keeps them under
the capacity `capacity.py` shows they need. Magentic-UI (arXiv:2507.22358)
has action guards but no approval-memory; this deliberately extends past
it, and stays in-session and structural, not the cross-session learned
store that belongs in `patterns/memory`.

A learned allow-list must not swallow a novel high-risk action just
because a cheaper cousin was approved before, so a hard safety ceiling
runs ahead of the memory lookup: any action `risk_classifier.py`'s rule
tier marks always-gate skips memory entirely and goes to the human, no
matter how warm the signature's history is.
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
    run_gate,
)
from patterns.human_in_the_loop.risk_classifier import rule_tier

DEFAULT_CONFIDENCE_K = 1
_REASON_KEYWORDS = (("shipping", "shipping"), ("duplicate", "duplicate_charge"), ("damaged", "damaged_item"))


def _amount_bucket(amount: float) -> str:
    """Coarsen a dollar amount into a range bucket."""
    if amount < 50:
        return "under_50"
    if amount < 200:
        return "50_to_200"
    if amount < 1000:
        return "200_to_1000"
    return "over_1000"


def action_signature(action: ToolCall) -> str:
    """Coarsened signature: tool name, amount bucket, reason keyword bucket.

    Two actions with different exact arguments (a $12 and a $9.50 refund,
    both for a reprinted shipping label) share a signature; a $12 refund
    for a damaged item does not, even though the amount matches.
    """
    amount = float(action.arguments.get("amount_usd", 0.0))
    reason = str(action.arguments.get("reason", "")).lower()
    reason_bucket = next((b for kw, b in _REASON_KEYWORDS if kw in reason), "other")
    return f"{action.name}:{_amount_bucket(amount)}:{reason_bucket}"


@dataclass
class MemoryEntry:
    """A remembered verdict for one signature.

    Attributes:
        kind: "approve" or "reject".
        count: How many times this verdict was given for this signature.
    """

    kind: str
    count: int


class ApprovalMemory:
    """In-session signature-to-verdict memory, cleared per run.

    Not a cross-session store: `patterns/memory` is where a persistent,
    embedding-based version would live. This is the gate-side structural
    half, a plain dict scoped to one run.
    """

    def __init__(self, confidence_k: int = DEFAULT_CONFIDENCE_K) -> None:
        """Args: confidence_k: matching verdicts needed before auto-resolving."""
        self.confidence_k = confidence_k
        self._entries: dict[str, MemoryEntry] = {}

    def lookup(self, signature: str) -> MemoryEntry | None:
        """Return the remembered entry for a signature, or None if unseen."""
        return self._entries.get(signature)

    def record(self, signature: str, kind: str) -> None:
        """Record a verdict; a repeat of the same kind increments its count.

        A verdict that flips kind (a prior reject now approved, or the
        reverse) resets the count to 1 instead of accumulating across
        contradictory decisions.
        """
        existing = self._entries.get(signature)
        if existing is not None and existing.kind == kind:
            existing.count += 1
        else:
            self._entries[signature] = MemoryEntry(kind=kind, count=1)


def run_memory_gate(
    request: ReviewRequest,
    registry: ToolRegistry,
    decision_source,
    audit_log: AuditLog,
    memory: ApprovalMemory,
    *,
    clock: Callable[[], float] = time.time,
) -> GateOutcome:
    """Consult the approval memory before the human, and learn from what the human decides.

    Args:
        request: The proposed action under review.
        registry: Where an approved action executes.
        decision_source: Where a human decision comes from, if memory does
            not already resolve this signature.
        audit_log: Append-only log, written to for every route.
        memory: The signature-to-verdict cache to consult and update.
        clock: Timestamp source.
    """
    action = request.action
    rule_verdict = rule_tier(action)
    if rule_verdict is not None and rule_verdict.route == "always_gate":
        # Safety ceiling: never resolved from memory, regardless of history.
        return run_gate(request, registry, decision_source, audit_log, clock=clock)

    signature = action_signature(action)
    entry = memory.lookup(signature)

    if entry is not None and entry.count >= memory.confidence_k:
        now = clock()
        if entry.kind == "approve":
            result = registry.execute(action)
            audit_log.append(AuditRecord(
                request.id, action.name, action.arguments, "auto_approved_from_memory", action.arguments,
                "approval_memory", f"signature {signature!r} approved {entry.count} time(s) before", now, now,
            ))
            return GateOutcome(kind="executed", tool_result=result, final_arguments=action.arguments)
        feedback = f"auto-rejected from memory: signature {signature!r} was rejected {entry.count} time(s) before"
        audit_log.append(AuditRecord(
            request.id, action.name, action.arguments, "auto_rejected_from_memory", None,
            "approval_memory", feedback, now, now,
        ))
        return GateOutcome(kind="rejected", tool_result=feedback, final_arguments=None)

    outcome = run_gate(request, registry, decision_source, audit_log, clock=clock)
    if outcome.kind in ("executed", "rejected"):
        memory.record(signature, "approve" if outcome.kind == "executed" else "reject")
    return outcome


def _shipping_refund(request_id: str, customer_id: str, amount_usd: float) -> ReviewRequest:
    """Build a small shipping-fee refund request, the demos' repeated signature."""
    action = ToolCall(id=request_id, name="send_refund", arguments={
        "customer_id": customer_id, "amount_usd": amount_usd, "reason": "shipping label reprint fee",
    })
    return ReviewRequest(id=request_id, action=action, context="shipping fee refund")


def run_learn_then_auto_demo() -> tuple[GateOutcome, GateOutcome, list]:
    """First shipping-fee refund is reviewed; a matching repeat auto-approves."""
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    memory = ApprovalMemory(confidence_k=1)
    decision_source = ScriptedDecisionSource(
        [Decision(kind="approve", reviewer="ops-lead-dana", reason="standard shipping fee refund")]
    )
    first = run_memory_gate(_shipping_refund("req-1", "c-01", 12.00), registry, decision_source, audit_log, memory)
    # No second decision is scripted: consulting the human here would raise.
    second = run_memory_gate(_shipping_refund("req-2", "c-02", 9.50), registry, decision_source, audit_log, memory)
    return first, second, ledger


def run_reject_memory_demo() -> tuple[GateOutcome, GateOutcome, list]:
    """A rejected signature auto-rejects a later match, with no new side effect."""
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    memory = ApprovalMemory(confidence_k=1)
    decision_source = ScriptedDecisionSource(
        [Decision(kind="reject", reviewer="ops-lead-dana", reason="policy no longer covers reprint fees")]
    )
    first = run_memory_gate(_shipping_refund("req-3", "c-03", 11.00), registry, decision_source, audit_log, memory)
    second = run_memory_gate(_shipping_refund("req-4", "c-04", 10.50), registry, decision_source, audit_log, memory)
    return first, second, ledger


def run_confidence_threshold_demo() -> tuple[list[GateOutcome], int]:
    """With k=2, the first two matches consult the human; the third auto-resolves."""
    registry, _ledger = build_refund_registry()
    audit_log = AuditLog()
    memory = ApprovalMemory(confidence_k=2)
    decision_source = ScriptedDecisionSource([
        Decision(kind="approve", reviewer="ops-lead-dana", reason="first sighting"),
        Decision(kind="approve", reviewer="ops-lead-dana", reason="confirms the pattern"),
    ])
    outcomes = [
        run_memory_gate(_shipping_refund(f"req-{i}", f"c-{i}", 10.00 + i), registry, decision_source, audit_log, memory)
        for i in range(3)
    ]
    return outcomes, len(decision_source.decisions_served)


def run_safety_ceiling_demo() -> tuple[GateOutcome, GateOutcome, int]:
    """A high-risk cousin of an approved signature is still gated, not auto-approved."""
    registry, _ledger = build_refund_registry()
    audit_log = AuditLog()
    memory = ApprovalMemory(confidence_k=1)
    decision_source = ScriptedDecisionSource([
        Decision(kind="approve", reviewer="ops-lead-dana", reason="duplicate charge confirmed in the log"),
        Decision(kind="reject", reviewer="ops-lead-dana", reason="six thousand dollars needs a second look regardless of history"),
    ])
    cousin = ReviewRequest(id="req-cousin", context="duplicate charge, moderate amount", action=ToolCall(
        id="req-cousin", name="send_refund",
        arguments={"customer_id": "c-10", "amount_usd": 1200.00, "reason": "duplicate charge on the order"},
    ))
    cousin_outcome = run_memory_gate(cousin, registry, decision_source, audit_log, memory)

    # Same signature bucket ("over_1000", "duplicate_charge"), but past the
    # risk_classifier hard-gate line: memory must not shortcut this one.
    risky = ReviewRequest(id="req-risky", context="duplicate charge, much larger amount", action=ToolCall(
        id="req-risky", name="send_refund",
        arguments={"customer_id": "c-11", "amount_usd": 6000.00, "reason": "duplicate charge on the order"},
    ))
    risky_outcome = run_memory_gate(risky, registry, decision_source, audit_log, memory)
    return cousin_outcome, risky_outcome, len(decision_source.decisions_served)


def run_load_falls_demo(stream_size: int = 10) -> tuple[list[GateOutcome], int]:
    """Ten identical-signature refunds, k=1: only the first ever consults a human."""
    registry, _ledger = build_refund_registry()
    audit_log = AuditLog()
    memory = ApprovalMemory(confidence_k=1)
    decision_source = ScriptedDecisionSource(
        [Decision(kind="approve", reviewer="ops-lead-dana", reason="standard shipping fee refund")]
    )
    outcomes = [
        run_memory_gate(_shipping_refund(f"req-load-{i}", f"c-load-{i}", 10.00), registry, decision_source, audit_log, memory)
        for i in range(stream_size)
    ]
    return outcomes, len(decision_source.decisions_served)
