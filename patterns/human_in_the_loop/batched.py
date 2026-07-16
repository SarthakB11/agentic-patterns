"""Sub-module: batched / queued review.

Several pending actions are collected and presented to a reviewer together
instead of one at a time, so a reviewer clears a whole batch in one pass
rather than context-switching between separate interruptions. Decisions
can come back in any order and must map to the correct pending action by
identifier, not by the order requests were submitted in.

`gate.ScriptedDecisionSource` already supports this by accepting a mapping
from request id to `Decision` instead of a plain queue; this module is
mostly about the review workflow around that: build the batch, hand it to
one reviewer pass, and apply whatever comes back.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from agentic_patterns import ToolCall, ToolRegistry
from patterns.human_in_the_loop.fake_tools import build_refund_registry
from patterns.human_in_the_loop.gate import (
    AuditLog,
    Decision,
    DecisionSourceExhausted,
    GateOutcome,
    ReviewRequest,
    ScriptedDecisionSource,
    UnauthorizedDecisionError,
    run_gate,
)


def run_batch_review(
    requests: list[ReviewRequest],
    registry: ToolRegistry,
    decisions: Mapping[str, Decision],
    audit_log: AuditLog,
    *,
    clock: Callable[[], float] = time.time,
) -> dict[str, GateOutcome | str]:
    """Present a batch of requests to one decision source and apply the results.

    Args:
        requests: The pending actions in the batch, in submission order.
        registry: Where an approved or edited action executes.
        decisions: A reviewer's rulings, keyed by `ReviewRequest.id`. Order
            does not need to match `requests`.
        audit_log: Append-only log, written to for every request.
        clock: Timestamp source.

    Returns:
        A dict keyed by request id. A resolved request maps to its
        `GateOutcome`; a request with no matching decision, or one that
        failed closed, maps to an error message string instead of raising,
        so one bad entry does not stop the rest of the batch from clearing.
    """
    decision_source = ScriptedDecisionSource(decisions)
    results: dict[str, GateOutcome | str] = {}
    for request in requests:
        try:
            results[request.id] = run_gate(request, registry, decision_source, audit_log, clock=clock)
        except (UnauthorizedDecisionError, DecisionSourceExhausted) as exc:
            results[request.id] = f"unresolved: {exc}"
    return results


def run_batched_review_demo() -> tuple[dict[str, GateOutcome | str], list]:
    """Three pending refunds, cleared in one reviewer pass, decisions out of order.

    The reviewer answers request 3 first, then request 1, then request 2,
    which is exactly the shape a batched review UI produces: the reviewer
    works through a list in whatever order catches their eye, not
    necessarily submission order.
    """
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()

    requests = [
        ReviewRequest(
            id="batch-1",
            action=ToolCall(id="batch-1", name="send_refund", arguments={
                "customer_id": "c-3001", "amount_usd": 55.00, "reason": "item arrived damaged",
            }),
            context="pending refund 1 of 3",
        ),
        ReviewRequest(
            id="batch-2",
            action=ToolCall(id="batch-2", name="send_refund", arguments={
                "customer_id": "c-3002", "amount_usd": 220.00, "reason": "order never arrived",
            }),
            context="pending refund 2 of 3",
        ),
        ReviewRequest(
            id="batch-3",
            action=ToolCall(id="batch-3", name="send_refund", arguments={
                "customer_id": "c-3003", "amount_usd": 610.00, "reason": "customer disputes the whole order",
            }),
            context="pending refund 3 of 3",
        ),
    ]

    # Decisions given in a mapping, deliberately not in request order, and
    # not all the same kind: request 3 is disputed and needs more digging.
    decisions = {
        "batch-3": Decision(
            kind="reject", reviewer="ops-lead-dana",
            reason="dispute is with the carrier, not us; route to the shipping claims team instead",
        ),
        "batch-1": Decision(
            kind="approve", reviewer="ops-lead-dana", reason="photo evidence attached, straightforward"
        ),
        "batch-2": Decision(
            kind="edit", reviewer="ops-lead-dana", reason="tracking shows partial delivery, not a full loss",
            arguments={"customer_id": "c-3002", "amount_usd": 130.00, "reason": "partial non-delivery, prorated"},
        ),
    }

    outcomes = run_batch_review(requests, registry, decisions, audit_log)
    return outcomes, ledger
