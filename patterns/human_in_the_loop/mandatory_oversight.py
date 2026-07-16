"""Sub-module: the Article 14 non-overridable gate and two-person rule.

EU AI Act Article 14 (verified against artificialintelligenceact.eu, ID:
Article 14, Human Oversight) becomes enforceable for high-risk systems on
2 August 2026, with fines up to 35M EUR or 6% of global turnover. Article
14(4) requires a human overseer be enabled to disregard, override, or
reverse a system's output, and to interrupt it through a stop that halts
in a safe state. Article 14(5) adds a two-person rule for biometric
identification: the action cannot proceed unless at least two competent
persons separately verify it.

This is a control-flow constraint, not a compliance memo: a class of
actions that no risk score, no judge verdict, and no learned allow-list
may auto-approve, ever. That is different from `policy_guard` (which
blocks outright) and from `risk_tier.py` or `risk_classifier.py` (which
may auto-approve): a mandatory-oversight action is neither blocked nor
auto-approved, it is forced to a human every time, and the audit trail
records that the human retained the standing capability to override and
to stop, per 14(4), independent of whether they exercised it. Automation-
bias training, interpretation tooling, and interface design are 14(4)
obligations too, but are HCI and process concerns, not gate mechanics;
this module models the enforceable control-flow subset only.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agentic_patterns import ToolCall, ToolRegistry
from patterns.human_in_the_loop.fake_tools import build_biometric_registry, build_support_ops_registry
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

ARTICLE_14_TOOLS = frozenset({"confirm_biometric_match"})
ARTICLE_14_AMOUNT_FLOOR = 10000.0


@dataclass
class OversightRecord:
    """Article 14(4) capability record: proves the human retained control.

    Attributes:
        request_id: The request this record covers.
        could_override: True if the human had the standing capability to
            disregard, override, or reverse the output at decision time.
        could_stop: True if the human had the standing capability to halt
            the run via the stop path at decision time.
        overridden: True if this decision reversed a prior output.
        reviewer: Reviewer identity, or a comma-joined set for a quorum.
    """

    request_id: str
    could_override: bool
    could_stop: bool
    overridden: bool
    reviewer: str


class OversightLog:
    """An append-only log of `OversightRecord` entries, parallel to `AuditLog`."""

    def __init__(self) -> None:
        self._records: list[OversightRecord] = []

    def append(self, record: OversightRecord) -> None:
        """Add a record."""
        self._records.append(record)

    @property
    def records(self) -> tuple[OversightRecord, ...]:
        """All records in append order."""
        return tuple(self._records)


def is_article_14_action(action: ToolCall) -> bool:
    """Rule classification: is this action in the Article-14 mandatory-oversight set."""
    if action.name in ARTICLE_14_TOOLS:
        return True
    return float(action.arguments.get("amount_usd", 0.0)) >= ARTICLE_14_AMOUNT_FLOOR


def run_mandatory_gate(
    request: ReviewRequest,
    registry: ToolRegistry,
    decision_source,
    audit_log: AuditLog,
    oversight_log: OversightLog,
    *,
    clock: Callable[[], float] = time.time,
    permissive_shortcut: Callable[[ToolCall], bool] | None = None,
) -> GateOutcome:
    """Force a human decision on an in-set action, bypassing any auto-approve shortcut.

    Args:
        request: The proposed action under review.
        registry: Where the action executes.
        decision_source: Where the human decision comes from.
        audit_log: Append-only log for the underlying gate outcome.
        oversight_log: Where the Article 14(4) capability record is written
            for an in-set decision.
        clock: Timestamp source.
        permissive_shortcut: Stands in for an upstream judge or approval
            memory that would otherwise auto-approve. Only ever consulted
            for an out-of-set action; an in-set action always ignores it,
            which is the invariant this module exists to enforce.
    """
    action = request.action
    if not is_article_14_action(action):
        if permissive_shortcut is not None and permissive_shortcut(action):
            now = clock()
            result = registry.execute(action)
            audit_log.append(AuditRecord(
                request.id, action.name, action.arguments, "auto_approved_out_of_scope", action.arguments,
                "mandatory_oversight", "not an Article 14 action, upstream shortcut applied", now, now,
            ))
            return GateOutcome(kind="executed", tool_result=result, final_arguments=action.arguments)
        return run_gate(request, registry, decision_source, audit_log, clock=clock)

    outcome = run_gate(request, registry, decision_source, audit_log, clock=clock)
    oversight_log.append(OversightRecord(
        request_id=request.id, could_override=True, could_stop=True, overridden=False,
        reviewer=audit_log.records[-1].reviewer,
    ))
    return outcome


def run_override(
    request: ReviewRequest,
    decision: Decision,
    registry: ToolRegistry,
    audit_log: AuditLog,
    oversight_log: OversightLog,
    *,
    clock: Callable[[], float] = time.time,
) -> GateOutcome:
    """Apply a reviewer's decision that reverses a prior output (Article 14(4) override).

    Models the overseer's standing capability to disregard, override, or
    reverse a system output, exercised here to actually reverse one.
    """
    decision_source = ScriptedDecisionSource([decision])
    outcome = run_gate(request, registry, decision_source, audit_log, clock=clock)
    oversight_log.append(OversightRecord(
        request_id=request.id, could_override=True, could_stop=True, overridden=True, reviewer=decision.reviewer,
    ))
    return outcome


@dataclass
class SafeStateRecord:
    """The run halted here, in a recorded safe state, at the reviewer's request.

    Attributes:
        request_id: The request the stop was raised on.
        reason: The reviewer's stated reason.
        reviewer: Reviewer identity.
        halted_at: Clock reading when the stop was recorded.
    """

    request_id: str
    reason: str
    reviewer: str
    halted_at: float


def run_with_stop_path(
    request: ReviewRequest,
    decision: Decision,
    registry: ToolRegistry,
    audit_log: AuditLog,
    oversight_log: OversightLog,
    *,
    clock: Callable[[], float] = time.time,
) -> GateOutcome | SafeStateRecord:
    """Apply a decision that may be a stop, Article 14(4)'s halt-to-a-safe-state path.

    A stop is distinct from a reject: a reject declines one action and the
    run continues; a stop halts the run itself. `decision.kind == "stop"`
    is a local vocabulary extension this module owns, not part of
    `gate.run_gate`'s four-kind contract.
    """
    if decision.kind == "stop":
        now = clock()
        record = SafeStateRecord(
            request_id=request.id, reason=decision.reason or "reviewer halted the run",
            reviewer=decision.reviewer, halted_at=now,
        )
        audit_log.append(AuditRecord(
            request.id, request.action.name, request.action.arguments, "stopped_safe_state", None,
            decision.reviewer, record.reason, now, now,
        ))
        oversight_log.append(OversightRecord(
            request_id=request.id, could_override=True, could_stop=True, overridden=False, reviewer=decision.reviewer,
        ))
        return record
    decision_source = ScriptedDecisionSource([decision])
    return run_gate(request, registry, decision_source, audit_log, clock=clock)


@dataclass
class QuorumVote:
    """One reviewer's vote toward a two-person quorum.

    Attributes:
        reviewer: Reviewer identity. The same identity voting twice does
            not count as two distinct approvers.
        approve: True to approve, False to veto outright.
        reason: Explanation for the vote.
    """

    reviewer: str
    approve: bool
    reason: str = ""


def run_two_person_gate(
    request: ReviewRequest,
    registry: ToolRegistry,
    votes: Sequence[QuorumVote],
    audit_log: AuditLog,
    oversight_log: OversightLog,
    *,
    required_approvals: int = 2,
    clock: Callable[[], float] = time.time,
) -> GateOutcome:
    """Require distinct-reviewer approvals; any veto blocks, a repeat identity does not count twice.

    Models Article 14(5)'s two-person rule for biometric identification:
    the action cannot proceed unless at least two competent persons
    separately verify it.

    Raises:
        UnauthorizedDecisionError: On any veto, or if fewer than
            `required_approvals` distinct reviewers approved.
    """
    seen: set[str] = set()
    for vote in votes:
        now = clock()
        if not vote.approve:
            audit_log.append(AuditRecord(
                request.id, request.action.name, request.action.arguments, "quorum_vetoed", None,
                vote.reviewer, vote.reason or "vetoed by a single reviewer", now, now,
            ))
            raise UnauthorizedDecisionError(f"no side effect: reviewer {vote.reviewer!r} vetoed request {request.id!r}")
        seen.add(vote.reviewer)

    if len(seen) < required_approvals:
        now = clock()
        audit_log.append(AuditRecord(
            request.id, request.action.name, request.action.arguments, "quorum_not_met", None,
            ",".join(sorted(seen)), f"only {len(seen)} distinct approver(s), needs {required_approvals}", now, now,
        ))
        raise UnauthorizedDecisionError(
            f"no side effect: request {request.id!r} did not reach a {required_approvals}-person quorum"
        )

    now = clock()
    result = registry.execute(request.action)
    reviewers = ",".join(sorted(seen))
    audit_log.append(AuditRecord(
        request.id, request.action.name, request.action.arguments, "quorum_approved", request.action.arguments,
        reviewers, f"{len(seen)} distinct reviewers approved", now, now,
    ))
    oversight_log.append(OversightRecord(
        request_id=request.id, could_override=True, could_stop=True, overridden=False, reviewer=reviewers,
    ))
    return GateOutcome(kind="executed", tool_result=result, final_arguments=request.action.arguments)


def run_non_overridable_demo() -> tuple[GateOutcome, OversightLog]:
    """An in-set action is gated even with a permissive shortcut standing by."""
    registry, _ledger = build_support_ops_registry()
    audit_log = AuditLog()
    oversight_log = OversightLog()
    decision_source = ScriptedDecisionSource(
        [Decision(kind="approve", reviewer="ops-lead-dana", reason="large refund confirmed against the dispute case")]
    )
    request = ReviewRequest(id="req-mandatory", context="refund above the Article 14 amount floor", action=ToolCall(
        id="req-mandatory", name="send_refund",
        arguments={
            "customer_id": "c-90",
            "amount_usd": 12000.00,
            "reason": "chargeback dispute settled in the customer's favor",
        },
    ))
    outcome = run_mandatory_gate(
        request, registry, decision_source, audit_log, oversight_log,
        clock=counting_clock(), permissive_shortcut=lambda _action: True,
    )
    return outcome, oversight_log


def run_two_person_demo() -> tuple[GateOutcome, list]:
    """Two distinct reviewers approve a biometric match; it executes."""
    registry, ledger = build_biometric_registry()
    audit_log = AuditLog()
    oversight_log = OversightLog()
    request = ReviewRequest(id="req-biometric", context="identity verification for account recovery", action=ToolCall(
        id="req-biometric", name="confirm_biometric_match",
        arguments={"candidate_id": "cand-77", "matched_identity": "member-4471"},
    ))
    votes = [
        QuorumVote(reviewer="ops-lead-dana", approve=True, reason="scan matches on file"),
        QuorumVote(reviewer="compliance-lead-marcus", approve=True, reason="independently confirmed"),
    ]
    outcome = run_two_person_gate(request, registry, votes, audit_log, oversight_log, clock=counting_clock())
    return outcome, ledger
