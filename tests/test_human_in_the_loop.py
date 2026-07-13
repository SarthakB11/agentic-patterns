"""Tests for the human-in-the-loop pattern.

Deterministic and offline: every test drives `MockProvider` scripts or
plain scripted decision sources through the pattern's own modules, with no
network call, no API key, and no stdin. `interactive.py` is intentionally
never imported here, since it is the one path that blocks on real input().
"""

from __future__ import annotations

import json

import pytest

from agentic_patterns import MockProvider, ToolCall, scripted_tool_call

from patterns.human_in_the_loop import approval_gate, batched, escalation, plan_review, post_hoc, resume, risk_tier
from patterns.human_in_the_loop.fake_tools import build_refund_registry, build_support_ops_registry
from patterns.human_in_the_loop.gate import (
    AuditLog,
    Decision,
    DecisionSourceExhausted,
    ReviewRequest,
    ScriptedDecisionSource,
    UnauthorizedDecisionError,
    counting_clock,
    run_gate,
)

# --- fake_tools.py: shared ledger schema --------------------------------


def test_refund_registry_and_support_ops_registry_share_ledger_schema() -> None:
    """`send_refund` must log the same entry shape no matter which registry ran it."""
    refund_registry, refund_ledger = build_refund_registry()
    ops_registry, ops_ledger = build_support_ops_registry()
    call = ToolCall(id="call_1", name="send_refund", arguments={
        "customer_id": "c-1", "amount_usd": 10.0, "reason": "test",
    })

    refund_registry.execute(call)
    ops_registry.execute(call)

    assert refund_ledger[0].keys() == ops_ledger[0].keys()
    assert refund_ledger[0]["type"] == "refund"
    assert ops_ledger[0]["type"] == "refund"


# --- gate.py mechanics: the four decisions -----------------------------


def _basic_request(amount: float = 50.0) -> ReviewRequest:
    action = ToolCall(id="call_1", name="send_refund", arguments={
        "customer_id": "c-1", "amount_usd": amount, "reason": "test refund",
    })
    return ReviewRequest(id="req-1", action=action, context="test scenario")


def test_approve_executes_once_and_records_approved_audit() -> None:
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    source = ScriptedDecisionSource([Decision(kind="approve", reviewer="dana")])
    request = _basic_request()

    outcome = run_gate(request, registry, source, audit_log, clock=counting_clock())

    assert outcome.kind == "executed"
    assert len(ledger) == 1
    assert ledger[0]["amount_usd"] == 50.0
    assert len(audit_log) == 1
    assert audit_log.records[0].decision_kind == "approve"
    assert audit_log.records[0].reviewer == "dana"


def test_reject_skips_side_effect_and_returns_feedback() -> None:
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    source = ScriptedDecisionSource([Decision(kind="reject", reviewer="dana", reason="outside policy window")])
    request = _basic_request()

    outcome = run_gate(request, registry, source, audit_log, clock=counting_clock())

    assert outcome.kind == "rejected"
    assert outcome.tool_result == "outside policy window"
    assert ledger == []
    assert audit_log.records[0].decision_kind == "reject"
    assert audit_log.records[0].final_arguments is None


def test_edit_runs_with_amended_arguments_and_audit_records_final_not_proposed() -> None:
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    edited_args = {"customer_id": "c-1", "amount_usd": 20.0, "reason": "capped to policy limit"}
    source = ScriptedDecisionSource([Decision(kind="edit", reviewer="dana", arguments=edited_args)])
    request = _basic_request(amount=999.0)

    outcome = run_gate(request, registry, source, audit_log, clock=counting_clock())

    assert outcome.kind == "executed"
    assert ledger[0]["amount_usd"] == 20.0
    record = audit_log.records[0]
    assert record.proposed_arguments["amount_usd"] == 999.0
    assert record.final_arguments["amount_usd"] == 20.0


def test_respond_returns_supplied_value_and_performs_no_side_effect() -> None:
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    source = ScriptedDecisionSource([Decision(kind="respond", reviewer="dana", value="c-1 is Gold tier")])
    request = _basic_request()

    outcome = run_gate(request, registry, source, audit_log, clock=counting_clock())

    assert outcome.kind == "responded"
    assert outcome.tool_result == "c-1 is Gold tier"
    assert ledger == []


# --- gate.py mechanics: fail-closed and policy -------------------------


def test_fail_closed_on_unknown_decision_kind_raises_and_no_side_effect() -> None:
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    source = ScriptedDecisionSource([Decision(kind="acknowledge", reviewer="dana")])
    request = _basic_request()

    with pytest.raises(UnauthorizedDecisionError):
        run_gate(request, registry, source, audit_log, clock=counting_clock())

    assert ledger == []
    assert len(audit_log) == 1
    assert audit_log.records[0].decision_kind == "invalid"


def test_fail_closed_on_edit_with_no_arguments_raises_and_no_side_effect() -> None:
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    source = ScriptedDecisionSource([Decision(kind="edit", reviewer="dana", arguments=None)])
    request = _basic_request()

    with pytest.raises(UnauthorizedDecisionError):
        run_gate(request, registry, source, audit_log, clock=counting_clock())

    assert ledger == []


def test_policy_guard_blocks_before_the_reviewer_is_ever_asked() -> None:
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    source = ScriptedDecisionSource([Decision(kind="approve", reviewer="dana")])
    request = _basic_request(amount=5000.0)

    def deny_large_amounts(action: ToolCall) -> str | None:
        if action.arguments["amount_usd"] > 1000:
            return "exceeds hard policy cap"
        return None

    with pytest.raises(UnauthorizedDecisionError):
        run_gate(request, registry, source, audit_log, policy_guard=deny_large_amounts, clock=counting_clock())

    assert ledger == []
    assert source.decisions_served == []  # the reviewer was never consulted
    assert audit_log.records[0].decision_kind == "blocked_by_policy"


def test_scripted_decision_source_raises_when_exhausted() -> None:
    source = ScriptedDecisionSource([])
    with pytest.raises(DecisionSourceExhausted):
        source.decide(_basic_request())


def test_scripted_decision_source_maps_by_request_id() -> None:
    source = ScriptedDecisionSource({"req-a": Decision(kind="approve"), "req-b": Decision(kind="reject")})
    decision_b = source.decide(ReviewRequest(id="req-b", action=_basic_request().action, context=""))
    assert decision_b.kind == "reject"


def test_audit_log_ordering_is_execution_order_with_nondecreasing_timestamps() -> None:
    registry, _ledger = build_refund_registry()
    audit_log = AuditLog()
    clock = counting_clock()
    for i in range(3):
        request = ReviewRequest(
            id=f"req-{i}",
            action=ToolCall(id=f"req-{i}", name="send_refund", arguments={
                "customer_id": f"c-{i}", "amount_usd": 10.0, "reason": "batch",
            }),
            context="ordering check",
        )
        run_gate(request, registry, ScriptedDecisionSource([Decision(kind="approve")]), audit_log, clock=clock)

    ids = [r.request_id for r in audit_log.records]
    assert ids == ["req-0", "req-1", "req-2"]
    timestamps = [r.decided_at for r in audit_log.records]
    assert timestamps == sorted(timestamps)


# --- approval_gate.py: the base variant, driven through MockProvider ---


def test_approval_gate_approve_demo_sends_expected_arguments_to_the_model() -> None:
    result = approval_gate.run_approve_demo()
    assert result.outcome.kind == "executed"
    assert result.ledger[0]["customer_id"] == "c-4471"


def test_approval_gate_reject_demo_feeds_reason_back_as_tool_observation() -> None:
    provider = MockProvider([
        {"tool": "send_refund", "args": {"customer_id": "c-1", "amount_usd": 900.0, "reason": "test"}},
        "understood, will not process this refund",
    ])
    result = approval_gate.run_reject_demo(provider)
    tool_message = provider.calls[1]["messages"][2]
    assert tool_message.role == "tool"
    assert tool_message.content == result.outcome.tool_result
    assert result.ledger == []


def test_approval_gate_fail_closed_demo_blocks_and_keeps_ledger_empty() -> None:
    result = approval_gate.run_fail_closed_demo()
    assert result.error is not None
    assert result.ledger == []
    assert result.outcome is None


# --- risk_tier.py --------------------------------------------------------


def test_low_risk_action_auto_approves_with_no_review_request() -> None:
    result = risk_tier.run_risk_tier_demo()
    assert result.low_risk_outcome.kind == "executed"
    assert result.reviewer_was_asked is False
    assert result.high_risk_outcome.kind == "executed"
    assert len(result.ledger) == 2


def test_flooding_demo_rubber_stamp_approves_malicious_but_policy_backstop_blocks_it() -> None:
    result = risk_tier.run_flooding_demo()
    slipped_through = any(e["amount_usd"] == result.malicious_amount for e in result.rubber_stamp_ledger)
    blocked = any(e["amount_usd"] == result.malicious_amount for e in result.guarded_ledger)
    assert slipped_through is True
    assert blocked is False


# --- resume.py -------------------------------------------------------------


def test_resume_after_suspend_matches_uninterrupted_run() -> None:
    result = resume.run_resume_demo()
    assert result.resumed_outcome.kind == "executed"
    assert result.resumed_outcome.tool_result == result.uninterrupted_outcome.tool_result
    assert result.resumed_ledger == result.uninterrupted_ledger


def test_gate_state_round_trips_through_plain_json() -> None:
    action = ToolCall(id="call_1", name="send_refund", arguments={
        "customer_id": "c-9", "amount_usd": 30.0, "reason": "test",
    })
    request = ReviewRequest(id="req-json", action=action, context="serialize me")
    state = resume.suspend_at_gate(request, timeout_seconds=60.0, clock=counting_clock())

    blob = json.dumps(state.to_dict())
    restored = resume.GateState.from_dict(json.loads(blob))

    assert restored.request.id == "req-json"
    assert restored.request.action.arguments["amount_usd"] == 30.0
    assert restored.deadline == state.deadline


def test_expired_decision_fails_closed_with_zero_side_effects() -> None:
    ledger, audit_log = resume.run_expired_demo()
    assert ledger == []
    assert audit_log.records[-1].decision_kind == "expired"


# --- escalation.py --------------------------------------------------------


def test_high_confidence_auto_approves_low_confidence_escalates() -> None:
    outcome_1, outcome_2, ledger = escalation.run_escalation_demo()
    assert outcome_1.kind == "executed"
    assert outcome_2.kind == "executed"  # escalated, then approved by the reviewer
    assert len(ledger) == 2


def test_confidence_gate_only_consults_reviewer_below_threshold() -> None:
    confident_call = scripted_tool_call(
        "send_refund", {"customer_id": "c-1", "amount_usd": 40.0, "reason": "clear-cut"}, call_id="call_1"
    )
    confident_call.raw = {"confidence": 0.99}
    unsure_call = scripted_tool_call(
        "send_refund", {"customer_id": "c-2", "amount_usd": 40.0, "reason": "unclear"}, call_id="call_2"
    )
    unsure_call.raw = {"confidence": 0.10}
    provider = MockProvider([confident_call, unsure_call])

    outcome_1, outcome_2, _ledger = escalation.run_escalation_demo(provider)

    assert outcome_1.kind == "executed"
    assert outcome_2.kind == "executed"


def test_async_escalation_does_not_block_other_tasks() -> None:
    result = escalation.run_async_escalation_demo()
    assert result.completed_immediately == ["task-a", "task-c"]
    assert result.queued_for_review == ["task-b"]
    assert result.resolved_after_wait == ["task-b"]
    assert len(result.ledger) == 3


# --- plan_review.py --------------------------------------------------------


def test_plan_approve_runs_every_step_in_order() -> None:
    outcomes, ledger, _audit_log = plan_review.run_plan_approve_demo()
    assert len(outcomes) == 2
    assert ledger[0]["type"] == "cancellation"
    assert ledger[1]["type"] == "refund"


def test_plan_edit_replaces_steps_before_any_step_executes() -> None:
    outcomes, ledger, _audit_log = plan_review.run_plan_edit_demo()
    assert len(outcomes) == 2
    refund_entry = next(e for e in ledger if e["type"] == "refund")
    assert refund_entry["amount_usd"] == 22.50  # not the originally proposed 40.00


def test_plan_reject_leaves_every_step_unexecuted() -> None:
    ledger, _audit_log = plan_review.run_plan_reject_demo()
    assert ledger == []


# --- post_hoc.py --------------------------------------------------------


def test_post_hoc_confirm_leaves_the_effect_in_place() -> None:
    _record, ledger, _audit_log = post_hoc.run_post_hoc_confirm_demo()
    assert len(ledger) == 1


def test_post_hoc_rollback_reverses_the_effect() -> None:
    _record, ledger, _audit_log = post_hoc.run_post_hoc_rollback_demo()
    assert ledger == []


def test_post_hoc_unknown_decision_raises_without_silently_confirming() -> None:
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    action = ToolCall(id="c1", name="send_refund", arguments={
        "customer_id": "c-1", "amount_usd": 10.0, "reason": "test",
    })
    record = post_hoc.execute_immediately(action, registry, audit_log, record_id="p1")
    decision = post_hoc.PostHocDecision(kind="escalate_further", reviewer="dana")

    with pytest.raises(UnauthorizedDecisionError):
        post_hoc.apply_post_hoc_review(record, decision, ledger, audit_log)

    assert len(ledger) == 1  # the original effect is neither confirmed nor rolled back automatically


# --- batched.py --------------------------------------------------------


def test_batch_review_maps_out_of_order_decisions_back_to_the_right_request() -> None:
    outcomes, ledger = batched.run_batched_review_demo()
    assert outcomes["batch-1"].kind == "executed"
    assert outcomes["batch-2"].kind == "executed"
    assert outcomes["batch-2"].final_arguments["amount_usd"] == 130.00
    assert outcomes["batch-3"].kind == "rejected"
    assert len(ledger) == 2


def test_batch_review_missing_decision_reports_unresolved_without_raising() -> None:
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    requests = [
        ReviewRequest(id="only-one", action=ToolCall(id="only-one", name="send_refund", arguments={
            "customer_id": "c-1", "amount_usd": 10.0, "reason": "test",
        }), context="test"),
        ReviewRequest(id="missing", action=ToolCall(id="missing", name="send_refund", arguments={
            "customer_id": "c-2", "amount_usd": 10.0, "reason": "test",
        }), context="test"),
    ]
    decisions = {"only-one": Decision(kind="approve", reviewer="dana")}

    results = batched.run_batch_review(requests, registry, decisions, audit_log)

    assert results["only-one"].kind == "executed"
    assert isinstance(results["missing"], str)
    assert len(ledger) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
