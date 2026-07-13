"""Human-in-the-loop pattern: pause an agent for a person to decide.

Human-in-the-loop is the pattern of pausing an autonomous agent at a
defined point so a person can inspect a proposed action, then approve it,
change it, reject it, or supply missing information, before execution
continues. The defining mechanic is a gate: the agent produces a proposed
action but performs no side effect until a human decision is recorded.

This demo runs every sub-variant end to end, entirely offline against
`MockProvider` with scripted, coherent conversations, no network call and
no API key:

1. The base approval gate: one task walks through all four decisions
   (approve, edit, reject, respond), plus the fail-closed default when a
   decision is missing or unrecognized.
2. Risk-tiered gating: a low-risk action auto-approves with no reviewer
   ever consulted; a high-risk one is gated. A second demo shows that
   gating everything, without a deterministic policy backstop, still lets
   a fatigued, rubber-stamping reviewer approve a malicious action.
3. Interrupt-and-resume: a gate suspends to a serializable state,
   reconstructs from it in a separate call, and resumes to the same
   result as an uninterrupted run. A decision that arrives past its
   deadline is refused.
4. Escalation on confidence: a confident proposal auto-approves; an
   unsure one escalates. A second demo shows the asynchronous shape, where
   other work continues while one escalation is outstanding.
5. Plan review: a whole multi-step plan is approved, edited, or rejected
   once, before any step in it executes.
6. Post-hoc review with override: an action executes immediately and is
   reviewed afterward, with the option to roll it back.
7. Batched review: several pending actions are cleared in one reviewer
   pass and decisions map back to the right action by identifier.
8. Model-judged risk classification: a cheap rule tier resolves the
   obvious ends (always-gate, never-gate); only the ambiguous middle is
   routed to a model judge, and only two of five actions cost a model call.
9. Load-aware capacity calibration: sweeping the escalation threshold
   traces an inverted-U safety curve; escalating everything floods the
   reviewer and is itself unsafe, not just wasteful.
10. Approval memory: a human's verdict on an action signature is cached,
    so a repeated, already-blessed action stops costing a new prompt,
    while a hard safety ceiling still forces a high-risk cousin to a human.
11. Mandatory oversight (EU AI Act Article 14): an in-set action is gated
    every time regardless of any upstream shortcut, and a two-person quorum
    is required for the highest-risk class.
12. Human-initiated interrupt: a reviewer takes over a running plan mid-run
    to edit the remaining steps, inject a step, or abort, instead of the
    agent hitting a predicate gate.

Run it from the repository root:

    python -m patterns.human_in_the_loop.main

Pass --interactive to try the one genuinely interactive path instead,
which reads a real decision from stdin with input(). That path is never
run by default and never exercised by the test suite, since both must
stay non-interactive:

    python -m patterns.human_in_the_loop.main --interactive

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key in the environment) to run the proposal side of
each demo against a real model instead of the mock. No source change is
required; every demo function builds its provider through
`agentic_patterns.get_provider`. The human decisions themselves stay
scripted either way, since a real reviewer is what --interactive is for.
"""

from __future__ import annotations

import argparse

from agentic_patterns import ToolCall

from patterns.human_in_the_loop import (
    approval_gate,
    approval_memory,
    batched,
    capacity,
    escalation,
    interrupt,
    mandatory_oversight,
    plan_review,
    post_hoc,
    resume,
    risk_classifier,
    risk_tier,
)
from patterns.human_in_the_loop.gate import AuditLog, ReviewRequest, run_gate
from patterns.human_in_the_loop.interactive import InteractiveDecisionSource
from patterns.human_in_the_loop.transcript import format_audit_log, format_outcome


def main(argv: list[str] | None = None) -> None:
    """Run every human-in-the-loop sub-variant demo and print a readable transcript."""
    parser = argparse.ArgumentParser(description="Human-in-the-loop pattern demo")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Read one real decision from stdin instead of running the scripted offline demos.",
    )
    args = parser.parse_args(argv)

    if args.interactive:
        _run_interactive_scenario()
        return

    print("HUMAN-IN-THE-LOOP PATTERN: approval gates for agent actions\n")

    print("=== 1. Approval gate: approve ===")
    result = approval_gate.run_approve_demo()
    print(f"task: {result.task}")
    print(f"proposed: send_refund({result.proposed_arguments})")
    print(format_outcome(result.outcome))
    print(f"agent: {result.follow_up}")
    assert result.ledger, "approve should have executed the refund"
    print()

    print("=== 1b. Approval gate: edit ===")
    result = approval_gate.run_edit_demo()
    print(f"proposed: send_refund({result.proposed_arguments})")
    print(format_outcome(result.outcome))
    print(f"final arguments used: {result.outcome.final_arguments}")
    assert result.outcome.final_arguments != result.proposed_arguments
    print()

    print("=== 1c. Approval gate: reject ===")
    result = approval_gate.run_reject_demo()
    print(f"proposed: send_refund({result.proposed_arguments})")
    print(format_outcome(result.outcome))
    print(f"agent: {result.follow_up}")
    assert not result.ledger, "reject must not execute the action"
    print()

    print("=== 1d. Approval gate: respond (ask-the-human) ===")
    result = approval_gate.run_respond_demo()
    print(f"proposed: lookup_customer_tier({result.proposed_arguments})")
    print(format_outcome(result.outcome))
    assert not result.ledger, "respond must not execute a side effect"
    print()

    print("=== 1e. Approval gate: fail-closed default ===")
    result = approval_gate.run_fail_closed_demo()
    print(f"proposed: send_refund({result.proposed_arguments})")
    print(f"error: {result.error}")
    assert not result.ledger, "an unrecognized decision must not execute the action"
    print()

    print("=== 2. Risk-tiered gating ===")
    tier_result = risk_tier.run_risk_tier_demo()
    print(f"low-risk ($15): {format_outcome(tier_result.low_risk_outcome)}")
    print(f"reviewer consulted for the low-risk request: {tier_result.reviewer_was_asked}")
    print(f"high-risk ($900): {format_outcome(tier_result.high_risk_outcome)}")
    assert tier_result.reviewer_was_asked is False
    print()

    print("=== 2b. Risk-tiered gating: escalation-fatigue failure mode ===")
    flood_result = risk_tier.run_flooding_demo()
    malicious_slipped_through = any(
        e["amount_usd"] == flood_result.malicious_amount for e in flood_result.rubber_stamp_ledger
    )
    malicious_blocked = any(
        e["amount_usd"] == flood_result.malicious_amount for e in flood_result.guarded_ledger
    )
    print(f"no policy backstop, rubber-stamping reviewer: malicious ${flood_result.malicious_amount:,.2f} refund sent = {malicious_slipped_through}")
    print(f"with a deterministic policy cap in front of the gate: same refund sent = {malicious_blocked}")
    assert malicious_slipped_through is True
    assert malicious_blocked is False
    print()

    print("=== 3. Interrupt-and-resume (durable pause) ===")
    resume_result = resume.run_resume_demo()
    print(f"resumed:       {format_outcome(resume_result.resumed_outcome)}")
    print(f"uninterrupted: {format_outcome(resume_result.uninterrupted_outcome)}")
    assert resume_result.resumed_outcome.tool_result == resume_result.uninterrupted_outcome.tool_result
    print()

    print("=== 3b. Interrupt-and-resume: expired decision fails closed ===")
    expired_ledger, expired_audit = resume.run_expired_demo()
    print(f"decision arrived after the deadline; refund count in ledger: {len(expired_ledger)}")
    print(format_audit_log(expired_audit))
    assert not expired_ledger
    print()

    print("=== 4. Escalation on confidence (synchronous) ===")
    outcome_1, outcome_2, esc_ledger = escalation.run_escalation_demo()
    print(f"high confidence (0.93): {format_outcome(outcome_1)}")
    print(f"low confidence (0.35):  {format_outcome(outcome_2)}")
    assert len(esc_ledger) == 2
    print()

    print("=== 4b. Escalation on confidence (asynchronous) ===")
    async_result = escalation.run_async_escalation_demo()
    print(f"completed immediately: {async_result.completed_immediately}")
    print(f"queued for review (agent kept working): {async_result.queued_for_review}")
    print(f"resolved after the wait: {async_result.resolved_after_wait}")
    print()

    print("=== 5. Plan review: approve ===")
    plan_outcomes, plan_ledger, _ = plan_review.run_plan_approve_demo()
    for outcome in plan_outcomes:
        print(format_outcome(outcome))
    assert len(plan_ledger) == 2
    print()

    print("=== 5b. Plan review: edit before any step executes ===")
    edit_outcomes, edit_ledger, _ = plan_review.run_plan_edit_demo()
    for outcome in edit_outcomes:
        print(format_outcome(outcome))
    refund_entry = next(e for e in edit_ledger if e["type"] == "refund")
    print(f"refund amount corrected during review: ${refund_entry['amount_usd']:.2f}")
    assert refund_entry["amount_usd"] == 22.50
    print()

    print("=== 5c. Plan review: reject leaves every step unexecuted ===")
    reject_ledger, _ = plan_review.run_plan_reject_demo()
    print(f"steps executed after rejection: {len(reject_ledger)}")
    assert not reject_ledger
    print()

    print("=== 6. Post-hoc review with override: confirm ===")
    _, confirm_ledger, _ = post_hoc.run_post_hoc_confirm_demo()
    print(f"refund stayed in effect after review: {len(confirm_ledger) == 1}")
    assert len(confirm_ledger) == 1
    print()

    print("=== 6b. Post-hoc review with override: rollback ===")
    _, rollback_ledger, _ = post_hoc.run_post_hoc_rollback_demo()
    print(f"refund reversed after review caught a duplicate: {len(rollback_ledger) == 0}")
    assert not rollback_ledger
    print()

    print("=== 7. Batched review ===")
    batch_outcomes, batch_ledger = batched.run_batched_review_demo()
    for request_id in ("batch-1", "batch-2", "batch-3"):
        outcome = batch_outcomes[request_id]
        rendered = format_outcome(outcome) if hasattr(outcome, "kind") else str(outcome)
        print(f"{request_id}: {rendered}")
    assert len(batch_ledger) == 2  # batch-3 was rejected, so only 1 and 2 sent
    print()

    print("=== 8. Model-judged risk classification ===")
    risk_result = risk_classifier.run_risk_classifier_demo()
    for name in ("always_gate", "never_gate", "trivial", "judge_gate", "judge_auto"):
        print(f"{name}: {format_outcome(risk_result.outcomes[name])}")
    print(f"model (judge) calls made for five actions: {risk_result.judge_calls_made}")
    assert risk_result.judge_calls_made == 2
    print()

    print("=== 9. Load-aware capacity calibration (inverted-U safety curve) ===")
    u_result = capacity.run_inverted_u_demo()
    nothing, everything = u_result.curve[-1], u_result.curve[0]
    print(f"escalate-nothing safety:    {nothing.safety:.2f} (threshold {nothing.threshold})")
    print(f"escalate-everything safety: {everything.safety:.2f} (threshold {everything.threshold})")
    print(f"safety-optimal safety:      {u_result.optimal.safety:.2f} (threshold {u_result.optimal.threshold})")
    flood_result = capacity.run_flooding_vs_optimal_demo()
    print(f"buried malicious action blocked by escalate-everything: {flood_result.blocked_by_escalate_everything}")
    print(f"buried malicious action blocked at the optimal threshold: {flood_result.blocked_at_optimal_threshold}")
    assert u_result.optimal.safety > nothing.safety
    assert u_result.optimal.safety > everything.safety
    print()

    print("=== 10. Approval memory: learn once, auto-resolve on repeat ===")
    first, second, mem_ledger = approval_memory.run_learn_then_auto_demo()
    print(f"first shipping refund:  {format_outcome(first)}")
    print(f"repeat, same signature: {format_outcome(second)}")
    _, _, ceiling_consultations = approval_memory.run_safety_ceiling_demo()
    print(f"high-risk cousin of an approved signature still consults a human: {ceiling_consultations == 2}")
    assert len(mem_ledger) == 2
    print()

    print("=== 11. Mandatory oversight (EU AI Act Article 14) ===")
    mandatory_outcome, oversight_log = mandatory_oversight.run_non_overridable_demo()
    print(f"in-set action, permissive shortcut ignored: {format_outcome(mandatory_outcome)}")
    print(f"oversight capability recorded: {oversight_log.records[0].could_override and oversight_log.records[0].could_stop}")
    quorum_outcome, bio_ledger = mandatory_oversight.run_two_person_demo()
    print(f"two-person biometric quorum: {format_outcome(quorum_outcome)}")
    assert len(bio_ledger) == 1
    print()

    print("=== 12. Human-initiated interrupt (real-time monitoring) ===")
    edit_result, edit_int_ledger = interrupt.run_edit_mid_run_demo()
    print(f"steps executed after a mid-run edit: {len(edit_result.outcomes)}")
    abort_result, abort_ledger = interrupt.run_abort_demo()
    print(f"steps executed after a mid-run abort: {len(abort_result.outcomes)} (of 3 proposed)")
    print(f"abort reason: {abort_result.stop_reason}")
    assert abort_result.aborted is True
    assert len(abort_ledger) == 1
    print()

    print("All twelve sub-variants completed without exhausting their scripts.")


def _run_interactive_scenario() -> None:
    """The one genuinely interactive path: a real person decides via input().

    Never called by the default flow and never exercised by tests. The
    proposal still comes from a scripted `MockProvider` turn so this needs
    no API key; only the decision itself is real.
    """
    from patterns.human_in_the_loop.fake_tools import build_refund_registry

    print("HUMAN-IN-THE-LOOP interactive mode: you are the reviewer.\n")
    registry, _ledger = build_refund_registry()
    audit_log = AuditLog()
    action = ToolCall(
        id="call_1", name="send_refund",
        arguments={"customer_id": "c-0001", "amount_usd": 175.00, "reason": "customer reports item never arrived"},
    )
    request = ReviewRequest(id="interactive-1", action=action, context="tracking shows no delivery scan in 12 days")
    outcome = run_gate(request, registry, InteractiveDecisionSource(), audit_log)
    print(f"\nresult: {format_outcome(outcome)}")
    print(format_audit_log(audit_log))


if __name__ == "__main__":
    main()
