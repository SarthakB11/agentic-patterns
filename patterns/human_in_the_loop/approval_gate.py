"""Sub-module: the approval gate base variant (approve / edit / reject / respond).

This is the pattern's base form and the single end-to-end example the
checklist calls for: one task walks through `run_gate` and the reader sees
the proposed action, the human decision, and the result, for each of the
four decisions in turn. A fifth demo shows the fail-closed default: a
malformed decision blocks the action instead of running it.

Every demo calls the provider twice: once so the model proposes a tool
call, and once more after the gate resolves, so the model's own reasoning
stays consistent with what actually happened, the way a resumed agent loop
would see it.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, Provider, get_provider, scripted_tool_call

from patterns.human_in_the_loop.fake_tools import build_refund_registry
from patterns.human_in_the_loop.gate import (
    AuditLog,
    Decision,
    GateOutcome,
    ReviewRequest,
    ScriptedDecisionSource,
    UnauthorizedDecisionError,
    counting_clock,
    run_gate,
)

_SYSTEM = (
    "You are a support agent empowered to issue refunds. For anything "
    "outside standard policy you propose the action and wait for review "
    "before it takes effect."
)


@dataclass
class ScenarioResult:
    """The full transcript of one gate scenario, for printing and assertions.

    Attributes:
        task: The user-facing task the agent was given.
        proposed_arguments: Arguments the model proposed before review.
        outcome: The gate's outcome, or None if the gate raised.
        error: The exception message if the gate raised, else None.
        ledger: The refund ledger after the scenario ran.
        audit_log: The audit log the scenario wrote to.
        follow_up: The model's final message after seeing the outcome.
    """

    task: str
    proposed_arguments: dict
    outcome: GateOutcome | None
    error: str | None
    ledger: list
    audit_log: AuditLog
    follow_up: str


def _run_scenario(provider: Provider, decision: Decision, task: str) -> ScenarioResult:
    """Drive one propose -> gate -> follow-up cycle and collect the transcript."""
    registry, ledger = build_refund_registry()
    audit_log = AuditLog()
    decision_source = ScriptedDecisionSource([decision])
    clock = counting_clock()

    proposal = provider.complete([Message.user(task)], system=_SYSTEM)
    call = proposal.tool_calls[0]
    request = ReviewRequest(id="req-1", action=call, context=task)

    outcome: GateOutcome | None = None
    error: str | None = None
    try:
        outcome = run_gate(request, registry, decision_source, audit_log, clock=clock)
        observation = outcome.tool_result
    except UnauthorizedDecisionError as exc:
        error = str(exc)
        observation = f"BLOCKED: {error}"

    follow_up = provider.complete(
        [
            Message.user(task),
            Message.assistant("", tool_calls=[call]),
            Message.tool(call.id, observation),
        ],
        system=_SYSTEM,
    )
    return ScenarioResult(
        task=task,
        proposed_arguments=call.arguments,
        outcome=outcome,
        error=error,
        ledger=ledger,
        audit_log=audit_log,
        follow_up=follow_up.content,
    )


def run_approve_demo(provider: Provider | None = None) -> ScenarioResult:
    """Approve: the action runs exactly as proposed."""
    task = (
        "Customer c-4471 was double-charged $42.50 for order #8823. "
        "Issue a refund for the duplicate charge."
    )
    if provider is None:
        provider = get_provider(
            script=[
                scripted_tool_call(
                    "send_refund",
                    {"customer_id": "c-4471", "amount_usd": 42.50, "reason": "duplicate charge on order #8823"},
                ),
                "Refund of $42.50 sent to c-4471 for the duplicate charge on order #8823. Ticket resolved.",
            ]
        )
    decision = Decision(
        kind="approve", reviewer="ops-lead-dana", reason="duplicate charge confirmed in the billing log"
    )
    return _run_scenario(provider, decision, task)


def run_edit_demo(provider: Provider | None = None) -> ScenarioResult:
    """Edit: the reviewer amends the arguments before the action runs."""
    task = (
        "Customer c-5190 asks for a $500 refund for a delayed shipment. "
        "Our policy caps delay-related refunds at $75; propose the refund."
    )
    if provider is None:
        provider = get_provider(
            script=[
                scripted_tool_call(
                    "send_refund",
                    {"customer_id": "c-5190", "amount_usd": 500.00, "reason": "delayed shipment goodwill refund"},
                ),
                "Refund adjusted to $75.00 per the delay-refund policy cap and sent to c-5190.",
            ]
        )
    decision = Decision(
        kind="edit",
        reviewer="ops-lead-dana",
        reason="capped to the policy limit for shipment delays",
        arguments={
            "customer_id": "c-5190",
            "amount_usd": 75.00,
            "reason": "delayed shipment goodwill refund, capped to policy limit",
        },
    )
    return _run_scenario(provider, decision, task)


def run_reject_demo(provider: Provider | None = None) -> ScenarioResult:
    """Reject: no side effect; the reviewer's reason is fed back to the model."""
    task = (
        "Customer c-2208 requests a $1200 refund for a purchase made 14 "
        "months ago, outside our 90-day refund window."
    )
    if provider is None:
        provider = get_provider(
            script=[
                scripted_tool_call(
                    "send_refund",
                    {"customer_id": "c-2208", "amount_usd": 1200.00, "reason": "refund request outside 90-day window"},
                ),
                (
                    "I will not process this refund; the purchase is outside the "
                    "90-day window. Offering c-2208 a store credit instead and "
                    "closing the refund request."
                ),
            ]
        )
    decision = Decision(
        kind="reject",
        reviewer="ops-lead-dana",
        reason="purchase is 14 months old, outside the 90-day refund window; offer store credit instead",
    )
    return _run_scenario(provider, decision, task)


def run_respond_demo(provider: Provider | None = None) -> ScenarioResult:
    """Respond: the gate fetches information rather than policing a side effect."""
    task = (
        "Before approving a goodwill credit for customer c-77, confirm "
        "their loyalty tier, since that sets the credit ceiling."
    )
    if provider is None:
        provider = get_provider(
            script=[
                scripted_tool_call("lookup_customer_tier", {"customer_id": "c-77"}),
                "c-77 is Platinum tier, so I can issue a goodwill credit of up to $150. Proceeding with $100.",
            ]
        )
    decision = Decision(
        kind="respond",
        reviewer="ops-lead-dana",
        value="c-77 is Platinum tier, eligible for up to $150 in goodwill credit.",
    )
    return _run_scenario(provider, decision, task)


def run_fail_closed_demo(provider: Provider | None = None) -> ScenarioResult:
    """Fail-closed: an unrecognized decision kind blocks the action."""
    task = "Customer c-9001 requests a $60 refund for a mis-shipped item."
    if provider is None:
        provider = get_provider(
            script=[
                scripted_tool_call(
                    "send_refund",
                    {"customer_id": "c-9001", "amount_usd": 60.00, "reason": "mis-shipped item"},
                ),
                "I could not confirm the refund went through; holding the ticket open for a human to check.",
            ]
        )
    # "acknowledge" is not a decision kind run_gate recognizes.
    decision = Decision(kind="acknowledge", reviewer="ops-lead-dana", reason="saw the request")
    return _run_scenario(provider, decision, task)
