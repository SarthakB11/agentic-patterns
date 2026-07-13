"""Sub-module: model-judged risk tiering with a cheap rule tier in front.

`risk_tier.py` decides with one scalar (`amount_usd > threshold`). Real
2025-2026 guards reserve rules for the obvious ends of the risk range and
route the ambiguous middle to a model judge instead. Magentic-UI (Mozannar
et al., arXiv:2507.22358): an action is always irreversible (gate by
rule), never irreversible (auto-allow by rule), or maybe irreversible, and
only the "maybe" case reaches an LLM judge. Spider-Sense (Yu et al.,
arXiv:2602.05386) generalizes this into "hierarchical adaptive screening":
a cheap tier resolves known-safe and known-bad patterns, only ambiguous
cases escalate to deeper model reasoning.

The escalation decision is itself just another `Provider.complete()` call.
Under `MockProvider` the judge's verdict is scripted text like any other
completion, parsed here into a `RiskVerdict`, so the two-tier screen is
fully deterministic offline: the direct rebuttal to the earlier README
claim (now corrected) that a learned risk trigger needs a trained detector
a scripted mock cannot produce. Only Spider-Sense's version that probes
model internals is out of reach offline; the screening shape is not.

The rule tier also bounds cost: the judge fires only on the ambiguous
middle, not on every action, which is cheaper than model-judging (or
gating) everything.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from agentic_patterns import Message, MockProvider, Provider, ToolCall, ToolRegistry

from patterns.human_in_the_loop.fake_tools import build_extended_ops_registry
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

ALWAYS_GATE_TOOLS = frozenset({"cancel_subscription"})
NEVER_GATE_TOOLS = frozenset({"lookup_customer_tier"})
HARD_GATE_AMOUNT = 5000.0
TRIVIAL_FLOOR_AMOUNT = 10.0

JUDGE_SYSTEM_PROMPT = (
    "You are a risk-tiering judge for a support agent's proposed action. Decide "
    "whether a human must review it before it runs. Reply in the exact form "
    "'GATE: <reason>' if a human should review it, or 'AUTO: <reason>' if it is "
    "safe to execute automatically. Optionally append a risk score in "
    "parentheses, for example 'GATE: ambiguous duplicate claim (score: 7)'."
)


@dataclass
class RiskVerdict:
    """Outcome of classifying one proposed action's risk.

    Attributes:
        tier: "rule" if resolved with no model call, else "judge".
        route: "always_gate" or "never_gate" for a rule verdict; "gate" or
            "auto" for a judge verdict.
        reason: Explanation, threaded into the review request's context
            when the route is "always_gate" or "gate".
        score: The judge's optional 0-10 risk score, else None.
    """

    tier: str
    route: str
    reason: str
    score: float | None = None


def rule_tier(
    action: ToolCall,
    *,
    always_gate_tools: frozenset[str] = ALWAYS_GATE_TOOLS,
    never_gate_tools: frozenset[str] = NEVER_GATE_TOOLS,
    hard_gate_amount: float = HARD_GATE_AMOUNT,
    trivial_floor: float = TRIVIAL_FLOOR_AMOUNT,
) -> RiskVerdict | None:
    """Resolve an action by rule alone, or return None if it is ambiguous.

    Args:
        action: The proposed tool call.
        always_gate_tools: Tool names that always require review.
        never_gate_tools: Read-only tool names that never require review.
        hard_gate_amount: Dollar amount past which review is always required.
        trivial_floor: Dollar amount under which review is never required.
    """
    amount = float(action.arguments.get("amount_usd", 0.0))
    if action.name in always_gate_tools:
        return RiskVerdict(tier="rule", route="always_gate", reason=f"{action.name} is in the always-gate tool set")
    if amount > hard_gate_amount:
        return RiskVerdict(
            tier="rule", route="always_gate",
            reason=f"amount ${amount:.2f} exceeds the hard gate line of ${hard_gate_amount:.2f}",
        )
    if action.name in never_gate_tools:
        return RiskVerdict(tier="rule", route="never_gate", reason=f"{action.name} is read-only")
    if "amount_usd" in action.arguments and amount < trivial_floor:
        return RiskVerdict(
            tier="rule", route="never_gate",
            reason=f"amount ${amount:.2f} is under the trivial floor of ${trivial_floor:.2f}",
        )
    return None


_VERDICT_PATTERN = re.compile(
    r"^(GATE|AUTO):\s*(.*?)(?:\s*\(score:\s*(\d+(?:\.\d+)?)\))?$", re.IGNORECASE | re.DOTALL
)


def parse_judge_verdict(text: str) -> RiskVerdict:
    """Parse a judge completion's content into a `RiskVerdict`.

    An unparseable verdict fails closed: treated as "gate", never "auto",
    the same fail-closed contract `run_gate` uses for a bad human decision.
    """
    match = _VERDICT_PATTERN.match(text.strip())
    if not match:
        return RiskVerdict(tier="judge", route="gate", reason=f"unparseable judge verdict, failing closed: {text!r}")
    kind, reason, score = match.groups()
    route = "gate" if kind.upper() == "GATE" else "auto"
    return RiskVerdict(tier="judge", route=route, reason=reason.strip(), score=float(score) if score else None)


def classify_risk(action: ToolCall, provider: Provider, *, context: str = "", **rule_kwargs: object) -> RiskVerdict:
    """Classify a proposed action's risk: rule tier first, judge only if ambiguous."""
    verdict = rule_tier(action, **rule_kwargs)  # type: ignore[arg-type]
    if verdict is not None:
        return verdict
    prompt = f"Action: {action.name}({action.arguments})\nContext: {context}"
    completion = provider.complete([Message.user(prompt)], system=JUDGE_SYSTEM_PROMPT)
    return parse_judge_verdict(completion.content)


def run_risk_classified_gate(
    request: ReviewRequest,
    registry: ToolRegistry,
    provider: Provider,
    decision_source,
    audit_log: AuditLog,
    *,
    clock: Callable[[], float] = time.time,
    **rule_kwargs: object,
) -> GateOutcome:
    """Route a proposed action through the rule tier, then the judge, then the gate.

    Args:
        request: The proposed action under review.
        registry: Where the action executes on a never-gate, auto, or
            human-approval route.
        provider: Consulted only for the judge tier, on an ambiguous action.
        decision_source: Where a human decision comes from, if gated.
        audit_log: Append-only log, written to for every route.
        clock: Timestamp source.
        **rule_kwargs: Forwarded to `rule_tier`.
    """
    action = request.action
    verdict = classify_risk(action, provider, context=request.context, **rule_kwargs)

    if verdict.route in ("never_gate", "auto"):
        now = clock()
        result = registry.execute(action)
        decision_kind = "auto_approved_rule_never_gate" if verdict.route == "never_gate" else "auto_approved_by_judge"
        reviewer = "risk_classifier" if verdict.route == "never_gate" else "risk_classifier_judge"
        audit_log.append(AuditRecord(
            request.id, action.name, action.arguments, decision_kind, action.arguments, reviewer, verdict.reason, now, now,
        ))
        return GateOutcome(kind="executed", tool_result=result, final_arguments=action.arguments)

    # verdict.route is "always_gate" (rule) or "gate" (judge): fall through to the human.
    gated = ReviewRequest(id=request.id, action=action, context=f"{request.context} | risk verdict: {verdict.reason}")
    return run_gate(gated, registry, decision_source, audit_log, clock=clock)


@dataclass
class RiskClassifierDemoResult:
    """Outcome of routing five actions through the two-tier risk screen.

    Attributes:
        outcomes: One `GateOutcome` per action, keyed by scenario name
            ("always_gate", "never_gate", "trivial", "judge_gate", "judge_auto").
        judge_calls_made: Provider calls made, proving the rule tier
            resolved three of five actions with zero model calls.
        ledger: The combined ledger after all five actions ran.
    """

    outcomes: dict[str, GateOutcome]
    judge_calls_made: int
    ledger: list


def run_risk_classifier_demo(provider: Provider | None = None) -> RiskClassifierDemoResult:
    """Five actions: three resolved by rule, two ambiguous ones routed to a judge."""
    registry, ledger = build_extended_ops_registry()
    audit_log = AuditLog()
    clock = counting_clock()
    if provider is None:
        provider = MockProvider([
            "GATE: refund reason is ambiguous relative to the order log (score: 7)",
            "AUTO: matches a routine, well-documented refund pattern (score: 2)",
        ])
    decision_source = ScriptedDecisionSource([
        Decision(kind="approve", reviewer="ops-lead-dana", reason="cancellation confirmed by the customer"),
        Decision(kind="approve", reviewer="ops-lead-dana", reason="checked against the order log, legitimate"),
    ])

    scenarios = [
        ("always_gate", "cancel_subscription", {"customer_id": "c-01", "reason": "customer requested cancellation"}),
        ("never_gate", "lookup_customer_tier", {"customer_id": "c-02"}),
        ("trivial", "send_refund", {"customer_id": "c-03", "amount_usd": 5.00, "reason": "shipping label reprint fee"}),
        ("judge_gate", "send_refund", {"customer_id": "c-04", "amount_usd": 300.00, "reason": "possible duplicate charge, unclear"}),
        ("judge_auto", "send_refund", {"customer_id": "c-05", "amount_usd": 150.00, "reason": "coupon applied twice"}),
    ]
    outcomes: dict[str, GateOutcome] = {}
    for name, tool_name, args in scenarios:
        request = ReviewRequest(id=f"req-{name}", action=ToolCall(id=f"req-{name}", name=tool_name, arguments=args), context=name)
        outcomes[name] = run_risk_classified_gate(request, registry, provider, decision_source, audit_log, clock=clock)

    judge_calls_made = len(provider.calls) if hasattr(provider, "calls") else -1
    return RiskClassifierDemoResult(outcomes=outcomes, judge_calls_made=judge_calls_made, ledger=ledger)
