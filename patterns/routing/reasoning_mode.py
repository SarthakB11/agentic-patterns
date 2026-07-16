"""Sub-module: reasoning-mode routing.

Routes whether the model should think step by step, not which model or
category answers. This is a distinct axis from capability selection: two
questions can go to the same model and still need different handling, one
answered directly and one worked through. "When to Reason: Semantic Router
for vLLM" (Wang et al., arXiv:2510.08731) reports a 10.2-point MMLU-Pro
gain with 47.1% lower latency and 48.5% fewer tokens versus always
reasoning, by routing only the prompts that benefit from chain-of-thought
into a reasoning mode. GPT-5.x's adaptive reasoning does a version of this
inside the provider; this module does the app-level equivalent with an
explicit, inspectable rule.

The classifier here is a lightweight heuristic (multi-step or analytical
language, or a long prompt), not a model call, since deciding whether to
spend the tokens on reasoning should itself stay cheap.

`enforce_reasoning_safety` is the correction the expansion research adds:
`escalation.py` already treats a sensitive-topic flag as an override that
always wins for the category/tier axis, but nothing stopped a sensitive or
adversarial-flavored prompt from landing on the direct (no-reasoning) mode,
which is the reasoning-axis version of the same misroute-downward failure
Kassem, Scholkopf, and Jin document for category routers (arXiv:2504.07113).
This reuses `escalation.is_sensitive` rather than duplicating the keyword
list, so both axes stay in sync.
"""

from __future__ import annotations

import re

from agentic_patterns import Message, Provider, get_provider
from patterns.routing.escalation import EscalationPolicy, is_sensitive
from patterns.routing.registry import RouteDecision

_REASONING_SIGNALS = re.compile(
    r"\b(calculate|compute|derive|compare|optimi[sz]e|how many|step by step|total cost|percentage|prove)\b",
    re.IGNORECASE,
)
_LONG_PROMPT_WORDS = 25

_DIRECT_SYSTEM = "Answer directly in one or two sentences. Do not show your reasoning steps."
_REASON_SYSTEM = (
    "Think through this step by step before answering. Show the key steps, "
    "then give the final answer on its own line prefixed with 'Answer:'."
)


def classify_reasoning_mode(text: str) -> RouteDecision:
    """Decide whether `text` benefits from an explicit reasoning pass.

    A prompt routes to "reason" if it contains multi-step or analytical
    language (see `_REASONING_SIGNALS`) or is long enough that it likely
    bundles more than one sub-question; otherwise it routes to "direct".
    """
    needs_reasoning = bool(_REASONING_SIGNALS.search(text)) or len(text.split()) > _LONG_PROMPT_WORDS
    if needs_reasoning:
        return RouteDecision(route="reason", score=1.0, method="reasoning_mode", metadata={"signal": "matched"})
    return RouteDecision(route="direct", score=0.0, method="reasoning_mode", metadata={"signal": "none"})


def enforce_reasoning_safety(
    decision: RouteDecision, text: str, policy: EscalationPolicy | None = None
) -> RouteDecision:
    """Override a "direct" decision to "reason" when `text` is sensitive.

    A sensitive or high-stakes prompt should never be routed to the mode
    that skips the reasoning pass, mirroring `escalation.apply_escalation`'s
    rule that a sensitive flag overrides a confident decision. A prompt
    already routed to "reason" is left alone; this only ever moves a
    decision up toward more care, never down.

    Args:
        decision: The upstream reasoning-mode decision.
        text: The original input, checked for sensitive-topic keywords.
        policy: Sensitivity keywords to check against; defaults to
            `escalation.EscalationPolicy()`'s keyword list, shared with the
            category/tier escalation axis.
    """
    policy = policy or EscalationPolicy()
    if decision.route == "direct" and is_sensitive(text, policy):
        return RouteDecision(
            route="reason",
            score=1.0,
            method="reasoning_mode",
            metadata={**decision.metadata, "safety_override": True},
        )
    return decision


def answer_with_reasoning_mode(text: str, provider: Provider, decision: RouteDecision | None = None) -> str:
    """Answer `text` using the system prompt the reasoning-mode route selects.

    Args:
        text: The question to answer.
        provider: Scripted with the answer matching the selected mode.
        decision: Precomputed routing decision; if omitted, classified fresh
            from `text`.
    """
    decision = decision or classify_reasoning_mode(text)
    system = _REASON_SYSTEM if decision.route == "reason" else _DIRECT_SYSTEM
    return provider.complete([Message.user(text)], system=system).content


def run_reasoning_mode_demo() -> tuple[RouteDecision, str, RouteDecision, str]:
    """Route and answer one simple lookup and one multi-step calculation.

    Returns:
        A (simple_decision, simple_answer, complex_decision, complex_answer)
        tuple, showing the direct route skipping chain-of-thought and the
        reason route showing its steps.
    """
    simple_question = "What is the capital of Japan?"
    simple_decision = classify_reasoning_mode(simple_question)
    simple_provider = get_provider(script=["Tokyo."])
    simple_answer = answer_with_reasoning_mode(simple_question, simple_provider, simple_decision)

    complex_question = (
        "A subscription costs $12/month and a customer upgraded halfway through a 30-day "
        "cycle to a $20/month plan; calculate their prorated total cost for the month."
    )
    complex_decision = classify_reasoning_mode(complex_question)
    complex_provider = get_provider(
        script=[
            "Step 1: 15 days at $12/month = $6.00. Step 2: 15 days at $20/month = $10.00. "
            "Step 3: $6.00 + $10.00 = $16.00.\nAnswer: $16.00"
        ]
    )
    complex_answer = answer_with_reasoning_mode(complex_question, complex_provider, complex_decision)

    return simple_decision, simple_answer, complex_decision, complex_answer
