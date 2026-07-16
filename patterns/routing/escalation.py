"""Sub-module: human-escalation routing.

The safe default when the machine should not decide alone: a route to a
person, taken when confidence is low or the topic is sensitive or
high-stakes, rather than forcing a low-confidence input onto whichever
route scored highest anyway.

Also encodes the safety correction from the expansion research: preference-
trained routers have been shown to misroute by category, sending
adversarial or sensitive inputs to a weaker handler (Kassem, Scholkopf,
Jin, "How Robust Are Router-LLMs?", arXiv:2504.07113). `apply_escalation`
therefore treats a sensitive-topic flag as an override that always wins,
regardless of how confident the upstream classifier was, and it only ever
escalates *up* toward "human": it never downgrades a decision to something
cheaper or weaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from patterns.routing.registry import RouteDecision

HUMAN_ROUTE = "human"

_SENSITIVE_KEYWORDS: tuple[str, ...] = (
    "self-harm",
    "suicide",
    "legal action",
    "lawsuit",
    "data breach",
    "threatening",
)


@dataclass
class EscalationPolicy:
    """Rules for when a `RouteDecision` should be overridden to human escalation.

    Attributes:
        threshold: A decision whose `score` is below this escalates, on the
            grounds that no route was confident enough to trust alone.
        sensitive_keywords: Substrings that flag an input as sensitive
            regardless of score, overriding even a fully confident decision.
    """

    threshold: float = 0.5
    sensitive_keywords: tuple[str, ...] = field(default_factory=lambda: _SENSITIVE_KEYWORDS)


def is_sensitive(text: str, policy: EscalationPolicy) -> bool:
    """Check whether `text` contains any of `policy.sensitive_keywords`.

    Whitespace is collapsed before matching (`"legal  action"`, extra
    spaces and all, still matches the keyword `"legal action"`), so a
    multi-word keyword's substring match does not depend on exact spacing.
    `robustness.py`'s whitespace perturbation is what surfaced this: a
    keyword match that only works on exact spacing should not be trusted
    to survive a paraphrase.
    """
    lowered = " ".join(text.lower().split())
    return any(keyword in lowered for keyword in policy.sensitive_keywords)


def apply_escalation(decision: RouteDecision, text: str, policy: EscalationPolicy | None = None) -> RouteDecision:
    """Override `decision` to human escalation if it fails the policy.

    Sensitive topics escalate even when `decision.score` is high, since a
    confident route to a weaker handler on a sensitive input is a worse
    failure than an unnecessary escalation. A decision with no score (for
    example, a rule-based miss) is treated as below threshold.

    Args:
        decision: The upstream classifier's decision.
        text: The original input, checked for sensitive-topic keywords.
        policy: Threshold and keyword list to apply. Defaults to
            `EscalationPolicy()`, built fresh per call rather than as a
            mutable default argument.
    """
    policy = policy or EscalationPolicy()
    sensitive = is_sensitive(text, policy)
    below_threshold = decision.score is None or decision.score < policy.threshold
    if not sensitive and not below_threshold:
        return decision

    reason = "sensitive_topic" if sensitive else "below_threshold"
    return RouteDecision(
        route=HUMAN_ROUTE,
        score=decision.score,
        method="escalation",
        attempts=decision.attempts,
        metadata={**decision.metadata, "escalation_reason": reason, "original_route": decision.route},
    )


def run_escalation_demo() -> tuple[RouteDecision, RouteDecision, RouteDecision]:
    """Show a low-score escalation, a sensitive-topic override, and a pass-through.

    Returns:
        A (low_score, sensitive_override, pass_through) triple. The middle
        case is the safety correction: the upstream decision is fully
        confident (score 0.9) and would normally dispatch straight to the
        "billing" route, but the input mentions legal action, so the
        policy overrides it to human escalation regardless of that
        confidence.
    """
    policy = EscalationPolicy()

    low_score_decision = RouteDecision(route="technical", score=0.31, method="semantic")
    low_score = apply_escalation(low_score_decision, "not sure what category this is", policy)

    confident_but_sensitive = RouteDecision(route="billing", score=0.9, method="semantic")
    sensitive_override = apply_escalation(
        confident_but_sensitive, "I'm considering legal action over this billing dispute", policy
    )

    confident_decision = RouteDecision(route="account", score=0.87, method="semantic")
    pass_through = apply_escalation(confident_decision, "I forgot my account password", policy)

    return low_score, sensitive_override, pass_through
