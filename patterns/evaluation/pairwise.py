"""Pairwise comparison judge with position-bias cancellation.

The judge sees two candidates and picks the better one, or declares a tie.
Comparative judgments are more stable than absolute scores, but pairwise
judging is the variant most exposed to position bias: a judge that leans
toward whichever candidate happens to appear first. This module runs every
comparison in both orderings (A,B) and (B,A) and aggregates them: only a
candidate preferred in both orderings wins outright, and orderings that
disagree fall back to a tie rather than trusting either single call
(arXiv:2306.05685).
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, Provider, get_provider

from patterns.evaluation.eval_set import EvalCase, get_case
from patterns.evaluation.verdict import PairwiseVerdict, parse_pairwise_verdict

_PAIRWISE_SYSTEM = (
    "Compare two candidate replies to a support task and pick the better "
    "one. Consider accuracy against any reference given and whether the "
    "reply names a concrete next step. End with a WINNER line: a, b, or "
    "tie."
)


@dataclass
class PairwiseResult:
    """The outcome of a bias-cancelled pairwise comparison.

    Attributes:
        order_ab: The judge's verdict when candidate_a was shown first.
        order_ba: The judge's verdict when candidate_b was shown first.
        winner: "candidate_a", "candidate_b", or "tie", after translating
            both orderings back to candidate identity and requiring
            agreement.
        position_bias_detected: True if the two orderings disagreed once
            translated back to candidate identity, meaning the raw
            preference tracked slot position rather than content.
    """

    order_ab: PairwiseVerdict
    order_ba: PairwiseVerdict
    winner: str
    position_bias_detected: bool


def _translate(verdict: PairwiseVerdict, *, swapped: bool) -> str:
    """Map a slot-relative winner ("a"/"b"/"tie") back to candidate identity."""
    if verdict.winner == "tie":
        return "tie"
    if not swapped:
        return "candidate_a" if verdict.winner == "a" else "candidate_b"
    return "candidate_b" if verdict.winner == "a" else "candidate_a"


def run_pairwise_judgment(
    provider: Provider,
    case: EvalCase,
    candidate_a: str,
    candidate_b: str,
    *,
    reference_mode: bool = True,
) -> PairwiseResult:
    """Compare two candidates in both presentation orders and aggregate.

    Args:
        provider: The model that plays the judge. Called exactly twice.
        case: The eval case both candidates are answering.
        candidate_a: First candidate's output.
        candidate_b: Second candidate's output.
        reference_mode: If True and `case.reference` is set, include it in
            the prompt (reference-based judging). If False, omit it
            (reference-free judging).
    """

    def build_prompt(first: str, second: str) -> str:
        parts = [f"Task:\n{case.input}"]
        if reference_mode and case.reference is not None:
            parts.append(f"Reference answer:\n{case.reference}")
        parts.append(f"Candidate A:\n{first}\n\nCandidate B:\n{second}")
        return "\n\n".join(parts)

    order_ab = parse_pairwise_verdict(
        provider.complete([Message.user(build_prompt(candidate_a, candidate_b))], system=_PAIRWISE_SYSTEM).content
    )
    order_ba = parse_pairwise_verdict(
        provider.complete([Message.user(build_prompt(candidate_b, candidate_a))], system=_PAIRWISE_SYSTEM).content
    )

    ab_pick = _translate(order_ab, swapped=False)
    ba_pick = _translate(order_ba, swapped=True)
    if ab_pick == ba_pick:
        winner, bias = ab_pick, False
    else:
        winner, bias = "tie", True

    return PairwiseResult(order_ab=order_ab, order_ba=order_ba, winner=winner, position_bias_detected=bias)


def run_pairwise_fair_demo(provider: Provider | None = None) -> PairwiseResult:
    """A comparison where candidate_b genuinely wins in both orderings.

    Args:
        provider: Judge provider. Defaults to a `MockProvider` scripted to
            prefer the specific candidate content (the one naming a next
            step), regardless of which slot it appears in.
    """
    case = get_case("cancel_subscription")
    candidate_a = "You can cancel any time from your account page."
    candidate_b = (
        "Go to Account Settings, select Subscription, and click Cancel. "
        "Access continues until the end of the current billing period."
    )
    if provider is None:
        provider = get_provider(
            script=[
                "Candidate B gives the exact menu path and states when access "
                "ends; Candidate A is vague about both.\nWINNER: b",
                "Candidate A here is vague about the menu path and access end "
                "date; Candidate B gives both precisely.\nWINNER: a",
            ]
        )
    return run_pairwise_judgment(provider, case, candidate_a, candidate_b)


def run_pairwise_biased_demo(provider: Provider | None = None) -> PairwiseResult:
    """A comparison where the judge always prefers whichever slot is first.

    Args:
        provider: Judge provider. Defaults to a `MockProvider` scripted to
            always return `WINNER: a`, modeling a judge with pure position
            bias and no real content preference. The two candidates are
            near-identical in quality on purpose, so a real judge would be
            expected to call it close either way.
    """
    case = get_case("cancel_subscription")
    candidate_a = "Cancel anytime from Account Settings under Subscription."
    candidate_b = "You can cancel your subscription from the Subscription tab in Account Settings."
    if provider is None:
        provider = get_provider(
            script=[
                "Candidate A is slightly more concise.\nWINNER: a",
                "Candidate A is slightly more concise.\nWINNER: a",
            ]
        )
    return run_pairwise_judgment(provider, case, candidate_a, candidate_b)
