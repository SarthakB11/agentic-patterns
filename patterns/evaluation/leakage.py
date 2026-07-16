"""Preference leakage: measuring judge bias toward a related generator.

Source: Li et al., "Preference Leakage: A Contamination Problem in
LLM-as-a-judge" (arXiv:2502.01534). `ensemble.py` already cites this paper
and states its mitigation rule, a jury only reduces bias if its members are
not related to the generator, but never measures the failure the rule
guards against. This module makes it observable: a preference-leakage score
is the win-rate gap a judge gives a *related* generator's output over an
*unrelated* generator's output, on two outputs authored to be equal in
quality. Any nonzero gap is then contamination, not merit.

The paper separates three relatedness tiers: same model, shared parent
(inheritance), and same family. This module models all three as three
scripted judges of decreasing bias strength, comparing the same pair of
equal-quality outputs, so the leakage score shrinks as relatedness weakens,
and ties the result back to `ensemble.py`'s mitigation: substituting an
unrelated judge for a related one collapses the score.

Reuses `pairwise.run_pairwise_judgment` (bias-cancelled by presentation
order, so a leakage measurement is not confounded with plain position bias)
and `aggregate.pairwise_win_rate` for the roll-up.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Provider, get_provider
from patterns.evaluation.aggregate import pairwise_win_rate
from patterns.evaluation.eval_set import EvalCase, get_case
from patterns.evaluation.pairwise import run_pairwise_judgment

DEFAULT_LEAKAGE_CEILING = 0.2


@dataclass
class LeakageResult:
    """The outcome of measuring one judge's preference leakage.

    Attributes:
        tier: Label for the relatedness tier under test, e.g. "same_model",
            "inheritance", or "unrelated".
        winners: One winner per case: "candidate_a" (the related output),
            "candidate_b" (the unrelated output), or "tie".
        leakage_score: Related generator's win rate minus the unrelated
            generator's win rate. Zero means no preference; positive means
            the judge leans toward the related generator.
        leakage_detected: True if `leakage_score` exceeds the ceiling.
    """

    tier: str
    winners: list[str]
    leakage_score: float
    leakage_detected: bool


def measure_preference_leakage(
    provider: Provider,
    cases: list[EvalCase],
    related_output: str,
    unrelated_output: str,
    *,
    tier: str = "same_model",
    ceiling: float = DEFAULT_LEAKAGE_CEILING,
) -> LeakageResult:
    """Compare a related and an unrelated generator's output across cases and score the gap.

    Args:
        provider: The judge under test. Called twice per case (both
            presentation orders) via `run_pairwise_judgment`.
        cases: Eval cases to run the comparison over. `related_output` and
            `unrelated_output` are compared identically on every case, so
            they should be authored to fit all of them.
        related_output: Output attributed to a generator related to the
            judge (shared weights, a shared parent, or a shared family).
            Always presented as candidate_a to `run_pairwise_judgment`.
        unrelated_output: Output attributed to an unrelated generator,
            matched in quality to `related_output` by construction. Always
            presented as candidate_b.
        tier: Label for the relatedness tier this call represents.
        ceiling: Leakage score above which `leakage_detected` is True.
    """
    winners = [
        run_pairwise_judgment(provider, case, related_output, unrelated_output, reference_mode=False).winner
        for case in cases
    ]
    related_win_rate = pairwise_win_rate(winners, "candidate_a")
    unrelated_win_rate = pairwise_win_rate(winners, "candidate_b")
    leakage_score = related_win_rate - unrelated_win_rate
    return LeakageResult(
        tier=tier, winners=winners, leakage_score=leakage_score, leakage_detected=leakage_score > ceiling
    )


def run_leakage_demo() -> tuple[LeakageResult, LeakageResult, LeakageResult]:
    """Compare the same equal-quality outputs under three judges of decreasing relatedness.

    Both outputs below name the same policy facts in different phrasing, a
    paraphrase pair authored to be equal in quality, so any preference gap
    a judge shows between them is leakage, not merit.

    - "same_model" always prefers the related output regardless of slot:
      leakage score 1.0.
    - "inheritance" prefers the related output on two of three cases and
      calls the third a tie (disagreeing across orders): leakage score
      2/3, weaker than same_model but still clearly leaking.
    - "unrelated" splits its preferences with no lean toward either
      generator: leakage score 0.0, `leakage_detected=False`. Substituting
      this judge for the same_model judge is exactly the mitigation
      `ensemble.py` recommends, and here it visibly collapses the score.
    """
    cases = [get_case("refund_policy"), get_case("cancel_subscription"), get_case("refund_investigation")]
    related_output = "Refunds are issued within 30 days of purchase with your receipt or order number."
    unrelated_output = "If it has been under 30 days since you bought it, show your receipt or order number."

    # Each pair of lines is (order_ab, order_ba) for one case. "Related" is
    # candidate_a in order_ab and candidate_b in order_ba, so a judge that
    # tracks the related output's *identity* rather than its slot names the
    # letter that currently holds it in both lines of a pair.
    same_model_provider = get_provider(
        script=[
            "Candidate A states the policy most directly.\nWINNER: a",
            "Candidate B here states the policy most directly.\nWINNER: b",
            "Candidate A states the policy most directly.\nWINNER: a",
            "Candidate B here states the policy most directly.\nWINNER: b",
            "Candidate A states the policy most directly.\nWINNER: a",
            "Candidate B here states the policy most directly.\nWINNER: b",
        ]
    )
    inheritance_provider = get_provider(
        script=[
            "Candidate A is a touch more direct.\nWINNER: a",
            "Candidate B here is a touch more direct.\nWINNER: b",
            "Candidate A reads slightly cleaner.\nWINNER: a",
            "Candidate B here reads slightly cleaner.\nWINNER: b",
            "Both cover the same facts equally well, marginal edge to A.\nWINNER: a",
            "Both cover the same facts equally well, marginal edge to A.\nWINNER: a",
        ]
    )
    unrelated_provider = get_provider(
        script=[
            "Both cover the same facts equally well, marginal edge to A.\nWINNER: a",
            "Both cover the same facts equally well, marginal edge to A.\nWINNER: a",
            "Candidate B leads with the condition, clearer phrasing.\nWINNER: b",
            "Candidate A here leads with the condition, clearer phrasing.\nWINNER: a",
            "Candidate A states the policy most directly.\nWINNER: a",
            "Candidate B here states the policy most directly.\nWINNER: b",
        ]
    )

    same_model = measure_preference_leakage(
        same_model_provider, cases, related_output, unrelated_output, tier="same_model"
    )
    inheritance = measure_preference_leakage(
        inheritance_provider, cases, related_output, unrelated_output, tier="inheritance"
    )
    unrelated = measure_preference_leakage(
        unrelated_provider, cases, related_output, unrelated_output, tier="unrelated"
    )
    return same_model, inheritance, unrelated
