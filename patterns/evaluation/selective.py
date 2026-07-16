"""Selective judging: confidence-calibrated abstention and escalation.

Source: Jung, Brahman, Choi, "Trust or Escalate" (arXiv:2407.18370). Every
judge elsewhere in this pattern answers every case unconditionally, so a
low-confidence verdict counts as much as a high-confidence one. This module
estimates a judge's confidence per case the paper's Simulated Annotators
way, resample the same judge `r` times and take the fraction agreeing with
the majority, and abstains below a threshold `tau`, trading coverage (the
fraction of cases answered) for reliability on the cases it does answer.
Resamples are scripted completions, so confidence is deterministic.

Boundary note: routing an abstained case through a cheaper-then-stronger
cascade is routing territory (`patterns/routing/`). This module keeps only
confidence calibration, abstention, and the coverage-versus-agreement
curve; `escalate` is an abstract callable, not a general router.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agentic_patterns import MockProvider, Provider, get_provider
from patterns.evaluation.aggregate import pass_rate
from patterns.evaluation.eval_set import EvalCase, get_case
from patterns.evaluation.meta import cohens_kappa
from patterns.evaluation.pointwise import build_pointwise_judge
from patterns.evaluation.verdict import Verdict

DEFAULT_RESAMPLES = 3
DEFAULT_TAU = 0.67


def estimate_confidence(resample_verdicts: list[Verdict]) -> tuple[bool, float]:
    """Return the majority `passed` value and confidence: the fraction of resamples agreeing with it.

    `resample_verdicts` are `r` calls to the same judge on the same case
    (the Simulated Annotators stand-in). Confidence 1.0 is unanimous; lower
    means the judge is unstable on this case.

    Raises:
        ValueError: If `resample_verdicts` is empty.
    """
    if not resample_verdicts:
        raise ValueError("estimate_confidence requires at least one resample")
    passed_votes = [bool(v.passed) for v in resample_verdicts]
    majority = sum(passed_votes) > len(passed_votes) / 2
    agreeing = sum(1 for p in passed_votes if p == majority)
    return majority, agreeing / len(passed_votes)


@dataclass
class SelectiveVerdict:
    """One case's outcome under selective judging.

    Attributes:
        case_id: The `EvalCase.id` this verdict belongs to.
        resample_verdicts: The `r` raw verdicts confidence was computed from.
        confidence: Fraction of resamples agreeing with the majority.
        answered: True if `confidence >= tau`.
        verdict: Representative verdict carrying the majority's score and
            reasoning, set only when `answered` is True.
        escalated: True if this case abstained and was routed to `escalate`.
    """

    case_id: str
    resample_verdicts: list[Verdict]
    confidence: float
    answered: bool
    verdict: Verdict | None
    escalated: bool = False


@dataclass
class SelectiveResult:
    """The outcome of running selective judging over a batch of cases.

    Attributes:
        verdicts: One `SelectiveVerdict` per case, in input order.
        coverage: Fraction of cases answered rather than abstained.
        selective_agreement: Chance-corrected agreement (`cohens_kappa`)
            over answered cases only. 0.0 if no case was answered.
        abstained_count: Number of cases that abstained.
        escalated_count: Number of abstained cases routed to `escalate`.
    """

    verdicts: list[SelectiveVerdict]
    coverage: float
    selective_agreement: float
    abstained_count: int
    escalated_count: int


def run_selective_judgment(
    provider: Provider,
    case: EvalCase,
    output: str,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    tau: float = DEFAULT_TAU,
    reference_mode: bool = True,
) -> SelectiveVerdict:
    """Judge one case with confidence-calibrated abstention.

    Calls the base pointwise judge `resamples` times on `(case, output)`,
    estimates confidence via `estimate_confidence`, and answers with the
    majority verdict only if confidence is at or above `tau`.
    """
    judge = build_pointwise_judge(provider, reference_mode=reference_mode)
    resample_verdicts = [judge(case, output) for _ in range(resamples)]
    majority_passed, confidence = estimate_confidence(resample_verdicts)
    answered = confidence >= tau
    representative = None
    if answered:
        representative = next((v for v in resample_verdicts if v.passed == majority_passed), resample_verdicts[0])
    return SelectiveVerdict(
        case_id=case.id,
        resample_verdicts=resample_verdicts,
        confidence=confidence,
        answered=answered,
        verdict=representative,
    )


def run_selective_evaluation(
    provider: Provider,
    cases_and_outputs: list[tuple[EvalCase, str]],
    human_labels: list[bool],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    tau: float = DEFAULT_TAU,
    escalate: Callable[[EvalCase, str], Verdict] | None = None,
) -> SelectiveResult:
    """Run selective judgment over a batch of cases and roll up coverage and agreement.

    Args:
        provider: The model that plays the judge.
        cases_and_outputs: One (case, candidate output) pair per case.
        human_labels: One human pass/fail label per case, same order.
        resamples: Passed through to `run_selective_judgment`.
        tau: Passed through to `run_selective_judgment`.
        escalate: Called once per abstained case, e.g. a stronger scripted
            judge or a human-label lookup. If None, abstained cases are
            simply left unanswered.

    Raises:
        ValueError: If `cases_and_outputs` and `human_labels` differ in length.
    """
    if len(cases_and_outputs) != len(human_labels):
        raise ValueError("cases_and_outputs and human_labels must be the same length")

    verdicts: list[SelectiveVerdict] = []
    for case, output in cases_and_outputs:
        sv = run_selective_judgment(provider, case, output, resamples=resamples, tau=tau)
        if not sv.answered and escalate is not None:
            escalate(case, output)
            sv.escalated = True
        verdicts.append(sv)

    coverage = pass_rate([sv.answered for sv in verdicts])
    answered_judge = [bool(sv.verdict.passed) for sv in verdicts if sv.answered and sv.verdict is not None]
    answered_human = [h for sv, h in zip(verdicts, human_labels) if sv.answered]
    selective_agreement = cohens_kappa(answered_judge, answered_human) if answered_judge else 0.0
    abstained_count = sum(1 for sv in verdicts if not sv.answered)
    escalated_count = sum(1 for sv in verdicts if sv.escalated)
    return SelectiveResult(
        verdicts=verdicts,
        coverage=coverage,
        selective_agreement=selective_agreement,
        abstained_count=abstained_count,
        escalated_count=escalated_count,
    )


def run_selective_demo() -> SelectiveResult:
    """Judge three cases: two confidently, one split enough to abstain and escalate.

    Cases 1 and 3 get unanimous 3-of-3 resample verdicts, both answered at
    the default `tau=0.67`. Case 2 gets a 2-1 split (confidence 2/3, just
    under tau) and abstains, escalating to a stand-in human-review stub.
    Coverage over the three cases is 2/3.
    """
    cases_and_outputs = [
        (get_case("refund_policy"), "Refunds are available within 30 days if you have your order number."),
        (get_case("order_status_lookup"), "Order 48213 is out for delivery."),
        (get_case("cancel_subscription"), "Go to Account Settings, select Subscription, and click Cancel."),
    ]
    human_labels = [True, True, True]
    provider = get_provider(
        script=[
            "Matches the reference on window and proof needed.\nSCORE: 9\nVERDICT: pass",
            "Matches the reference on window and proof needed.\nSCORE: 8\nVERDICT: pass",
            "Matches the reference on window and proof needed.\nSCORE: 9\nVERDICT: pass",
            "Names the order and a status.\nSCORE: 8\nVERDICT: pass",
            "States a status but the wording is ambiguous about delivery.\nSCORE: 5\nVERDICT: fail",
            "Names the order and a status.\nSCORE: 8\nVERDICT: pass",
            "Names the exact menu path.\nSCORE: 9\nVERDICT: pass",
            "Names the exact menu path.\nSCORE: 8\nVERDICT: pass",
            "Names the exact menu path.\nSCORE: 9\nVERDICT: pass",
        ]
    )

    def escalate(case: EvalCase, output: str) -> Verdict:
        return Verdict(score=None, passed=True, reasoning="escalated to human review", raw="", malformed=False)

    return run_selective_evaluation(provider, cases_and_outputs, human_labels, escalate=escalate)


def run_coverage_tau_curve_demo() -> list[SelectiveResult]:
    """Run one fixed judge script at two thresholds to show the coverage-agreement tradeoff.

    Two cases are unanimous (confidence 1.0, agreeing with their human
    label); two are 2-1 splits (confidence 0.67, one agreeing, one not). At
    `tau=0.5` all four answer, and the split disagreement pulls
    `selective_agreement` down. At `tau=0.7` only the unanimous pair clears
    the bar: coverage falls 1.0 to 0.5 while `selective_agreement` rises to
    perfect agreement, the coverage-reliability tradeoff the paper names.
    """
    cases_and_outputs = [
        (get_case("refund_policy"), "unanimous pass output"),
        (get_case("cancel_subscription"), "unanimous fail output"),
        (get_case("order_status_lookup"), "split pass output, human disagrees"),
        (get_case("refund_investigation"), "split fail output, human agrees"),
    ]
    human_labels = [True, False, False, False]
    script = [
        "Solid reply.\nSCORE: 9\nVERDICT: pass",
        "Solid reply.\nSCORE: 8\nVERDICT: pass",
        "Solid reply.\nSCORE: 9\nVERDICT: pass",
        "Misses the point.\nSCORE: 3\nVERDICT: fail",
        "Misses the point.\nSCORE: 2\nVERDICT: fail",
        "Misses the point.\nSCORE: 3\nVERDICT: fail",
        "Reads fine on its own.\nSCORE: 7\nVERDICT: pass",
        "Missing a detail a human would flag.\nSCORE: 5\nVERDICT: fail",
        "Reads fine on its own.\nSCORE: 7\nVERDICT: pass",
        "Skips a verification step.\nSCORE: 4\nVERDICT: fail",
        "Borderline, could pass.\nSCORE: 6\nVERDICT: pass",
        "Skips a verification step.\nSCORE: 4\nVERDICT: fail",
    ]
    return [
        run_selective_evaluation(MockProvider(list(script)), cases_and_outputs, human_labels, tau=tau)
        for tau in (0.5, 0.7)
    ]
