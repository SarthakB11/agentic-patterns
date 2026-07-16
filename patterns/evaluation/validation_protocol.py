"""Judge validation: the Minimum Viable Validation protocol.

Source: Norman, Rivera, Hughes, "Reliability without Validity: A Systematic,
Large-Scale Evaluation of LLM-as-a-Judge Models Across Agreement,
Consistency, and Bias" (arXiv:2606.19544).

`meta.py` already measures two of the paper's three validation axes:
chance-corrected agreement (`cohens_kappa`) and stability across repeated
runs (`test_retest_rate`). This module adds the third axis the folder was
missing, position bias, and the paper's actual deliverable: a single
accept-or-reject decision that requires all three axes to pass at once. A
judge is never trusted on the strength of one axis alone. The module also
names the paper's consistency-bias paradox directly: two production-deployed
judges in the study held test-retest reliability above 0.95 while also
carrying position bias above 0.10, meaning a judge can be highly
self-consistent and still be unfit to trust, since it is consistently
biased rather than consistently correct.

`position_bias_rate` reuses `pairwise.run_pairwise_judgment`'s own
order-cancellation logic (`PairwiseResult.position_bias_detected`) rather
than reimplementing the slot-to-identity translation: a judge validation
protocol should measure with the same mechanism the pattern already uses to
mitigate the failure, not a second copy of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import get_provider
from patterns.evaluation.eval_set import get_case
from patterns.evaluation.pairwise import PairwiseResult, run_pairwise_judgment

DEFAULT_AGREEMENT_FLOOR = 0.4
DEFAULT_CONSISTENCY_FLOOR = 0.95
DEFAULT_BIAS_CEILING = 0.10


def position_bias_rate(results: list[PairwiseResult]) -> float:
    """Return the fraction of comparisons where the two orderings disagreed.

    Args:
        results: One `PairwiseResult` per compared pair, each already
            carrying its own order-cancelled `position_bias_detected` flag.

    Raises:
        ValueError: If `results` is empty.
    """
    if not results:
        raise ValueError("position_bias_rate requires at least one comparison")
    return sum(1 for r in results if r.position_bias_detected) / len(results)


@dataclass
class JudgeValidation:
    """The outcome of the Minimum Viable Validation protocol on one judge.

    Attributes:
        kappa: Chance-corrected agreement with human labels
            (`meta.cohens_kappa`).
        test_retest: Same-verdict rate across two runs of the judge
            (`meta.test_retest_rate`).
        bias_rate: Fraction of swapped-order comparisons that disagreed
            (`position_bias_rate`).
        agreement_ok: True if `kappa` meets the agreement floor.
        consistency_ok: True if `test_retest` meets the consistency floor.
        bias_ok: True if `bias_rate` is at or under the bias ceiling.
        accepted: True only if all three axes passed. Never derived from
            any single axis.
        paradox_flag: True when `consistency_ok` holds but `bias_ok` does
            not: the study's named consistency-bias paradox, a judge that
            looks trustworthy on stability alone while carrying position
            bias severe enough to reject it.
    """

    kappa: float
    test_retest: float
    bias_rate: float
    agreement_ok: bool
    consistency_ok: bool
    bias_ok: bool
    accepted: bool
    paradox_flag: bool


def validate_judge(
    kappa: float,
    test_retest: float,
    bias_rate: float,
    *,
    agreement_floor: float = DEFAULT_AGREEMENT_FLOOR,
    consistency_floor: float = DEFAULT_CONSISTENCY_FLOOR,
    bias_ceiling: float = DEFAULT_BIAS_CEILING,
) -> JudgeValidation:
    """Run the joint accept/reject decision over all three validation axes.

    Args:
        kappa: Chance-corrected agreement with human labels.
        test_retest: Same-verdict rate across two runs of the judge.
        bias_rate: Fraction of swapped-order comparisons that disagreed.
        agreement_floor: Minimum acceptable kappa.
        consistency_floor: Minimum acceptable test-retest rate.
        bias_ceiling: Maximum acceptable bias rate.
    """
    agreement_ok = kappa >= agreement_floor
    consistency_ok = test_retest >= consistency_floor
    bias_ok = bias_rate <= bias_ceiling
    accepted = agreement_ok and consistency_ok and bias_ok
    paradox_flag = consistency_ok and not bias_ok
    return JudgeValidation(
        kappa=kappa,
        test_retest=test_retest,
        bias_rate=bias_rate,
        agreement_ok=agreement_ok,
        consistency_ok=consistency_ok,
        bias_ok=bias_ok,
        accepted=accepted,
        paradox_flag=paradox_flag,
    )


def run_position_bias_demo() -> tuple[list[PairwiseResult], float]:
    """Run three swapped-order comparisons: two flip, one agrees.

    Two pairs are scripted with a judge that always prefers whichever slot
    is first, so the two orderings disagree once translated back to
    candidate identity (position bias). The third pair is scripted with a
    judge that genuinely prefers one candidate's content regardless of
    slot, so the two orderings agree. The bias rate over the three pairs is
    2/3.
    """
    case = get_case("cancel_subscription")
    candidate_a = "Cancel anytime from Account Settings under Subscription."
    candidate_b = "You can cancel your subscription from the Subscription tab in Account Settings."
    provider = get_provider(
        script=[
            # Pair 1: slot-first bias, both orders pick slot "a".
            "Candidate A reads slightly cleaner.\nWINNER: a",
            "Candidate A reads slightly cleaner.\nWINNER: a",
            # Pair 2: slot-first bias again.
            "Candidate A is marginally more direct.\nWINNER: a",
            "Candidate A is marginally more direct.\nWINNER: a",
            # Pair 3: genuine content preference, agrees across orders.
            "Candidate B names the exact tab; more specific.\nWINNER: b",
            "Candidate A here names the exact tab; more specific.\nWINNER: a",
        ]
    )
    results = [
        run_pairwise_judgment(provider, case, candidate_a, candidate_b, reference_mode=False) for _ in range(3)
    ]
    return results, position_bias_rate(results)


def run_validation_protocol_demo() -> tuple[JudgeValidation, JudgeValidation]:
    """Validate a healthy judge and a paradox judge under the same protocol.

    The healthy judge is given a moderate kappa (0.6), a high test-retest
    rate (0.97), and a low, literal bias rate (0.05): it clears all three
    floors and is accepted with no paradox. The paradox judge is given the
    same kappa and test-retest rate but the bias rate measured mechanically
    by `run_position_bias_demo` (2/3, well past the 0.10 ceiling): despite
    near-perfect self-consistency, it is rejected, and `paradox_flag` is
    True, the exact pattern arXiv:2606.19544 names.
    """
    healthy = validate_judge(kappa=0.6, test_retest=0.97, bias_rate=0.05)
    _, paradox_bias_rate = run_position_bias_demo()
    paradox = validate_judge(kappa=0.6, test_retest=0.97, bias_rate=paradox_bias_rate)
    return healthy, paradox
