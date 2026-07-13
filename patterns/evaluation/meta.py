"""Meta-evaluation: judging the judge.

A judge is only useful once it has itself been measured against a small
human-labeled slice. Report chance-corrected agreement (Cohen's kappa)
rather than raw exact-match agreement: "Reliability without Validity"
(arXiv:2606.19544) found kappa runs 33 to 41 points below raw agreement on
MT-Bench, so a raw-agreement number like the frequently-cited "~80%
judge-human agreement" overstates how much the judge is actually adding
past chance. Separately, Stureborg, Alikaniotis, and Suhara (arXiv:2405.01724)
found same-verdict rates for one judge asked the same question twice fall
from above 95% at temperature 0 to near 70% at temperature 1 (a finding
Norman et al. cite; their own protocol fixed judge temperature at 0),
motivating the test-retest check here as a first-class metric alongside
kappa, not an afterthought.

Kappa and test-retest are two of the paper's three validation axes, not a
sufficient judge validation on their own: a judge can hold excellent
test-retest reliability while carrying severe position bias, the study's
named consistency-bias paradox. See `validation_protocol.py` for the third
axis (position bias) and the joint accept/reject decision that requires all
three axes to pass at once.
"""

from __future__ import annotations


def cohens_kappa(judge_labels: list[bool], human_labels: list[bool]) -> float:
    """Compute Cohen's kappa between a judge's verdicts and human labels.

    Chance-corrected agreement: `(observed_agreement - expected_agreement)
    / (1 - expected_agreement)`, where expected agreement is computed from
    each rater's marginal pass rate rather than assumed to be 0.5. This is
    the number the meta-evaluation should report, since raw exact-match
    agreement does not account for how often two raters would agree by
    chance alone given how often each one says "pass".

    Args:
        judge_labels: The judge's pass/fail verdict per case.
        human_labels: A human's pass/fail label for the same cases, in the
            same order.

    Raises:
        ValueError: If the two lists differ in length or either is empty.
    """
    if not judge_labels or not human_labels:
        raise ValueError("cohens_kappa requires at least one labeled case")
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must be the same length")

    n = len(judge_labels)
    observed_agreement = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n
    p_judge_pass = sum(judge_labels) / n
    p_human_pass = sum(human_labels) / n
    expected_agreement = p_judge_pass * p_human_pass + (1 - p_judge_pass) * (1 - p_human_pass)

    if expected_agreement >= 1.0:
        # Every rater's marginal is unanimous in the same direction: chance
        # agreement is already 100%, so kappa is undefined by the usual
        # formula. Treat a perfect observed match as full agreement and any
        # mismatch (impossible here, but guarded) as none.
        return 1.0 if observed_agreement >= 1.0 else 0.0

    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def test_retest_rate(run_1: list[bool], run_2: list[bool]) -> float:
    """Return the fraction of matching verdicts between two runs of a judge.

    Both runs should score the same cases in the same order; a rate well
    below 1.0 signals the judge is not stable at the sampling temperature
    it was run at, independent of whether it agrees with a human.

    Raises:
        ValueError: If the two lists differ in length or either is empty.
    """
    if not run_1 or not run_2:
        raise ValueError("test_retest_rate requires at least one case")
    if len(run_1) != len(run_2):
        raise ValueError("run_1 and run_2 must be the same length")
    return sum(1 for a, b in zip(run_1, run_2) if a == b) / len(run_1)


def run_meta_evaluation_demo() -> float:
    """Compute kappa on a small fixed slice of judge verdicts vs human labels.

    Five cases, human-labeled by hand. The judge and the human agree on
    four of five (raw agreement 0.8), but both raters lean toward "pass"
    (judge 3/5, human 2/5), so chance agreement is not negligible and kappa
    comes out below the raw rate.
    """
    judge_labels = [True, True, False, True, False]
    human_labels = [True, False, False, True, False]
    return cohens_kappa(judge_labels, human_labels)


def run_test_retest_demo() -> float:
    """Compute the same-verdict rate across two scripted resamplings.

    Three cases scored twice each, simulating a judge run at temperature
    above zero: two cases land the same verdict both times, one flips,
    giving a same-verdict rate of 2/3, below the near-perfect rate expected
    at temperature 0.
    """
    run_1 = [True, True, False]
    run_2 = [True, False, False]
    return test_retest_rate(run_1, run_2)
