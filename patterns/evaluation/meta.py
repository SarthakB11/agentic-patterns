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

The last three functions measure a different property: whether a judge's
probability can be trusted, not only its verdict. A judge that emits a
probability (a decision model such as TypeSafe's Jev, see
`decision_judge.py`, or any judge whose confidence feeds an escalation
rule like `selective.py`) needs three numbers. `auroc` asks whether good
answers score above bad ones at all, independent of any cutoff.
`brier_score` and `expected_calibration_error` ask whether a 0.9 means
right about 90% of the time. A judge can rank perfectly (AUROC 1.0) and
still be badly calibrated, which is exactly the case where tuning the
cutoff fixes accuracy and leaving it at 0.5 does not.
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


def _check_probabilities(probabilities: list[float], labels: list[bool], name: str) -> None:
    """Validate a probability list against its labels, shared by the calibration metrics."""
    if not probabilities or not labels:
        raise ValueError(f"{name} requires at least one labeled case")
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must be the same length")
    if any(not (0.0 <= p <= 1.0) for p in probabilities):  # also rejects NaN
        raise ValueError("every probability must lie in [0, 1]")


def auroc(scores: list[float], labels: list[bool]) -> float:
    """Area under the ROC curve: the chance a random positive outscores a random negative.

    Computed directly as the Mann-Whitney statistic over every (positive,
    negative) pair, a tie counting half, which is exact for small labeled
    slices and needs no curve construction. 1.0 is a perfect ranking, 0.5
    is no better than chance. It never looks at a cutoff, so it separates
    "the judge cannot tell good from bad" from "the judge is fine but the
    threshold is in the wrong place".

    Raises:
        ValueError: If the lists differ in length, are empty, or the labels
            hold only one class (AUROC is undefined without both).
    """
    if not scores or len(scores) != len(labels):
        raise ValueError("auroc requires non-empty scores and labels of the same length")
    if any(s != s for s in scores):  # NaN is the only value not equal to itself
        raise ValueError("auroc scores must not contain NaN")
    positives = [s for s, y in zip(scores, labels) if y]
    negatives = [s for s, y in zip(scores, labels) if not y]
    if not positives or not negatives:
        raise ValueError("auroc requires at least one positive and one negative label")
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def brier_score(probabilities: list[float], labels: list[bool]) -> float:
    """Mean squared error between a predicted probability of pass and the 0/1 label.

    Lower is better; 0.0 is perfect, and always answering 0.5 scores 0.25.
    It penalizes both a wrong ranking and a right ranking at the wrong
    confidence, so an underconfident judge pays for hedging even when
    every verdict is correct.

    Raises:
        ValueError: If the lists differ in length, are empty, or hold a
            value outside [0, 1].
    """
    _check_probabilities(probabilities, labels, "brier_score")
    return sum((p - float(y)) ** 2 for p, y in zip(probabilities, labels)) / len(labels)


def expected_calibration_error(probabilities: list[float], labels: list[bool], bins: int = 10) -> float:
    """Weighted gap between stated confidence and observed pass rate, over equal-width bins.

    Each probability falls into one of `bins` equal-width bins over [0, 1]
    (1.0 lands in the top bin). Per bin, the gap is `|mean probability -
    fraction labeled pass|`; ECE is those gaps weighted by bin size. 0.0
    means "0.8" really is right 80% of the time. With small slices a few
    bins hold one case each, so treat ECE as a direction, not a decimal.

    This is the binary reliability form: it bins P(pass) and compares it
    with the observed pass rate. Papers on classifiers often report the
    top-label form instead (Guo et al., arXiv:1706.04599), binning the
    confidence of the predicted label against accuracy; the two differ, so
    say which one a reported number is.

    Raises:
        ValueError: If `bins` is below 1 or the probabilities are invalid.
    """
    _check_probabilities(probabilities, labels, "expected_calibration_error")
    if bins < 1:
        raise ValueError("bins must be at least 1")
    grouped: dict[int, list[tuple[float, bool]]] = {}
    for p, y in zip(probabilities, labels):
        grouped.setdefault(min(int(p * bins), bins - 1), []).append((p, y))
    total = len(labels)
    ece = 0.0
    for members in grouped.values():
        mean_p = sum(p for p, _ in members) / len(members)
        pass_rate = sum(1 for _, y in members if y) / len(members)
        ece += len(members) / total * abs(mean_p - pass_rate)
    return ece
