"""Ensemble / jury of judges: majority vote across independent judges.

Several judges score the same output and a majority vote is taken, reducing
any single judge's bias. The judges here are three independent
`Provider` instances, standing in for distinct model families: a jury only
reduces bias if its members are not the same model (or a fine-tune of it)
grading itself, since a judge favors outputs from any generator it is
related to by shared weights or lineage, not only its own literal output
("Preference Leakage," arXiv:2502.01534). This module does not simulate
that failure mode; it demonstrates the mitigation, independent judges voting,
that the separation rule motivates.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Provider, get_provider

from patterns.evaluation.eval_set import EvalCase, get_case
from patterns.evaluation.pointwise import build_pointwise_judge
from patterns.evaluation.verdict import Verdict


@dataclass
class JuryResult:
    """The outcome of a jury vote.

    Attributes:
        verdicts: One `Verdict` per juror, in the order the jurors were
            given.
        pass_votes: How many jurors returned `passed=True`.
        majority_passed: True if more than half the jurors passed the
            output. Ties (an even jury split evenly) fail closed.
    """

    verdicts: list[Verdict]
    pass_votes: int
    majority_passed: bool


def run_jury(providers: list[Provider], case: EvalCase, output: str, *, reference_mode: bool = True) -> JuryResult:
    """Score `output` with one independent judge per provider and vote.

    Args:
        providers: One provider per juror. Each builds its own pointwise
            judge via `build_pointwise_judge`, so jurors do not share state.
        case: The eval case being judged.
        output: The candidate output to score.
        reference_mode: Passed through to each juror's pointwise judge.
    """
    verdicts = [build_pointwise_judge(p, reference_mode=reference_mode)(case, output) for p in providers]
    pass_votes = sum(1 for v in verdicts if v.passed)
    majority_passed = pass_votes > len(verdicts) / 2
    return JuryResult(verdicts=verdicts, pass_votes=pass_votes, majority_passed=majority_passed)


def run_jury_demo() -> JuryResult:
    """Run a 3-juror vote where two jurors pass the reply and one dissents.

    The dissenting juror focuses on a real gap (no explicit next step) that
    the other two treat as minor, a realistic disagreement rather than a
    parsing failure, and the majority vote still resolves to pass.
    """
    case = get_case("refund_policy")
    reply = "Refunds are available within 30 days if you have your order number."

    juror_a = get_provider(script=["Matches the reference on window and proof needed.\nSCORE: 9\nVERDICT: pass"])
    juror_b = get_provider(script=["Accurate and concise, minor style nit only.\nSCORE: 8\nVERDICT: pass"])
    juror_c = get_provider(
        script=[
            "Accurate but never tells the customer what to do next, e.g. "
            "where to start the refund. That gap matters for a support "
            "reply.\nSCORE: 6\nVERDICT: fail"
        ]
    )

    return run_jury([juror_a, juror_b, juror_c], case, reply)
