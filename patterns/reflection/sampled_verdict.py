"""Sub-module: self-consistent judging, denoising one critic by sampling it.

Every other critic in this folder is called once per round. A single LLM
judge call is noisy: the same critic, on the same draft, can hand back a
harsh outlier or a lenient one depending on the sample. This wraps one
critic so each round samples it `n` times on the same draft and aggregates
to a lower-variance verdict before the loop decides anything. Distinct axis
from `multi_critic.py`: that module widens coverage across lenses; this one
reduces variance on a single lens. Width versus depth.

Aggregation: the combined score is the *median* of the sampled scores, not
the mean, so one harsh or lenient outlier does not swing the verdict;
`approved` is a majority vote (an approval fraction at or above a quorum).
`comments` come from whichever sample's score is nearest the median, a real
critique to act on rather than a synthetic blend of `n` critiques.

Source: Generative Verifiers / GenRM-CoT (Zhang et al., arXiv:2408.15240),
which boosts verification accuracy by sampling several chain-of-thought
verifications and majority-voting the verdict, self-consistency (Wang et
al. 2022, arXiv:2203.11171) applied to the judge instead of the solver.
Online, the `n` samples come from temperature above zero; offline under
`MockProvider` the diversity is authored directly into the script.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

from agentic_patterns import Provider, get_provider
from patterns.reflection.loop import Critique, ReflectionResult, run_reflection_loop
from patterns.reflection.prompting import make_critique, make_generate, make_refine

_TASK = (
    "Explain in one paragraph why the sky looks blue, for a curious "
    "10-year-old, correctly naming the physical cause."
)

_GENERATOR_SYSTEM = "You write short, accurate science explanations for children. Reply with only the paragraph."

_CRITIC_SYSTEM = (
    "You are a science-accuracy reviewer with no attachment to this draft; "
    "you did not write it. Check whether the explanation correctly "
    "attributes the sky's color to Rayleigh scattering of shorter "
    "wavelengths, not a vaguer or wrong cause. Reply with a SCORE out of "
    "10 and comments. If it is correct and clear, start your reply with "
    "APPROVED."
)


def make_sampled_critic(
    base_critique: Callable[[str], Critique],
    *,
    n: int,
    quorum: float = 0.5,
    sample_log: list[list[float | None]] | None = None,
) -> Callable[[str], Critique]:
    """Wrap a single critic so each call samples it `n` times and aggregates.

    Args:
        base_critique: A critic callable, typically built by
            `prompting.make_critique`, that makes one provider call per
            invocation. Called `n` times per round here.
        n: Number of samples to draw per round.
        quorum: Minimum fraction of samples that must be `approved` for the
            aggregate to be approved.
        sample_log: If given, each round's list of sampled scores (in
            sample order, `None` for unscored samples) is appended here, so
            a caller can show the spread across rounds.

    Returns:
        A callable usable as `run_reflection_loop`'s `critique` argument.
    """

    def critique(draft: str) -> Critique:
        samples = [base_critique(draft) for _ in range(n)]
        if sample_log is not None:
            sample_log.append([s.score for s in samples])

        scored = [(s.score, s) for s in samples if s.score is not None]
        median_score = statistics.median(score for score, _s in scored) if scored else None

        approvals = sum(1 for s in samples if s.approved)
        approved = n > 0 and (approvals / n) >= quorum

        if scored and median_score is not None:
            _closest_score, representative = min(scored, key=lambda pair: abs(pair[0] - median_score))
            comments = representative.comments
        else:
            comments = samples[0].comments if samples else ""

        return Critique(comments=comments, score=median_score, approved=approved)

    return critique


def run_sampled_verdict_demo(
    generator_provider: Provider | None = None,
    critic_provider: Provider | None = None,
    *,
    n: int = 3,
) -> tuple[ReflectionResult, list[list[float | None]]]:
    """Run a loop whose critic is one lens sampled `n` times per round.

    Round one: samples score 8, 8, 2; the median (8) is not sunk by the
    harsh outlier, but none used the APPROVED sentinel, so the aggregate is
    unapproved and the loop refines. Round two: samples score 9, 9, 4; two
    of three approve, a majority, so the aggregate is approved and the loop
    stops even though one sample dissented.

    Args:
        generator_provider: Drives generate and refine. Defaults to a
            `MockProvider` scripted with a vague first draft and an
            accurate revision.
        critic_provider: Drives every sampled critique call (`n` per
            round). Defaults to a `MockProvider` scripted with the score
            pattern above.
        n: Number of samples per round.

    Returns:
        The loop result plus the per-round list of sampled scores, so a
        caller can show the spread.
    """
    if generator_provider is None:
        generator_provider = get_provider(
            script=[
                "The sky is blue because of how light bounces around in the air.",
                "The sky looks blue because sunlight is made of many colors, and air "
                "molecules scatter blue light (which has a shorter wavelength) much more "
                "than red light, a phenomenon called Rayleigh scattering, so blue light "
                "reaches your eyes from all directions.",
            ]
        )
    if critic_provider is None:
        critic_provider = get_provider(
            script=[
                "SCORE: 8\nDirectionally right but never names the mechanism.",
                "SCORE: 8\nClose, but 'bounces around' is too vague for the actual cause.",
                "SCORE: 2\nThis does not name Rayleigh scattering or wavelength at all.",
                "APPROVED: yes\nSCORE: 9\nNames Rayleigh scattering and wavelength correctly.",
                "APPROVED: yes\nSCORE: 9\nCorrect mechanism, clear for a child.",
                "SCORE: 4\nStill too technical in one clause for a 10-year-old.",
            ]
        )
    sample_log: list[list[float | None]] = []
    base_critique = make_critique(critic_provider, _TASK, system=_CRITIC_SYSTEM, external_framing=True)
    critique = make_sampled_critic(base_critique, n=n, sample_log=sample_log)

    generate = make_generate(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    refine = make_refine(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    result = run_reflection_loop(generate, critique, refine, max_iterations=3)
    return result, sample_log
