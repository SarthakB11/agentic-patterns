"""Sub-module: rubric-based structured critique and score-gated stopping.

The critic scores the draft on named dimensions (accuracy, completeness,
clarity) and reports an overall SCORE line alongside its comments. The loop
stops once the score clears a threshold, rather than waiting for the critic
to say a bare "approved". This makes the stop condition testable: it is a
number comparison, not a judgment call about wording.

This demo also shows why best-so-far tracking matters. The scripted score
sequence rises, peaks, and then drops: round three over-corrects and buries
the main point under extra hedging language, the kind of round-count
over-reflection reported for reasoning models re-checking and
self-overturning already-correct output (survey arXiv:2505.00551) and
catalogued as measured overthinking degradation (survey arXiv:2508.02120).
This is not "structure snowballing" (arXiv:2604.06066): that finding is
specifically about constrained decoding forcing a structured output format,
not about hedging that appears after extra rounds. The loop runs to the
iteration cap without ever crossing the threshold, and the result returned
is the round-two draft, the highest scored one, not the regressed
round-three draft that happened to run last.
"""

from __future__ import annotations

from agentic_patterns import Provider, get_provider

from patterns.reflection.loop import ReflectionResult, run_reflection_loop
from patterns.reflection.prompting import make_critique, make_generate, make_refine

_TASK = (
    "Write a 3-bullet executive summary of Q2 churn for a leadership "
    "readout. Facts: churn rose from 4% to 6%; the increase is driven by "
    "onboarding drop-off in the mobile app; billing-related support "
    "ticket volume doubled."
)

_GENERATOR_SYSTEM = "You write tight executive summaries. Reply with only the bullets."

_CRITIC_SYSTEM = (
    "You score executive summaries on three dimensions: accuracy "
    "(matches the given facts), completeness (names the primary driver "
    "and a next step), and clarity (a leader can act on it in ten "
    "seconds). Report ACCURACY, COMPLETENESS, and CLARITY each out of 10, "
    "an overall SCORE out of 10, and comments naming the weakest "
    "dimension."
)

_SCORE_THRESHOLD = 9.5


def _generator_script() -> list[str]:
    """Three drafts: too thin, a strong revision, then an over-corrected one."""
    draft_1 = (
        "- Churn rose from 4% to 6% in Q2.\n"
        "- Support tickets about billing doubled.\n"
        "- We should investigate."
    )
    draft_2 = (
        "- Q2 churn rose from 4% to 6%, driven primarily by onboarding "
        "drop-off in the mobile app.\n"
        "- Billing-related support tickets doubled over the same period, a "
        "secondary signal worth watching.\n"
        "- Next step: fix the mobile onboarding flow first; it is the "
        "larger lever on churn."
    )
    draft_3 = (
        "- Q2 churn rose from 4% to 6%, which could reflect onboarding "
        "drop-off in the mobile app, seasonal effects, pricing changes "
        "from last quarter, or a mix of all three, though onboarding "
        "looks like the largest single factor based on available data.\n"
        "- Billing-related support ticket volume doubled, which may or "
        "may not be connected to the churn increase and merits its own "
        "separate investigation before drawing conclusions.\n"
        "- Next steps could include reviewing onboarding, auditing "
        "billing flows, or commissioning a churn survey, pending "
        "prioritization discussion."
    )
    return [draft_1, draft_2, draft_3]


def _critic_script() -> list[str]:
    """Three critiques: thin, strong, then penalized for burying the point."""
    critique_1 = (
        "ACCURACY: 8 COMPLETENESS: 4 CLARITY: 6\n"
        "SCORE: 6\n"
        "Completeness is the weak dimension: the summary never names the "
        "primary driver (mobile onboarding) or a concrete next step, so a "
        "leader cannot act on it."
    )
    critique_2 = (
        "ACCURACY: 9 COMPLETENESS: 9 CLARITY: 8\n"
        "SCORE: 8.5\n"
        "Clarity is the weakest dimension now: still strong overall, "
        "names the driver and a next step, but could be tightened "
        "further before it is publish-ready."
    )
    critique_3 = (
        "ACCURACY: 7 COMPLETENESS: 6 CLARITY: 5\n"
        "SCORE: 7\n"
        "Clarity regressed: hedging language buries the primary driver "
        "under alternative explanations, and the next step is now three "
        "vague options instead of one clear recommendation."
    )
    return [critique_1, critique_2, critique_3]


def run_rubric_demo(
    generator_provider: Provider | None = None,
    critic_provider: Provider | None = None,
) -> ReflectionResult:
    """Run a rubric-scored, threshold-gated reflection loop.

    Args:
        generator_provider: Drives generate and refine. Defaults to a
            `MockProvider` scripted with three drafts: thin, strong, then
            over-corrected.
        critic_provider: Drives critique. Defaults to a `MockProvider`
            scripted with matching per-dimension scores.
    """
    if generator_provider is None:
        generator_provider = get_provider(script=_generator_script())
    if critic_provider is None:
        critic_provider = get_provider(script=_critic_script())

    generate = make_generate(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    critique = make_critique(critic_provider, _TASK, system=_CRITIC_SYSTEM)
    refine = make_refine(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    return run_reflection_loop(
        generate, critique, refine, max_iterations=3, score_threshold=_SCORE_THRESHOLD
    )
