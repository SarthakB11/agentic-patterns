"""Sub-module: separate generator and critic (with external framing).

The generator and the critic are two independent `Provider` instances here,
each scripted with its own persona, instead of one model wearing two hats.
Splitting the roles keeps the critic from being biased toward defending
text it just wrote itself, and lets the critic apply a standard the
generator never saw.

This module also demonstrates external framing: the critic is shown the
draft as "a submission from another author" rather than "your previous
answer." Models have a measured self-correction blind spot: Tsui's
Self-Correction Bench (arXiv:2507.02778) reports a 64.5% average blind-spot
rate across 14 non-reasoning models, missing an injected error in their own
output that they catch when the same error is shown as another author's
text. Cross-Context Review (arXiv:2603.12123) confirms the fix direction, a
fresh review session beats same-session self-review, and adds a caution:
reviewing twice in the same session did not beat reviewing once, so extra
in-context rounds are not free improvement. Generator/critic separation
plus external framing is the practical answer to the blind spot. Worth
noting as a cheaper alternative: the same Self-Correction Bench paper finds
that simply appending the word "Wait" to a model's own output cuts the
blind spot by 89.3%, a one-token activation instead of a second provider
and a role split.
"""

from __future__ import annotations

from agentic_patterns import Provider, get_provider
from patterns.reflection.loop import ReflectionResult, run_reflection_loop
from patterns.reflection.prompting import make_critique, make_generate, make_refine

_TASK = (
    "Write a one-paragraph product description for a noise-cancelling "
    "travel pillow, aimed at frequent flyers, for an e-commerce listing."
)

_GENERATOR_SYSTEM = (
    "You are a product copywriter. Write a compelling one-paragraph "
    "listing description. Reply with only the paragraph."
)

_CRITIC_SYSTEM = (
    "You are a skeptical marketing reviewer with no attachment to this "
    "draft; you did not write it and are judging a submission from another "
    "author. Check for a specific, verifiable claim (not just adjectives), "
    "a clear statement of who it is for, and a call to action. Reply with "
    "a SCORE out of 10 and comments on what is missing. If the draft meets "
    "the bar, start your reply with APPROVED."
)


def _generator_script() -> list[str]:
    """Two turns from the generator: the first draft and the revision."""
    draft_1 = (
        "Our travel pillow is amazing and super comfortable. You will "
        "love it on every flight. Buy it today!"
    )
    draft_2 = (
        "This noise-cancelling travel pillow blocks up to 25 dB of cabin "
        "noise with a built-in active filter, so frequent flyers can sleep "
        "through engine drone on long-haul flights. The memory-foam collar "
        "supports your neck in any seat position and folds flat into its "
        "own pouch for the overhead bin. Add it to your carry-on and land "
        "rested instead of stiff."
    )
    return [draft_1, draft_2]


def _critic_script() -> list[str]:
    """Two turns from the critic: reject the vague draft, approve the fixed one."""
    critique_1 = (
        "SCORE: 4\n"
        "This submission is all adjectives (amazing, super comfortable) "
        "with no specific claim a reader can check, no named use case, and "
        "the call to action is generic. Add a concrete spec (noise "
        "reduction number, material) and say who this is for."
    )
    critique_2 = (
        "APPROVED: yes\n"
        "SCORE: 9\n"
        "This submission now has a specific claim (25 dB, memory foam), "
        "names the audience (frequent flyers on long-haul flights), and "
        "closes with a clear action. Ready to publish."
    )
    return [critique_1, critique_2]


def run_generator_critic_demo(
    generator_provider: Provider | None = None,
    critic_provider: Provider | None = None,
) -> ReflectionResult:
    """Run a loop with an independent generator and critic.

    Args:
        generator_provider: Drives the generate and refine steps. Defaults
            to a `MockProvider` scripted with a weak first draft and a
            fixed revision.
        critic_provider: Drives the critique step only, kept separate so it
            never sees its own prior turns as "self". Defaults to a
            `MockProvider` scripted to reject the weak draft and approve
            the revision.
    """
    if generator_provider is None:
        generator_provider = get_provider(script=_generator_script())
    if critic_provider is None:
        critic_provider = get_provider(script=_critic_script())

    generate = make_generate(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    critique = make_critique(critic_provider, _TASK, system=_CRITIC_SYSTEM, external_framing=True)
    refine = make_refine(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    return run_reflection_loop(generate, critique, refine, max_iterations=3)
