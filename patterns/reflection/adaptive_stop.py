"""Sub-module: revision-necessity gate plus diminishing-returns stop.

Every other variant here always pays for at least one critique round and
keeps refining until approval or the iteration cap. This adds two controls
the base loop lacks, wired through `run_reflection_loop`'s optional `gate`,
`diminishing_epsilon`, and `diminishing_patience` parameters (`loop.py`):
a pre-critique **revision gate**, a cheap binary "does this plausibly need
revision" check run once before the first critique that returns the draft
unchanged with zero critique calls when it is already fine (mirrors the
OpenAI Agents SDK's conditional guardrail execution, run the evaluator only
when warranted); and a **diminishing-returns stop**, tracking the score
delta across rounds and stopping once the marginal gain falls below an
epsilon for a patience window, even below `score_threshold`. The no-change
guard in `loop.py` only fires on byte-for-byte identical refinements; real
over-reflection produces different but no-better text, which this stop
catches and the no-change guard does not.

Both answer the same over-reflection evidence: reasoning models
re-checking and self-overturning already-correct answers (survey,
arXiv:2505.00551), overthinking as measured degradation (survey,
arXiv:2508.02120), and Cross-Context Review's finding that reviewing twice
in one session did not beat reviewing once (arXiv:2603.12123). Forcing
explicit reflection can also cost through a different mechanism, structure
snowballing under constrained decoding (arXiv:2604.06066), which is why the
gate here is a plain binary reply, not a structured schema.
"""

from __future__ import annotations

from collections.abc import Callable

from agentic_patterns import Message, Provider, get_provider
from patterns.reflection.loop import ReflectionResult, run_reflection_loop
from patterns.reflection.prompting import make_critique, make_generate, make_refine

_TASK = "Write a one-sentence caption for a product photo of a stainless steel water bottle."
_GENERATOR_SYSTEM = "You write short, plain product captions. Reply with only the caption."
_GATE_SYSTEM = (
    "You are a fast pre-check, not a full reviewer. Reply with exactly one "
    "word: OK if the draft is already acceptable, or REVISE if a full "
    "review is warranted."
)
_CRITIC_SYSTEM = (
    "You review product captions for a specific, checkable detail (material, "
    "size, or use case) and a plain tone. SCORE out of 10 and comments; "
    "start with APPROVED if the bar is met."
)


def _parse_gate(text: str) -> bool:
    """Parse a gate reply into "needs revision" (True) or "OK" (False)."""
    return not text.strip().upper().startswith("OK")


def make_gate(provider: Provider, task: str, *, system: str = _GATE_SYSTEM) -> Callable[[str], bool]:
    """Build a `gate` callable usable as `run_reflection_loop`'s `gate` argument.

    Args:
        provider: The model that plays the fast pre-checker role.
        task: The original task description, given to the gate for context.
        system: System prompt describing the gate's binary standard.

    Returns:
        A callable returning True if the draft plausibly needs revision,
        False if it is already fine.
    """

    def gate(draft: str) -> bool:
        completion = provider.complete([Message.user(f"Task:\n{task}\n\nDraft:\n{draft}")], system=system)
        return _parse_gate(completion.content)

    return gate


def run_gate_skip_demo(
    generator_provider: Provider | None = None, gate_provider: Provider | None = None
) -> ReflectionResult:
    """Run a loop whose gate reports the first draft is already fine.

    The gate returns OK, so the loop stops immediately: zero critique
    calls, the draft unchanged, `stop_reason="gated_no_revision"`.

    Args:
        generator_provider: Drives generate. Defaults to a `MockProvider`
            scripted with one already-good caption.
        gate_provider: Drives the gate check. Defaults to a `MockProvider`
            scripted with a single "OK" reply.
    """
    if generator_provider is None:
        generator_provider = get_provider(
            script=["Double-walled stainless steel bottle, keeps drinks cold for 24 hours."]
        )
    if gate_provider is None:
        gate_provider = get_provider(script=["OK"])
    return _run_gated_loop(generator_provider, gate_provider, critic_provider=generator_provider)


def run_gate_pass_through_demo(
    generator_provider: Provider | None = None,
    gate_provider: Provider | None = None,
    critic_provider: Provider | None = None,
) -> ReflectionResult:
    """Run a loop whose gate reports the first draft needs a full review.

    The gate returns REVISE, so the loop runs its normal rounds exactly as
    if no gate had been passed at all.

    Args:
        generator_provider: Drives generate and refine. Defaults to a
            `MockProvider` scripted with a vague draft and a specific one.
        gate_provider: Drives the gate check. Defaults to a `MockProvider`
            scripted with a single "REVISE" reply.
        critic_provider: Drives critique. Defaults to a `MockProvider`
            scripted to reject then approve.
    """
    if generator_provider is None:
        generator_provider = get_provider(
            script=["Great water bottle.", "Double-walled stainless steel bottle, keeps drinks cold for 24 hours."]
        )
    if gate_provider is None:
        gate_provider = get_provider(script=["REVISE"])
    if critic_provider is None:
        critic_provider = get_provider(
            script=["SCORE: 3\nNo material, size, or use case named.", "APPROVED: yes\nSCORE: 9\nSpecific and plain."]
        )
    return _run_gated_loop(generator_provider, gate_provider, critic_provider=critic_provider)


def _run_gated_loop(
    generator_provider: Provider, gate_provider: Provider, *, critic_provider: Provider
) -> ReflectionResult:
    """Shared wiring for both gate demos: same task, same three roles, gate attached."""
    generate = make_generate(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    critique = make_critique(critic_provider, _TASK, system=_CRITIC_SYSTEM)
    refine = make_refine(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    gate = make_gate(gate_provider, _TASK)
    return run_reflection_loop(generate, critique, refine, max_iterations=3, gate=gate)


def run_diminishing_returns_demo(
    generator_provider: Provider | None = None, critic_provider: Provider | None = None
) -> ReflectionResult:
    """Run a loop that stops on plateaued score gain rather than the iteration cap.

    Scripted scores rise 5, then 7, then 7.2: the round-three gain (0.2) is
    below epsilon (0.5), so the loop stops with
    `stop_reason="diminishing_returns"` after round three, never reaching
    the round-five cap it was budgeted for. Each round's refinement is
    worded differently, so this is not the no-change guard firing.

    Args:
        generator_provider: Drives generate and refine. Defaults to a
            `MockProvider` scripted with three distinct captions.
        critic_provider: Drives critique. Defaults to a `MockProvider`
            scripted with the 5, 7, 7.2 score sequence, never approving.
    """
    if generator_provider is None:
        generator_provider = get_provider(
            script=[
                "Water bottle, good for drinks.",
                "Stainless steel water bottle, keeps drinks cold.",
                "Stainless steel water bottle, keeps drinks cold for hours on end.",
            ]
        )
    if critic_provider is None:
        critic_provider = get_provider(
            script=[
                "SCORE: 5\nNo material named, too generic.",
                "SCORE: 7\nNames the material, could use a concrete duration.",
                "SCORE: 7.2\nMarginal wording change, still no concrete duration.",
            ]
        )
    generate = make_generate(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    critique = make_critique(critic_provider, _TASK, system=_CRITIC_SYSTEM)
    refine = make_refine(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    return run_reflection_loop(
        generate, critique, refine, max_iterations=5, diminishing_epsilon=0.5, diminishing_patience=1
    )
