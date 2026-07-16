"""Prompt-building helpers that turn a `Provider` into loop callables.

`loop.run_reflection_loop` takes plain callables so it stays independent of
any provider or prompt shape. These functions build the common case: a
single provider asked, in turn, to generate, critique, and refine a draft
for a fixed task description. Variant modules that need something other
than the common case (two separate providers, a local checker instead of a
critic call, memory carried across attempts) write their own callables
instead of using these.
"""

from __future__ import annotations

from collections.abc import Callable

from agentic_patterns import Message, Provider
from patterns.reflection.loop import Critique, parse_critique


def make_generate(provider: Provider, task: str, *, system: str) -> Callable[[], str]:
    """Build a `generate` callable that asks `provider` to draft from scratch."""

    def generate() -> str:
        completion = provider.complete([Message.user(task)], system=system)
        return completion.content

    return generate


def make_critique(
    provider: Provider,
    task: str,
    *,
    system: str,
    external_framing: bool = False,
) -> Callable[[str], Critique]:
    """Build a `critique` callable that asks `provider` to review a draft.

    Args:
        provider: The model that plays the critic role.
        task: The original task description, given to the critic for
            context.
        system: System prompt describing the critic's persona and standard.
        external_framing: If True, present the draft as text submitted by
            someone else rather than "your previous answer". Models show a
            self-correction blind spot: they miss errors in their own prior
            output that they catch when the same error is shown as another
            author's text (the "self-correction blind spot" finding, 2025).
            Framing the draft as external input works around that bias.
    """

    def critique(draft: str) -> Critique:
        if external_framing:
            prompt = (
                f"Task given to the author:\n{task}\n\n"
                f"Below is a draft submitted by another author for review:\n{draft}"
            )
        else:
            prompt = f"Task:\n{task}\n\nDraft to review:\n{draft}"
        completion = provider.complete([Message.user(prompt)], system=system)
        return parse_critique(completion.content)

    return critique


def make_refine(provider: Provider, task: str, *, system: str) -> Callable[[str, Critique], str]:
    """Build a `refine` callable that asks `provider` to revise a draft."""

    def refine(draft: str, crit: Critique) -> str:
        prompt = (
            f"Task:\n{task}\n\nPrevious draft:\n{draft}\n\n"
            f"Critique to address:\n{crit.comments}\n\n"
            "Rewrite the draft so it fully addresses the critique."
        )
        completion = provider.complete([Message.user(prompt)], system=system)
        return completion.content

    return refine
