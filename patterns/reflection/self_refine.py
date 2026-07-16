"""Sub-module: single-model self-refinement (Self-Refine).

One model plays all three roles: generator, critic, and refiner. There is
no second agent and no external tool; the same provider is called three
times with three different prompts. This is the simplest form of
reflection and the baseline every other variant improves on. See Madaan et
al., "Self-Refine: Iterative Refinement with Self-Feedback," NeurIPS 2023.

Two demos live here:

- `run_self_refine_demo` shows the full loop: a weak first draft, a
  critique that names a specific gap, a refined draft that fixes it, and an
  approval that stops the loop.
- `run_guard_demo` shows the empty-critique guard: the critic returns
  nothing usable, and the loop stops on round one and returns the original
  draft unrefined, instead of guessing what to change.
"""

from __future__ import annotations

from agentic_patterns import Provider, get_provider
from patterns.reflection.loop import ReflectionResult, run_reflection_loop
from patterns.reflection.prompting import make_critique, make_generate, make_refine

_TASK = (
    "Explain what a hash table is, in 2-3 sentences, for a first-year "
    "computer science student who already knows what an array is."
)

_GENERATOR_SYSTEM = (
    "You write short, precise technical explanations for students. "
    "Write only the explanation, no preamble."
)

_CRITIC_SYSTEM = (
    "You review technical explanations for a first-year CS course. "
    "Judge whether the explanation names the mechanism (not just the "
    "behavior) and includes a concrete example. Reply with a SCORE out of "
    "10 and, if the draft is not yet good enough, specific comments on "
    "what to add or fix. If the draft is ready, start your reply with "
    "APPROVED."
)


def _self_refine_script() -> list[str]:
    """The four scripted turns: draft, critique, refined draft, approval."""
    draft_1 = "A hash table stores data using keys. It is fast. Hash tables are used in many programs."
    critique_1 = (
        "SCORE: 5\n"
        "The explanation never says how a hash table achieves speed: it "
        "does not mention that a hash function maps each key to an array "
        "index. It also gives no concrete example. Add one sentence on the "
        "hash function's role and a small example."
    )
    draft_2 = (
        "A hash table stores each key-value pair at an array index computed "
        "by a hash function, which converts the key into a number in "
        "roughly constant time. For example, storing a student's name and "
        "grade lets you look the grade up directly by name instead of "
        "scanning every entry. Because lookups jump straight to an index, "
        "hash tables average O(1) time for insert, delete, and search."
    )
    critique_2 = (
        "APPROVED: yes\n"
        "SCORE: 9\n"
        "Names the hash function, gives a concrete example, and states the "
        "time complexity. This is clear and complete for a first-year "
        "student."
    )
    return [draft_1, critique_1, draft_2, critique_2]


def run_self_refine_demo(provider: Provider | None = None) -> ReflectionResult:
    """Run the baseline self-refine loop and return its result.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with a coherent draft, critique,
            refinement, and approval, so the demo runs offline.
    """
    if provider is None:
        provider = get_provider(script=_self_refine_script())
    generate = make_generate(provider, _TASK, system=_GENERATOR_SYSTEM)
    critique = make_critique(provider, _TASK, system=_CRITIC_SYSTEM)
    refine = make_refine(provider, _TASK, system=_GENERATOR_SYSTEM)
    return run_reflection_loop(generate, critique, refine, max_iterations=3)


def run_guard_demo(provider: Provider | None = None) -> ReflectionResult:
    """Run a loop where the critic returns an empty response.

    Demonstrates the guard: an empty critique stops the loop on round one
    and the original draft is returned unrefined, rather than the loop
    inventing a refinement with nothing to act on.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with one draft and one blank critique.
    """
    if provider is None:
        provider = get_provider(script=["A hash table maps keys to array slots using a hash function.", ""])
    generate = make_generate(provider, _TASK, system=_GENERATOR_SYSTEM)
    critique = make_critique(provider, _TASK, system=_CRITIC_SYSTEM)
    refine = make_refine(provider, _TASK, system=_GENERATOR_SYSTEM)
    return run_reflection_loop(generate, critique, refine, max_iterations=3)
