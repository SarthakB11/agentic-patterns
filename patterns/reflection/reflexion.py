"""Sub-module: memory-augmented (Reflexion-style) reflection.

Reflexion (Shinn et al., NeurIPS 2023) separates three roles across
repeated attempts at one task: an actor that acts, an evaluator that scores
the attempt against an external signal, and a self-reflection step that
turns a failed attempt into a verbal lesson written to episodic memory. The
next attempt conditions on that memory instead of starting cold, carrying
learning across trials with no weight updates.

This demo reuses `run_reflection_loop` rather than a separate loop: each
"refine" call here is really "retry the task with memory," which fits the
loop's generate/critique/refine shape even though nothing edits the prior
attempt's text directly. The evaluator is a deterministic checker, in the
same tool-grounded spirit as `tool_grounded.py`, so this stays offline and
the pass/fail signal is not the model's own opinion. To keep the demo
small, the checker's failure message doubles as the verbal reflection that
gets written to memory, rather than a separate reflection call.
"""

from __future__ import annotations

import re

from agentic_patterns import Message, Provider, get_provider
from patterns.reflection.loop import Critique, ReflectionResult, run_reflection_loop

_TASK = (
    "A bakery sells cupcakes in boxes of 6. Rina needs at least 45 "
    "cupcakes for a party. State the minimum number of boxes she must buy, "
    "and how many cupcakes will be left over after using exactly 45."
)

_ACTOR_SYSTEM = (
    "You solve short arithmetic word problems. Reply with one sentence "
    "stating the number of boxes and the leftover count."
)

_BOXES_RE = re.compile(r"(\d+)\s*boxes?", re.IGNORECASE)
_LEFTOVER_RE = re.compile(r"(\d+)\s*(?:cupcakes?\s*)?left\s*over", re.IGNORECASE)

_BOX_SIZE = 6
_NEEDED = 45


def _evaluate_attempt(answer: str) -> Critique:
    """Deterministically check an attempt against the arithmetic ground truth.

    Plays the evaluator role: this is the external reward Reflexion pairs
    with the self-reflection step, not a model judging its own answer.
    """
    boxes_match = _BOXES_RE.search(answer)
    leftover_match = _LEFTOVER_RE.search(answer)
    correct_boxes = -(-_NEEDED // _BOX_SIZE)  # ceiling division
    correct_leftover = correct_boxes * _BOX_SIZE - _NEEDED

    if not boxes_match:
        return Critique(comments="Lesson: the answer never states a number of boxes.", score=0.0)

    boxes = int(boxes_match.group(1))
    if boxes * _BOX_SIZE < _NEEDED:
        return Critique(
            comments=(
                f"Lesson: {boxes} boxes only provides {boxes * _BOX_SIZE} cupcakes, "
                f"short of the {_NEEDED} needed. Round the division up, not down: "
                f"the minimum is {correct_boxes} boxes."
            ),
            score=2.0,
        )

    leftover = int(leftover_match.group(1)) if leftover_match else None
    if leftover != correct_leftover:
        return Critique(
            comments=(
                f"Lesson: with {boxes} boxes ({boxes * _BOX_SIZE} cupcakes) the "
                f"leftover after using {_NEEDED} is {correct_leftover}, not "
                f"{leftover}. Recompute boxes * 6 - 45."
            ),
            score=5.0,
        )

    return Critique(comments="Correct: minimum boxes and leftover both match.", score=10.0, approved=True)


def _make_actor(provider: Provider):
    """Build the generate callable: solve the task cold, with no memory yet."""

    def generate() -> str:
        completion = provider.complete([Message.user(_TASK)], system=_ACTOR_SYSTEM)
        return completion.content

    return generate


def _make_retry_with_memory(provider: Provider, memory: list[str]):
    """Build the refine callable: retry the task, conditioned on memory so far.

    Appends the current attempt's evaluator lesson to `memory`, then asks
    the actor to try again with every lesson accumulated so far visible in
    the prompt. This is the step that carries learning across attempts
    without updating any weights.
    """

    def refine(_draft: str, critique: Critique) -> str:
        memory.append(critique.comments)
        lessons = "\n".join(f"- {lesson}" for lesson in memory)
        prompt = (
            f"{_TASK}\n\n"
            f"Lessons from earlier attempts:\n{lessons}\n\n"
            "Try again, applying the lessons above."
        )
        completion = provider.complete([Message.user(prompt)], system=_ACTOR_SYSTEM)
        return completion.content

    return refine


def run_reflexion_demo(provider: Provider | None = None) -> tuple[ReflectionResult, list[str]]:
    """Run a memory-augmented reflection loop across repeated attempts.

    Args:
        provider: Drives the actor across attempts. Defaults to a
            `MockProvider` scripted with a wrong first attempt and a
            corrected second attempt.

    Returns:
        The loop result plus the episodic memory list accumulated across
        attempts, so a caller can show the lesson that carried forward.
    """
    if provider is None:
        provider = get_provider(
            script=[
                "Rina needs 7 boxes, with 3 cupcakes left over.",
                "Rina needs 8 boxes, with 3 cupcakes left over.",
            ]
        )
    memory: list[str] = []
    generate = _make_actor(provider)
    refine = _make_retry_with_memory(provider, memory)
    result = run_reflection_loop(generate, _evaluate_attempt, refine, max_iterations=3)
    return result, memory
