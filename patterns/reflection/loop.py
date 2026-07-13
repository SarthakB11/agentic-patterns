"""Reusable engine for the generate, critique, refine loop.

This module holds the pattern's mechanics, kept separate from any one
provider or demo script: parsing a critic's free-form output into a
structured `Critique`, and running the iterate-until-stopped loop with
best-so-far tracking. Prompt-building helpers live in `prompting.py`;
transcript rendering lives in `transcript.py`.

The loop is often called the evaluator-optimizer workflow in current
framework docs (LangGraph, OpenAI Agents SDK): a generator produces work,
an evaluator scores it, and the two alternate until the evaluator is
satisfied or a budget runs out. That is the same loop implemented here
under the name reflection.

Stop conditions, checked in this order each round: (0) optional revision
gate, evaluated once before the first critique, stop immediately and keep
the draft unrevised if the gate reports the draft plausibly needs no
revision at all; (1) empty critique guard, stop and keep the draft
unrevised if the critic returns nothing usable; (2) approval, either an
explicit approved flag or a score at or above a threshold; (3) optional
diminishing-returns stop, stop when the score gain across rounds falls
below an epsilon for a patience window, even below threshold; (4) iteration
cap, `max_iterations` rounds with no approval; (5) no-change convergence, a
refine call that returns the draft unchanged; (6) blank refinement guard,
stop and keep the last good draft if a refine call returns nothing. Across
rounds the loop tracks the best-scoring draft seen, not simply the most
recent one, since a refinement can regress.

The no-change guard (5) only catches a refinement that comes back
byte-for-byte identical to the prior draft. Real over-reflection more often
produces different but no-better text round after round, a degradation
documented for reasoning models specifically: models re-checking and
self-overturning already-correct answers (survey coverage, arXiv:2505.00551)
and measured overthinking as rounds accumulate (survey, arXiv:2508.02120).
Two things in this file catch that case where the no-change guard cannot:
best-so-far tracking, which recovers the highest-scoring draft even when a
later round regresses, and the optional diminishing-returns stop (3) above,
which stops before wasting a plateaued round at all. See
`patterns/reflection/adaptive_stop.py` for the gate and diminishing-returns
mechanics.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

_SCORE_RE = re.compile(r"score:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_APPROVED_FIELD_RE = re.compile(r"approved:\s*(yes|true|no|false)", re.IGNORECASE)
_APPROVAL_PHRASES = ("no changes needed", "no issues found", "looks good", "ship it")


@dataclass
class Critique:
    """A critic's structured judgment of a draft.

    Attributes:
        comments: The critic's free-form remarks, kept in full for the
            refine step to read.
        score: A numeric score if the critic's output contained one, else
            None. Drives threshold-based stopping.
        approved: True if the critic's output contained an explicit
            approval signal (an "approved" field or sentinel phrase).
    """

    comments: str
    score: float | None = None
    approved: bool = False


def parse_critique(text: str) -> Critique:
    """Parse a critic's free-form response into a structured `Critique`.

    Recognizes an optional `SCORE: <number>` line and either an
    `APPROVED: yes|no` field or a bare approval sentinel such as "APPROVED"
    or "no changes needed" appearing in the text. Anything not recognized is
    kept verbatim as `comments`. An empty or whitespace-only input parses to
    an empty, unapproved, unscored `Critique`, which the loop treats as the
    empty-critique guard condition.

    Args:
        text: The critic's raw response text.
    """
    stripped = text.strip()
    if not stripped:
        return Critique(comments="")

    score_match = _SCORE_RE.search(stripped)
    score = float(score_match.group(1)) if score_match else None

    field_match = _APPROVED_FIELD_RE.search(stripped)
    if field_match:
        approved = field_match.group(1).lower() in ("yes", "true")
    else:
        lowered = stripped.lower()
        approved = lowered.startswith("approved") or any(p in lowered for p in _APPROVAL_PHRASES)

    return Critique(comments=stripped, score=score, approved=approved)


@dataclass
class ReflectionIteration:
    """One round of the loop: the draft that was reviewed and what happened.

    Attributes:
        index: 1-based round number.
        draft: The draft text that was critiqued this round.
        critique: The critic's structured judgment of `draft`.
        note: Short human-readable description of what the loop did next,
            e.g. "refine" or "stop: approved".
    """

    index: int
    draft: str
    critique: Critique
    note: str


@dataclass
class ReflectionResult:
    """The outcome of a full reflection loop run.

    Attributes:
        initial_draft: The first draft produced, before any refinement.
        best_draft: The highest-scoring draft seen (or the last draft
            reviewed, when the critic never returned a numeric score).
        best_score: The score attached to `best_draft`, or None if no
            critique in the run carried a score.
        final_draft: The draft the loop was working on when it stopped.
            Usually equal to `best_draft`, but can differ when the last
            round regressed and best-so-far tracking kept the earlier draft.
        iterations: One `ReflectionIteration` per critique round that ran.
        stop_reason: Why the loop stopped. One of "gated_no_revision",
            "empty_critique", "approved", "score_threshold",
            "diminishing_returns", "no_change", "blank_refinement", or
            "max_iterations".
    """

    initial_draft: str
    best_draft: str
    best_score: float | None
    final_draft: str
    iterations: list[ReflectionIteration] = field(default_factory=list)
    stop_reason: str = "max_iterations"


def run_reflection_loop(
    generate: Callable[[], str],
    critique: Callable[[str], Critique],
    refine: Callable[[str, Critique], str],
    *,
    max_iterations: int = 3,
    score_threshold: float | None = None,
    gate: Callable[[str], bool] | None = None,
    diminishing_epsilon: float | None = None,
    diminishing_patience: int = 1,
) -> ReflectionResult:
    """Run the generate, critique, refine loop to a stop condition.

    Args:
        generate: Produces the initial draft. Takes no arguments.
        critique: Reviews a draft and returns a structured `Critique`.
        refine: Produces a revised draft from the current draft and its
            critique.
        max_iterations: Maximum number of critique rounds to run before
            stopping unconditionally.
        score_threshold: If set, a critique with `score >= score_threshold`
            stops the loop even without an explicit approval.
        gate: Optional cheap pre-check run once on the initial draft, before
            the first critique. Returns True if the draft plausibly needs
            revision. When it returns False, the loop stops immediately with
            `stop_reason="gated_no_revision"` and never calls `critique` or
            `refine`. None (the default) always proceeds to the normal loop,
            matching every variant module that does not pass this argument.
        diminishing_epsilon: If set, track the score delta between
            consecutive rounds; once the delta stays below this epsilon for
            `diminishing_patience` consecutive rounds, stop with
            `stop_reason="diminishing_returns"` and return the best-so-far
            draft, even if the score is still below `score_threshold`. None
            (the default) disables this stop.
        diminishing_patience: Number of consecutive plateaued rounds
            required before the diminishing-returns stop fires. Ignored when
            `diminishing_epsilon` is None.

    Returns:
        A `ReflectionResult` with the best draft found, the full transcript,
        and why the loop stopped.
    """
    initial = generate()
    if gate is not None and not gate(initial):
        return ReflectionResult(
            initial_draft=initial,
            best_draft=initial,
            best_score=None,
            final_draft=initial,
            iterations=[],
            stop_reason="gated_no_revision",
        )

    current = initial
    best_draft = current
    best_score: float | None = None
    iterations: list[ReflectionIteration] = []
    stop_reason = "max_iterations"
    prev_score: float | None = None
    plateau_rounds = 0

    for i in range(1, max_iterations + 1):
        crit = critique(current)

        if not crit.comments and crit.score is None and not crit.approved:
            iterations.append(
                ReflectionIteration(i, current, crit, "stop: empty critique, draft kept unchanged")
            )
            stop_reason = "empty_critique"
            break

        if crit.score is not None and (best_score is None or crit.score > best_score):
            best_draft, best_score = current, crit.score
        elif crit.score is None and best_score is None:
            best_draft = current

        threshold_hit = score_threshold is not None and crit.score is not None and crit.score >= score_threshold
        if crit.approved or threshold_hit:
            reason = "approved" if crit.approved else "score_threshold"
            iterations.append(ReflectionIteration(i, current, crit, f"stop: {reason}"))
            stop_reason = reason
            break

        if diminishing_epsilon is not None and crit.score is not None:
            if prev_score is not None and (crit.score - prev_score) < diminishing_epsilon:
                plateau_rounds += 1
            else:
                plateau_rounds = 0
            prev_score = crit.score
            if plateau_rounds >= diminishing_patience:
                iterations.append(
                    ReflectionIteration(i, current, crit, "stop: diminishing returns, gain below epsilon")
                )
                stop_reason = "diminishing_returns"
                break

        if i == max_iterations:
            iterations.append(ReflectionIteration(i, current, crit, "stop: max_iterations reached"))
            stop_reason = "max_iterations"
            break

        refined = refine(current, crit)
        if not refined.strip():
            iterations.append(
                ReflectionIteration(i, current, crit, "stop: refinement came back blank, draft kept")
            )
            stop_reason = "blank_refinement"
            break
        if refined.strip() == current.strip():
            iterations.append(
                ReflectionIteration(i, current, crit, "stop: refinement identical to draft (convergence)")
            )
            stop_reason = "no_change"
            break

        iterations.append(ReflectionIteration(i, current, crit, "refine"))
        current = refined

    return ReflectionResult(
        initial_draft=initial,
        best_draft=best_draft,
        best_score=best_score,
        final_draft=current,
        iterations=iterations,
        stop_reason=stop_reason,
    )
