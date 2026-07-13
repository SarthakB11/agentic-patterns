"""Trajectory-level self-consistency: run n rollouts, vote on the answer.

Self-consistency (Wang et al., 2022) runs several independent samples and
takes a majority vote instead of trusting a single one. This module lifts
that idea from a single completion to a whole ReAct episode: each rollout is
an independent call to `text_loop.run_react` or `native_loop.run_native_react`
against its own scripted provider, and the vote is over each rollout's final
answer rather than a token-level sample. Soft self-consistency (Wang et al.,
arXiv:2402.13212) and TrACE (Sethi et al., arXiv:2604.08369) both spend the
extra rollouts adaptively: this module supports an early stop once one answer
has an unbeatable lead, and an optional confidence-weighted vote in place of
plain counting.

No loop logic lives here; every rollout is a normal call to an existing
`run_react`/`run_native_react`, so this module is pure orchestration on top.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agentic_patterns import Provider, ToolRegistry, get_provider

from patterns.react.text_loop import run_react
from patterns.react.world import build_registry


@dataclass
class RolloutOutcome:
    """One rollout's contribution to the vote.

    Attributes:
        index: 1-based rollout number, in the order it ran.
        answer: The rollout's final answer, or None if it abstained.
        abstained: True if the rollout stopped without reaching Finish.
    """

    index: int
    answer: str | None
    abstained: bool


@dataclass
class SelfConsistencyResult:
    """Outcome of a self-consistency vote across rollouts.

    Attributes:
        answer: The winning answer's original (non-normalized) text, or None
            if every rollout abstained.
        votes: Normalized-answer text mapped to its vote weight (1.0 per
            rollout for a plain vote, or the supplied confidence for a soft vote).
        rollouts: Every rollout's outcome, in order, including any abstains.
        rollouts_run: Number of rollouts actually run, which is less than
            `len(providers)` when early stop fires.
        stopped_early: True if voting stopped before every provider was used.
    """

    answer: str | None
    votes: dict[str, float]
    rollouts: list[RolloutOutcome]
    rollouts_run: int
    stopped_early: bool


def _normalize(answer: str) -> str:
    """Fold trivially different answer strings together before counting."""
    return answer.strip().lower()


def _has_clear_leader(counts: dict[str, float], margin: float) -> bool:
    """True once the top vote total leads the runner-up by at least `margin`."""
    totals = sorted(counts.values(), reverse=True)
    if not totals:
        return False
    runner_up = totals[1] if len(totals) > 1 else 0.0
    return (totals[0] - runner_up) >= margin


def run_self_consistency(
    providers: Sequence[Provider],
    tools: ToolRegistry,
    goal: str,
    *,
    run_fn: Callable[..., Any] = run_react,
    agreement_margin: float = 2.0,
    weights: Sequence[float] | None = None,
    **run_kwargs: Any,
) -> SelfConsistencyResult:
    """Run up to `len(providers)` independent rollouts and vote on the answer.

    Args:
        providers: One provider per rollout, each scripted independently.
            Rollouts run in order and stop early once a clear leader emerges.
        tools: Registry of tools available to every rollout.
        goal: The question or task every rollout attempts.
        run_fn: The loop to run each rollout with. Must return an object with
            `.answer` and `.stopped_reason`; `run_react` and
            `run_native_react` both qualify.
        agreement_margin: Stop once the leading answer's vote total is ahead
            of the runner-up by at least this much.
        weights: Optional per-rollout confidence, same length as `providers`,
            for a soft vote in place of one vote per rollout. `weights[i]` is
            the weight of the rollout run against `providers[i]`.
        **run_kwargs: Extra keyword arguments passed through to `run_fn`.

    Returns:
        A SelfConsistencyResult with the winning answer, the vote breakdown,
        and every rollout's outcome.
    """
    outcomes: list[RolloutOutcome] = []
    counts: dict[str, float] = {}
    original_text: dict[str, str] = {}
    first_seen: dict[str, int] = {}

    for i, provider in enumerate(providers, start=1):
        result = run_fn(provider, tools, goal, **run_kwargs)
        if result.stopped_reason != "finish" or result.answer is None:
            outcomes.append(RolloutOutcome(index=i, answer=None, abstained=True))
            continue
        outcomes.append(RolloutOutcome(index=i, answer=result.answer, abstained=False))
        norm = _normalize(result.answer)
        original_text.setdefault(norm, result.answer)
        first_seen.setdefault(norm, i)
        weight = weights[i - 1] if weights is not None else 1.0
        counts[norm] = counts.get(norm, 0.0) + weight

        if _has_clear_leader(counts, agreement_margin):
            return _tally(counts, first_seen, original_text, outcomes, stopped_early=len(outcomes) < len(providers))

    if not counts:
        return SelfConsistencyResult(None, {}, outcomes, len(outcomes), stopped_early=False)
    return _tally(counts, first_seen, original_text, outcomes, stopped_early=False)


def _tally(
    counts: dict[str, float],
    first_seen: dict[str, int],
    original_text: dict[str, str],
    outcomes: list[RolloutOutcome],
    *,
    stopped_early: bool,
) -> SelfConsistencyResult:
    """Build the final result: pick the winner and render votes back to original text."""
    votes = {original_text[k]: v for k, v in counts.items()}
    return SelfConsistencyResult(_pick_winner(counts, first_seen, original_text), votes, outcomes, len(outcomes), stopped_early)


def _pick_winner(counts: dict[str, float], first_seen: dict[str, int], original_text: dict[str, str]) -> str:
    """Return the original text of the highest-count answer.

    Ties are broken by whichever tied answer was seen in an earlier rollout,
    so the vote is deterministic even when two answers end up level.
    """
    max_count = max(counts.values())
    tied = [k for k, v in counts.items() if v == max_count]
    winner = min(tied, key=lambda k: first_seen[k])
    return original_text[winner]


def demo_self_consistency() -> SelfConsistencyResult:
    """Vote across three rollouts: two answer "Paris, France", one answers "Paris".

    Normalization folds "Paris" and "Paris, France" into distinct votes here
    (they are not equal after stripping and lowercasing), so the winner is
    decided by count: two rollouts for "Paris, France" beat one for "Paris".
    `agreement_margin=2` means the vote only stops early once a clear lead
    of 2 exists, so all three rollouts run.
    """
    tools = build_registry()
    finishes = ["Finish[Paris, France]", "Finish[Paris]", "Finish[Paris, France]"]
    rollout_scripts = [["Action: search[eiffel tower]", f"Thought: Done.\nAction: {finish}"] for finish in finishes]
    providers = [get_provider(script=script) for script in rollout_scripts]
    return run_self_consistency(providers, tools, "Where is the Eiffel Tower located?", agreement_margin=2.0)
