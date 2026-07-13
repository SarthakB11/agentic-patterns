"""Metrics aggregation: mean score, pass rate, win rate, and Elo ranking.

Per-case scores and pairwise verdicts are only useful once rolled up into
run-level metrics that a regression gate or a leaderboard can compare. This
module holds the pure aggregation functions; nothing here calls a provider.

The Elo-style ranking implements the same idea the LMSYS Chatbot Arena uses
to turn many local pairwise comparisons into one global ranking: each
candidate starts at a fixed rating, and every match nudges the winner's
rating up and the loser's down by an amount proportional to how surprising
the result was.
"""

from __future__ import annotations

DEFAULT_ELO_RATING = 1000.0
DEFAULT_ELO_K = 32.0


def mean_score(scores: list[float]) -> float:
    """Return the arithmetic mean of `scores`, or 0.0 for an empty list."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def pass_rate(results: list[bool]) -> float:
    """Return the fraction of `results` that are True, or 0.0 if empty."""
    if not results:
        return 0.0
    return sum(1 for r in results if r) / len(results)


def pairwise_win_rate(winners: list[str], candidate: str) -> float:
    """Return `candidate`'s win rate across a list of match winners.

    Args:
        winners: One entry per match: the winning candidate's label, or
            "tie".
        candidate: The candidate to compute a win rate for. Ties count as
            half a win, the standard convention for win-rate leaderboards.

    Returns:
        (wins + 0.5 * ties_involving_candidate) / total_matches, or 0.0 if
        `winners` is empty.
    """
    if not winners:
        return 0.0
    wins = sum(1 for w in winners if w == candidate)
    ties = sum(1 for w in winners if w == "tie")
    return (wins + 0.5 * ties) / len(winners)


def update_elo(
    rating_a: float, rating_b: float, outcome_a: float, *, k: float = DEFAULT_ELO_K
) -> tuple[float, float]:
    """Update two Elo ratings after one match between them.

    Args:
        rating_a: Candidate A's current rating.
        rating_b: Candidate B's current rating.
        outcome_a: A's actual result: 1.0 for a win, 0.0 for a loss, 0.5 for
            a tie.
        k: Maximum rating change per match.

    Returns:
        The updated (rating_a, rating_b).
    """
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    new_a = rating_a + k * (outcome_a - expected_a)
    new_b = rating_b + k * ((1.0 - outcome_a) - (1.0 - expected_a))
    return new_a, new_b


def compute_rankings(
    matches: list[tuple[str, str, str]], *, initial_rating: float = DEFAULT_ELO_RATING, k: float = DEFAULT_ELO_K
) -> dict[str, float]:
    """Roll up pairwise verdicts into an Elo ranking.

    Args:
        matches: One entry per match: (candidate_a_label, candidate_b_label,
            winner), where winner is one of candidate_a_label,
            candidate_b_label, or "tie".
        initial_rating: Starting rating for every candidate seen.
        k: Passed through to `update_elo`.

    Returns:
        A mapping from candidate label to final Elo rating, covering every
        label that appeared in `matches`.
    """
    ratings: dict[str, float] = {}
    for label_a, label_b, winner in matches:
        ratings.setdefault(label_a, initial_rating)
        ratings.setdefault(label_b, initial_rating)
        if winner == label_a:
            outcome_a = 1.0
        elif winner == label_b:
            outcome_a = 0.0
        else:
            outcome_a = 0.5
        ratings[label_a], ratings[label_b] = update_elo(ratings[label_a], ratings[label_b], outcome_a, k=k)
    return ratings
