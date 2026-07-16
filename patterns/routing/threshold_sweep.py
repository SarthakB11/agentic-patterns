"""Continuous route score against a swept cost threshold: RouteLLM's operating shape.

Every router elsewhere in this folder returns a hard label: cheap or
strong, this route or that one. RouteLLM (Ong et al., arXiv:2406.18665)
and the tuned-threshold cascades (Zellinger, Liu, Thomson, arXiv:2502.09054)
operate differently: a router predicts a continuous score (a win-rate
estimate, a confidence), and a single threshold decides cheap-versus-strong.
Sweeping the threshold traces a cost-quality frontier, and picking the
threshold that meets a cost budget is how a production router is actually
run day to day.

This module builds that operating shape without a trained model. RouteLLM's
score comes from a model trained on preference data (out of scope here, see
the folder README's Skipped section); `score_query` is a small, inspectable
heuristic standing in for it, reusing `cascade._HARD_SIGNALS` the same way
`cascade.select_tier` does, plus prompt length and digit presence. The
mechanism this module teaches, a score plus a swept threshold, is exactly
what a trained router would expose at inference time; only the score's
origin is simplified.
"""

from __future__ import annotations

from dataclasses import dataclass

from patterns.routing import cascade, router_eval

_HARD_SIGNAL_SATURATION = 2  # 2+ hard-signal words already means "definitely strong"
_LONG_PROMPT_WORDS = 30

# A fixed, deterministic sweep grid. 0.0 and above-1.0 are included on
# purpose: they are the degenerate ends where the sweep collapses onto
# always-strong and always-cheap respectively.
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.1)


def score_query(query: str) -> float:
    """Estimate, in [0, 1], how much the strong tier is likely to help `query`.

    A deterministic stand-in for a trained win-rate predictor, combining
    three signals: hard-signal word density (derive, compare, prove, and
    the rest of `cascade._HARD_SIGNALS`), prompt length, and whether the
    prompt contains a digit (numeric questions often need careful
    arithmetic). Weighted and clamped to [0, 1]; the weights are chosen so
    two or more hard-signal words alone are enough to saturate the
    hard-signal component, without any model call.
    """
    lowered = query.lower()
    hard_hits = sum(1 for signal in cascade._HARD_SIGNALS if signal in lowered)
    hard_density = min(hard_hits / _HARD_SIGNAL_SATURATION, 1.0)
    length_signal = min(len(query.split()) / _LONG_PROMPT_WORDS, 1.0)
    digit_signal = 1.0 if any(ch.isdigit() for ch in query) else 0.0
    score = 0.55 * hard_density + 0.30 * length_signal + 0.15 * digit_signal
    return max(0.0, min(1.0, score))


def route_at_threshold(score: float, t: float) -> str:
    """Strong if `score >= t`, else cheap. The single knob a swept router operates."""
    return "strong" if score >= t else "cheap"


@dataclass
class FrontierPoint:
    """One point on the swept cost-quality frontier.

    Attributes:
        t: The threshold this point was computed at.
        accuracy: Fraction of rows routed to their labeled tier.
        cost: Total charged cost at this threshold (`router_eval.route_cost`).
        strong_fraction: Fraction of rows routed to "strong".
        flips_from_previous: Number of rows whose route differs from the
            previous (next-lower) threshold in the sweep; 0 for the first point.
    """

    t: float
    accuracy: float
    cost: float
    strong_fraction: float
    flips_from_previous: int


def sweep(
    dataset: list[tuple[str, str]], thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID
) -> list[FrontierPoint]:
    """Route every row in `dataset` at each threshold in `thresholds` and score it.

    Args:
        dataset: Rows of (question, correct_tier), e.g. `cascade._DIFFICULTY_DATASET`.
        thresholds: Fixed, ascending grid of thresholds to sweep.

    Returns:
        One `FrontierPoint` per threshold, in the same order as `thresholds`.
    """
    scores = [score_query(question) for question, _ in dataset]
    labels = [label for _, label in dataset]

    points: list[FrontierPoint] = []
    previous_routes: list[str] | None = None
    for t in thresholds:
        routes = [route_at_threshold(s, t) for s in scores]
        correct = sum(1 for route, label in zip(routes, labels) if route == label)
        cost = sum(router_eval.route_cost(route) for route in routes)
        strong_fraction = sum(1 for route in routes if route == "strong") / len(routes)
        flips = 0 if previous_routes is None else sum(1 for a, b in zip(routes, previous_routes) if a != b)
        points.append(
            FrontierPoint(
                t=t,
                accuracy=correct / len(labels),
                cost=cost,
                strong_fraction=strong_fraction,
                flips_from_previous=flips,
            )
        )
        previous_routes = routes
    return points


def pick_operating_point(frontier: list[FrontierPoint], cost_budget: float) -> FrontierPoint:
    """Return the highest-accuracy point whose cost is at or under `cost_budget`.

    Ties on accuracy are broken by lower cost (the cheaper of two equally
    accurate operating points), so the choice is deterministic.

    Raises:
        ValueError: If every point in `frontier` exceeds `cost_budget`.
    """
    affordable = [p for p in frontier if p.cost <= cost_budget]
    if not affordable:
        raise ValueError(f"No operating point fits within cost_budget={cost_budget}")
    return max(affordable, key=lambda p: (p.accuracy, -p.cost))


def run_threshold_sweep_demo() -> tuple[list[FrontierPoint], FrontierPoint, FrontierPoint]:
    """Sweep the tier dataset's frontier and pick a tight-budget and a loose-budget point.

    Returns:
        A (frontier, tight_point, loose_point) tuple. The tight budget
        forces a mostly-cheap operating point; the loose budget allows more
        strong-tier routing, making the cost-quality tradeoff the threshold
        knob controls visible on the same frontier.
    """
    frontier = sweep(cascade._DIFFICULTY_DATASET)
    tight_point = pick_operating_point(frontier, cost_budget=20.0)
    loose_point = pick_operating_point(frontier, cost_budget=50.0)
    return frontier, tight_point, loose_point
