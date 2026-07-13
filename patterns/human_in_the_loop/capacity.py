"""Sub-module: load-aware guard calibration and the inverted-U safety curve.

The flooding demo in `risk_tier.py` shows one point: gating everything
without a deterministic policy backstop lets a fatigued, rubber-stamping
reviewer approve a malicious action. Turan, "Oversight Has a Capacity"
(arXiv:2606.08919), shows the actual result is a curve, not a point: as
the escalation rate rises, realized safety climbs, peaks, then falls,
because a reviewer forced past their attentive capacity rubber-stamps
everything past it. The safety-optimal escalation rate sits strictly
between escalate-nothing and escalate-everything; gating more can make a
system less safe, not more.

The reviewer here is modeled as endogenous: a capacity `C` of actions per
pass they can review attentively, and a deterministic (not sampled)
fatigue rule: a decision is correct while its position in the escalated
queue is at or under `C`, and a rubber-stamp (always approve) after. Risk
scores and true labels are fixed in a scripted stream, so the whole sweep
is arithmetic and reproducible; no model call is required for the
simulation itself, though `risk_classifier.py`'s judge could supply the
scores in a real build.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CAPACITY = 4


@dataclass(frozen=True)
class ScoredAction:
    """One action in the scripted stream, in arrival order.

    Attributes:
        id: Identifier, for pointing a demo or test at one specific action.
        true_label: "safe" or "malicious". Hidden from the guard; used only
            to score realized safety after the fact.
        risk_score: A 0-10 score, standing in for `risk_classifier.py`'s
            judge score or any other scalar risk signal.
    """

    id: str
    true_label: str
    risk_score: float


@dataclass(frozen=True)
class SweepPoint:
    """Realized safety at one escalation threshold.

    Attributes:
        threshold: Actions with `risk_score >= threshold` escalate; the
            rest auto-approve with no reviewer.
        safety: Fraction of malicious actions actually blocked.
        escalated_count: How many actions escalated at this threshold.
    """

    threshold: float
    safety: float
    escalated_count: int


def build_action_stream() -> list[ScoredAction]:
    """A 13-action stream: 4 malicious actions buried among 9 safe ones.

    `m3` sits at position 6, deep enough that escalating everything pushes
    it past a capacity of 4. `m4` shares its risk score with four filler
    safe actions ahead of it, so only a capacity generous enough to reach
    position 8 ever catches it: this is what drives the capacity
    monotonicity result, not just the threshold sweep alone.
    """
    return [
        ScoredAction("m1", "malicious", 9),
        ScoredAction("s1", "safe", 3),
        ScoredAction("m2", "malicious", 8),
        ScoredAction("s2", "safe", 3),
        ScoredAction("s3", "safe", 2),
        ScoredAction("m3", "malicious", 7),
        ScoredAction("s4", "safe", 4),
        ScoredAction("s5", "safe", 4),
        ScoredAction("s6", "safe", 4),
        ScoredAction("s7", "safe", 4),
        ScoredAction("m4", "malicious", 4),
        ScoredAction("s8", "safe", 2),
        ScoredAction("s9", "safe", 3),
    ]


def _simulate_escalation(stream: list[ScoredAction], threshold: float, capacity: int) -> dict[str, bool]:
    """Escalate actions at or above `threshold`, in stream order, and decide each.

    Returns a map from escalated action id to whether the reviewer's
    decision on it was correct: True blocks a malicious action and passes
    a safe one; False is a rubber-stamp, which only ever misses a
    malicious action, since a safe one is approved either way. The
    decision rule is deterministic arithmetic on position, not sampling:
    correct while `position <= capacity`, rubber-stamp after.
    """
    escalated = [a for a in stream if a.risk_score >= threshold]
    return {a.id: (position <= capacity) for position, a in enumerate(escalated, start=1)}


def realized_safety(stream: list[ScoredAction], threshold: float, capacity: int) -> SweepPoint:
    """Compute the safety fraction and escalated count at one threshold."""
    outcomes = _simulate_escalation(stream, threshold, capacity)
    malicious = [a for a in stream if a.true_label == "malicious"]
    blocked = sum(1 for a in malicious if outcomes.get(a.id, False))
    safety = blocked / len(malicious) if malicious else 0.0
    return SweepPoint(threshold=threshold, safety=safety, escalated_count=len(outcomes))


def is_action_blocked(stream: list[ScoredAction], action_id: str, threshold: float, capacity: int) -> bool:
    """Whether one specific action was blocked at a given threshold and capacity."""
    return _simulate_escalation(stream, threshold, capacity).get(action_id, False)


def sweep_escalation_thresholds(stream: list[ScoredAction], capacity: int) -> list[SweepPoint]:
    """Sweep every distinct risk score as a threshold, plus one above the max.

    The lowest swept threshold (the minimum score in the stream) escalates
    everyone, since every score is at or above it: escalate-everything.
    The highest (one past the maximum score) escalates no one:
    escalate-nothing. Both are the curve's endpoints, in ascending order.
    """
    scores = sorted({a.risk_score for a in stream})
    thresholds = [*scores, scores[-1] + 1]
    return [realized_safety(stream, t, capacity) for t in thresholds]


def safety_optimal_threshold(curve: list[SweepPoint]) -> SweepPoint:
    """Argmax safety over a swept curve; ties favor the higher threshold.

    A tie is broken toward less reviewer load (fewer escalations), which
    is the load-aware preference the whole module is built around: do not
    escalate more than the safety curve actually pays for.
    """
    return max(curve, key=lambda p: (p.safety, p.threshold))


def capacity_fitted_threshold(stream: list[ScoredAction], capacity: int) -> SweepPoint:
    """Pick the lowest threshold whose escalated count still fits `capacity`.

    Unlike `safety_optimal_threshold`, this needs no true-label knowledge:
    it is the greedy guard a real system can actually run, fitting the
    escalation rate to a reviewer's stated capacity rather than to
    hindsight safety. Falls back to escalate-nothing if even the smallest
    escalated set exceeds capacity.
    """
    curve = sweep_escalation_thresholds(stream, capacity)
    fitting = [p for p in curve if p.escalated_count <= capacity]
    return min(fitting, key=lambda p: p.threshold) if fitting else curve[-1]


@dataclass
class InvertedUDemoResult:
    """Outcome of sweeping the escalation threshold at one capacity.

    Attributes:
        capacity: The reviewer capacity the curve was computed for.
        curve: Every swept point, ascending by threshold.
        optimal: The safety-optimal point (argmax of `curve`).
    """

    capacity: int
    curve: list[SweepPoint]
    optimal: SweepPoint


def run_inverted_u_demo(capacity: int = DEFAULT_CAPACITY) -> InvertedUDemoResult:
    """Sweep the stream's escalation threshold and find the safety-optimal point.

    Escalate-nothing misses every malicious action; escalate-everything
    floods the reviewer past capacity and rubber-stamps the deepest ones.
    The optimal threshold sits strictly between the two, which is the
    inverted-U: safety is not monotonic in the escalation rate.
    """
    stream = build_action_stream()
    curve = sweep_escalation_thresholds(stream, capacity)
    return InvertedUDemoResult(capacity=capacity, curve=curve, optimal=safety_optimal_threshold(curve))


@dataclass
class FloodingComparisonResult:
    """Outcome of contrasting escalate-everything against the optimal threshold.

    Attributes:
        buried_action_id: The malicious action used as the demo's example.
        blocked_by_escalate_everything: Whether it was blocked when every
            action escalates.
        blocked_at_optimal_threshold: Whether it was blocked at the
            safety-optimal threshold for the same capacity.
    """

    buried_action_id: str
    blocked_by_escalate_everything: bool
    blocked_at_optimal_threshold: bool


def run_flooding_vs_optimal_demo(capacity: int = DEFAULT_CAPACITY) -> FloodingComparisonResult:
    """`m3` is buried deep enough that flooding the reviewer misses it.

    Escalating everything gives the reviewer no way to prioritize: `m3`
    lands past `capacity` in the queue and is rubber-stamped. The
    safety-optimal threshold escalates a narrower, higher-precision set
    that still contains `m3`, keeping it within the reviewer's capacity.
    """
    stream = build_action_stream()
    curve = sweep_escalation_thresholds(stream, capacity)
    escalate_everything = curve[0]
    optimal = safety_optimal_threshold(curve)
    return FloodingComparisonResult(
        buried_action_id="m3",
        blocked_by_escalate_everything=is_action_blocked(stream, "m3", escalate_everything.threshold, capacity),
        blocked_at_optimal_threshold=is_action_blocked(stream, "m3", optimal.threshold, capacity),
    )


def run_capacity_monotonicity_demo(small_capacity: int = 4, large_capacity: int = 8) -> tuple[SweepPoint, SweepPoint]:
    """Raising capacity lets the safety-optimal threshold escalate more.

    At the small capacity, catching `m4` would flood the reviewer with
    filler safe actions ahead of it for no safety gain, so the optimum
    stays at the higher, narrower threshold. At the larger capacity the
    reviewer can absorb that flood, `m4` is caught too, and the optimum
    moves to the lower threshold that captures it.
    """
    stream = build_action_stream()
    small = safety_optimal_threshold(sweep_escalation_thresholds(stream, small_capacity))
    large = safety_optimal_threshold(sweep_escalation_thresholds(stream, large_capacity))
    return small, large
