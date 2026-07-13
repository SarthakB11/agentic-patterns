"""Process reward: step-level scoring as a trajectory evaluator.

Source: Lightman et al., "Let's Verify Step by Step" (arXiv:2305.20050),
process supervision and step-level scoring, used here at inference time as
an evaluator, not for training a reward model. Failure-step localization is
motivated by Zhu et al., "Where LLM Agents Fail and How They can Learn From
Failures" (arXiv:2509.25370).

`trajectory.py` grades a whole trace with one holistic verdict, so it
cannot say *which* step is weak or apply a "the chain is only as strong as
its worst step" rule. This module scores each step independently, then
aggregates under a selectable rule: `min` gates on the single worst step,
catching a confidently-wrong middle step that `mean` washes out; `product`
compounds every weak step's penalty instead of gating on only the worst
one; `last` reduces to outcome-only, final-answer judging, the degenerate
case this module contrasts against. `weakest_step_index` localizes the
failure to a step, which a holistic verdict cannot do.

This is process reward reused as an eval-time evaluator, not the training
loop PRM800K was built for; see `docs/research/evaluation_deep.md` for why
training a process reward model offline against a mock teaches nothing the
scripted step scorer does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_patterns import Message, Provider, get_provider

from patterns.evaluation.eval_set import get_case
from patterns.evaluation.trajectory import TrajectoryStep
from patterns.evaluation.verdict import parse_pointwise_verdict

_STEP_SYSTEM = (
    "You grade one step of an agent's trajectory in isolation. Given the goal, "
    "the steps taken so far, and this step's action and observation, judge "
    "whether this step is a necessary, correctly-grounded contribution toward "
    "the goal, not whether the final answer eventually sounds right. Reason "
    "briefly, then end with a SCORE line (0-10)."
)

AggregationRule = Literal["min", "product", "mean", "last"]

DEFAULT_PASS_THRESHOLD = 6.0
_SCALE_MAX = 10.0


@dataclass
class ProcessRewardResult:
    """The outcome of scoring one trajectory step by step.

    Attributes:
        step_scores: One score per step, in trajectory order.
        aggregate_score: `step_scores` combined under `rule`.
        rule: The aggregation rule used.
        weak_step_index: Index of the lowest-scoring step (first minimum on
            a tie), the localized weak point in the trajectory.
        passed: True if `aggregate_score` meets the pass threshold.
    """

    step_scores: list[float]
    aggregate_score: float
    rule: AggregationRule
    weak_step_index: int
    passed: bool


def score_steps(provider: Provider, goal: str, steps: list[TrajectoryStep]) -> list[float]:
    """Score each step of a trajectory independently, one provider call per step.

    Each step is graded with the steps before it shown as context but not
    scored jointly with them: each call's `SCORE` line judges only the one
    step under evaluation.
    """
    scores: list[float] = []
    history: list[TrajectoryStep] = []
    for step in steps:
        prior = (
            "\n".join(f"{i}. {s.action} -> {s.observation}" for i, s in enumerate(history, start=1))
            if history
            else "(none yet)"
        )
        prompt = (
            f"Goal: {goal}\n\nSteps so far:\n{prior}\n\n"
            f"Step to grade: {step.action} -> {step.observation}"
        )
        completion = provider.complete([Message.user(prompt)], system=_STEP_SYSTEM)
        verdict = parse_pointwise_verdict(completion.content)
        scores.append(verdict.score if verdict.score is not None else 0.0)
        history.append(step)
    return scores


def aggregate_step_scores(scores: list[float], rule: AggregationRule, *, scale_max: float = _SCALE_MAX) -> float:
    """Combine per-step scores into one trajectory-level score under `rule`.

    `rule` is "min", "product" (scores normalized to 0-1 and multiplied,
    then rescaled), "mean", or "last". `scale_max` is the scale each score
    is on, used to normalize for "product".

    Raises:
        ValueError: If `scores` is empty or `rule` is not recognized.
    """
    if not scores:
        raise ValueError("aggregate_step_scores requires at least one score")
    if rule == "min":
        return min(scores)
    if rule == "mean":
        return sum(scores) / len(scores)
    if rule == "last":
        return scores[-1]
    if rule == "product":
        normalized = 1.0
        for s in scores:
            normalized *= max(0.0, min(1.0, s / scale_max))
        return normalized * scale_max
    raise ValueError(f"Unknown aggregation rule: {rule!r}")


def weakest_step_index(scores: list[float]) -> int:
    """Return the index of the lowest-scoring step, first minimum on a tie.

    Raises:
        ValueError: If `scores` is empty.
    """
    if not scores:
        raise ValueError("weakest_step_index requires at least one score")
    best_index = 0
    for i in range(1, len(scores)):
        if scores[i] < scores[best_index]:
            best_index = i
    return best_index


def _build_result(scores: list[float], rule: AggregationRule, threshold: float) -> ProcessRewardResult:
    aggregate_score = aggregate_step_scores(scores, rule)
    return ProcessRewardResult(
        step_scores=scores,
        aggregate_score=aggregate_score,
        rule=rule,
        weak_step_index=weakest_step_index(scores),
        passed=aggregate_score >= threshold,
    )


def evaluate_process_reward(
    provider: Provider,
    goal: str,
    steps: list[TrajectoryStep],
    *,
    rule: AggregationRule = "min",
    threshold: float = DEFAULT_PASS_THRESHOLD,
) -> ProcessRewardResult:
    """Score a trajectory step by step and aggregate under `rule`.

    `threshold` is the minimum `aggregate_score` to count as a pass, so
    this evaluator plugs into the same regression gate as the other
    scorers in this pattern.
    """
    scores = score_steps(provider, goal, steps)
    return _build_result(scores, rule, threshold)


def run_process_reward_demo(provider: Provider | None = None) -> tuple[ProcessRewardResult, ProcessRewardResult]:
    """Score one trajectory with an unsupported middle step, aggregated by mean and by min.

    The trajectory looks up the order (grounded, score 9), then infers
    damage from the customer's tone alone rather than confirming it (an
    unsupported leap, score 3), then issues the refund (reasonable given
    policy, but resting on the unconfirmed claim, score 8). Scored by
    `mean` the trajectory passes (6.67, above threshold): the one bad step
    is washed out. Scored by `min` it fails (3): the unsupported step gates
    the whole trajectory, the reason min-aggregation exists.

    Args:
        provider: Judge provider. Defaults to a `MockProvider` scripted
            with exactly the 9, 3, 8 scores above, one call per step.
    """
    case = get_case("refund_investigation")
    steps = [
        TrajectoryStep("look up order 48213", "order found: status=delivered, item=headphones"),
        TrajectoryStep("assume damage from the customer's tone", "customer sounded upset, likely damaged"),
        TrajectoryStep("issue the refund", "refund of $49.99 issued to the original payment method"),
    ]
    if provider is None:
        provider = get_provider(
            script=[
                "The order lookup is necessary and its result is grounded in a real record.\nSCORE: 9",
                "This step never confirms damage; it infers it from tone alone, an unsupported leap.\nSCORE: 3",
                "Issuing the refund follows policy, but it rests on the prior step's unconfirmed claim.\nSCORE: 8",
            ]
        )
    scores = score_steps(provider, case.input, steps)
    mean_result = _build_result(scores, "mean", DEFAULT_PASS_THRESHOLD)
    min_result = _build_result(scores, "min", DEFAULT_PASS_THRESHOLD)
    return mean_result, min_result
