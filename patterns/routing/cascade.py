"""Sub-module: cost/quality cascade and capability model selection.

Two related but distinct variants live here, both routing across model
*tiers* rather than task categories:

- **Cascade**: try the cheap tier first, run a deterministic quality check
  against its answer, and escalate to the strong tier only when the check
  fails. This is FrugalGPT's shape (Chen, Zaharia, Zou, 2023): a router, a
  quality estimator, and a stop judge, except the estimator here is a
  small local heuristic rather than a second model call, so escalation
  decisions cost nothing beyond the cheap answer itself.
- **Capability model selection**: decide the tier up front from the
  question's difficulty, without trying the cheap tier at all. This is the
  Anthropic model-selection example: easy questions go straight to a small
  model, hard ones straight to a capable one.

`run_baseline_comparison_demo` adds the two baselines the expansion section
calls for: a random-choice router and an always-strong router. A learned or
heuristic router only earns its complexity if it beats random on quality
and beats always-strong on cost; the demo makes both comparisons explicit
on a small fixed dataset instead of asserting the claim.
"""

from __future__ import annotations

import random

from agentic_patterns import Message, Provider, get_provider

from patterns.routing.registry import RouteDecision

_WEAK_SIGNALS = ("i'm not sure", "i don't know", "unclear", "cannot determine", "not enough information")

_STRONG_SYSTEM = "You are a careful expert. Answer precisely and show the key reasoning step."
_CHEAP_SYSTEM = "Answer briefly."

# (question, minimum words needed for a good answer, tier a difficulty heuristic should pick)
_DIFFICULTY_DATASET: list[tuple[str, str]] = [
    ("What is the capital of France?", "cheap"),
    ("What year did the company IPO?", "cheap"),
    ("Derive the break-even price given fixed cost, variable cost, and volume.", "strong"),
    ("Compare the tax implications of the two acquisition structures and recommend one.", "strong"),
]

_HARD_SIGNALS = ("derive", "compare", "prove", "optimize", "recommend", "tax implications", "break-even")


def quality_check(answer: str) -> bool:
    """Deterministic stand-in for FrugalGPT's quality estimator.

    A cheap-tier answer passes if it is reasonably substantial and does not
    contain a hedge phrase signaling the model could not actually answer.
    Real systems might instead score against a rubric or a second model
    call; this stays a local heuristic so escalation decisions are visible
    and free.
    """
    lowered = answer.lower()
    return len(answer.strip()) >= 40 and not any(signal in lowered for signal in _WEAK_SIGNALS)


def run_cascade(question: str, provider: Provider) -> RouteDecision:
    """Try the cheap tier, escalate to the strong tier if the quality check fails.

    Args:
        question: The question to answer.
        provider: Scripted with the cheap-tier answer first and, only
            consumed if escalation happens, the strong-tier answer second.
    """
    cheap_answer = provider.complete([Message.user(question)], system=_CHEAP_SYSTEM).content
    if quality_check(cheap_answer):
        return RouteDecision(route="cheap", score=1.0, method="cascade", attempts=1, metadata={"answer": cheap_answer})

    strong_answer = provider.complete([Message.user(question)], system=_STRONG_SYSTEM).content
    return RouteDecision(
        route="strong",
        score=1.0,
        method="cascade",
        attempts=2,
        metadata={"cheap_answer": cheap_answer, "answer": strong_answer, "escalated": True},
    )


def select_tier(question: str) -> RouteDecision:
    """Pick a tier up front from a difficulty heuristic, with no model call.

    Multi-step or analytical phrasing (see `_HARD_SIGNALS`) routes straight
    to "strong"; everything else routes to "cheap". Unlike `run_cascade`,
    this never tries the cheap tier and checks its output; it decides
    before making any call at all.
    """
    lowered = question.lower()
    if any(signal in lowered for signal in _HARD_SIGNALS):
        return RouteDecision(route="strong", score=1.0, method="capability_selection", metadata={"reason": "hard_signal"})
    return RouteDecision(route="cheap", score=1.0, method="capability_selection", metadata={"reason": "no_hard_signal"})


def run_cascade_demo() -> tuple[RouteDecision, RouteDecision]:
    """Run one cascade that passes on the cheap tier and one that escalates.

    Returns:
        A (passed, escalated) pair of `RouteDecision`. The call count on
        each provider (`provider.calls`) shows the cascade's point: the
        strong tier is only ever invoked on the failing question.
    """
    passing_provider = get_provider(
        script=["The invoice total is $482.10, due on the 15th, covering the March and April subscription periods."]
    )
    passed = run_cascade("What does my latest invoice total and what does it cover?", passing_provider)

    escalating_provider = get_provider(
        script=[
            "I'm not sure, I don't have enough information to calculate that precisely.",
            (
                "Break-even price = (fixed cost / volume) + variable cost per unit. "
                "With $120,000 fixed cost, 8,000 units, and $6 variable cost per unit, "
                "break-even price is $21.00 per unit."
            ),
        ]
    )
    escalated = run_cascade(
        "Derive the break-even price given $120,000 fixed cost, 8,000 units, and $6 variable cost per unit.",
        escalating_provider,
    )
    return passed, escalated


def run_capability_selection_demo() -> list[RouteDecision]:
    """Route every question in `_DIFFICULTY_DATASET` by the up-front heuristic."""
    return [select_tier(question) for question, _ in _DIFFICULTY_DATASET]


def run_baseline_comparison_demo(seed: int = 0) -> dict[str, float]:
    """Compare the difficulty heuristic against random-choice and always-strong.

    Scores each router's tier choice on `_DIFFICULTY_DATASET` against the
    dataset's known correct tier, and counts how often each router picks
    "strong" (a proxy for cost, since the strong tier is the expensive
    one). A router earns its complexity only if it beats random on
    accuracy and beats always-strong on strong-tier usage.

    Args:
        seed: Seed for the random-choice baseline, kept fixed so this demo
            is deterministic.

    Returns:
        A flat dict of metric name to value for all three routers.
    """
    rng = random.Random(seed)
    tiers = ("cheap", "strong")

    heuristic_correct = 0
    heuristic_strong_calls = 0
    random_correct = 0
    random_strong_calls = 0
    always_strong_correct = 0

    for question, correct_tier in _DIFFICULTY_DATASET:
        heuristic_tier = select_tier(question).route
        heuristic_correct += heuristic_tier == correct_tier
        heuristic_strong_calls += heuristic_tier == "strong"

        random_tier = rng.choice(tiers)
        random_correct += random_tier == correct_tier
        random_strong_calls += random_tier == "strong"

        always_strong_correct += correct_tier == "strong"

    n = len(_DIFFICULTY_DATASET)
    return {
        "heuristic_accuracy": heuristic_correct / n,
        "heuristic_strong_calls": heuristic_strong_calls,
        "random_accuracy": random_correct / n,
        "random_strong_calls": random_strong_calls,
        "always_strong_accuracy": always_strong_correct / n,
        "always_strong_calls": n,
    }
