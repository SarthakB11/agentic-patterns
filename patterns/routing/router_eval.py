"""Benchmark every router in this folder against baselines and an oracle.

The folder builds five routers and never asked whether any of them earns
its complexity. LLMRouterBench (Li et al., arXiv:2601.07206) unifies 400K
instances across 21 datasets and finds that many routers, commercial ones
included, fail to reliably beat a simple baseline, and that most of the
gap to an oracle comes from model recall failing, not from routing being
clever. This module reproduces that methodology offline: score every
router against random-choice, always-cheapest, always-strongest, and an
oracle that always picks the correct answer.

Two label spaces, not one: `rule_based`, `semantic`, and `llm_classifier`
route to a *category* (billing, technical, account, general); `cascade`
and `select_tier` route to a *cost tier* (cheap, strong). These are not
interchangeable, so this module scores each router only against its own
label space (`_CATEGORY_DATASET` or `cascade._DIFFICULTY_DATASET`), sharing
the methodology, not the labels: random/oracle baselines everywhere, and
the always-cheap/always-strong cost axis only where cheap and strong are
real prices (the tier track).

Cost accounting is the other piece `cascade.run_baseline_comparison_demo`
skipped: every router here is charged its own classifier overhead
(`_CLASSIFIER_OVERHEAD`), not just the handler it dispatches to, and a
cascade that escalates is charged the burned cheap attempt plus the strong
tier, not the strong tier alone.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from agentic_patterns import Provider, get_provider
from patterns.routing import cascade, llm_classifier, rule_based, semantic
from patterns.routing.registry import RouteDecision

Dataset = Literal["category", "tier"]

_TIER_COSTS: dict[str, float] = {"cheap": 1.0, "strong": 10.0, "human": 20.0}
_CATEGORY_COST = 1.0  # flat handler cost; support categories have no cost ordering

# Cost of running the classifier itself, on top of the handler it dispatches
# to. Rule-based and semantic are model-free; an LLM classifier spends one
# call on its own classification.
_CLASSIFIER_OVERHEAD: dict[str, float] = {
    "rule_based": 0.0, "semantic": 0.0, "llm_classifier": 1.0, "cascade": 0.0, "select_tier": 0.0,
}

# semantic's below-threshold route and the others' "general" default mean
# the same thing ("no specific category fits"); normalize before scoring.
_NO_MATCH_ALIASES: dict[str, str] = {"no_match": "general"}

_ACCURACY_TOLERANCE = 0.1  # "beats always-strong on cost" tolerates this much accuracy loss

_CATEGORY_DATASET: list[tuple[str, str]] = [
    ("I was charged twice for my subscription this month", "billing"),
    ("the app crashes every time I open settings", "technical"),
    ("I forgot my password and I'm locked out of my account", "account"),
    ("do you offer support in French", "general"),
    ("how do I get a refund for last invoice", "billing"),
    ("I am getting an error installing the latest update", "technical"),
    ("I am locked out of my account after too many login attempts", "account"),
    ("what are your support hours", "general"),
]


@dataclass
class RouterScore:
    """One router's (or baseline's) benchmark result.

    Attributes:
        name: Router or baseline name.
        dataset: Which label space this was scored against.
        accuracy: Fraction of rows where the chosen route matched the label.
        total_cost: Sum of per-row charged cost, including classifier
            overhead and any burned cascade attempts.
        beats_random: True if `accuracy` beats the random baseline's.
        beats_always_strong_on_cost: For the tier dataset, True if
            `total_cost` beats always-strong's while `accuracy` stays
            within `_ACCURACY_TOLERANCE` of it. None on the category
            dataset, where "always-strong" has no meaning.
        oracle_gap: Oracle accuracy minus this router's accuracy.
        earns_complexity: True only for a non-baseline router that clears
            every baseline that applies to its dataset.
        is_baseline: True for random/oracle/always-cheap/always-strong rows.
    """

    name: str
    dataset: Dataset
    accuracy: float
    total_cost: float
    beats_random: bool
    beats_always_strong_on_cost: bool | None
    oracle_gap: float = 0.0
    earns_complexity: bool = False
    is_baseline: bool = False


def route_cost(route: str) -> float:
    """Unit cost of a route: tier price if it names a tier, else flat category cost."""
    return _TIER_COSTS.get(route, _CATEGORY_COST)


def _normalize(route: str) -> str:
    """Map an alias route name (e.g. semantic's "no_match") to its canonical label."""
    return _NO_MATCH_ALIASES.get(route, route)


def _build_score(
    name: str, dataset: Dataset, routes: list[str], labels: list[str], costs: list[float], *, is_baseline: bool = False
) -> RouterScore:
    """Turn per-row routes/labels/costs into accuracy and total cost; flags filled in by `_finalize`."""
    correct = sum(1 for r, label in zip(routes, labels) if _normalize(r) == label)
    return RouterScore(
        name=name, dataset=dataset, accuracy=correct / len(labels), total_cost=sum(costs),
        beats_random=False, beats_always_strong_on_cost=None, is_baseline=is_baseline,
    )


# --- category-track routers (rule_based, semantic, llm_classifier) ------------


def _score_no_model_router(
    name: str, classify_fn: Callable[[str], RouteDecision], dataset: list[tuple[str, str]] = _CATEGORY_DATASET
) -> RouterScore:
    """Score a no-model classifier (`rule_based.classify` or `semantic.classify`)."""
    routes = [classify_fn(query).route for query, _ in dataset]
    costs = [_CLASSIFIER_OVERHEAD[name] + route_cost(r) for r in routes]
    return _build_score(name, "category", routes, [label for _, label in dataset], costs)


def score_rule_based(dataset: list[tuple[str, str]] = _CATEGORY_DATASET) -> RouterScore:
    """Score `rule_based.classify` against `dataset`'s category labels."""
    return _score_no_model_router("rule_based", rule_based.classify, dataset)


def score_semantic(dataset: list[tuple[str, str]] = _CATEGORY_DATASET) -> RouterScore:
    """Score `semantic.classify` against `dataset`'s category labels."""
    return _score_no_model_router("semantic", semantic.classify, dataset)


def score_llm_classifier(provider: Provider, dataset: list[tuple[str, str]] = _CATEGORY_DATASET) -> RouterScore:
    """Score `llm_classifier.classify` against `dataset`'s category labels.

    Args:
        provider: Scripted with one `ROUTE: <name>` reply per row, in order.
        dataset: Rows to classify; pass a mislabeling script to test how
            the benchmark catches a weak classifier.
    """
    routes = [llm_classifier.classify(query, provider).route for query, _ in dataset]
    costs = [_CLASSIFIER_OVERHEAD["llm_classifier"] + route_cost(r) for r in routes]
    return _build_score("llm_classifier", "category", routes, [label for _, label in dataset], costs)


def _default_llm_classifier_provider() -> Provider:
    """Scripted provider that labels every `_CATEGORY_DATASET` row correctly."""
    return get_provider(script=[f"ROUTE: {label}" for _, label in _CATEGORY_DATASET])


# --- tier-track routers (cascade, select_tier) ---------------------------------


def _cascade_script_for(dataset: list[tuple[str, str]]) -> list[str]:
    """Script so `cascade.run_cascade` lands on each row's labeled tier.

    A "cheap" row gets one substantial, hedge-free answer, which passes
    `cascade.quality_check` and stops there. A "strong" row gets a hedging
    cheap answer first (fails the check), then a substantial strong answer.
    """
    script: list[str] = []
    for _, tier in dataset:
        if tier == "cheap":
            script.append("This is a complete, confident answer with enough detail to pass review.")
        else:
            script.append("I'm not sure, I don't have enough information to answer that precisely.")
            script.append("After working through the calculation in full: this is the detailed strong-tier answer.")
    return script


def score_cascade(dataset: list[tuple[str, str]] = cascade._DIFFICULTY_DATASET) -> RouterScore:
    """Score `cascade.run_cascade` against `dataset`'s tier labels.

    Cost honesty: a row that escalates is charged both the burned cheap
    attempt and the strong tier, not the strong tier alone.
    """
    provider = get_provider(script=_cascade_script_for(dataset))
    routes = [cascade.run_cascade(question, provider).route for question, _ in dataset]
    costs = [_TIER_COSTS["cheap"] + _TIER_COSTS["strong"] if r == "strong" else _TIER_COSTS["cheap"] for r in routes]
    return _build_score("cascade", "tier", routes, [label for _, label in dataset], costs)


def score_select_tier(dataset: list[tuple[str, str]] = cascade._DIFFICULTY_DATASET) -> RouterScore:
    """Score `cascade.select_tier` against `dataset`'s tier labels."""
    routes = [cascade.select_tier(question).route for question, _ in dataset]
    costs = [_CLASSIFIER_OVERHEAD["select_tier"] + route_cost(r) for r in routes]
    return _build_score("select_tier", "tier", routes, [label for _, label in dataset], costs)


# --- baselines -----------------------------------------------------------------


def oracle_score(dataset: list[tuple[str, str]], kind: Dataset) -> RouterScore:
    """The quality ceiling: route == label by construction, no burned attempts."""
    labels = [label for _, label in dataset]
    score = _build_score(
        "oracle", kind, list(labels), labels, [route_cost(label) for label in labels], is_baseline=True
    )
    score.accuracy = 1.0
    return score


def random_score(dataset: list[tuple[str, str]], kind: Dataset, choices: list[str], seed: int = 0) -> RouterScore:
    """Fixed-seed random choice among `choices`, for a reproducible baseline."""
    rng = random.Random(seed)
    routes = [rng.choice(choices) for _ in dataset]
    return _build_score(
        "random", kind, routes, [label for _, label in dataset], [route_cost(r) for r in routes], is_baseline=True
    )


def always_score(name: str, dataset: list[tuple[str, str]], kind: Dataset, fixed_route: str) -> RouterScore:
    """Every row routes to `fixed_route` regardless of its label."""
    routes = [fixed_route] * len(dataset)
    return _build_score(
        name,
        kind,
        routes,
        [label for _, label in dataset],
        [route_cost(fixed_route)] * len(dataset),
        is_baseline=True,
    )


def _finalize(
    score: RouterScore, oracle: RouterScore, random_baseline: RouterScore, always_strong: RouterScore | None
) -> RouterScore:
    """Fill in the comparison flags on `score`, including baselines against each other."""
    score.beats_random = score.accuracy > random_baseline.accuracy
    if always_strong is not None:
        score.beats_always_strong_on_cost = (
            score.total_cost < always_strong.total_cost
            and score.accuracy >= always_strong.accuracy - _ACCURACY_TOLERANCE
        )
    score.oracle_gap = oracle.accuracy - score.accuracy
    if not score.is_baseline:
        score.earns_complexity = score.beats_random and (
            always_strong is None or bool(score.beats_always_strong_on_cost)
        )
    return score


def run_benchmark(seed: int = 0) -> list[RouterScore]:
    """Run every router in this folder against baselines and an oracle.

    Returns:
        Category-track routers and baselines first, then tier-track.
        Deterministic: the LLM classifier's script always labels every row
        correctly, the random baselines use `seed`, and every other router
        is a pure or scripted-deterministic function of its dataset.
    """
    category_labels = ["billing", "technical", "account", "general"]
    category_oracle = oracle_score(_CATEGORY_DATASET, "category")
    category_random = random_score(_CATEGORY_DATASET, "category", category_labels, seed=seed)
    category_scores = [score_rule_based(), score_semantic(), score_llm_classifier(_default_llm_classifier_provider())]
    for score in [category_random, *category_scores]:
        _finalize(score, category_oracle, category_random, always_strong=None)
    category_oracle.beats_random, category_oracle.oracle_gap = category_oracle.accuracy > category_random.accuracy, 0.0

    tier_dataset = cascade._DIFFICULTY_DATASET
    tier_oracle = oracle_score(tier_dataset, "tier")
    tier_random = random_score(tier_dataset, "tier", ["cheap", "strong"], seed=seed)
    tier_always_cheap = always_score("always_cheap", tier_dataset, "tier", "cheap")
    tier_always_strong = always_score("always_strong", tier_dataset, "tier", "strong")
    tier_scores = [score_cascade(), score_select_tier()]
    for score in [tier_random, tier_always_cheap, tier_always_strong, *tier_scores]:
        _finalize(score, tier_oracle, tier_random, tier_always_strong)
    tier_oracle.beats_random, tier_oracle.oracle_gap = tier_oracle.accuracy > tier_random.accuracy, 0.0

    return [
        category_oracle, category_random, *category_scores,
        tier_oracle, tier_random, tier_always_cheap, tier_always_strong, *tier_scores,
    ]


def render_table(scores: list[RouterScore]) -> str:
    """Render `run_benchmark`'s scores as an aligned, readable table."""
    header = (
        f"{'router':<16} {'dataset':<9} {'accuracy':>8} {'cost':>7} {'beats_random':>13} "
        f"{'beats_strong$':>14} {'oracle_gap':>11} {'verdict':>17}"
    )
    lines = [header]
    for s in scores:
        beats_strong = "n/a" if s.beats_always_strong_on_cost is None else str(s.beats_always_strong_on_cost)
        verdict = "(baseline)" if s.is_baseline else ("earns_complexity" if s.earns_complexity else "near_baseline")
        lines.append(
            f"{s.name:<16} {s.dataset:<9} {s.accuracy:>8.3f} {s.total_cost:>7.1f} "
            f"{str(s.beats_random):>13} {beats_strong:>14} {s.oracle_gap:>11.3f} {verdict:>17}"
        )
    return "\n".join(lines)
