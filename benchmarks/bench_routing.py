"""Benchmark: router accuracy vs an oracle, with a cost narrative.

Three ways to decide whether a prompt needs the strong model tier, scored
against 24 hand-labeled prompts (12 "cheap", 12 "hard", ground truth below):

- `semantic`: embed the prompt, nearest-centroid match against a few labeled
  example prompts per tier. Reuses `patterns.routing.semantic.route_scores`
  and `classify_scores` with a tier registry instead of the pattern's
  default category registry.
- `llm_classifier`: ask Flash-Lite to read a rubric and reply `ROUTE:
  cheap` or `ROUTE: hard`. Reuses `patterns.routing.llm_classifier.
  parse_route_label` for the same validate-or-fallback behavior the
  pattern's category classifier gets.
- `always_hard`: the naive baseline, every prompt routed to the strong tier.

The headline is routing accuracy against the oracle labels. The cost
narrative in `detail` projects, from `harness.PRICING` and a fixed assumed
answer size per tier, what it would cost to then answer all 24 tasks under
each router's tier choices: a router that is merely decent at telling easy
from hard already beats always-hard on cost while keeping the hard tasks on
the strong model. No task is actually answered on the strong model here;
only the routing decision is measured and the answer cost is projected.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Embedder, Message, Provider
from benchmarks.harness import (
    PRICING,
    BenchProvider,
    BenchResult,
    finalize,
    live_provider,
    mock_embedder,
    mock_provider,
)
from patterns.routing.llm_classifier import parse_route_label
from patterns.routing.registry import Route, RouteDecision, RouteRegistry
from patterns.routing.semantic import classify_scores, route_scores

CHEAP_MODEL = "gemini-3.1-flash-lite"
HARD_MODEL = "gemini-3.5-flash"

# Assumed token size of one answered task, used only to project cost from
# the routing decision, never to actually answer anything on the strong
# model. Hard tasks are assumed to need a longer, more reasoned answer.
_ANSWER_TOKENS: dict[str, tuple[int, int]] = {
    "cheap": (200, 150),   # (input, output) tokens for a short factual/format task
    "hard": (400, 600),    # a multi-step or analytical answer runs longer
}

_DEFAULT_THRESHOLD = 0.15

# --- ground truth: 24 prompts, 12 cheap / 12 hard, hand-labeled -----------------

TASKS: list[tuple[str, str]] = [
    # cheap: short factual lookups, formatting, simple rewrites
    ("What is the capital of Australia?", "cheap"),
    ("What year did World War II end?", "cheap"),
    ("Convert 'hello world' to uppercase.", "cheap"),
    ("List the days of the week.", "cheap"),
    ("What is the chemical symbol for gold?", "cheap"),
    ("Rewrite 'i cant beleive its monday again' with correct spelling.", "cheap"),
    ("What is 12 plus 7?", "cheap"),
    ("Name the largest planet in the solar system.", "cheap"),
    ("Format this date as YYYY-MM-DD: March 3, 2026.", "cheap"),
    ("What is the boiling point of water in Celsius?", "cheap"),
    ("Turn this into a bullet list: apples, bananas, cherries.", "cheap"),
    ("Who wrote the play Romeo and Juliet?", "cheap"),
    # hard: multi-step reasoning, tricky math, careful analysis
    ("A train leaves at 2pm going 60mph and another leaves at 3pm going 90mph on the "
     "same route; at what time does the second train catch the first?", "hard"),
    ("Compare the tax implications of an asset sale versus a stock sale for a company "
     "acquisition and recommend which structure minimizes buyer risk.", "hard"),
    ("Derive the break-even unit price given a fixed cost of $50,000, a variable cost "
     "of $12 per unit, and a target volume of 5,000 units.", "hard"),
    ("Prove that the square root of 2 is irrational.", "hard"),
    ("Design a database schema for a multi-tenant SaaS billing system, explaining the "
     "tradeoffs between row-level and schema-level tenant isolation.", "hard"),
    ("Optimize a marketing budget of $100,000 across three channels with different "
     "diminishing-returns curves to maximize total conversions.", "hard"),
    ("Walk through the failure modes of a distributed cache that could cause stale "
     "reads under concurrent writes, and propose a fix for each.", "hard"),
    ("A store offers 20% off, then an additional 10% off the discounted price, then "
     "charges 8% sales tax; what percentage of the original price does a customer pay?", "hard"),
    ("Analyze the second-order effects of raising a product's price by 15% on churn, "
     "referral volume, and support load, and recommend whether to proceed.", "hard"),
    ("Explain step by step why a race condition can occur in this scenario: two threads "
     "increment the same unlocked counter variable 1000 times each, and predict the range "
     "of possible final values.", "hard"),
    ("Given quarterly revenues of 120, 135, 128, and 150 (in thousands), forecast next "
     "quarter using a reasoned trend argument, not just the average.", "hard"),
    ("Critique this argument for a logical fallacy: 'Our competitor raised prices and "
     "grew revenue, so raising our prices will also grow our revenue.'", "hard"),
]

assert sum(1 for _, label in TASKS if label == "cheap") == 12
assert sum(1 for _, label in TASKS if label == "hard") == 12

_TIER_REGISTRY = RouteRegistry(
    [
        Route(
            name="cheap",
            description="Short factual lookups, simple formatting, or a straightforward rewrite. "
            "A small model answers these correctly without multi-step reasoning.",
            utterances=[
                "What is the capital of Japan?",
                "Convert this sentence to lowercase.",
                "What year did the company IPO?",
                "List the primary colors.",
                "What is 15% of 200?",
            ],
            tier="cheap",
        ),
        Route(
            name="hard",
            description="Multi-step reasoning, tricky math, or careful analysis that requires "
            "working through several dependent steps or weighing tradeoffs.",
            utterances=[
                "Derive the break-even price given fixed cost, variable cost, and volume.",
                "Compare the tax implications of two acquisition structures and recommend one.",
                "Optimize the marketing budget allocation across three channels to maximize ROI.",
                "Prove that the sum of the first n odd numbers equals n squared.",
                "Analyze the second-order effects of a price change and recommend a decision.",
            ],
            tier="hard",
        ),
    ]
)

_CLASSIFIER_SYSTEM = (
    "You are a routing classifier that decides which model tier should answer a prompt. "
    "Reply with exactly one line: ROUTE: cheap or ROUTE: hard. Reply with nothing else.\n\n"
    "Rubric:\n"
    "- cheap: short factual lookups, simple formatting, or a straightforward rewrite. "
    "A small model can answer correctly in one step.\n"
    "- hard: multi-step reasoning, tricky math, or careful analysis that needs working "
    "through several dependent steps or weighing tradeoffs."
)


@dataclass
class _RouterOutcome:
    """One router's routing decisions across every task."""

    variant: str
    routes: list[str]
    accuracy: float
    projected_cost_usd: float


def _projected_cost(routes: list[str]) -> float:
    """Project the $ cost of answering every task under a router's tier choices.

    Uses `harness.PRICING` list prices and the fixed `_ANSWER_TOKENS` size
    assumption per tier. Never answers anything; this only prices the
    routing decision that was already made.
    """
    total = 0.0
    for route in routes:
        model = CHEAP_MODEL if route == "cheap" else HARD_MODEL
        in_tokens, out_tokens = _ANSWER_TOKENS[route if route in _ANSWER_TOKENS else "hard"]
        in_price, out_price = PRICING[model]
        total += in_tokens / 1e6 * in_price + out_tokens / 1e6 * out_price
    return total


def _accuracy(routes: list[str], labels: list[str]) -> float:
    """Fraction of rows where the router's chosen tier matched the oracle label."""
    correct = sum(1 for route, label in zip(routes, labels) if route == label)
    return correct / len(labels)


def _run_semantic(embedder: Embedder, tasks: list[tuple[str, str]]) -> _RouterOutcome:
    """Score the semantic (nearest-centroid) router over `tasks`."""
    routes: list[str] = []
    for prompt, _ in tasks:
        scores = route_scores(prompt, _TIER_REGISTRY, embedder=embedder)
        decision = classify_scores(scores, threshold=_DEFAULT_THRESHOLD)
        routes.append(decision.route if decision.route in ("cheap", "hard") else "hard")
    labels = [label for _, label in tasks]
    return _RouterOutcome("semantic", routes, _accuracy(routes, labels), _projected_cost(routes))


def _run_llm_classifier(provider: Provider, tasks: list[tuple[str, str]]) -> _RouterOutcome:
    """Score the LLM-classifier router over `tasks`. One provider call per task."""
    routes: list[str] = []
    for prompt, _ in tasks:
        completion = provider.complete([Message.user(prompt)], system=_CLASSIFIER_SYSTEM)
        decision: RouteDecision = parse_route_label(completion.content, _TIER_REGISTRY, default="hard")
        routes.append(decision.route)
    labels = [label for _, label in tasks]
    return _RouterOutcome("llm_classifier", routes, _accuracy(routes, labels), _projected_cost(routes))


def _run_always_hard(tasks: list[tuple[str, str]]) -> _RouterOutcome:
    """The naive baseline: every task routed to the strong tier."""
    routes = ["hard"] * len(tasks)
    labels = [label for _, label in tasks]
    return _RouterOutcome("always_hard", routes, _accuracy(routes, labels), _projected_cost(routes))


def _run(provider: BenchProvider, embedder: Embedder) -> BenchResult:
    """Shared logic: score every variant over `TASKS`, mock and live alike."""
    semantic = _run_semantic(embedder, TASKS)
    llm = _run_llm_classifier(provider, TASKS)
    always_hard = _run_always_hard(TASKS)
    outcomes = [semantic, llm, always_hard]

    oracle_cost = _projected_cost([label for _, label in TASKS])
    tasks_rows = [
        {
            "id": i,
            "prompt": prompt,
            "label": label,
            "semantic": semantic.routes[i],
            "llm_classifier": llm.routes[i],
            "always_hard": always_hard.routes[i],
            "semantic_correct": semantic.routes[i] == label,
            "llm_classifier_correct": llm.routes[i] == label,
        }
        for i, (prompt, label) in enumerate(TASKS)
    ]

    best = max(outcomes, key=lambda o: o.accuracy)
    headline = (
        f"{best.variant} routes {best.accuracy:.0%} of {len(TASKS)} prompts to the correct tier "
        f"(vs semantic {semantic.accuracy:.0%}, llm_classifier {llm.accuracy:.0%}, oracle 100%), "
        f"projecting ${best.projected_cost_usd:.5f} to answer all tasks versus "
        f"${always_hard.projected_cost_usd:.5f} for always_hard "
        f"({(1 - best.projected_cost_usd / always_hard.projected_cost_usd):.0%} cheaper)."
    )

    return BenchResult(
        name="routing",
        model=provider.model,
        n=len(TASKS),
        variants={o.variant: o.accuracy for o in outcomes},
        headline=headline,
        detail={
            "oracle_accuracy": 1.0,
            "oracle_projected_cost_usd": round(oracle_cost, 5),
            "projected_cost_usd": {o.variant: round(o.projected_cost_usd, 5) for o in outcomes},
            "cheap_model": CHEAP_MODEL,
            "hard_model": HARD_MODEL,
            "assumed_answer_tokens": _ANSWER_TOKENS,
        },
        tasks=tasks_rows,
    )


def run_mock() -> BenchResult:
    """Run every variant against the free mock provider and embedder.

    The classifier script replies with the oracle label for every task in
    order, so `llm_classifier` scores 100% on mock: this proves the
    plumbing (parsing, tier validation, cost projection) runs end to end
    for free. `semantic` still runs against the real nearest-centroid logic
    with a hash embedder, so its mock accuracy is a genuine, if noisier,
    number rather than a scripted one.
    """
    script = [f"ROUTE: {label}" for _, label in TASKS]
    provider = mock_provider(script)
    embedder = mock_embedder()
    result = _run(provider, embedder)
    return finalize(result, provider)


def run_live() -> BenchResult:
    """Run every variant against live Flash-Lite and the Gemini embedder.

    Budgeted at $0.50; `llm_classifier` spends one short Flash-Lite call
    per task (24 calls), `semantic` spends one embedding call per unique
    prompt (cached across runs by `CachedEmbedder`), and `always_hard`
    spends nothing.
    """
    from benchmarks.harness import gemini_embedder

    provider = live_provider(model=CHEAP_MODEL, budget_usd=0.5)
    embedder = gemini_embedder()
    result = _run(provider, embedder)
    return finalize(result, provider)


if __name__ == "__main__":
    result = run_mock()
    print(f"routing benchmark: n={result.n} model={result.model}")
    for variant, accuracy in result.variants.items():
        cost = result.detail["projected_cost_usd"][variant]
        print(f"  {variant:<15} accuracy={accuracy:.2%}  projected_answer_cost=${cost:.5f}")
    print(f"  {'oracle':<15} accuracy=100.00%  projected_answer_cost=${result.detail['oracle_projected_cost_usd']:.5f}")
    print(f"cost: ${result.usage.get('cost_usd', 0.0):.4f} (mock, always $0)")
    print(result.headline)
