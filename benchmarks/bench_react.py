"""Benchmark: multi-hop QA, direct answer vs ReAct vs ReAct+verify.

A tiny invented knowledge base holds two-hop facts (e.g. a fictional river's
country, then that country's capital) that cannot exist in any model's
training data. `direct` answers from parametric knowledge alone with no
tools, so it is expected to fail nearly every task: this is the point, since
it makes the tool-using variants' advantage real rather than the model
reciting memorized geography. `react` chains `search`/`lookup` calls through
`patterns.react.text_loop.run_react`. `react_verify` runs the same loop
gated by `patterns.react.verify.run_react_with_verification`, which rejects
an unsupported Finish and forces another lookup before accepting an answer.

Ground truth is embedded below as (question, gold_answer, hop keys) tuples.
All three variants share one `_run_task` dispatcher; `run_mock` and
`run_live` differ only in which `Provider` they feed it.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any

from agentic_patterns import Message, Provider, Tool, ToolRegistry
from benchmarks.harness import BenchProvider, BenchResult, finalize, live_provider, mock_provider
from patterns.react.text_loop import FEW_SHOT_SYSTEM_PROMPT, run_react
from patterns.react.verify import run_react_with_verification

# ---------------------------------------------------------------------------
# Invented knowledge base. None of these entities exist in the real world, so
# a model answering with no tools cannot know them; it can only guess.
# ---------------------------------------------------------------------------

_FACTS: dict[str, str] = {
    "kelvorin river": "The Kelvorin River flows through the country of Ostrenia.",
    "ostrenia capital": "The capital of Ostrenia is Marveth.",
    "duskfall mountains": "The Duskfall Mountains are located in the country of Palvaria.",
    "palvaria currency": "The currency of Palvaria is the drenn.",
    "azurine lake": "Lake Azurine sits within the country of Thallowmere.",
    "thallowmere language": "The official language of Thallowmere is Vessic.",
    "cindrapass": "Cindrapass is a mountain pass in the country of Grendale.",
    "grendale capital": "The capital of Grendale is Northarrow.",
    "orven delta": "The Orven Delta empties into the sea within the country of Kestrilan.",
    "kestrilan founder": "Kestrilan was founded by the explorer Adair Voss.",
    "sablewood forest": "Sablewood Forest lies in the country of Marrowfen.",
    "marrowfen capital": "The capital of Marrowfen is Duskhollow.",
    "thornvale pass": "Thornvale Pass crosses into the country of Ebrenna.",
    "ebrenna anthem": "The national anthem of Ebrenna is titled 'Song of the Wide Field'.",
    "glasmere strait": "The Glasmere Strait borders the country of Voskaria.",
    "voskaria capital": "The capital of Voskaria is Greyholt.",
    "ironreach canyon": "Ironreach Canyon is located in the country of Tammerlin.",
    "tammerlin export": "Tammerlin's chief export is silverglass.",
    "windmere plateau": "The Windmere Plateau lies within the country of Corvassa.",
    "corvassa capital": "The capital of Corvassa is Fenhall.",
}

# id, question, gold answer, (search_key, lookup_key) hop keys the ideal
# trajectory visits, in order.
_TASKS: list[tuple[str, str, str, tuple[str, str]]] = [
    ("t01", "What is the capital of the country the Kelvorin River flows through?",
     "Marveth", ("kelvorin river", "ostrenia capital")),
    ("t02", "What is the currency of the country the Duskfall Mountains are located in?",
     "drenn", ("duskfall mountains", "palvaria currency")),
    ("t03", "What language is spoken in the country that contains Lake Azurine?",
     "Vessic", ("azurine lake", "thallowmere language")),
    ("t04", "What is the capital of the country Cindrapass is in?",
     "Northarrow", ("cindrapass", "grendale capital")),
    ("t05", "Who founded the country the Orven Delta empties into?",
     "Adair Voss", ("orven delta", "kestrilan founder")),
    ("t06", "What is the capital of the country Sablewood Forest lies in?",
     "Duskhollow", ("sablewood forest", "marrowfen capital")),
    ("t07", "What is the national anthem of the country Thornvale Pass crosses into?",
     "Song of the Wide Field", ("thornvale pass", "ebrenna anthem")),
    ("t08", "What is the capital of the country the Glasmere Strait borders?",
     "Greyholt", ("glasmere strait", "voskaria capital")),
    ("t09", "What is the chief export of the country Ironreach Canyon is in?",
     "silverglass", ("ironreach canyon", "tammerlin export")),
    ("t10", "What is the capital of the country the Windmere Plateau lies within?",
     "Fenhall", ("windmere plateau", "corvassa capital")),
    ("t11", "What is the capital of the country the Kelvorin River flows through, precisely?",
     "Marveth", ("kelvorin river", "ostrenia capital")),
    ("t12", "Which country's capital is Northarrow, given it contains Cindrapass?",
     "Northarrow", ("cindrapass", "grendale capital")),
    ("t13", "What currency is used where the Duskfall Mountains stand?",
     "drenn", ("duskfall mountains", "palvaria currency")),
    ("t14", "What tongue is spoken around Lake Azurine's country?",
     "Vessic", ("azurine lake", "thallowmere language")),
    ("t15", "Name the capital of the nation holding Sablewood Forest.",
     "Duskhollow", ("sablewood forest", "marrowfen capital")),
]

_VARIANTS = ("direct", "react", "react_verify")


def search(query: str) -> str:
    """Search the invented knowledge base for a topic.

    Args:
        query: Free-text search query.

    Returns:
        The matching fact, or a not-found message.
    """
    key = query.strip().lower()
    if key in _FACTS:
        return _FACTS[key]
    for fact_key, fact_value in _FACTS.items():
        if key and (key in fact_key or fact_key in key):
            return fact_value
    return f"No results found for '{query}'."


def lookup(term: str) -> str:
    """Look up one entity-and-attribute term, e.g. 'ostrenia capital'.

    Args:
        term: Combined entity and attribute text, mirroring an in-page
            Ctrl+F search over the knowledge base's fact keys.

    Returns:
        The matching fact, or a not-found message.
    """
    return search(term)


def build_kb_registry() -> ToolRegistry:
    """Build the two-tool registry (`search`, `lookup`) over the invented KB."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="search",
            description="Search the knowledge base for a topic and return a short fact.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Topic to search for."}},
                "required": ["query"],
            },
            fn=search,
        )
    )
    registry.register(
        Tool(
            name="lookup",
            description="Look up one entity plus attribute, e.g. 'ostrenia capital'.",
            parameters={
                "type": "object",
                "properties": {"term": {"type": "string", "description": "Entity and attribute to look up."}},
                "required": ["term"],
            },
            fn=lookup,
        )
    )
    return registry


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for a lenient answer comparison."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _is_correct(answer: str | None, gold: str) -> bool:
    """Normalized substring match: gold must appear in the answer, or vice versa."""
    if not answer:
        return False
    norm_answer, norm_gold = _normalize(answer), _normalize(gold)
    return bool(norm_gold) and (norm_gold in norm_answer or norm_answer in norm_gold)


@dataclass
class _Outcome:
    """One variant's result on one task."""

    task_id: str
    variant: str
    answer: str | None
    correct: bool
    model_calls: int


def _run_direct(provider: Provider, task_id: str, question: str, gold: str) -> _Outcome:
    """Answer with a single completion and no tools, straight from the model's own knowledge."""
    completion = provider.complete(
        [Message.user(question)],
        system="Answer the question directly and concisely. Do not explain your reasoning.",
    )
    return _Outcome(task_id, "direct", completion.content, _is_correct(completion.content, gold), 1)


def _run_react_variant(provider: Provider, task_id: str, question: str, gold: str) -> _Outcome:
    """Run the text-parsing ReAct loop (no verification gate)."""
    result = run_react(provider, build_kb_registry(), question, system_prompt=FEW_SHOT_SYSTEM_PROMPT)
    return _Outcome(task_id, "react", result.answer, _is_correct(result.answer, gold), result.steps_taken)


def _run_react_verify_variant(provider: Provider, task_id: str, question: str, gold: str) -> _Outcome:
    """Run the same loop gated by verify-before-finish.

    Uses a system prompt that adds one sentence about the verify gate to
    `FEW_SHOT_SYSTEM_PROMPT`. This is accurate (the loop really does reject
    an unsupported Finish) and it also keeps this variant's first request
    distinct from plain `react`'s, so the two never collide in the shared
    on-disk response cache when a task's initial trajectory would otherwise
    be identical.
    """
    system_prompt = (
        f"{FEW_SHOT_SYSTEM_PROMPT}\n\nNote: a verifier checks every Finish against the trajectory "
        "before accepting it, so only finish once your observations actually support the answer."
    )
    result = run_react_with_verification(provider, build_kb_registry(), question, system_prompt=system_prompt)
    calls = result.steps_taken + result.verify_calls
    return _Outcome(task_id, "react_verify", result.answer, _is_correct(result.answer, gold), calls)


def _run_task(provider: Provider, variant: str, task_id: str, question: str, gold: str) -> _Outcome:
    """Dispatch one (variant, task) pair to the right runner. Shared by mock and live."""
    if variant == "direct":
        return _run_direct(provider, task_id, question, gold)
    if variant == "react":
        return _run_react_variant(provider, task_id, question, gold)
    return _run_react_verify_variant(provider, task_id, question, gold)


def _build_result(outcomes: list[_Outcome], model: str) -> BenchResult:
    """Aggregate per-task outcomes into a `BenchResult`, before `finalize` attaches usage."""
    variant_success: dict[str, float] = {}
    variant_calls: dict[str, float] = {}
    for variant in _VARIANTS:
        rows = [o for o in outcomes if o.variant == variant]
        variant_success[variant] = sum(o.correct for o in rows) / len(rows)
        variant_calls[variant] = statistics.mean(o.model_calls for o in rows)

    best_tool_variant = max(("react", "react_verify"), key=lambda v: variant_success[v])
    headline = (
        f"On {len(_TASKS)} invented two-hop questions, direct answering succeeded on "
        f"{variant_success['direct']:.0%} of tasks versus react {variant_success['react']:.0%} and "
        f"react_verify {variant_success['react_verify']:.0%}, with {best_tool_variant} best; "
        f"mean model calls per task were direct {variant_calls['direct']:.1f}, "
        f"react {variant_calls['react']:.1f}, react_verify {variant_calls['react_verify']:.1f}."
    )
    tasks = [
        {"id": o.task_id, "variant": o.variant, "answer": o.answer, "correct": o.correct, "model_calls": o.model_calls}
        for o in outcomes
    ]
    return BenchResult(
        name="bench_react",
        model=model,
        n=len(_TASKS),
        variants=variant_success,
        headline=headline,
        detail={"mean_model_calls": variant_calls},
        tasks=tasks,
    )


def _mock_scripts_for(task_id: str, gold: str, hops: tuple[str, str]) -> dict[str, list[Any]]:
    """Build the scripted turns for one task, per variant, so every path finishes cleanly.

    `direct` is scripted to answer with a plausible but wrong guess (proving
    the invented facts are unguessable). `react` chains two tool calls then
    finishes correctly. `react_verify` rejects a premature Finish once, then
    accepts the corrected one, exercising the verify gate for real.
    """
    search_key, lookup_key = hops
    return {
        "direct": [f"I believe the answer is {task_id}-guess (no knowledge of this exists)."],
        "react": [
            f"Thought: I need to look up {search_key}.\nAction: search[{search_key}]",
            f"Thought: Now I need {lookup_key}.\nAction: lookup[{lookup_key}]",
            f"Thought: I have enough information.\nAction: Finish[{gold}]",
        ],
        "react_verify": [
            f"Thought: I recall {search_key} roughly.\nAction: Finish[wrong-guess-{task_id}]",
            "REJECT: the trajectory has no observations supporting this answer",
            f"Thought: I need to look up {search_key}.\nAction: search[{search_key}]",
            f"Thought: Now I need {lookup_key}.\nAction: lookup[{lookup_key}]",
            f"Thought: This is supported now.\nAction: Finish[{gold}]",
            "ACCEPT",
        ],
    }


def run_mock() -> BenchResult:
    """Run every variant against `mock_provider` with scripted turns. Free, deterministic.

    Each (variant, task) pair gets its own freshly scripted `MockProvider`,
    since every task needs a different scripted trajectory. The last one
    built is what `finalize` tallies (mock usage is always zero-cost).
    """
    outcomes: list[_Outcome] = []
    provider: BenchProvider | None = None
    for variant in _VARIANTS:
        for task_id, question, gold, hops in _TASKS:
            script = _mock_scripts_for(task_id, gold, hops)[variant]
            provider = mock_provider(script)
            outcomes.append(_run_task(provider, variant, task_id, question, gold))
    assert provider is not None, "_TASKS must be non-empty"
    result = _build_result(outcomes, model=provider.model)
    return finalize(result, provider)


def run_live() -> BenchResult:
    """Run every variant against `live_provider`, budgeted at $0.50 total.

    All tasks and all variants share one `BenchProvider` so the disk cache
    and the budget ceiling apply across the whole run, not per task.
    """
    provider = live_provider(model="gemini-3.1-flash-lite", budget_usd=0.5)
    outcomes: list[_Outcome] = []
    for variant in _VARIANTS:
        for task_id, question, gold, _hops in _TASKS:
            outcomes.append(_run_task(provider, variant, task_id, question, gold))
    result = _build_result(outcomes, model=provider.model)
    return finalize(result, provider)


if __name__ == "__main__":
    result = run_mock()
    print("bench_react (mock smoke test)")
    print(f"  n={result.n} model={result.model}")
    for variant, score in result.variants.items():
        calls = result.detail["mean_model_calls"][variant]
        print(f"  {variant:<12} success={score:.0%}  mean_model_calls={calls:.1f}")
    print(f"  cost=${result.usage.get('cost_usd', 0.0):.4f}")
