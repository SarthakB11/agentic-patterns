"""Benchmark: a decision model (TypeSafe Jev) as a judge, alone and in a cascade.

Runs `patterns/evaluation/decision_judge.py` against the live Jev API and
compares it with the LLM correctness judge from `bench_evaluation.py`
(Gemini 3.1 Flash-Lite), on two human-labeled slices:

- `factual`: the same 20 question/answer pairs `bench_evaluation.py` uses,
  10 good and 10 bad (wrong fact or non-responsive).
- `reasoning`: 12 new pairs, 6 good and 6 bad, where checking the answer
  takes a computation or a short deduction (multiplication, a classic
  trick question, primality, clock arithmetic, polygon angles, a syllogism).
  Li et al. (arXiv:2609.26550) report decision models losing ground on
  math and logic, so this slice is where a gap should show if there is one.

No reference answers are given to either judge: both read only the question
and the reply.

Reported variants (from Jev's first sample per item):

- `jev_auroc`, `jev_brier`, `jev_ece`: calibration of Jev's P(yes) over all 32.
- `jev_accuracy_cutoff_0.5`: Jev's verdict at the default cutoff.
- `jev_accuracy_tuned_crossfit`: 2-fold cross-fitted tuned cutoff (tune on
  half the question pairs, score the other half, and the reverse), so the
  tuned number is never scored on the slice it was picked from.
- `llm_accuracy`: the Gemini judge on every item.
- `cascade_accuracy`, `cascade_coverage`: `decision_judge.judge_with_decision_model`
  at tau=0.9, escalating to the Gemini judge's verdict for that item.

Jev has no temperature setting, so every item is also asked a second time
and `detail["retest"]` records how much the two samples differ: verdict
flips at 0.5 (`meta.test_retest_rate`), the largest probability drift,
items whose accept-or-escalate decision flips at tau, and the cascade's
accuracy under the second sample.

Cost. `detail["cost"]` prices what it takes to produce these numbers from a
cold start, from recorded token counts for both judges, so it is the same
on a cold run and a cached re-run: Jev input tokens at $0.042 per million
with output free (https://docs.typesafe.ai/models), Gemini tokens at the
harness list price. `cascade_usd` is the first Jev sample plus only the
escalated Gemini calls, what the cascade would really spend.
`billed_this_run_usd` is what this particular run spent (zero when fully
cached). Jev spend is capped at `JEV_BUDGET_USD`.

Run live (needs TYPESAFE_API_KEY and GEMINI_KEY in the environment or .env):

    python3 -m benchmarks.bench_decision_judge --live

Jev and Gemini responses are cached under `benchmarks/.cache/`, so a re-run
reproduces the same numbers for free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from agentic_patterns import Completion, Message, Provider
from benchmarks.bench_evaluation import JUDGE_ITEMS, JudgeItem, _build_correctness_judge
from benchmarks.harness import (
    CACHE_DIR,
    PRICING,
    BenchProvider,
    BenchResult,
    _read_key,
    finalize,
    live_provider,
    mock_provider,
)
from patterns.evaluation.decision_judge import (
    TYPESAFE_MODEL,
    DecisionVerdict,
    ScriptedDecisionModel,
    TypeSafeDecisionModel,
    judge_with_decision_model,
    tune_cutoff,
)
from patterns.evaluation.eval_set import EvalCase
from patterns.evaluation.meta import auroc, brier_score, expected_calibration_error, test_retest_rate
from patterns.evaluation.verdict import Verdict

LLM_MODEL = "gemini-3.1-flash-lite"
LLM_BUDGET_USD = 0.5
JEV_PRICE_PER_M_INPUT = 0.042
JEV_BUDGET_USD = 0.05
JEV_SAMPLES = 2
TAU = 0.9

REASONING_ITEMS: list[JudgeItem] = [
    JudgeItem("r01", "What is 17 multiplied by 23?", "391.", True),
    JudgeItem("r02", "What is 17 multiplied by 23?", "381.", False),
    JudgeItem(
        "r03",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "5 cents.",
        True,
    ),
    JudgeItem(
        "r04",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "10 cents.",
        False,
    ),
    JudgeItem("r05", "Is 221 a prime number?", "No, 221 is 13 times 17.", True),
    JudgeItem("r06", "Is 221 a prime number?", "Yes, 221 is prime.", False),
    JudgeItem(
        "r07", "A train leaves at 2:45 pm and the trip takes 1 hour 50 minutes. When does it arrive?", "4:35 pm.", True
    ),
    JudgeItem(
        "r08", "A train leaves at 2:45 pm and the trip takes 1 hour 50 minutes. When does it arrive?", "4:25 pm.", False
    ),
    JudgeItem("r09", "What is the sum of the interior angles of a hexagon?", "720 degrees.", True),
    JudgeItem("r10", "What is the sum of the interior angles of a hexagon?", "540 degrees.", False),
    JudgeItem("r11", "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?", "Yes.", True),
    JudgeItem(
        "r12",
        "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
        "No, not necessarily.",
        False,
    ),
]

SLICES: dict[str, list[JudgeItem]] = {"factual": JUDGE_ITEMS, "reasoning": REASONING_ITEMS}


class JevBudgetExceeded(RuntimeError):
    """Raised before a live Jev call that would start past the run's Jev ceiling."""


class CachedDecisionModel:
    """Disk cache over a live decision model, keyed per sample so a repeat is a real second call.

    Sample 0 uses the plain key, so earlier single-sample caches stay valid.
    `recorded_input_tokens` sums the tokens each answer cost when it was
    first fetched (cache hit or not), which prices a cold run;
    `billed_input_tokens` counts only calls made in this run.
    """

    def __init__(self, inner: TypeSafeDecisionModel, budget_usd: float = JEV_BUDGET_USD) -> None:
        self.inner = inner
        self.budget_usd = budget_usd
        self.cache_dir = CACHE_DIR / "jev"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.recorded_input_tokens = 0
        self.billed_input_tokens = 0
        self.api_calls = 0
        self.cache_hits = 0
        self.sample = 0

    def noul(self, state: str, instructions: str) -> float:
        """Return cached P(yes) for the current sample when present, else call Jev and cache it."""
        key: dict[str, Any] = {"model": self.inner.model, "state": state, "q": instructions}
        if self.sample:
            key["sample"] = self.sample
        blob = json.dumps(key, sort_keys=True)
        path = self.cache_dir / f"{hashlib.sha256(blob.encode()).hexdigest()}.json"
        if path.exists():
            record = json.loads(path.read_text())
            self.cache_hits += 1
        else:
            if self.billed_input_tokens / 1e6 * JEV_PRICE_PER_M_INPUT >= self.budget_usd:
                raise JevBudgetExceeded(f"Jev spend reached the ${self.budget_usd:.2f} ceiling for this run")
            before = self.inner.input_tokens
            p = self.inner.noul(state, instructions)
            record = {"p_yes": p, "input_tokens": self.inner.input_tokens - before}
            path.write_text(json.dumps(record))
            self.api_calls += 1
            self.billed_input_tokens += int(record["input_tokens"])
        self.recorded_input_tokens += int(record["input_tokens"])
        return float(record["p_yes"])


class TokenRecorder(Provider):
    """Forward to a provider and keep each call's reported token usage, cached or live.

    `BenchProvider` charges nothing on a cache hit, which is right for its
    budget but makes a re-run look free. A cached completion still carries
    the usage it was produced with, so recording it here prices every call
    the same way on a cold run and a warm one.
    """

    def __init__(self, inner: Provider) -> None:
        self.inner = inner
        self.last_tokens = (0, 0)

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        completion = self.inner.complete(
            messages, tools=tools, system=system, temperature=temperature, max_tokens=max_tokens
        )
        usage = (completion.raw or {}).get("usage", {}) if isinstance(completion.raw, dict) else {}
        self.last_tokens = (
            int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        )
        return completion


@dataclass
class _ItemOutcome:
    item: JudgeItem
    slice_name: str
    llm: Verdict
    llm_cost: float
    samples: list[DecisionVerdict]  # one cascade verdict per Jev sample

    @property
    def llm_passed(self) -> bool:
        # A malformed verdict (no VERDICT line) counts as a fail, as in
        # bench_evaluation.py; `detail["llm_malformed"]` says how often.
        return bool(self.llm.passed)


def _cascade_metrics(outcomes: list[_ItemOutcome], sample: int) -> tuple[float, float]:
    """Cascade accuracy and coverage under one Jev sample."""
    labels = [o.item.human_label for o in outcomes]
    verdicts = [o.samples[sample] for o in outcomes]
    accuracy = sum(v.passed is not None and v.passed == y for v, y in zip(verdicts, labels)) / len(outcomes)
    return accuracy, sum(v.accepted for v in verdicts) / len(outcomes)


def _score_slice(outcomes: list[_ItemOutcome]) -> dict[str, float]:
    labels = [o.item.human_label for o in outcomes]
    probs = [o.samples[0].p_yes for o in outcomes]
    n = len(outcomes)
    # 2-fold cross-fit: tune on one half, score the other half, both ways.
    # Items come in (good, bad) pairs on the same question, so folds split by
    # pair: each fold keeps both labels, and no question sits in both folds.
    # (Splitting by item parity would put every good answer in one fold and
    # every bad one in the other, and tune a cutoff on a single class.)
    crossfit_correct = 0
    for held in (0, 1):
        tune = [(p, y) for i, (p, y) in enumerate(zip(probs, labels)) if (i // 2) % 2 != held]
        test = [(p, y) for i, (p, y) in enumerate(zip(probs, labels)) if (i // 2) % 2 == held]
        cut, _ = tune_cutoff([p for p, _ in tune], [y for _, y in tune])
        crossfit_correct += sum((p >= cut) == y for p, y in test)
    cascade_accuracy, cascade_coverage = _cascade_metrics(outcomes, 0)
    return {
        "jev_auroc": auroc(probs, labels),
        "jev_brier": brier_score(probs, labels),
        "jev_ece": expected_calibration_error(probs, labels),
        "jev_accuracy_cutoff_0.5": sum(o.samples[0].decision_passed == y for o, y in zip(outcomes, labels)) / n,
        "jev_accuracy_tuned_crossfit": crossfit_correct / n,
        "llm_accuracy": sum(o.llm_passed == y for o, y in zip(outcomes, labels)) / n,
        "cascade_accuracy": cascade_accuracy,
        "cascade_coverage": cascade_coverage,
    }


def _retest(outcomes: list[_ItemOutcome]) -> dict[str, object]:
    first = [o.samples[0] for o in outcomes]
    second = [o.samples[1] for o in outcomes]
    cascade_accuracy_2, _ = _cascade_metrics(outcomes, 1)
    return {
        "verdict_retest_rate_at_0.5": test_retest_rate(
            [v.decision_passed for v in first], [v.decision_passed for v in second]
        ),
        "max_abs_drift": round(max(abs(a.p_yes - b.p_yes) for a, b in zip(first, second)), 4),
        "accept_flips_at_tau": [o.item.id for o, a, b in zip(outcomes, first, second) if a.accepted != b.accepted],
        "cascade_accuracy_sample_2": cascade_accuracy_2,
    }


def _run(
    decision: CachedDecisionModel | ScriptedDecisionModel, provider: BenchProvider
) -> tuple[dict[str, float], dict[str, object], list[dict[str, object]], float]:
    """Judge every item with both judges, then score each slice and the whole set.

    The Gemini judge runs on every item once (that is the LLM-only
    baseline). The cascade is the module's own `judge_with_decision_model`,
    whose fallback returns that same Gemini verdict for the item, so the
    cascade is charged only for the items it escalates.

    Returns:
        (variants, detail, tasks, escalated_llm_cost_usd).
    """
    recorder = TokenRecorder(provider)
    llm_judge = _build_correctness_judge(recorder)
    in_price, out_price = PRICING.get(provider.model, (0.0, 0.0))
    outcomes: list[_ItemOutcome] = []
    for slice_name, items in SLICES.items():
        for item in items:
            case = EvalCase(id=item.id, input=item.question)
            llm = llm_judge(case, item.answer)
            tokens_in, tokens_out = recorder.last_tokens
            llm_cost = tokens_in / 1e6 * in_price + tokens_out / 1e6 * out_price
            samples = []
            for sample in range(JEV_SAMPLES):
                if isinstance(decision, CachedDecisionModel):
                    decision.sample = sample
                samples.append(
                    judge_with_decision_model(
                        decision, case, item.answer, tau=TAU, escalate=lambda _c, _o, v=llm: v
                    )
                )
            outcomes.append(_ItemOutcome(item, slice_name, llm, llm_cost, samples))

    variants = _score_slice(outcomes)
    detail: dict[str, object] = {
        name: _score_slice([o for o in outcomes if o.slice_name == name]) for name in SLICES
    }
    detail["retest"] = _retest(outcomes)
    detail["llm_malformed"] = sum(o.llm.passed is None for o in outcomes)
    escalated = [o for o in outcomes if o.samples[0].escalated]
    detail["escalated_ids"] = [o.item.id for o in escalated]
    detail["llm_all_usd"] = sum(o.llm_cost for o in outcomes)
    tasks: list[dict[str, object]] = [
        {
            "id": o.item.id,
            "slice": o.slice_name,
            "question": o.item.question,
            "answer": o.item.answer,
            "human_label": o.item.human_label,
            "jev_p_yes": round(o.samples[0].p_yes, 4),
            "jev_p_yes_sample_2": round(o.samples[1].p_yes, 4),
            "jev_passed_at_0.5": o.samples[0].decision_passed,
            "llm_passed": o.llm_passed,
            "llm_malformed": o.llm.passed is None,
            "escalated": o.samples[0].escalated,
            "cascade_passed": o.samples[0].passed,
        }
        for o in outcomes
    ]
    return variants, detail, tasks, sum(o.llm_cost for o in escalated)


def _headline(variants: dict[str, float], detail: dict[str, object], prefix: str = "") -> str:
    factual = detail["factual"]
    reasoning = detail["reasoning"]
    assert isinstance(factual, dict) and isinstance(reasoning, dict)
    return (
        f"{prefix}Jev at cutoff 0.5 scored {factual['jev_accuracy_cutoff_0.5']:.0%} on factual and "
        f"{reasoning['jev_accuracy_cutoff_0.5']:.0%} on reasoning answers (AUROC {variants['jev_auroc']:.2f} "
        f"overall) against the LLM judge's {variants['llm_accuracy']:.0%}; the tau={TAU} cascade reached "
        f"{variants['cascade_accuracy']:.0%} "
        f"while settling {variants['cascade_coverage']:.0%} of items without the LLM."
    )


def run_mock() -> BenchResult:
    """Free plumbing check: scripted Jev probabilities and scripted Gemini verdicts.

    The scripted probabilities are decisive and right on the factual slice
    and uncertain on the reasoning slice, so every metric and the
    escalation path are exercised. The two Jev samples per item are
    identical. Not a finding.
    """
    items = [item for items in SLICES.values() for item in items]
    probs: list[float] = []
    for item in items:
        p = (0.95 if item.human_label else 0.05) if item.id.startswith("q") else (0.7 if item.human_label else 0.6)
        probs.extend([p] * JEV_SAMPLES)
    script = [
        "Step 1: checked.\nSCORE: 9\nVERDICT: pass" if item.human_label else "Step 1: wrong.\nSCORE: 2\nVERDICT: fail"
        for item in items
    ]
    provider = mock_provider(script, model=LLM_MODEL)
    variants, detail, tasks, escalated_cost = _run(ScriptedDecisionModel(probs), provider)
    detail.pop("llm_all_usd")
    detail["cost"] = {"jev_usd": 0.0, "llm_all_usd": 0.0, "cascade_usd": escalated_cost, "billed_this_run_usd": 0.0}
    result = BenchResult(
        name="bench_decision_judge",
        model=f"{TYPESAFE_MODEL} + {LLM_MODEL}",
        n=len(items),
        variants=variants,
        headline=_headline(variants, detail, prefix="[mock] "),
        detail=detail,
        tasks=tasks,
    )
    return finalize(result, provider)


def run_live() -> BenchResult:
    """Run against live Jev and live Gemini, each under a budget and a disk cache."""
    decision = CachedDecisionModel(TypeSafeDecisionModel(_read_key(("TYPESAFE_API_KEY",))))
    provider = live_provider(model=LLM_MODEL, budget_usd=LLM_BUDGET_USD)
    variants, detail, tasks, escalated_cost = _run(decision, provider)
    jev_all_usd = decision.recorded_input_tokens / 1e6 * JEV_PRICE_PER_M_INPUT
    jev_one_sample_usd = jev_all_usd / JEV_SAMPLES
    llm_all = float(detail.pop("llm_all_usd"))  # type: ignore[arg-type]
    detail["cost"] = {
        "jev_usd": round(jev_one_sample_usd, 6),
        "jev_input_tokens_per_sample": decision.recorded_input_tokens // JEV_SAMPLES,
        "llm_all_usd": round(llm_all, 6),
        "cascade_usd": round(jev_one_sample_usd + escalated_cost, 6),
        "retest_sample_usd": round(jev_all_usd - jev_one_sample_usd, 6),
        "billed_this_run_usd": round(
            decision.billed_input_tokens / 1e6 * JEV_PRICE_PER_M_INPUT + provider.usage.cost_usd(provider.model), 6
        ),
        "jev_api_calls_this_run": decision.api_calls,
        "jev_cache_hits_this_run": decision.cache_hits,
        "jev_calls_without_usage": decision.inner.calls_without_usage,
    }
    result = BenchResult(
        name="bench_decision_judge",
        model=f"{TYPESAFE_MODEL} + {LLM_MODEL}",
        n=len(tasks),
        variants=variants,
        headline=_headline(variants, detail),
        detail=detail,
        tasks=tasks,
    )
    return finalize(result, provider)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision-model judge benchmark (TypeSafe Jev).")
    parser.add_argument("--live", action="store_true", help="call live Jev and Gemini instead of the mock path")
    outcome = run_live() if parser.parse_args().live else run_mock()
    print(f"bench_decision_judge ({outcome.model}, n={outcome.n})")
    for variant_name, value in outcome.variants.items():
        print(f"  {variant_name}: {value:.3f}")
    for slice_name in SLICES:
        print(f"  [{slice_name}] {json.dumps(outcome.detail[slice_name])}")
    for key in ("retest", "llm_malformed", "escalated_ids", "cost"):
        print(f"  {key}: {json.dumps(outcome.detail.get(key))}")
    print(outcome.headline)
