"""Shared benchmark harness: a caching, budgeted wrapper over a real provider.

The design goals are honesty and low cost. Honesty: every reported number
comes from the actual pattern code running against a real model, and the raw
per-task outcome is written to `results/` so it can be checked. Low cost: a
disk cache means a completed call never repeats, a hard budget ceiling aborts
a run before it overspends, and a free mock path lets a benchmark's plumbing
be verified before a single paid call.

Nothing here needs a code change in `agentic_patterns` or `patterns`. A
`BenchProvider` wraps any `Provider` and forwards `complete()` unchanged, so
the pattern code cannot tell it is being measured.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from agentic_patterns import Completion, Message, MockProvider, Provider, ToolCall
from agentic_patterns.core.embeddings import Embedder, HashEmbedder, OpenAIEmbedder
from agentic_patterns.core.providers import AnthropicProvider, OpenAICompatibleProvider

GEMINI_EMBED_MODEL = "gemini-embedding-001"

BENCH_DIR = pathlib.Path(__file__).resolve().parent
CACHE_DIR = BENCH_DIR / ".cache"
RESULTS_DIR = BENCH_DIR / "results"

# List prices in USD per one million tokens (input, output), as of July 2026.
# Preview model prices can change; the run stamps the model and date so a
# number can always be re-derived from the recorded token counts.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3.5-flash": (1.50, 9.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class BudgetExceeded(RuntimeError):
    """Raised when a call would push a run past its USD ceiling."""


def _read_key(names: tuple[str, ...], env_path: pathlib.Path | None = None) -> str:
    """Read the first of `names` found in the environment or the gitignored .env.

    Keys are never written to a tracked file. They are read at call time and
    passed straight to the provider.
    """
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    path = env_path or (BENCH_DIR.parent / ".env")
    if path.exists():
        for line in path.read_text().splitlines():
            for name in names:
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError(f"None of {names} found in environment or .env. Benchmarks need a key to run live.")


def load_key(env_path: pathlib.Path | None = None) -> str:
    """Read the Gemini key from the environment or .env."""
    return _read_key(("GEMINI_KEY", "GEMINI_API_KEY"), env_path)


def load_anthropic_key(env_path: pathlib.Path | None = None) -> str:
    """Read the Anthropic key from the environment or .env."""
    return _read_key(("ANTHROPIC_KEY", "ANTHROPIC_API_KEY"), env_path)


@dataclass
class Usage:
    """Token and cost tally for a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    cache_hits: int = 0

    def cost_usd(self, model: str) -> float:
        """Cost of the tallied tokens at `model`'s list price."""
        in_price, out_price = PRICING.get(model, (0.0, 0.0))
        return self.input_tokens / 1e6 * in_price + self.output_tokens / 1e6 * out_price


def _cache_key(model: str, messages: list[Message], tools: Any, system: str | None, params: dict[str, Any]) -> str:
    """Stable hash over everything that determines a completion."""
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content, "tool_call_id": m.tool_call_id,
                      "tool_calls": [asdict(tc) for tc in m.tool_calls]} for m in messages],
        "tools": tools,
        "system": system,
        "params": params,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _completion_to_dict(c: Completion, usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": c.content,
        "reasoning": c.reasoning,
        "stop_reason": c.stop_reason,
        "tool_calls": [asdict(tc) for tc in c.tool_calls],
        "usage": usage,
    }


def _completion_from_dict(d: dict[str, Any]) -> Completion:
    return Completion(
        content=d["content"],
        reasoning=d.get("reasoning", ""),
        stop_reason=d["stop_reason"],
        tool_calls=[ToolCall(**tc) for tc in d.get("tool_calls", [])],
        raw={"usage": d.get("usage", {})},
    )


class BenchProvider(Provider):
    """Wraps a `Provider`, adding a disk cache, a token tally, and a budget stop.

    A cache hit returns instantly and costs nothing. A miss calls the wrapped
    provider, records the reported token usage, checks the running total
    against the ceiling, and writes the result to the cache before returning.
    """

    def __init__(
        self,
        inner: Provider,
        model: str,
        budget_usd: float = 2.0,
        namespace: str = "live",
        use_cache: bool = True,
    ) -> None:
        self.inner = inner
        self.model = model
        self.budget_usd = budget_usd
        self.usage = Usage()
        # The disk cache exists to avoid re-spending on live calls. A scripted
        # MockProvider is free and serves a strict in-order queue, so caching
        # it is pointless and actively harmful: a warm-cache hit skips a call
        # and desyncs the queue from the tasks. The mock path disables the
        # cache, which keeps mock runs free and exactly reproducible.
        self.use_cache = use_cache
        # Mock and live responses share a model id but must never share a cache
        # slot, or a free scripted dry run would silently replay as a "live"
        # result. The namespace keeps the two apart.
        self.cache_dir = CACHE_DIR / namespace
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        params = {"temperature": temperature, "max_tokens": max_tokens}
        cache_file = None
        if self.use_cache:
            key = _cache_key(self.model, messages, tools, system, params)
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                self.usage.cache_hits += 1
                return _completion_from_dict(json.loads(cache_file.read_text()))

        if self.usage.cost_usd(self.model) >= self.budget_usd:
            raise BudgetExceeded(
                f"Run reached the ${self.budget_usd:.2f} ceiling "
                f"(spent ${self.usage.cost_usd(self.model):.4f}). Raise the budget or trim the task set."
            )

        completion = self.inner.complete(
            messages, tools=tools, system=system, temperature=temperature, max_tokens=max_tokens
        )
        usage = (completion.raw or {}).get("usage", {}) if isinstance(completion.raw, dict) else {}
        # OpenAI-compatible endpoints report prompt_tokens/completion_tokens;
        # the Anthropic Messages API reports input_tokens/output_tokens.
        self.usage.input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.usage.output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        self.usage.api_calls += 1
        if cache_file is not None:
            cache_file.write_text(json.dumps(_completion_to_dict(completion, usage)))
        return completion


def live_provider(model: str = DEFAULT_MODEL, budget_usd: float = 2.0) -> BenchProvider:
    """A budgeted, cached provider for a benchmark's live run.

    Defaults to Gemini through its OpenAI-compatible endpoint. Set the env var
    BENCH_BACKEND=anthropic to run the same benchmark against Anthropic instead
    (model from BENCH_MODEL, defaulting to Claude Haiku 4.5), so a second model
    can be measured with no change to any benchmark module. The cache key
    includes the model, so different models never share a cached response.
    """
    backend = os.environ.get("BENCH_BACKEND", "gemini")
    if backend == "anthropic":
        anthropic_model = os.environ.get("BENCH_MODEL", DEFAULT_ANTHROPIC_MODEL)
        inner: Provider = AnthropicProvider(model=anthropic_model, api_key=load_anthropic_key())
        return BenchProvider(inner, model=anthropic_model, budget_usd=budget_usd)
    inner = OpenAICompatibleProvider(model=model, api_key=load_key(), base_url=GEMINI_BASE_URL)
    return BenchProvider(inner, model=model, budget_usd=budget_usd)


def mock_provider(script: Sequence[Any], model: str = DEFAULT_MODEL) -> BenchProvider:
    """A budgeted wrapper over a scripted `MockProvider` for free dry runs.

    Mock completions carry no usage, so a dry run reports zero cost while
    still exercising every line of the metric and result-writing path. The
    ceiling is set to infinity because the mock path cannot spend.
    """
    return BenchProvider(MockProvider(script), model=model, budget_usd=float("inf"), namespace="mock", use_cache=False)


class CachedEmbedder(Embedder):
    """An `Embedder` that caches each text's vector on disk, keyed by model and text.

    A corpus is embedded once and reused across every retrieval variant and
    every re-run, so the dense-retrieval benchmark stays real (a trained
    embedding model, not the teaching hash stand-in) while costing almost
    nothing after the first pass.
    """

    def __init__(self, inner: Embedder, model: str) -> None:
        self.inner = inner
        self.model = model
        self.calls = 0
        self._dir = CACHE_DIR / "embed"
        self._dir.mkdir(parents=True, exist_ok=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            key = hashlib.sha256(f"{self.model}\x00{text}".encode()).hexdigest()
            f = self._dir / f"{key}.json"
            if f.exists():
                out[i] = json.loads(f.read_text())
            else:
                misses.append((i, text))
        if misses:
            self.calls += 1
            vectors = self.inner.embed([t for _, t in misses])
            for (i, text), vec in zip(misses, vectors):
                out[i] = vec
                key = hashlib.sha256(f"{self.model}\x00{text}".encode()).hexdigest()
                (self._dir / f"{key}.json").write_text(json.dumps(vec))
        return [v for v in out if v is not None]


def gemini_embedder(model: str = GEMINI_EMBED_MODEL) -> CachedEmbedder:
    """A disk-cached embedder backed by Gemini's OpenAI-compatible embeddings endpoint."""
    inner = OpenAIEmbedder(model=model, api_key=load_key(), base_url=GEMINI_BASE_URL)
    return CachedEmbedder(inner, model=model)


def mock_embedder(dim: int = 256) -> CachedEmbedder:
    """A disk-cached wrapper over the stdlib hash embedder, for free dry runs."""
    return CachedEmbedder(HashEmbedder(dim=dim), model=f"hash-{dim}")


@dataclass
class BenchResult:
    """One benchmark's outcome, ready to serialize and tabulate.

    Attributes:
        name: Benchmark name, used as the results filename.
        model: Model the numbers were produced with.
        n: Number of tasks in the set.
        variants: Metric value per variant, e.g. {"naive": 0.55, "hybrid": 0.80}.
        headline: One sentence stating the finding, for the README and resume.
        detail: Any extra per-variant numbers (tokens, cost, secondary metrics).
        tasks: Per-task outcomes, the provenance record.
        usage: Token and cost tally for the whole benchmark.
    """

    name: str
    model: str
    n: int
    variants: dict[str, float]
    headline: str
    detail: dict[str, Any] = field(default_factory=dict)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)

    def save(self, subdir: str = "") -> pathlib.Path:
        """Write the result to results/[<subdir>/]<name>.json and return the path.

        A live second-model run passes a subdir (for example "haiku") so its
        results land in their own folder instead of overwriting the first
        model's. A mock dry run passes "mock" so it never overwrites live
        results at all.
        """
        out_dir = RESULTS_DIR / subdir if subdir else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.name}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path


def finalize(result: BenchResult, provider: BenchProvider) -> BenchResult:
    """Attach the provider's usage tally to a result and save it.

    A mock dry run (a provider with caching off) saves to results/mock/ so a
    free plumbing check can never overwrite a committed live result. A live
    run saves to results/ or, for a second model, results/<BENCH_RESULTS_SUBDIR>/.
    """
    result.usage = {
        "model": provider.model,
        "input_tokens": provider.usage.input_tokens,
        "output_tokens": provider.usage.output_tokens,
        "api_calls": provider.usage.api_calls,
        "cache_hits": provider.usage.cache_hits,
        "cost_usd": round(provider.usage.cost_usd(provider.model), 4),
        "stamped_at": time.strftime("%Y-%m-%d", time.gmtime()) if not os.environ.get("BENCH_NO_DATE") else "",
    }
    subdir = "mock" if not provider.use_cache else os.environ.get("BENCH_RESULTS_SUBDIR", "")
    result.save(subdir)
    return result
