"""Context-isolation economics: the token cost that decides topology.

Anthropic's multi-agent research post reports single agents use about 4x a
chat turn's tokens and multi-agent systems about 15x, and that token usage
alone explains 80% of BrowseComp's performance variance. Cognition's "Don't
Build Multi-Agents" argues the other side: parallel workers on partial
context make conflicting implicit decisions, so a single-threaded agent is
the default and a fan-out has to earn its keep. `state.py` implements the
single-writer half of that argument but never measures the cost curve that
decides which side wins for a given task. This module measures it: it runs
one goal two ways, a single-threaded agent and the `supervisor.py` fan-out,
and reports the token multiple and the context-isolation benefit (the
largest single input any one agent saw) for each path.

Token counts are an approximation (character count over 4) applied to every
message, tool schema, and completion a call sent or received; a real build
swaps in the provider's own tokenizer. Core-compatible workaround:
`MockProvider.calls` snapshots only what was sent, not the `Completion`
returned, so output tokens cannot be read back after the fact.
`TrackedProvider` wraps any `Provider` and records both sides itself,
rather than changing core.

The demo's multiple comes out modest (a bit over 1x, not 15x): each worker
here answers in one call with no tool-use loop of its own, while a real
system pays for many exploratory tool-calling turns per worker plus
repeated large tool schemas. This demonstrates the multiple is measurable
and real, not that this demo reproduces Anthropic's reported ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Completion, Message, Provider, get_provider, scripted_tool_call
from patterns.multi_agent import aggregation, supervisor
from patterns.multi_agent.worker import Subtask, Worker, dispatch_parallel

SINGLE_THREADED_SYSTEM = (
    "You are a single agent researching a goal end to end, one section at a time, building on "
    "everything written so far, then writing the final report."
)


def _approx_tokens(text: str) -> int:
    """Approximate token count as character count over 4."""
    return len(text) // 4


def _render_input(messages: list[Message], tools: list[dict] | None, system: str | None) -> str:
    """Render the text a provider call actually sent, for token approximation."""
    parts = [system or "", str(tools) if tools else ""]
    for m in messages:
        parts.append(m.content)
        parts.extend(str(tc.arguments) for tc in m.tool_calls)
    return "\n".join(parts)


@dataclass
class CallTally:
    """Approximate input and output token count for one provider call."""

    input_tokens: int
    output_tokens: int


class TrackedProvider(Provider):
    """Wraps a `Provider`, recording an approximate token tally per call.

    See the module docstring: this exists because `MockProvider.calls` has
    no output side to read after the fact.
    """

    def __init__(self, inner: Provider) -> None:
        self._inner = inner
        self.calls: list[CallTally] = []

    def complete(
        self, messages: list[Message], *, tools: list[dict] | None = None, system: str | None = None,
        temperature: float = 0.0, max_tokens: int = 1024,
    ) -> Completion:
        completion = self._inner.complete(
            messages, tools=tools, system=system, temperature=temperature, max_tokens=max_tokens
        )
        input_tokens = _approx_tokens(_render_input(messages, tools, system))
        output_tokens = _approx_tokens(completion.content)
        self.calls.append(CallTally(input_tokens, output_tokens))
        return completion


@dataclass
class Accounting:
    """Tallied tokens for one tracked provider's calls."""

    input_tokens: int
    output_tokens: int
    call_count: int
    peak_input_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def account(tracked: TrackedProvider) -> Accounting:
    """Sum a `TrackedProvider`'s recorded calls into one `Accounting`."""
    input_tokens = sum(c.input_tokens for c in tracked.calls)
    output_tokens = sum(c.output_tokens for c in tracked.calls)
    peak = max((c.input_tokens for c in tracked.calls), default=0)
    return Accounting(input_tokens, output_tokens, len(tracked.calls), peak)


@dataclass
class EconomicsReport:
    """The measured cost and isolation comparison between the two paths.

    `supervised_tokens` is `supervisor_tokens` plus `sum(worker_tokens.values())`;
    `multiple` is `supervised_tokens / single_threaded_tokens`. The `_peak_context`
    fields are each path's largest single input, in tokens: for the single-threaded
    path that is its last (biggest) call, since its context grows every call; for
    the supervised path it is the max over the supervisor and every worker.
    """

    single_threaded_tokens: int
    supervisor_tokens: int
    worker_tokens: dict[str, int]
    supervised_tokens: int
    multiple: float
    single_threaded_peak_context: int
    supervised_peak_context: int
    worker_peak_contexts: dict[str, int]
    worker_count: int
    single_threaded_call_count: int
    supervised_call_count: int


def run_single_threaded(task: str, provider: Provider) -> tuple[str, TrackedProvider]:
    """Run one agent through the whole task in a linear loop, tracking its tokens.

    Each call's prompt includes everything written so far, so context grows
    call over call, unlike a fan-out where each worker only sees its own
    subtask. This is the baseline Cognition's "Don't Build Multi-Agents"
    argues for.

    Args:
        task: The goal to research end to end.
        provider: Scripted with one completion per research section plus one final synthesis.
    """
    tracked = TrackedProvider(provider)
    transcript = f"Goal: {task}"
    for section in ("market positioning", "technical differentiation", "competitive risk"):
        prompt = f"{transcript}\n\nNow research: {section}."
        completion = tracked.complete([Message.user(prompt)], system=SINGLE_THREADED_SYSTEM)
        transcript += f"\n\n[{section}]\n{completion.content}"
    completion = tracked.complete(
        [Message.user(f"{transcript}\n\nNow write the final one-page report synthesizing everything above.")],
        system=SINGLE_THREADED_SYSTEM,
    )
    return completion.content, tracked


_SUBTASK_SPECS = [
    ("market", "market_researcher", "Summarize the top 3 competitors' pricing and target users", "3 bullet points"),
    ("tech", "tech_researcher", "Summarize technical differentiators: offline support, sync, API", "3 bullet points"),
    ("risk", "risk_analyst", "Flag the single biggest competitive risk to our roadmap", "one sentence"),
]
_WORKER_SCRIPTS = {
    "market": "Notion is $10/user/month, Obsidian is a $50 one-time purchase, Evernote is $15/month.",
    "tech": "Obsidian stores plain markdown files offline; Notion and Evernote both require network sync.",
    "risk": "Obsidian's local-first, no-lock-in model removes the switching cost our retention plan depends on.",
}


def run_supervised(task: str) -> tuple[str, TrackedProvider, dict[str, TrackedProvider]]:
    """Run the `supervisor.py` fan-out on the same task, tracking every agent's tokens.

    Reuses `supervisor.decompose`, `worker.dispatch_parallel`, and
    `aggregation.model_synthesize` so this measures the real mechanism.
    """
    subtask_args = [
        {"id": sid, "role": role, "objective": obj, "output_format": fmt} for sid, role, obj, fmt in _SUBTASK_SPECS
    ]
    synthesis = (
        "Notion ($10/mo) and Evernote ($15/mo) both require network sync and lock content into "
        "proprietary formats, while Obsidian's one-time $50 price and local markdown storage "
        "remove our biggest lock-in argument, which is also our top competitive risk."
    )
    supervisor_tracked = TrackedProvider(
        get_provider(script=[scripted_tool_call("delegate_subtasks", {"subtasks": subtask_args}), synthesis])
    )
    subtasks = supervisor.decompose(supervisor_tracked, task)

    worker_trackeds: dict[str, TrackedProvider] = {}
    assignments: list[tuple[Worker, Subtask]] = []
    for subtask in subtasks:
        tracked = TrackedProvider(get_provider(script=[_WORKER_SCRIPTS[subtask.id]]))
        worker_trackeds[subtask.id] = tracked
        assignments.append((Worker(subtask.role, f"You are a {subtask.role}.", tracked), subtask))

    results = dispatch_parallel(assignments)
    final_report = aggregation.model_synthesize(
        supervisor_tracked, results, goal=task, system=supervisor.SYNTHESIS_SYSTEM
    )
    return final_report, supervisor_tracked, worker_trackeds


def compute_report(
    single: TrackedProvider, sup: TrackedProvider, workers: dict[str, TrackedProvider]
) -> EconomicsReport:
    """Tally both paths' `TrackedProvider`s into one `EconomicsReport`."""
    single_acc, sup_acc = account(single), account(sup)
    worker_accs = {sid: account(t) for sid, t in workers.items()}
    worker_tokens = {sid: acc.total_tokens for sid, acc in worker_accs.items()}
    worker_peaks = {sid: acc.peak_input_tokens for sid, acc in worker_accs.items()}
    supervised_tokens = sup_acc.total_tokens + sum(worker_tokens.values())
    return EconomicsReport(
        single_threaded_tokens=single_acc.total_tokens,
        supervisor_tokens=sup_acc.total_tokens,
        worker_tokens=worker_tokens,
        supervised_tokens=supervised_tokens,
        multiple=supervised_tokens / single_acc.total_tokens if single_acc.total_tokens else float("inf"),
        single_threaded_peak_context=single_acc.peak_input_tokens,
        supervised_peak_context=max([sup_acc.peak_input_tokens, *worker_peaks.values()], default=0),
        worker_peak_contexts=worker_peaks,
        worker_count=len(workers),
        single_threaded_call_count=single_acc.call_count,
        supervised_call_count=sup_acc.call_count + sum(acc.call_count for acc in worker_accs.values()),
    )


# --- demo ------------------------------------------------------------------


def run_economics_demo() -> EconomicsReport:
    """Run the note-taking-apps brief both ways and report the cost comparison."""
    single_provider = get_provider(
        script=[
            "Notion is $10/mo, Obsidian is a one-time $50, Evernote is $15/mo.",
            "Obsidian stores markdown offline; the others need network sync.",
            "Obsidian's no-lock-in model is the biggest roadmap risk.",
            "Ship a local-export path before the next renewal cycle.",
        ]
    )
    _, single_tracked = run_single_threaded(supervisor.GOAL, single_provider)
    _, supervisor_tracked, worker_trackeds = run_supervised(supervisor.GOAL)
    return compute_report(single_tracked, supervisor_tracked, worker_trackeds)
