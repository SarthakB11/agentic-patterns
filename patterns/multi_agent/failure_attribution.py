"""MAST taxonomy and automated failure attribution over a run trace.

Every other module in this folder either succeeds or fails a demo run and
moves on. This module is what happens after a failed run: it turns the MAST
taxonomy (Cemri et al., "Why Do Multi-Agent LLM Systems Fail?",
arXiv:2503.13657) from a citation into working code. MAST hand-annotated
1600-plus multi-agent traces across seven frameworks and grouped 14 recurring
failure modes into three categories: specification and system-design issues
(41.8%), inter-agent misalignment (36.9%), and task verification (21.3%).
`MAST_MODES` below is that table, with a stable id per mode so a caller can
look up a category without re-deriving it from free text.

Naming which agent and which step caused a failure is its own problem, not
solved by having the taxonomy. Zhang et al., "Which Agent Causes Task
Failures and When?" (arXiv:2505.00212) built the Who&When benchmark and
compared three attribution strategies: All-at-Once (one pass over the whole
log), Step-by-Step (walk each step in order until the first decisive one),
and Binary-Search (recursively halve the step range). Their finding,
reproduced here as a demo rather than asserted as a fact: All-at-Once tends
to name the right agent more often, Step-by-Step tends to pinpoint the right
step more often, and even strong models score modestly in absolute terms
(roughly 53% agent-level, 14% step-level). Chen et al., "Seeing the Whole
Elephant" (arXiv:2604.22708) add that attributing over the full trace beats
output-only attribution by up to 76%, the argument for attributing over
`state.SharedState.trace` instead of a run's final answer alone.

Every attributor verdict here is one `Provider.complete()` call, scripted
under `MockProvider` exactly like every other completion in this repo. This
module does not implement a trained judge or a real accuracy benchmark: the
"model" is scripted text, so what it demonstrates is the three strategies'
control flow (call count, narrowing, stop condition), not their real-world
accuracy. A reader wiring this to a live provider gets a genuine judge; a
reader running it offline gets a deterministic replay of one.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, MockProvider, Provider

from patterns.multi_agent.state import SharedState, TraceEntry

CATEGORY_SPECIFICATION = "specification"
CATEGORY_INTER_AGENT = "inter_agent"
CATEGORY_VERIFICATION = "verification"


@dataclass(frozen=True)
class MastMode:
    """One entry in the MAST failure-mode taxonomy.

    Attributes:
        id: Stable MAST id, e.g. "FM-2.3".
        label: Short human-readable description of the mode.
        category: One of `CATEGORY_SPECIFICATION`, `CATEGORY_INTER_AGENT`,
            `CATEGORY_VERIFICATION`.
    """

    id: str
    label: str
    category: str


MAST_MODES: dict[str, MastMode] = {
    m.id: m
    for m in [
        MastMode("FM-1.1", "Disobey task specification", CATEGORY_SPECIFICATION),
        MastMode("FM-1.2", "Disobey role specification", CATEGORY_SPECIFICATION),
        MastMode("FM-1.3", "Step repetition", CATEGORY_SPECIFICATION),
        MastMode("FM-1.4", "Loss of conversation history", CATEGORY_SPECIFICATION),
        MastMode("FM-1.5", "Unaware of termination conditions", CATEGORY_SPECIFICATION),
        MastMode("FM-2.1", "Conversation reset", CATEGORY_INTER_AGENT),
        MastMode("FM-2.2", "Fail to ask for clarification", CATEGORY_INTER_AGENT),
        MastMode("FM-2.3", "Task derailment", CATEGORY_INTER_AGENT),
        MastMode("FM-2.4", "Information withholding", CATEGORY_INTER_AGENT),
        MastMode("FM-2.5", "Ignored other agent's input", CATEGORY_INTER_AGENT),
        MastMode("FM-2.6", "Reasoning-action mismatch", CATEGORY_INTER_AGENT),
        MastMode("FM-3.1", "Premature termination", CATEGORY_VERIFICATION),
        MastMode("FM-3.2", "No or incomplete verification", CATEGORY_VERIFICATION),
        MastMode("FM-3.3", "Incorrect verification", CATEGORY_VERIFICATION),
    ]
}


@dataclass
class Attribution:
    """One attributor's verdict on which agent and step caused a failure.

    Attributes:
        agent: Name of the agent judged responsible.
        step: 1-based step number (matches `TraceEntry.seq`) judged decisive.
        mode_id: The MAST mode id the attributor named.
        category: Resolved from `MAST_MODES`, never trusted from the model
            directly, so the category label cannot drift from the table.
        strategy: One of "all_at_once", "step_by_step", "binary_search".
    """

    agent: str
    step: int
    mode_id: str
    category: str
    strategy: str


def trace_steps(source: SharedState | list[TraceEntry]) -> list[TraceEntry]:
    """Return a numbered list of trace steps from a `SharedState` or a raw list.

    `TraceEntry` already carries (seq, actor, action, detail), so this is a
    thin view rather than a new type: it lets every attributor accept either
    a live `SharedState` or a hand-built list of entries.
    """
    return source.trace if isinstance(source, SharedState) else list(source)


def _resolve_mode(mode_id: str) -> MastMode:
    """Look up a MAST mode id, raising if the model named one that does not exist."""
    mode = MAST_MODES.get(mode_id)
    if mode is None:
        raise ValueError(f"Unknown MAST mode id {mode_id!r}; not present in MAST_MODES")
    return mode


def _render_trace(goal: str, steps: list[TraceEntry]) -> str:
    lines = [f"Goal: {goal}", "Trace:"]
    lines.extend(f"{s.seq}. {s.actor} {s.action}: {s.detail}" for s in steps)
    return "\n".join(lines)


def _field(lines: list[str], key: str) -> str:
    prefix = f"{key.upper()}:"
    for line in lines:
        if line.strip().upper().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


ALL_AT_ONCE_SYSTEM = (
    "You are a failure-attribution judge. Read the goal and the full numbered trace in one "
    "pass. Name the agent most responsible for the run's failure, the step where the decisive "
    "error occurred, and the MAST failure mode id. Reply with exactly three lines: "
    "'AGENT: <name>', 'STEP: <n>', 'MODE: <FM-id>'."
)

STEP_BY_STEP_SYSTEM = (
    "You are a failure-attribution judge scanning one step at a time. Given the goal, the full "
    "trace for context, and one candidate step, decide whether the decisive error occurred at "
    "or before that step. Reply 'VERDICT: NO' if not, or 'VERDICT: YES' plus a second line "
    "'MODE: <FM-id>' if this step is where it happened."
)

BINARY_SEARCH_SYSTEM = (
    "You are a failure-attribution judge doing a binary search over the trace. Given the goal, "
    "the full trace, and a candidate step range, decide whether the decisive error falls in the "
    "first half or the second half of that range. Reply 'HALF: first' or 'HALF: second', plus a "
    "second line 'MODE: <FM-id>' with your best current guess at the failure mode."
)


def attribute_all_at_once(provider: Provider, goal: str, steps: list[TraceEntry]) -> Attribution:
    """Attribute a failure with one pass over the whole trace: one provider call.

    Args:
        provider: Provider for the attributor, scripted with one verdict.
        goal: The goal the run was working toward.
        steps: The trace to attribute over, in order.
    """
    completion = provider.complete([Message.user(_render_trace(goal, steps))], system=ALL_AT_ONCE_SYSTEM)
    lines = completion.content.splitlines()
    agent, step_text, mode_id = _field(lines, "AGENT"), _field(lines, "STEP"), _field(lines, "MODE")
    if not agent or not step_text or not mode_id:
        raise ValueError(f"malformed all-at-once verdict: {completion.content!r}")
    mode = _resolve_mode(mode_id)
    return Attribution(agent=agent, step=int(step_text), mode_id=mode.id, category=mode.category, strategy="all_at_once")


def attribute_step_by_step(provider: Provider, goal: str, steps: list[TraceEntry]) -> Attribution:
    """Attribute a failure by walking each step in order, one call per step.

    Stops at the first step the attributor flags as decisive, so the call
    count equals the decisive step's position, not the trace length.

    Raises:
        ValueError: If every step is scanned with no decisive verdict.
    """
    trace_text = _render_trace(goal, steps)
    for step in steps:
        prompt = f"{trace_text}\n\nCandidate step: {step.seq}. {step.actor} {step.action}: {step.detail}"
        completion = provider.complete([Message.user(prompt)], system=STEP_BY_STEP_SYSTEM)
        lines = completion.content.splitlines()
        verdict = _field(lines, "VERDICT").upper()
        if verdict == "NO":
            continue
        if verdict == "YES":
            mode = _resolve_mode(_field(lines, "MODE"))
            return Attribution(agent=step.actor, step=step.seq, mode_id=mode.id, category=mode.category, strategy="step_by_step")
        raise ValueError(f"malformed step-by-step verdict: {completion.content!r}")
    raise ValueError("step-by-step attribution scanned every step without a decisive verdict")


def attribute_binary_search(provider: Provider, goal: str, steps: list[TraceEntry]) -> Attribution:
    """Attribute a failure by recursively halving the step range: log(n) calls.

    Raises:
        ValueError: If the search converges without ever receiving a MODE.
    """
    trace_text = _render_trace(goal, steps)
    lo, hi = 0, len(steps) - 1
    last_mode_id: str | None = None
    while lo < hi:
        mid = (lo + hi) // 2
        window = steps[lo : hi + 1]
        prompt = f"{trace_text}\n\nCandidate range: steps {window[0].seq} to {window[-1].seq}."
        completion = provider.complete([Message.user(prompt)], system=BINARY_SEARCH_SYSTEM)
        lines = completion.content.splitlines()
        half = _field(lines, "HALF").lower()
        mode_id = _field(lines, "MODE")
        if mode_id:
            last_mode_id = mode_id
        if half == "first":
            hi = mid
        elif half == "second":
            lo = mid + 1
        else:
            raise ValueError(f"malformed binary-search verdict: {completion.content!r}")
    if last_mode_id is None:
        raise ValueError("binary-search attribution converged without ever receiving a MODE")
    mode = _resolve_mode(last_mode_id)
    decisive = steps[lo]
    return Attribution(agent=decisive.actor, step=decisive.seq, mode_id=mode.id, category=mode.category, strategy="binary_search")


# --- demo ----------------------------------------------------------------


def run_failure_attribution_demo() -> dict[str, Attribution]:
    """Attribute one deliberately broken run with all three strategies.

    Builds a six-step trace for the note-taking-apps competitive brief where
    `market_researcher` twice returns an off-topic finding (attorney billing
    rates, not note-taking app pricing: an FM-2.3 "task derailment" mode) and
    the supervisor's synthesis folds it in without flagging it. All-at-Once
    and Step-by-Step are scripted to land on the first occurrence (step 2);
    Binary-Search is scripted to converge on the second occurrence (step 4).
    All three name the same agent, matching Zhang et al.'s agent-level
    finding; they differ on the exact step, matching the step-level
    trade-off the paper reports between the strategies.
    """
    goal = "Produce a one-page competitive brief on note-taking apps for the product team."
    state = SharedState(goal=goal)
    state.record("supervisor", "decompose", "3 subtasks proposed: market, tech, risk")
    state.record(
        "market_researcher",
        "produce",
        "Law firms typically bill $300-600/hour for partner time and $150-250/hour for associates.",
    )
    state.record(
        "tech_researcher",
        "produce",
        "Obsidian is offline-first with local markdown files; Notion and Evernote require network sync.",
    )
    state.record(
        "market_researcher",
        "revise",
        "Billing structures vary by practice area, with contingency fees common in litigation.",
    )
    state.record("risk_analyst", "produce", "Obsidian's no-lock-in model is the single biggest risk to our retention plan.")
    state.record("supervisor", "synthesize", "Combined findings into a final report, including the market_researcher's billing-rate notes.")

    steps = trace_steps(state)
    all_at_once = attribute_all_at_once(
        MockProvider(script=["AGENT: market_researcher\nSTEP: 2\nMODE: FM-2.3"]), goal, steps
    )
    step_by_step = attribute_step_by_step(
        MockProvider(script=["VERDICT: NO", "VERDICT: YES\nMODE: FM-2.3"]), goal, steps
    )
    binary_search = attribute_binary_search(
        MockProvider(
            script=[
                "HALF: second\nMODE: FM-2.3",
                "HALF: first\nMODE: FM-2.3",
                "HALF: first\nMODE: FM-2.3",
            ]
        ),
        goal,
        steps,
    )
    return {"all_at_once": all_at_once, "step_by_step": step_by_step, "binary_search": binary_search}
