"""Summarize-and-continue context management for a running ReAct scratchpad.

`scratchpad.py`'s `max_observation_chars` truncation is the naive version of
context management: it caps one observation but never reclaims transcript
length once many steps accumulate. This module is the upgrade Anthropic's
"Effective context engineering for AI agents" and "Effective harnesses for
long-running agents" (September and November 2025) describe as current
practice: once the running transcript crosses a size threshold, fold the
oldest steps into a model-written summary note and keep only the most recent
steps verbatim, so the prompt stops growing without losing everything earlier.

`CompactingScratchpad` wraps a plain `scratchpad.Scratchpad` rather than
subclassing or modifying it, so `Scratchpad`'s own behavior stays exactly as
it was. The driving loop below duplicates the small shape of
`text_loop.run_react`'s loop body, since that loop builds its own plain
`Scratchpad` internally with no seam to swap in a compacting one without
changing `run_react`'s behavior; parsing and tool-argument handling are
reused from `text_loop` and `parser`, not reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry, get_provider
from patterns.react.parser import ActionParseError, parse_action
from patterns.react.scratchpad import Scratchpad, Step
from patterns.react.text_loop import DEFAULT_FORCE_MESSAGE, FEW_SHOT_SYSTEM_PROMPT, _build_arguments, _prompt
from patterns.react.world import build_registry

COMPACTION_SYSTEM_PROMPT = (
    "You are compacting an agent's trajectory. Summarize the given steps into "
    "one short notes block that preserves the facts a later step would need."
)


@dataclass
class CompactionEvent:
    """Record of one fold, for assertions and observability.

    Attributes:
        folded_count: Number of steps folded into the new note.
        pre_size: Transcript size, in characters, immediately before the fold.
        post_size: Transcript size immediately after the fold.
        summary: The scripted summary text the note step carries.
    """

    folded_count: int
    pre_size: int
    post_size: int
    summary: str


@dataclass
class CompactingScratchpad:
    """A `Scratchpad` wrapper that folds old steps into a summary once size crosses a threshold.

    Attributes:
        pad: The underlying `Scratchpad` steps live in and render from.
        threshold_chars: Total transcript size, summed over all steps' fields,
            that triggers a fold.
        fold_count: How many oldest steps to fold into one summary note.
        keep_recent: Minimum number of most-recent steps a fold must always
            leave verbatim; folding stops once only this many (plus the
            would-be-folded ones) remain.
        compactions: Every fold that has run, in order.
    """

    pad: Scratchpad = field(default_factory=Scratchpad)
    threshold_chars: int = 400
    fold_count: int = 2
    keep_recent: int = 2
    compactions: list[CompactionEvent] = field(default_factory=list)

    @property
    def steps(self) -> list[Step]:
        return self.pad.steps

    def size(self) -> int:
        """Approximate transcript size: summed character counts over all step fields."""
        return sum(len(s.thought) + len(s.action) + len(s.action_input) + len(s.observation) for s in self.pad.steps)

    def render(self) -> str:
        return self.pad.render()

    def add(self, step: Step, provider: Provider) -> None:
        """Append a step, then fold the oldest steps while over the size threshold.

        Args:
            step: The step to append.
            provider: Model provider used for the scripted summary call(s) a
                fold needs. Not called at all if no fold is triggered.
        """
        self.pad.add(step)
        while self.size() > self.threshold_chars and len(self.pad.steps) > self.keep_recent + self.fold_count:
            self._fold(provider)

    def _fold(self, provider: Provider) -> None:
        """Summarize the oldest `fold_count` steps into one note step."""
        pre_size = self.size()
        to_fold = self.pad.steps[: self.fold_count]
        remaining = self.pad.steps[self.fold_count :]
        rendered = "\n".join(f"Action: {s.action}[{s.action_input}] -> Observation: {s.observation}" for s in to_fold)
        summary_prompt = Message.user(f"Steps to summarize:\n{rendered}")
        completion = provider.complete([summary_prompt], system=COMPACTION_SYSTEM_PROMPT)
        summary = completion.content
        note = Step(thought="(compacted)", action="note", action_input="", observation=summary)
        self.pad.steps = [note, *remaining]
        self.compactions.append(CompactionEvent(self.fold_count, pre_size, self.size(), summary))


@dataclass
class CompactionReactResult:
    """Outcome of a text-parsing ReAct run driven with a `CompactingScratchpad`.

    Attributes:
        answer: The final answer, or None if the loop stopped without one.
        pad: The compacting scratchpad, including its fold history.
        steps_taken: Number of main-loop model calls made.
        stopped_reason: One of "finish", "max_iterations_force", "loop_detected".
    """

    answer: str | None
    pad: CompactingScratchpad
    steps_taken: int
    stopped_reason: str


def run_react_with_compaction(
    provider: Provider,
    tools: ToolRegistry,
    goal: str,
    *,
    system_prompt: str = FEW_SHOT_SYSTEM_PROMPT,
    max_iterations: int = 8,
    threshold_chars: int = 400,
    fold_count: int = 2,
    keep_recent: int = 2,
) -> CompactionReactResult:
    """Run the text-parsing ReAct loop with a compacting scratchpad in place of a plain one.

    Args:
        provider: Model provider, used for both loop steps and fold summaries.
        tools: Registry of tools the model may invoke by name.
        goal: The question or task to solve.
        system_prompt: Instruction plus, for few-shot, a worked example trajectory.
        max_iterations: Maximum Thought/Action/Observation steps before stopping.
        threshold_chars: Size threshold passed to the `CompactingScratchpad`.
        fold_count: Steps folded per compaction.
        keep_recent: Minimum verbatim steps a fold must leave.

    Returns:
        A CompactionReactResult describing how the loop ended and every fold that ran.
    """
    pad = CompactingScratchpad(threshold_chars=threshold_chars, fold_count=fold_count, keep_recent=keep_recent)
    for step_num in range(1, max_iterations + 1):
        completion = provider.complete([Message.user(_prompt(goal, pad.pad))], system=system_prompt)
        try:
            parsed = parse_action(completion.content)
        except ActionParseError:
            error = "ERROR: could not parse an action. Use the format 'Action: Tool[args]'."
            pad.add(Step("", "(unparsed)", completion.content, observation=error), provider)
            continue

        if parsed.is_finish:
            return CompactionReactResult(parsed.final_answer, pad, step_num, "finish")

        try:
            arguments = _build_arguments(tools, parsed.tool, parsed.args_text)
        except Exception as exc:  # unknown tool or a tool with the wrong argument shape
            observation = f"ERROR: {exc}"
        else:
            call = ToolCall(id=f"call_{step_num}", name=parsed.tool, arguments=arguments)
            observation = tools.execute(call)

        pad.add(Step(parsed.thought, parsed.tool, parsed.args_text, observation=observation), provider)
        if pad.pad.is_repeating():
            return CompactionReactResult(None, pad, step_num, "loop_detected")

    return CompactionReactResult(DEFAULT_FORCE_MESSAGE, pad, max_iterations, "max_iterations_force")


def demo_compaction() -> CompactionReactResult:
    """Run a four-hop research task with a small threshold, so a fold happens mid-run.

    The first two searches answer the question; two more (Eiffel Tower,
    Beijing's population) are extra research that bulks up the transcript.
    By the fourth step the transcript is over the threshold, so `add`
    folds the oldest two steps into one note before the fifth call. The
    fold does not change the answer: the model still reaches Finish[Beijing].
    """
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: First, the Great Wall's country.\nAction: search[great wall]",
            "Thought: Now that country's capital.\nAction: search[capital of china]",
            "Thought: Also note the Eiffel Tower's location for the record.\nAction: search[eiffel tower]",
            "Thought: And Beijing's population, for completeness.\nAction: search[population of beijing]",
            # Fold triggers here, before the next call: the scripted summary.
            "Beijing is the capital of China, where the Great Wall is located.",
            "Thought: I have everything I need.\nAction: Finish[Beijing]",
        ]
    )
    goal = "What is the capital of the country where the Great Wall is located?"
    return run_react_with_compaction(provider, tools, goal, threshold_chars=200, fold_count=2, keep_recent=1)
