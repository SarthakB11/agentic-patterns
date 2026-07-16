"""Native tool-calling ReAct that carries a reasoning channel verbatim, no Thought scaffold.

`native_loop.run_native_react` already carries `Completion.content` across
turns unmodified, which is most of what a reasoning model needs. The one
thing it drops is `Completion.reasoning`: `Message.assistant(completion.content,
completion.tool_calls)` never passes `reasoning=`, so a real reasoning-model
adapter's thinking blocks (and their signatures) would be silently discarded
turn to turn. This module is the small, deliberate fix: the same control
flow as `run_native_react`, reusing its system prompt, registry, and repeat
guard, with `reasoning` threaded through and never parsed or rewritten.

It also does not add a `Thought:` instruction anywhere. Anthropic's
interleaved thinking (beta `interleaved-thinking-2025-05-14`, May 2025) lets
a model reason in its own channel between tool calls; forcing that same
model through the `text_loop.py` text grammar's explicit `Thought:` line is
redundant scaffolding on top of a channel that already exists. NoWait (Wang
et al., arXiv:2506.08343) found that suppressing self-reflection tokens like
"Wait" cuts chain-of-thought length 27 to 51% with no accuracy loss, and
Rana et al.'s "Model-First Reasoning LLM Agents" (arXiv:2512.14474) argue a
model should build an explicit problem model rather than externalize
free-text thoughts into a template. The practical rule this module follows:
for a reasoning model, let it think in `reasoning`, do not also ask it to
narrate a `Thought:` line in `content`.

`thinking_budget` is accepted and recorded on the result for observability.
Offline it only annotates the call; a real provider adapter would pass it to
the API and enforce it there.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Completion, Message, Provider, ToolCall, ToolRegistry, get_provider
from patterns.react.native_loop import NATIVE_SYSTEM_PROMPT, _repeats_previous_call, build_native_registry

DEFAULT_FORCE_MESSAGE = "Stopped: could not reach an answer within the iteration budget."


@dataclass
class ReasoningReactResult:
    """Outcome of a native tool-calling ReAct run with reasoning carried across turns.

    Attributes:
        answer: The final answer, or None if the loop stopped without one.
        messages: The full conversation, including reasoning on each assistant turn.
        steps_taken: Number of main-loop model calls made.
        stopped_reason: One of "finish", "max_iterations_force", "loop_detected".
        thinking_budget: The budget passed in, recorded for observability. Not
            enforced offline; a real provider adapter would enforce it.
    """

    answer: str | None
    messages: list[Message]
    steps_taken: int
    stopped_reason: str
    thinking_budget: int | None


def run_reasoning_react(
    provider: Provider,
    tools: ToolRegistry,
    goal: str,
    *,
    system_prompt: str = NATIVE_SYSTEM_PROMPT,
    max_iterations: int = 6,
    thinking_budget: int | None = None,
) -> ReasoningReactResult:
    """Run native tool-calling ReAct, carrying each turn's reasoning verbatim into the next.

    Args:
        provider: Model provider to call each iteration.
        tools: Registry of tools the model may invoke, including `finish`.
        goal: The question or task to solve.
        system_prompt: Instruction describing the task. Deliberately has no
            `Thought:` directive; a reasoning model reasons in `reasoning` instead.
        max_iterations: Maximum model turns before stopping.
        thinking_budget: Optional thinking-token budget, recorded on the
            result but not enforced by `MockProvider`.

    Returns:
        A ReasoningReactResult describing how the loop ended.
    """
    messages: list[Message] = [Message.user(goal)]
    for step_num in range(1, max_iterations + 1):
        completion: Completion = provider.complete(messages, tools=tools.specs(), system=system_prompt)
        messages.append(Message.assistant(completion.content, completion.tool_calls, reasoning=completion.reasoning))

        if not completion.tool_calls:
            return ReasoningReactResult(completion.content, messages, step_num, "finish", thinking_budget)

        if _repeats_previous_call(messages):
            return ReasoningReactResult(None, messages, step_num, "loop_detected", thinking_budget)

        finished_answer: str | None = None
        for call in completion.tool_calls:
            observation = tools.execute(call)
            messages.append(Message.tool(call.id, observation))
            if call.name == "finish":
                finished_answer = call.arguments.get("answer", observation)
        if finished_answer is not None:
            return ReasoningReactResult(finished_answer, messages, step_num, "finish", thinking_budget)

    return ReasoningReactResult(
        DEFAULT_FORCE_MESSAGE, messages, max_iterations, "max_iterations_force", thinking_budget
    )


def demo_reasoning() -> ReasoningReactResult:
    """Run the same two-hop Great Wall question as `native_loop.demo_native`, with reasoning carried.

    Each assistant turn's `reasoning` is opaque text distinct from its
    user-visible `content`, standing in for a reasoning model's thinking
    blocks. The second call's messages carry the first call's exact
    reasoning string unmodified, never parsed or rewritten.
    """
    tools = build_native_registry()
    provider = get_provider(
        script=[
            Completion(
                content="I'll start by finding the Great Wall's country.",
                reasoning="The question needs two hops: a landmark's country, then that country's capital.",
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": "great wall"})],
                stop_reason="tool_use",
            ),
            Completion(
                content="Now the capital of that country.",
                reasoning="China was named; capital-of-china is the natural next lookup.",
                tool_calls=[ToolCall(id="call_2", name="search", arguments={"query": "capital of china"})],
                stop_reason="tool_use",
            ),
            Completion(
                content="I have enough information to answer.",
                reasoning="Beijing is confirmed as the capital of China.",
                tool_calls=[ToolCall(id="call_3", name="finish", arguments={"answer": "Beijing"})],
                stop_reason="tool_use",
            ),
        ]
    )
    goal = "What is the capital of the country where the Great Wall is located?"
    return run_reasoning_react(provider, tools, goal, thinking_budget=2000)
