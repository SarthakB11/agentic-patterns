"""Native tool-calling ReAct: the same reason-act loop with structured tool calls.

The model returns `Completion.tool_calls` directly; there is no Thought/
Action grammar to parse and no regex to break on. Reasoning that accompanies
a tool call lives in `Completion.content` and is carried back into the
transcript unmodified, never re-parsed or rewritten - which is also the
right way to handle a reasoning model's thinking output, since stripping or
reformatting it can discard context (or break a provider-signed block) the
next call needs. This is the production-common shape the brief calls out as
`create_agent`, `Runner`, and similar framework loops.

Finish is modeled as an ordinary tool call (`finish`); a model turn with no
tool calls at all is treated as a final answer too, matching the canonical
control flow's "no tool call and a final message" stop condition.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Completion, Message, Provider, Tool, ToolCall, ToolRegistry, get_provider

from patterns.react.world import build_registry

NATIVE_SYSTEM_PROMPT = (
    "You are a research agent. Call tools to gather the facts you need, then "
    "call `finish` with your answer once you have enough information. Do not "
    "guess at facts you have not looked up."
)


def finish(answer: str) -> str:
    """Terminal tool: records the final answer and ends the loop.

    Args:
        answer: The agent's final answer to the user's question.

    Returns:
        The answer text, echoed back as the tool result.
    """
    return answer


def build_native_registry() -> ToolRegistry:
    """Build the shared demo registry plus a `finish` tool for native loops."""
    registry = build_registry()
    registry.register(
        Tool(
            name="finish",
            description="Call this with the final answer once you are done researching.",
            parameters={
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "The final answer."}},
                "required": ["answer"],
            },
            fn=finish,
        )
    )
    return registry


@dataclass
class NativeReactResult:
    """Outcome of a native tool-calling ReAct run.

    Attributes:
        answer: The final answer, or None if the loop stopped without one.
        messages: The full conversation, including tool calls and results.
        steps_taken: Number of main-loop model calls made (excludes the extra
            tool-free call a "generate" stop makes).
        stopped_reason: One of "finish", "max_iterations_force",
            "max_iterations_generate", "loop_detected".
    """

    answer: str | None
    messages: list[Message]
    steps_taken: int
    stopped_reason: str


def _tool_call_signature(message: Message) -> list[tuple[str, tuple[tuple[str, object], ...]]]:
    """A hashable summary of a message's tool calls, for repeat detection."""
    return [(tc.name, tuple(sorted(tc.arguments.items()))) for tc in message.tool_calls]


def _repeats_previous_call(messages: list[Message]) -> bool:
    """Detect two consecutive assistant turns that request the identical tool call(s)."""
    assistant_turns = [m for m in messages if m.role == "assistant" and m.tool_calls]
    if len(assistant_turns) < 2:
        return False
    return _tool_call_signature(assistant_turns[-1]) == _tool_call_signature(assistant_turns[-2])


def run_native_react(
    provider: Provider,
    tools: ToolRegistry,
    goal: str,
    *,
    system_prompt: str = NATIVE_SYSTEM_PROMPT,
    max_iterations: int = 6,
    on_max_iterations: str = "force",
    force_message: str = "Stopped: could not reach an answer within the iteration budget.",
) -> NativeReactResult:
    """Run the native tool-calling ReAct loop to completion.

    One model turn may return multiple tool calls at once; every call in the
    batch is executed before the next model turn, which is what lets a
    "programmatic" batched turn (see `programmatic.py`) cut round trips.

    Args:
        provider: Model provider to call each iteration.
        tools: Registry of tools the model may invoke, including `finish`.
        goal: The question or task to solve.
        system_prompt: Instruction describing the task and the `finish` convention.
        max_iterations: Maximum model turns before stopping.
        on_max_iterations: "force" returns `force_message` immediately;
            "generate" makes one further tool-free model call asking it to
            answer from the work done so far.
        force_message: The fixed message returned when `on_max_iterations="force"`.

    Returns:
        A NativeReactResult describing how the loop ended.
    """
    messages: list[Message] = [Message.user(goal)]
    for step_num in range(1, max_iterations + 1):
        completion = provider.complete(messages, tools=tools.specs(), system=system_prompt)
        messages.append(Message.assistant(completion.content, completion.tool_calls))

        if not completion.tool_calls:
            return NativeReactResult(
                answer=completion.content, messages=messages, steps_taken=step_num, stopped_reason="finish"
            )

        if _repeats_previous_call(messages):
            return NativeReactResult(answer=None, messages=messages, steps_taken=step_num, stopped_reason="loop_detected")

        finished_answer: str | None = None
        for call in completion.tool_calls:
            observation = tools.execute(call)
            messages.append(Message.tool(call.id, observation))
            if call.name == "finish":
                finished_answer = call.arguments.get("answer", observation)
        if finished_answer is not None:
            return NativeReactResult(answer=finished_answer, messages=messages, steps_taken=step_num, stopped_reason="finish")

    return _stop_at_max_iterations_native(provider, messages, system_prompt, on_max_iterations, force_message, max_iterations)


def _stop_at_max_iterations_native(
    provider: Provider,
    messages: list[Message],
    system_prompt: str,
    on_max_iterations: str,
    force_message: str,
    max_iterations: int,
) -> NativeReactResult:
    """Apply the configured early-stop policy once the iteration cap is hit."""
    if on_max_iterations == "generate":
        final_messages = [*messages, Message.user("You are out of tool-call budget. Answer from the work above, in plain text.")]
        completion = provider.complete(final_messages, system=system_prompt)
        return NativeReactResult(
            answer=completion.content, messages=final_messages, steps_taken=max_iterations, stopped_reason="max_iterations_generate"
        )
    if on_max_iterations == "force":
        return NativeReactResult(
            answer=force_message, messages=messages, steps_taken=max_iterations, stopped_reason="max_iterations_force"
        )
    raise ValueError(f"Unknown on_max_iterations policy: {on_max_iterations!r}")


def demo_native() -> NativeReactResult:
    """Run the same two-hop Great Wall question as `text_loop.demo_few_shot`.

    Uses native tool calls instead of parsed text, so the two transcripts can
    be compared directly for an identical task.
    """
    tools = build_native_registry()
    provider = get_provider(
        script=[
            Completion(
                content="I need to find which country the Great Wall is located in.",
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": "great wall"})],
                stop_reason="tool_use",
            ),
            Completion(
                content="Now I need the capital of that country.",
                tool_calls=[ToolCall(id="call_2", name="search", arguments={"query": "capital of china"})],
                stop_reason="tool_use",
            ),
            Completion(
                content="I have enough information to answer.",
                tool_calls=[ToolCall(id="call_3", name="finish", arguments={"answer": "Beijing"})],
                stop_reason="tool_use",
            ),
        ]
    )
    goal = "What is the capital of the country where the Great Wall is located?"
    return run_native_react(provider, tools, goal)
