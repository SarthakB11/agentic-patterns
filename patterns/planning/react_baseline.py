"""ReAct-style interleaved baseline, for contrast with plan-then-execute.

Here the model gets one tool call at a time: think, act, observe, decide the
next action, repeat, with no separate planning phase and no upfront `Plan`
object at all. This module exists so the pattern's demos can compare the two
control-flow shapes on the same task and see the difference in model-call
count directly: plan-then-execute (`sequential_executor.py`) needs one
planner call and one solver call regardless of step count, while this loop
needs one call per step plus one to produce the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, Provider, ToolRegistry

SYSTEM = (
    "You are a trip-planning agent. Call one tool at a time to gather what "
    "you need. Once you have enough information, reply with plain text and "
    "no tool call to give the final answer."
)


@dataclass
class ReactRun:
    """The outcome of an interleaved ReAct-style run.

    Attributes:
        transcript: The full conversation, including every tool call and
            observation, unlike the compact `StepResult` lists the
            plan-then-execute variants return.
        final_answer: The model's final plain-text reply.
        model_calls: Total number of `provider.complete` calls made.
    """

    transcript: list[Message]
    final_answer: str
    model_calls: int


def run_react(provider: Provider, goal: str, registry: ToolRegistry, max_steps: int = 6) -> ReactRun:
    """Run an interleaved think-act-observe loop until the model stops calling tools.

    Args:
        provider: Called once per loop iteration; no separate planner phase.
        goal: The user's goal, sent as the first user message.
        registry: Tools the model may call, offered on every turn.
        max_steps: Safety cap on loop iterations.

    Raises:
        RuntimeError: If the loop exceeds `max_steps` without a final answer.
    """
    messages: list[Message] = [Message.user(goal)]
    model_calls = 0
    for model_calls in range(1, max_steps + 1):
        completion = provider.complete(messages, tools=registry.specs(), system=SYSTEM)
        if not completion.tool_calls:
            messages.append(Message.assistant(completion.content))
            return ReactRun(transcript=messages, final_answer=completion.content, model_calls=model_calls)
        messages.append(Message.assistant(completion.content, tool_calls=completion.tool_calls))
        for call in completion.tool_calls:
            observation = registry.execute(call)
            messages.append(Message.tool(call.id, observation))
    raise RuntimeError(f"ReAct loop exceeded max_steps={max_steps} without a final answer")


def demo() -> None:
    """Run the ReAct baseline on the same weather-and-attractions goal and print the transcript."""
    from agentic_patterns import get_provider
    from patterns.planning.tools import build_travel_registry

    goal = "What's the weather and what are the top attractions in Lisbon?"
    script = [
        {"tool": "get_weather", "args": {"city": "Lisbon"}},
        {"tool": "search_attractions", "args": {"city": "Lisbon"}},
        "Lisbon is sunny and warm with no rain, and the top attractions are "
        "Belem Tower, the Alfama district, and Time Out Market.",
    ]
    provider = get_provider(script=script)
    registry = build_travel_registry()

    print("=== ReAct baseline (interleaved, no upfront plan) ===")
    print(f"Goal: {goal}")
    run = run_react(provider, goal, registry)
    for message in run.transcript:
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                print(f"  [call] {call.name}({call.arguments})")
        elif message.role == "tool":
            print(f"  [observation] {message.content}")
    print(f"Final answer: {run.final_answer}")
    print(f"Total model calls: {run.model_calls} (one per step, unlike ReWOO's fixed 2)")


if __name__ == "__main__":
    demo()
