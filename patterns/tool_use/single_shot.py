"""Single-shot tool use: the base case.

The model returns exactly one tool call, the app runs it, and the result
goes straight back for one final answer. No branching, no dependency
between calls: this is the smallest complete instance of the canonical
control flow in docs/research/tool_use.md, and the shape every other module
in this pattern builds on.
"""

from __future__ import annotations

from agentic_patterns import Message, get_provider, scripted_tool_call

from patterns.tool_use.catalog import SYSTEM_PROMPT, build_registry
from patterns.tool_use.loop import ToolLoopResult, run_tool_loop


def demo_single_shot() -> ToolLoopResult:
    """Run a one-call, one-answer tool use exchange.

    The user asks for Tokyo's weather; the model has no way to know that
    without calling `get_weather`, so it calls it once and then answers
    using the observation.
    """
    registry = build_registry()
    provider = get_provider(
        script=[
            scripted_tool_call("get_weather", {"city": "Tokyo"}),
            "It's 18C with light rain in Tokyo right now, so an umbrella is worth bringing.",
        ]
    )
    messages = [Message.user("What's the weather in Tokyo?")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=2)

    print("=== 1. Single-shot tool call ===")
    print(f"user:  {messages[0].content}")
    call = result.rounds[0].calls[0]
    print(f"  call: {call.call.name}({call.call.arguments})")
    print(f"  observation: {call.observation}")
    print(f"final: {result.final_answer}")
    print(f"stop_reason={result.stop_reason}, rounds={len(result.rounds)}")
    print()
    return result


if __name__ == "__main__":
    demo_single_shot()
