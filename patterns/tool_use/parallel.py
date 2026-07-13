"""Parallel tool calls: several independent calls in one turn.

When the calls in a turn do not depend on each other, the model can (if the
provider supports it) return them all at once instead of one per round trip.
The runtime here executes such a turn's calls concurrently with a thread
pool and recombines the observations in the order the model asked for them,
regardless of which call happened to finish first, so the model always sees
a stable, deterministic order.
"""

from __future__ import annotations

from agentic_patterns import Completion, Message, ToolCall, get_provider

from patterns.tool_use.catalog import SYSTEM_PROMPT, build_registry
from patterns.tool_use.loop import ToolLoopResult, run_tool_loop


def demo_parallel() -> ToolLoopResult:
    """Run one turn with three independent weather calls, executed concurrently.

    The three cities do not depend on each other, so a provider that
    supports parallel tool calls returns all three requests in a single
    completion. `run_tool_loop` executes them with a thread pool and appends
    their observations back in call order.
    """
    registry = build_registry()
    provider = get_provider(
        script=[
            Completion(
                tool_calls=[
                    ToolCall("call_1", "get_weather", {"city": "Tokyo"}),
                    ToolCall("call_2", "get_weather", {"city": "Paris"}),
                    ToolCall("call_3", "get_weather", {"city": "San Francisco"}),
                ],
                stop_reason="tool_use",
            ),
            "Tokyo is 18C with light rain, Paris is 21C and clear, and San "
            "Francisco is 16C with fog.",
        ]
    )
    messages = [Message.user("What's the weather in Tokyo, Paris, and San Francisco?")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=2)

    print("=== 3. Parallel tool calls (one turn, three independent calls) ===")
    print(f"user:  {messages[0].content}")
    for record in result.rounds[0].calls:
        print(f"  {record.call.id}: {record.call.name}({record.call.arguments}) -> {record.observation}")
    print(f"final: {result.final_answer}")
    print(f"stop_reason={result.stop_reason}, calls_in_round_1={len(result.rounds[0].calls)}")
    print()
    return result


if __name__ == "__main__":
    demo_parallel()
