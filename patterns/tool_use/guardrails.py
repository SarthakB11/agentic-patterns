"""Guardrails: tool errors, unknown tools, retry caps, and iteration caps.

Four ways a tool-calling loop can go wrong, and how `run_tool_loop` keeps
each one from taking the whole run down:

- A tool function raises. `ToolRegistry.execute` (core) catches the
  exception and turns it into an "ERROR: ..." observation, so the loop
  treats a runtime failure exactly like any other observation and keeps
  going instead of crashing.
- The model names a tool that is not registered. `run_tool_loop` catches
  the registry's `KeyError` the same way and continues.
- The model keeps sending invalid arguments. `retry_limit` bounds how many
  repair turns the whole run will grant; once exhausted, further invalid
  calls become a terminal observation instead of another chance.
- The model never stops calling tools. `max_iterations` bounds how many
  model round trips the loop will make before it gives up.
"""

from __future__ import annotations

from agentic_patterns import Message, get_provider, scripted_tool_call

from patterns.tool_use.catalog import SYSTEM_PROMPT, build_registry
from patterns.tool_use.loop import ToolLoopResult, run_tool_loop


def demo_tool_error() -> ToolLoopResult:
    """A tool raises on an unknown city; the exception becomes an observation, not a crash."""
    registry = build_registry()
    provider = get_provider(
        script=[
            scripted_tool_call("get_weather", {"city": "Atlantis"}),
            "I don't have weather data for Atlantis in this system; could you confirm the city name?",
        ]
    )
    messages = [Message.user("What's the weather in Atlantis?")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=2)

    print("=== 7a. Guardrail: tool-execution error becomes an observation ===")
    print(f"user:  {messages[0].content}")
    print(f"  observation: {result.rounds[0].calls[0].observation} (outcome={result.rounds[0].calls[0].outcome})")
    print(f"final: {result.final_answer}")
    print()
    return result


def demo_unknown_tool() -> ToolLoopResult:
    """The model calls a tool that was never registered; the loop reports it, then continues."""
    registry = build_registry()
    provider = get_provider(
        script=[
            scripted_tool_call("get_forecast", {"city": "Tokyo"}),
            "A multi-day forecast isn't available here; current conditions are what I can offer instead.",
        ]
    )
    messages = [Message.user("Give me the 5-day forecast for Tokyo.")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=2)

    print("=== 7b. Guardrail: unknown tool name ===")
    print(f"user:  {messages[0].content}")
    print(f"  observation: {result.rounds[0].calls[0].observation} (outcome={result.rounds[0].calls[0].outcome})")
    print(f"final: {result.final_answer}")
    print()
    return result


def demo_retry_cap() -> ToolLoopResult:
    """The model keeps sending an invalid amount; retry_limit=1 cuts off after one repair attempt."""
    registry = build_registry()
    provider = get_provider(
        script=[
            scripted_tool_call("convert_currency", {"amount": "100", "from_currency": "USD", "to_currency": "EUR"}),
            scripted_tool_call("convert_currency", {"amount": "100", "from_currency": "USD", "to_currency": "EUR"}),
            "I wasn't able to validate a numeric amount for that conversion; "
            "please resend the amount as a plain number.",
        ]
    )
    messages = [Message.user("Convert 100 USD to EUR.")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=3, retry_limit=1)

    print("=== 7c. Guardrail: repair budget (retry_limit) exhausted ===")
    print(f"user:  {messages[0].content}")
    for round_record in result.rounds:
        for record in round_record.calls:
            print(f"  round {round_record.index} [{record.outcome}]: {record.observation}")
    print(f"final: {result.final_answer}")
    print()
    return result


def demo_iteration_cap() -> ToolLoopResult:
    """A model that never stops calling tools is halted by max_iterations, not left to run forever."""
    registry = build_registry()
    provider = get_provider(
        script=[
            scripted_tool_call("get_weather", {"city": "Tokyo"}),
            scripted_tool_call("get_weather", {"city": "Paris"}),
            scripted_tool_call("get_weather", {"city": "San Francisco"}),
            scripted_tool_call("get_weather", {"city": "Tokyo"}),
            scripted_tool_call("get_weather", {"city": "Paris"}),
        ]
    )
    messages = [Message.user("Keep checking weather until I say stop.")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=3)

    print("=== 7d. Guardrail: max_iterations halts a model that never stops calling ===")
    print(f"user:  {messages[0].content}")
    print(f"rounds run: {len(result.rounds)} (cap was 3; script had 5 scripted calls available)")
    print(f"stop_reason={result.stop_reason}")
    print()
    return result


if __name__ == "__main__":
    demo_tool_error()
    demo_unknown_tool()
    demo_retry_cap()
    demo_iteration_cap()
