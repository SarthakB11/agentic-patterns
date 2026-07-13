"""Programmatic (batched) tool calling: many tool calls in one model turn.

Real programmatic tool calling has the model write code that a sandbox
executes, orchestrating many tool calls without a round trip back to the
model between each one. This repository's frozen core contract has no code
sandbox, so this module demonstrates the part of that idea that matters for
the loop: `native_loop.run_native_react` already executes every `ToolCall` in
a turn before calling the model again, so a "program" here is simply a
`Completion` whose `tool_calls` list has more than one entry. The loop logic
is unchanged; only the script shape differs from `native_loop.demo_native`.
"""

from __future__ import annotations

from agentic_patterns import Completion, ToolCall, get_provider

from patterns.react.native_loop import NativeReactResult, build_native_registry, run_native_react

PROGRAMMATIC_SYSTEM_PROMPT = (
    "You are a research agent. When you already know every fact you need to "
    "look up, request all of the tool calls in a single turn instead of one "
    "at a time, then call `finish`."
)


def demo_batched_lookup() -> NativeReactResult:
    """Answer a two-fact question in one batched turn plus a finish turn.

    Question: "Where is the Eiffel Tower, and what is the population of
    Beijing?" Both facts are independent, so the first model turn requests
    both `search` calls at once instead of one at a time. Compare
    `steps_taken` here (2) to `native_loop.demo_native` (3), whose two
    searches are sequential because the second query depends on the first
    observation.
    """
    tools = build_native_registry()
    provider = get_provider(
        script=[
            Completion(
                content="Both facts are independent, so I can look them up in one turn.",
                tool_calls=[
                    ToolCall(id="call_1", name="search", arguments={"query": "eiffel tower"}),
                    ToolCall(id="call_2", name="search", arguments={"query": "population of beijing"}),
                ],
                stop_reason="tool_use",
            ),
            Completion(
                content="Both observations are in, I can answer now.",
                tool_calls=[
                    ToolCall(
                        id="call_3",
                        name="finish",
                        arguments={
                            "answer": "The Eiffel Tower is in Paris, France; Beijing has about 21.5 million people."
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
        ]
    )
    goal = "Where is the Eiffel Tower, and what is the population of Beijing?"
    return run_native_react(provider, tools, goal, system_prompt=PROGRAMMATIC_SYSTEM_PROMPT)
