"""ReAct: reason and act loop.

ReAct interleaves natural-language reasoning with tool use inside a single
loop. At each step the model emits a Thought (why it is doing what it is
about to do), then an Action (a tool call), and the runtime returns an
Observation (the tool's result). The Thought/Action/Observation triple is
appended to a running scratchpad and the model is prompted again with the
accumulated history. The loop repeats until the model emits a terminal
Finish action or a stop condition (max iterations, loop detection) fires.

This module is the runnable entrypoint. It drives five sub-variants against
a scripted `MockProvider` and prints a readable transcript for each:

1. Few-shot text-parsing ReAct (the canonical form, with a worked example
   in the prompt), on a two-hop lookup question.
2. Zero-shot text-parsing ReAct (no worked example), on a one-hop question.
3. Native tool-calling ReAct, using structured tool calls instead of a
   parsed text grammar, on the same two-hop question as (1).
4. ReAct plus Reflexion: a first episode fails, the agent critiques its own
   trajectory, and a second episode succeeds.
5. Programmatic (batched) tool calling: one model turn requests two
   independent tool calls at once instead of one call per round trip.

Run:
    python3 -m patterns.react.main

No environment variables or API keys are required; every demo above runs
against MockProvider. Set AGENTIC_PATTERNS_PROVIDER=openai or =anthropic
(plus the matching API key) to run the same loop logic against a real
provider instead; the pattern modules never special-case the mock.
"""

from __future__ import annotations

from patterns.react.native_loop import NativeReactResult, demo_native
from patterns.react.programmatic import demo_batched_lookup
from patterns.react.reflexion import demo_reflexion
from patterns.react.text_loop import ReactResult, demo_few_shot
from patterns.react.zero_shot import demo_zero_shot


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_text_transcript(result: ReactResult) -> None:
    """Print a text-parsing ReAct run's scratchpad, one line per field."""
    for i, step in enumerate(result.scratchpad.steps, start=1):
        print(f"Step {i}")
        print(f"  Thought:     {step.thought}")
        print(f"  Action:      {step.action}[{step.action_input}]")
        if step.observation:
            print(f"  Observation: {step.observation}")
    print(f"Stopped: {result.stopped_reason} after {result.steps_taken} step(s)")
    print(f"Answer:  {result.answer}")


def _print_native_transcript(result: NativeReactResult) -> None:
    """Print a native tool-calling ReAct run's message history, grouped by turn."""
    step = 0
    for message in result.messages:
        if message.role == "assistant":
            step += 1
            print(f"Step {step}")
            if message.content:
                print(f"  Reasoning: {message.content}")
            for call in message.tool_calls:
                print(f"  Tool call: {call.name}({call.arguments})")
        elif message.role == "tool":
            print(f"  Observation: {message.content}")
    print(f"Stopped: {result.stopped_reason} after {result.steps_taken} step(s)")
    print(f"Answer:  {result.answer}")


def main() -> None:
    print("ReAct pattern demo: reason + act loop, driven against a scripted MockProvider.")

    _print_header("1. Few-shot text-parsing ReAct (canonical, two-hop lookup)")
    _print_text_transcript(demo_few_shot())

    _print_header("2. Zero-shot text-parsing ReAct (one-hop lookup)")
    _print_text_transcript(demo_zero_shot())

    _print_header("3. Native tool-calling ReAct (same two-hop question as #1)")
    _print_native_transcript(demo_native())

    _print_header("4. ReAct plus Reflexion (retry after a failed episode)")
    reflexion_result = demo_reflexion()
    for i, reflection in enumerate(reflexion_result.reflections, start=1):
        print(f"Reflection {i}: {reflection}")
    _print_text_transcript(reflexion_result.final)
    print(f"Episodes run: {reflexion_result.episodes_run}")

    _print_header("5. Programmatic (batched) tool calling")
    _print_native_transcript(demo_batched_lookup())

    print("\nAll ReAct demos completed.")


if __name__ == "__main__":
    main()
