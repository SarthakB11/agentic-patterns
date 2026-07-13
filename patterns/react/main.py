"""ReAct: reason and act loop.

ReAct interleaves natural-language reasoning with tool use inside a single
loop. At each step the model emits a Thought (why it is doing what it is
about to do), then an Action (a tool call), and the runtime returns an
Observation (the tool's result). The Thought/Action/Observation triple is
appended to a running scratchpad and the model is prompted again with the
accumulated history. The loop repeats until the model emits a terminal
Finish action or a stop condition (max iterations, loop detection) fires.

This module is the runnable entrypoint. It drives eleven sub-variants against
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
6. Best-first tree search (LATS-lite): a two-branch fork where the losing
   branch is abandoned and the search backtracks to the winning sibling.
7. Summarize-and-continue compaction: a fold replaces old steps with a
   scripted summary note once the transcript crosses a size threshold.
8. Self-consistency: three independent rollouts vote on the final answer.
9. Verify-before-finish: a wrong Finish is rejected and corrected before
   the loop accepts an answer.
10. Derailment detection and recovery: an oscillating trajectory is nudged
    back on track instead of stopping outright.
11. Reasoning carried verbatim: the same two-hop task as (3), with a
    reasoning channel threaded across turns and no `Thought:` scaffolding.

Run:
    python3 -m patterns.react.main

No environment variables or API keys are required; every demo above runs
against MockProvider. Set AGENTIC_PATTERNS_PROVIDER=openai or =anthropic
(plus the matching API key) to run the same loop logic against a real
provider instead; the pattern modules never special-case the mock.
"""

from __future__ import annotations

from patterns.react.compaction import CompactionReactResult, demo_compaction
from patterns.react.derailment import DerailmentResult, demo_derailment
from patterns.react.native_loop import NativeReactResult, demo_native
from patterns.react.programmatic import demo_batched_lookup
from patterns.react.reasoning_loop import ReasoningReactResult, demo_reasoning
from patterns.react.reflexion import demo_reflexion
from patterns.react.self_consistency import SelfConsistencyResult, demo_self_consistency
from patterns.react.text_loop import ReactResult, demo_few_shot
from patterns.react.tree_search import TreeSearchResult, demo_tree_search
from patterns.react.verify import VerifyResult, demo_verify
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


def _print_tree_search_transcript(result: TreeSearchResult) -> None:
    for node in result.tree:
        marker = "*" if node is result.best_node else " "
        kind = "terminal" if node.terminal else "frontier"
        print(f"{marker} node {node.id}: score={node.score:g} ({kind})")
    print(f"Expansion order: {result.expansion_order}")
    print(f"Stopped: {result.stopped_reason} after {result.nodes_expanded} expansion(s)")
    print(f"Answer:  {result.answer}")


def _print_compaction_transcript(result: CompactionReactResult) -> None:
    print(f"Folds: {len(result.pad.compactions)}")
    for event in result.pad.compactions:
        print(f"  Folded {event.folded_count} step(s): {event.pre_size} -> {event.post_size} chars")
        print(f"  Note: {event.summary}")
    print(f"Stopped: {result.stopped_reason} after {result.steps_taken} step(s)")
    print(f"Answer:  {result.answer}")


def _print_self_consistency_transcript(result: SelfConsistencyResult) -> None:
    for outcome in result.rollouts:
        status = "abstained" if outcome.abstained else outcome.answer
        print(f"Rollout {outcome.index}: {status}")
    print(f"Votes: {result.votes}")
    print(f"Ran {result.rollouts_run} rollout(s), stopped_early={result.stopped_early}")
    print(f"Answer: {result.answer}")


def _print_verify_transcript(result: VerifyResult) -> None:
    print(f"Stopped: {result.stopped_reason}, first_try={result.first_try}, verify_calls={result.verify_calls}")
    print(f"Answer:  {result.answer}")


def _print_derailment_transcript(result: DerailmentResult) -> None:
    print(f"Detector fired: {result.detector_fired}, recovery_attempted={result.recovery_attempted}")
    print(f"Stopped: {result.stopped_reason} after {result.steps_taken} step(s)")
    print(f"Answer:  {result.answer}")


def _print_reasoning_transcript(result: ReasoningReactResult) -> None:
    for message in result.messages:
        if message.role == "assistant" and message.reasoning:
            print(f"  Reasoning (carried verbatim): {message.reasoning}")
    print(f"Stopped: {result.stopped_reason}, thinking_budget={result.thinking_budget}")
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

    _print_header("6. Best-first tree search (LATS-lite)")
    _print_tree_search_transcript(demo_tree_search())

    _print_header("7. Summarize-and-continue compaction")
    _print_compaction_transcript(demo_compaction())

    _print_header("8. Self-consistency (vote across 3 rollouts)")
    _print_self_consistency_transcript(demo_self_consistency())

    _print_header("9. Verify-before-finish")
    _print_verify_transcript(demo_verify())

    _print_header("10. Derailment detection and recovery")
    _print_derailment_transcript(demo_derailment())

    _print_header("11. Reasoning carried verbatim (same task as #3, no Thought scaffold)")
    _print_reasoning_transcript(demo_reasoning())

    print("\nAll ReAct demos completed.")


if __name__ == "__main__":
    main()
