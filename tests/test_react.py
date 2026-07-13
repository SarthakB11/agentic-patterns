"""Tests for the ReAct pattern.

Deterministic and offline: every test drives `MockProvider` scripts through
the pattern logic directly, with no network call and no API key. Tests cover
loop mechanics (iterations, stop conditions, tool dispatch, error observation
paths), the text-parsing grammar's edge cases, and each variant module.
"""

from __future__ import annotations

import pytest

from agentic_patterns import Completion, Tool, ToolRegistry, get_provider

from patterns.react.native_loop import build_native_registry, demo_native, run_native_react
from patterns.react.parser import ActionParseError, parse_action
from patterns.react.programmatic import demo_batched_lookup
from patterns.react.reflexion import demo_reflexion, run_with_reflexion
from patterns.react.scratchpad import Scratchpad, Step
from patterns.react.text_loop import demo_few_shot, run_react
from patterns.react.world import build_registry
from patterns.react.zero_shot import ZERO_SHOT_SYSTEM_PROMPT, demo_zero_shot


# --- parser -----------------------------------------------------------------


def test_parse_action_well_formed() -> None:
    parsed = parse_action("Thought: I should search.\nAction: search[eiffel tower]")
    assert parsed.thought == "I should search."
    assert parsed.tool == "search"
    assert parsed.args_text == "eiffel tower"
    assert parsed.is_finish is False
    assert parsed.final_answer is None


def test_parse_action_finish_extracts_final_answer() -> None:
    parsed = parse_action("Thought: Done.\nAction: Finish[Beijing]")
    assert parsed.is_finish is True
    assert parsed.final_answer == "Beijing"


def test_parse_action_missing_action_line_raises() -> None:
    with pytest.raises(ActionParseError):
        parse_action("Thought: I am thinking but forgot to act.")


def test_parse_action_missing_thought_still_parses() -> None:
    parsed = parse_action("Action: search[capital of china]")
    assert parsed.thought == ""
    assert parsed.tool == "search"


# --- scratchpad ---------------------------------------------------------------


def test_scratchpad_truncates_long_observations() -> None:
    pad = Scratchpad(max_observation_chars=10)
    pad.add(Step(thought="t", action="search", action_input="x", observation="y" * 50))
    assert pad.steps[0].observation == "y" * 10 + "... [truncated]"


def test_scratchpad_is_repeating_true_for_identical_pair() -> None:
    pad = Scratchpad()
    pad.add(Step(thought="t1", action="search", action_input="monument", observation="not found"))
    pad.add(Step(thought="t2", action="search", action_input="monument", observation="not found"))
    assert pad.is_repeating() is True


def test_scratchpad_is_repeating_false_for_distinct_steps() -> None:
    pad = Scratchpad()
    pad.add(Step(thought="t1", action="search", action_input="a", observation="obs a"))
    pad.add(Step(thought="t2", action="search", action_input="b", observation="obs b"))
    assert pad.is_repeating() is False


# --- text_loop: happy path and multi-step ------------------------------------


def test_run_react_happy_path_single_tool_call() -> None:
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: Look it up.\nAction: search[eiffel tower]",
            "Thought: Done.\nAction: Finish[Paris, France]",
        ]
    )
    result = run_react(provider, tools, "Where is the Eiffel Tower?")
    assert result.answer == "Paris, France"
    assert result.stopped_reason == "finish"
    assert len(result.scratchpad.steps) == 1
    assert result.scratchpad.steps[0].action == "search"
    assert result.scratchpad.steps[0].action_input == "eiffel tower"


def test_run_react_two_hop_second_query_depends_on_first_observation() -> None:
    result = demo_few_shot()
    assert result.answer == "Beijing"
    steps = result.scratchpad.steps
    assert [s.action for s in steps] == ["search", "search"]
    assert steps[0].action_input == "Great Wall"
    assert steps[0].observation == "The Great Wall is located in China."
    # The second query is informed by the first observation (mentions China).
    assert steps[1].action_input == "capital of china"
    assert steps[1].observation == "The capital of China is Beijing."


def test_run_react_zero_shot_uses_zero_shot_system_prompt() -> None:
    result = demo_zero_shot()
    assert result.answer == "Paris, France"
    # No exemplar trajectory in this system prompt, unlike the few-shot one.
    assert "Example:" not in ZERO_SHOT_SYSTEM_PROMPT


# --- text_loop: error and edge-case handling ---------------------------------


def test_run_react_malformed_output_yields_recoverable_observation() -> None:
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: I forgot the Action line.",
            "Thought: Retrying.\nAction: Finish[recovered]",
        ]
    )
    result = run_react(provider, tools, "anything")
    assert result.answer == "recovered"
    assert result.scratchpad.steps[0].action == "(unparsed)"
    assert "ERROR" in result.scratchpad.steps[0].observation


def test_run_react_unknown_tool_yields_error_observation_and_continues() -> None:
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: Try a tool that does not exist.\nAction: fetch[eiffel tower]",
            "Thought: Fall back to a real tool.\nAction: Finish[gave up]",
        ]
    )
    result = run_react(provider, tools, "anything")
    assert result.scratchpad.steps[0].observation.startswith("ERROR")
    assert result.answer == "gave up"


def test_run_react_bad_argument_shape_yields_error_observation() -> None:
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="two_arg",
            description="A tool with two parameters, which the Tool[args] grammar cannot address.",
            parameters={"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}},
            fn=lambda a, b: f"{a}-{b}",
        )
    )
    provider = get_provider(
        script=[
            "Thought: Call the bad tool.\nAction: two_arg[x]",
            "Thought: Give up.\nAction: Finish[done]",
        ]
    )
    result = run_react(provider, tools, "anything")
    assert result.scratchpad.steps[0].observation.startswith("ERROR")


def test_run_react_tool_raises_is_caught_into_observation() -> None:
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="explode",
            description="A tool that always raises.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            fn=lambda query: (_ for _ in ()).throw(ValueError("boom")),
        )
    )
    provider = get_provider(
        script=[
            "Thought: Call it.\nAction: explode[anything]",
            "Thought: Recovered.\nAction: Finish[ok]",
        ]
    )
    result = run_react(provider, tools, "anything")
    assert "ERROR" in result.scratchpad.steps[0].observation
    assert "boom" in result.scratchpad.steps[0].observation
    assert result.answer == "ok"


# --- text_loop: max iterations and loop detection ----------------------------


def test_run_react_max_iterations_force_stop() -> None:
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: keep looking.\nAction: search[topic one]",
            "Thought: keep looking.\nAction: search[topic two]",
            "Thought: keep looking.\nAction: search[topic three]",
        ]
    )
    result = run_react(provider, tools, "anything", max_iterations=3, on_max_iterations="force")
    assert result.stopped_reason == "max_iterations_force"
    assert result.steps_taken == 3
    assert result.answer == "Stopped: could not reach an answer within the iteration budget."


def test_run_react_max_iterations_generate_stop_makes_one_extra_call() -> None:
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: keep looking.\nAction: search[unrelated topic]",
            "Thought: keep looking.\nAction: search[another topic]",
            "Best guess based on the search results: Beijing.",
        ]
    )
    result = run_react(provider, tools, "anything", max_iterations=2, on_max_iterations="generate")
    assert result.stopped_reason == "max_iterations_generate"
    assert result.answer == "Best guess based on the search results: Beijing."
    assert len(provider.calls) == 3  # two loop iterations plus one tool-free generate call
    assert provider.calls[-1]["tools"] is None


def test_run_react_loop_detection_stops_on_identical_repeat() -> None:
    tools = build_registry()
    provider = get_provider(script=["Thought: try again.\nAction: search[monument]"] * 5)
    result = run_react(provider, tools, "anything", max_iterations=5)
    assert result.stopped_reason == "loop_detected"
    assert result.steps_taken == 2  # stops as soon as steps 1 and 2 match, not at the cap


# --- native_loop --------------------------------------------------------------


def test_run_native_react_happy_path_dispatches_registered_tool() -> None:
    tools = build_native_registry()
    provider = get_provider(
        script=[
            {"tool": "search", "args": {"query": "eiffel tower"}},
            {"tool": "finish", "args": {"answer": "Paris, France"}},
        ]
    )
    result = run_native_react(provider, tools, "Where is the Eiffel Tower?")
    assert result.answer == "Paris, France"
    assert result.stopped_reason == "finish"
    tool_calls = [tc for m in result.messages if m.role == "assistant" for tc in m.tool_calls]
    assert [tc.name for tc in tool_calls] == ["search", "finish"]
    assert tool_calls[0].arguments == {"query": "eiffel tower"}


def test_demo_native_two_hop_matches_text_loop_answer() -> None:
    result = demo_native()
    assert result.answer == "Beijing"
    tool_calls = [tc for m in result.messages if m.role == "assistant" for tc in m.tool_calls]
    assert [tc.name for tc in tool_calls] == ["search", "search", "finish"]


def test_run_native_react_no_tool_calls_ends_the_loop() -> None:
    tools = build_native_registry()
    provider = get_provider(script=[Completion(content="I already know the answer: Paris.", stop_reason="stop")])
    result = run_native_react(provider, tools, "Where is the Eiffel Tower?")
    assert result.stopped_reason == "finish"
    assert result.answer == "I already know the answer: Paris."


def test_run_native_react_unknown_tool_yields_error_observation() -> None:
    tools = build_native_registry()
    provider = get_provider(
        script=[
            {"tool": "fetch", "args": {"query": "eiffel tower"}},
            {"tool": "finish", "args": {"answer": "gave up"}},
        ]
    )
    result = run_native_react(provider, tools, "anything")
    tool_messages = [m for m in result.messages if m.role == "tool"]
    assert tool_messages[0].content.startswith("ERROR")
    assert result.answer == "gave up"


def test_run_native_react_max_iterations_force_stop() -> None:
    tools = build_native_registry()
    provider = get_provider(
        script=[
            {"tool": "search", "args": {"query": "topic one"}},
            {"tool": "search", "args": {"query": "topic two"}},
            {"tool": "search", "args": {"query": "topic three"}},
        ]
    )
    result = run_native_react(provider, tools, "anything", max_iterations=3, on_max_iterations="force")
    assert result.stopped_reason == "max_iterations_force"
    assert result.steps_taken == 3


def test_run_native_react_loop_detection_stops_on_identical_repeat() -> None:
    tools = build_native_registry()
    provider = get_provider(script=[{"tool": "search", "args": {"query": "monument"}}] * 4)
    result = run_native_react(provider, tools, "anything", max_iterations=4)
    assert result.stopped_reason == "loop_detected"
    assert result.steps_taken == 2


# --- reflexion ------------------------------------------------------------


def test_run_with_reflexion_retries_after_failure_and_succeeds() -> None:
    result = demo_reflexion()
    assert result.final.answer == "Paris, France"
    assert result.episodes_run == 2
    assert len(result.reflections) == 1
    assert "vague" in result.reflections[0]


def test_run_with_reflexion_succeeds_first_episode_writes_no_reflection() -> None:
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: Search directly.\nAction: search[eiffel tower]",
            "Thought: Done.\nAction: Finish[Paris, France]",
        ]
    )
    result = run_with_reflexion(provider, tools, "Where is the Eiffel Tower?", max_episodes=2)
    assert result.episodes_run == 1
    assert result.reflections == []


# --- programmatic -----------------------------------------------------------


def test_demo_batched_lookup_dispatches_two_tools_in_one_turn() -> None:
    result = demo_batched_lookup()
    assert result.steps_taken == 2
    first_assistant = next(m for m in result.messages if m.role == "assistant")
    assert len(first_assistant.tool_calls) == 2
    assert {tc.name for tc in first_assistant.tool_calls} == {"search"}
    tool_messages = [m for m in result.messages if m.role == "tool"]
    assert len(tool_messages) == 3  # two searches plus the finish call's result
