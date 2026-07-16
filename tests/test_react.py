"""Tests for the ReAct pattern.

Deterministic and offline: every test drives `MockProvider` scripts through
the pattern logic directly, with no network call and no API key. Tests cover
loop mechanics (iterations, stop conditions, tool dispatch, error observation
paths), the text-parsing grammar's edge cases, and each variant module.
"""

from __future__ import annotations

import pytest

from agentic_patterns import Completion, Tool, ToolCall, ToolRegistry, get_provider
from patterns.react.compaction import demo_compaction, run_react_with_compaction
from patterns.react.derailment import (
    demo_derailment,
    detect_error_storm,
    detect_no_progress,
    detect_oscillation,
    run_react_with_derailment_recovery,
)
from patterns.react.native_loop import build_native_registry, demo_native, run_native_react
from patterns.react.parser import ActionParseError, parse_action
from patterns.react.programmatic import demo_batched_lookup
from patterns.react.reasoning_loop import NATIVE_SYSTEM_PROMPT, demo_reasoning, run_reasoning_react
from patterns.react.reflexion import demo_reflexion, run_with_reflexion
from patterns.react.scratchpad import Scratchpad, Step
from patterns.react.self_consistency import run_self_consistency
from patterns.react.text_loop import demo_few_shot, run_react
from patterns.react.tree_search import demo_tree_search, run_tree_search
from patterns.react.verify import demo_verify, run_react_with_verification
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


# --- tree_search ------------------------------------------------------------


def test_run_tree_search_two_branch_pick_expands_good_branch_first() -> None:
    tools = build_native_registry()
    provider = get_provider(
        script=[
            {"tool": "search", "args": {"query": "great wall"}},  # branch A: scored higher
            {"tool": "search", "args": {"query": "population of beijing"}},  # branch B: scored lower
            "8",
            "2",
            {"tool": "finish", "args": {"answer": "Beijing"}},  # A's child
            {"tool": "search", "args": {"query": "great wall"}},  # A's decoy child
            "9",
            "1",
        ]
    )
    result = run_tree_search(provider, tools, "goal", branching_factor=2, node_budget=5)
    branch_a_id = result.tree[1].id
    assert result.expansion_order == [0, branch_a_id]
    assert result.answer == "Beijing"


def test_run_tree_search_backtracks_to_higher_scored_sibling() -> None:
    result = demo_tree_search()
    assert result.expansion_order == [0, 1, 2]  # root, A, B: A's children (ids 3, 4) are never expanded
    assert result.answer == "Beijing"
    assert result.stopped_reason == "terminal_best"


def test_run_tree_search_returns_higher_scored_terminal() -> None:
    tools = build_native_registry()
    provider = get_provider(
        script=[
            {"tool": "finish", "args": {"answer": "wrong"}},
            {"tool": "finish", "args": {"answer": "right"}},
            "3",
            "8",
        ]
    )
    result = run_tree_search(provider, tools, "goal", branching_factor=2, node_budget=5)
    assert result.answer == "right"
    assert result.stopped_reason == "terminal_best"


def test_run_tree_search_budget_exhaustion_returns_best_partial() -> None:
    tools = build_native_registry()
    provider = get_provider(script=[{"tool": "search", "args": {"query": "great wall"}}, "5"])
    result = run_tree_search(provider, tools, "goal", branching_factor=1, node_budget=1, on_budget_exhausted="force")
    assert result.stopped_reason == "budget_force"
    assert result.best_node.score == 5.0


def test_run_tree_search_deterministic_across_runs() -> None:
    tools = build_native_registry()

    def build_script() -> list[object]:
        return [
            {"tool": "finish", "args": {"answer": "A"}},
            {"tool": "finish", "args": {"answer": "B"}},
            "4",
            "9",
        ]

    r1 = run_tree_search(get_provider(script=build_script()), tools, "goal", branching_factor=2, node_budget=5)
    r2 = run_tree_search(get_provider(script=build_script()), tools, "goal", branching_factor=2, node_budget=5)
    assert r1.answer == r2.answer == "B"
    assert [n.score for n in r1.tree] == [n.score for n in r2.tree]
    assert r1.expansion_order == r2.expansion_order


# --- compaction ---------------------------------------------------------------


def test_compacting_scratchpad_no_fold_under_threshold() -> None:
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: t.\nAction: search[eiffel tower]",
            "Thought: done.\nAction: Finish[Paris, France]",
        ]
    )
    result = run_react_with_compaction(provider, tools, "Where is the Eiffel Tower?", threshold_chars=10_000)
    assert result.pad.compactions == []
    assert result.answer == "Paris, France"


def test_compacting_scratchpad_single_fold_matches_scripted_summary() -> None:
    result = demo_compaction()
    assert len(result.pad.compactions) == 1
    event = result.pad.compactions[0]
    assert event.folded_count == 2
    assert event.summary == "Beijing is the capital of China, where the Great Wall is located."
    assert len(result.pad.steps) == 3  # 4 raw steps folded down to 1 note + 2 verbatim


def test_compacting_scratchpad_recency_preserved() -> None:
    result = demo_compaction()
    assert result.pad.steps[-1].action_input == "population of beijing"
    assert result.pad.steps[-2].action_input == "eiffel tower"


def test_run_react_with_compaction_answer_parity_vs_high_threshold() -> None:
    tools = build_registry()
    script = [
        "Thought: First, the Great Wall's country.\nAction: search[great wall]",
        "Thought: Now that country's capital.\nAction: search[capital of china]",
        "Thought: Also note the Eiffel Tower's location for the record.\nAction: search[eiffel tower]",
        "Thought: And Beijing's population, for completeness.\nAction: search[population of beijing]",
        "Thought: I have everything I need.\nAction: Finish[Beijing]",
    ]
    provider = get_provider(script=script)
    result = run_react_with_compaction(
        provider, tools, "What is the capital of the country where the Great Wall is located?", threshold_chars=10_000
    )
    assert result.pad.compactions == []
    assert result.answer == "Beijing"


def test_compacting_scratchpad_repeated_folds() -> None:
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="search",
            description="Returns a fixed-length padding string, for exact size control.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            fn=lambda query: "x" * 40,
        )
    )
    provider = get_provider(
        script=[
            "Thought: t.\nAction: search[q1]",
            "Thought: t.\nAction: search[q2]",
            "Thought: t.\nAction: search[q3]",
            "Thought: t.\nAction: search[q4]",
            "s1",
            "Thought: t.\nAction: search[q5]",
            "s2",
            "Thought: done.\nAction: Finish[ok]",
        ]
    )
    result = run_react_with_compaction(provider, tools, "goal", threshold_chars=90, fold_count=2, keep_recent=1, max_iterations=10)
    assert result.answer == "ok"
    assert len(result.pad.compactions) == 2


# --- self_consistency ---------------------------------------------------------


def test_run_self_consistency_majority_over_noise() -> None:
    tools = build_registry()
    scripts = [
        ["Thought: t.\nAction: search[eiffel tower]", "Thought: d.\nAction: Finish[Paris]"],
        ["Thought: t.\nAction: search[eiffel tower]", "Thought: d.\nAction: Finish[Paris]"],
        ["Thought: t.\nAction: search[eiffel tower]", "Thought: d.\nAction: Finish[Rome]"],
    ]
    providers = [get_provider(script=s) for s in scripts]
    result = run_self_consistency(providers, tools, "goal", agreement_margin=3.0)
    assert result.answer == "Paris"
    assert result.votes["Paris"] == 2.0
    assert result.rollouts_run == 3


def test_run_self_consistency_early_stop_skips_remaining_rollout() -> None:
    tools = build_registry()
    scripts = [
        ["Thought: t.\nAction: search[eiffel tower]", "Thought: d.\nAction: Finish[Paris]"],
        ["Thought: t.\nAction: search[eiffel tower]", "Thought: d.\nAction: Finish[Paris]"],
        ["Thought: t.\nAction: search[eiffel tower]", "Thought: d.\nAction: Finish[Rome]"],
    ]
    providers = [get_provider(script=s) for s in scripts]
    result = run_self_consistency(providers, tools, "goal", agreement_margin=2.0)
    assert result.rollouts_run == 2
    assert result.stopped_early is True
    assert result.answer == "Paris"


def test_run_self_consistency_abstain_handling() -> None:
    tools = build_registry()
    scripts = [
        ["Thought: t.\nAction: search[eiffel tower]", "Thought: d.\nAction: Finish[Paris]"],
        ["Thought: t.\nAction: search[nonsense]", "Thought: t2.\nAction: search[nonsense2]"],
    ]
    providers = [get_provider(script=s) for s in scripts]
    result = run_self_consistency(providers, tools, "goal", agreement_margin=5.0, max_iterations=2)
    assert result.rollouts[1].abstained is True
    assert result.answer == "Paris"
    assert result.rollouts_run == 2


def test_run_self_consistency_tie_break_first_to_reach_count() -> None:
    tools = build_registry()
    scripts = [
        ["Thought: t.\nAction: search[eiffel tower]", "Thought: d.\nAction: Finish[Paris]"],
        ["Thought: t.\nAction: search[eiffel tower]", "Thought: d.\nAction: Finish[Rome]"],
    ]
    providers = [get_provider(script=s) for s in scripts]
    result = run_self_consistency(providers, tools, "goal", agreement_margin=5.0)
    assert result.votes == {"Paris": 1.0, "Rome": 1.0}
    assert result.answer == "Paris"


def test_run_self_consistency_soft_vote_flips_winner() -> None:
    tools = build_registry()
    scripts = [
        ["Thought: t.\nAction: search[eiffel tower]", "Thought: d.\nAction: Finish[Paris]"],
        ["Thought: t.\nAction: search[eiffel tower]", "Thought: d.\nAction: Finish[Paris]"],
        ["Thought: t.\nAction: search[eiffel tower]", "Thought: d.\nAction: Finish[Rome]"],
    ]
    providers = [get_provider(script=s) for s in scripts]
    result = run_self_consistency(providers, tools, "goal", agreement_margin=10.0, weights=[0.1, 0.1, 5.0])
    assert result.answer == "Rome"
    assert result.votes["Rome"] == 5.0


# --- verify -------------------------------------------------------------------


def test_run_react_with_verification_first_try_accept() -> None:
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: t.\nAction: search[eiffel tower]",
            "Thought: d.\nAction: Finish[Paris, France]",
            "ACCEPT",
        ]
    )
    result = run_react_with_verification(provider, tools, "Where is the Eiffel Tower?")
    assert result.answer == "Paris, France"
    assert result.first_try is True
    assert result.verify_calls == 1


def test_run_react_with_verification_reject_then_fix() -> None:
    result = demo_verify()
    assert result.answer == "Beijing"
    assert result.first_try is False
    assert result.verify_calls == 2


def test_run_react_with_verification_cycle_cap() -> None:
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: t.\nAction: search[eiffel tower]",
            "Thought: d.\nAction: Finish[wrong]",
            "REJECT: not supported",
            "Thought: retry.\nAction: Finish[still wrong]",
            "REJECT: still not supported",
        ]
    )
    result = run_react_with_verification(provider, tools, "goal", max_verify_cycles=2)
    assert result.stopped_reason == "verify_cap"
    assert result.answer is None
    assert result.verify_calls == 2


def test_run_react_with_verification_reason_threaded_back() -> None:
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: t.\nAction: search[great wall]",
            "Thought: d.\nAction: Finish[Beijing]",
            "REJECT: only shows China, not the capital",
            "Thought: retry.\nAction: search[capital of china]",
            "Thought: d2.\nAction: Finish[Beijing]",
            "ACCEPT",
        ]
    )
    result = run_react_with_verification(provider, tools, "goal")
    assert result.answer == "Beijing"
    retry_call_messages = provider.calls[3]["messages"]
    assert any("verification failed" in m.content for m in retry_call_messages)


# --- derailment -----------------------------------------------------------


def test_detect_oscillation_fires_where_exact_repeat_would_not() -> None:
    pad = Scratchpad()
    pad.add(Step(thought="t", action="search", action_input="a", observation="oa"))
    pad.add(Step(thought="t", action="search", action_input="b", observation="ob"))
    pad.add(Step(thought="t", action="search", action_input="a", observation="oa"))
    pad.add(Step(thought="t", action="search", action_input="b", observation="ob"))
    assert pad.is_repeating() is False
    assert detect_oscillation(pad.steps) is True


def test_detect_no_progress_fires_on_distinct_actions_repeated_observation() -> None:
    steps = [
        Step(thought="t", action="search", action_input="a", observation="no results"),
        Step(thought="t", action="search", action_input="b", observation="no results"),
        Step(thought="t", action="search", action_input="c", observation="no results"),
    ]
    assert detect_no_progress(steps) is True


def test_run_react_with_derailment_recovery_recovers_and_finishes() -> None:
    result = demo_derailment()
    assert result.stopped_reason == "finish"
    assert result.detector_fired == "oscillation"
    assert result.recovery_attempted is True
    assert result.answer == "Paris, France"


def test_run_react_with_derailment_recovery_gives_up_after_second_flag() -> None:
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: t.\nAction: search[monument]",
            "Thought: t.\nAction: search[landmark]",
            "Thought: t.\nAction: search[monument]",
            "Thought: t.\nAction: search[landmark]",
            # recovery nudge injected here; the model repeats the same stuck pattern anyway
            "Thought: t.\nAction: search[monument]",
            "Thought: t.\nAction: search[landmark]",
            "Thought: t.\nAction: search[monument]",
            "Thought: t.\nAction: search[landmark]",
        ]
    )
    result = run_react_with_derailment_recovery(provider, tools, "goal", max_iterations=10)
    assert result.stopped_reason == "derailed"
    assert result.recovery_attempted is True


def test_derailment_detectors_false_positive_guard_on_healthy_trajectory() -> None:
    result = demo_few_shot()
    steps = result.scratchpad.steps
    assert detect_oscillation(steps) is False
    assert detect_no_progress(steps) is False
    assert detect_error_storm(steps) is False


# --- reasoning_loop -------------------------------------------------------


def test_run_reasoning_react_reasoning_carried_verbatim_to_next_call() -> None:
    tools = build_native_registry()
    provider = get_provider(
        script=[
            Completion(
                content="c1",
                reasoning="thinking-one",
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": "eiffel tower"})],
                stop_reason="tool_use",
            ),
            Completion(
                content="c2",
                reasoning="thinking-two",
                tool_calls=[ToolCall(id="call_2", name="finish", arguments={"answer": "Paris"})],
                stop_reason="tool_use",
            ),
        ]
    )
    run_reasoning_react(provider, tools, "goal")
    second_call_messages = provider.calls[1]["messages"]
    assistant_message = next(m for m in second_call_messages if m.role == "assistant")
    assert assistant_message.reasoning == "thinking-one"


def test_run_reasoning_react_no_thought_scaffolding_in_system_prompt() -> None:
    assert "Thought:" not in NATIVE_SYSTEM_PROMPT


def test_run_reasoning_react_thinking_budget_recorded() -> None:
    tools = build_native_registry()
    provider = get_provider(script=[Completion(content="answer", stop_reason="stop")])
    result = run_reasoning_react(provider, tools, "goal", thinking_budget=4096)
    assert result.thinking_budget == 4096


def test_demo_reasoning_matches_native_demo_answer() -> None:
    result = demo_reasoning()
    native_result = demo_native()
    assert result.answer == native_result.answer == "Beijing"
