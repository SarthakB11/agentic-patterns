"""Tests for the planning (plan-then-execute) pattern.

Deterministic and offline: every test drives the pattern logic with
`MockProvider` scripts and local tool functions, no network calls, no API
keys. Tests cover the shared plan/parser/validator mechanics, each executor
variant's control flow, and the replan, offload, and subagent extensions.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from agentic_patterns import MockProvider, Tool, ToolRegistry

from patterns.planning.context_offload import load_state, run_with_offload, save_state
from patterns.planning.dag_executor import run_dag
from patterns.planning.parser import PlanParseError, parse_plan
from patterns.planning.plan import Plan, Step, StepResult, substitute_args, topological_waves
from patterns.planning.plan_and_solve import run_plan_and_solve
from patterns.planning.react_baseline import run_react
from patterns.planning.replanning import ReplanBudgetExceeded, run_with_replanning
from patterns.planning.rewoo import run_rewoo
from patterns.planning.sequential_executor import run_sequential
from patterns.planning.subagent_executor import run_with_subagents
from patterns.planning.todo_list import run_todo_list
from patterns.planning.tools import build_travel_registry
from patterns.planning.validator import PlanValidationError, validate_plan

# --- Parser ----------------------------------------------------------------


def test_parse_plan_builds_exact_step_objects() -> None:
    raw = (
        '[{"id": "step1", "tool": "get_weather", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "step2", "tool": "draft_itinerary",'
        '  "args": {"weather": "$step1"}, "depends_on": ["step1"]}]'
    )
    plan = parse_plan("goal text", raw)
    assert plan.goal == "goal text"
    assert plan.steps == [
        Step(id="step1", tool="get_weather", args={"city": "Paris"}, depends_on=[]),
        Step(id="step2", tool="draft_itinerary", args={"weather": "$step1"}, depends_on=["step1"]),
    ]


def test_parse_plan_rejects_malformed_json() -> None:
    with pytest.raises(PlanParseError):
        parse_plan("goal", "not json at all")


def test_parse_plan_rejects_non_array() -> None:
    with pytest.raises(PlanParseError):
        parse_plan("goal", '{"id": "step1"}')


def test_parse_plan_rejects_missing_required_field() -> None:
    with pytest.raises(PlanParseError):
        parse_plan("goal", '[{"id": "step1", "tool": "get_weather"}]')


# --- Validator ---------------------------------------------------------


def test_validate_plan_rejects_unknown_tool() -> None:
    registry = build_travel_registry()
    plan = Plan(goal="g", steps=[Step(id="s1", tool="not_a_real_tool", args={}, depends_on=[])])
    with pytest.raises(PlanValidationError):
        validate_plan(plan, registry)


def test_validate_plan_rejects_dangling_dependency() -> None:
    registry = build_travel_registry()
    plan = Plan(
        goal="g",
        steps=[Step(id="s1", tool="get_weather", args={"city": "Paris"}, depends_on=["nonexistent"])],
    )
    with pytest.raises(PlanValidationError):
        validate_plan(plan, registry)


def test_validate_plan_rejects_cycle() -> None:
    registry = build_travel_registry()
    plan = Plan(
        goal="g",
        steps=[
            Step(id="a", tool="get_weather", args={"city": "Paris"}, depends_on=["b"]),
            Step(id="b", tool="get_weather", args={"city": "Paris"}, depends_on=["a"]),
        ],
    )
    with pytest.raises(PlanValidationError):
        validate_plan(plan, registry)


def test_validate_plan_accepts_a_valid_plan() -> None:
    registry = build_travel_registry()
    plan = Plan(goal="g", steps=[Step(id="a", tool="get_weather", args={"city": "Paris"}, depends_on=[])])
    validate_plan(plan, registry)  # should not raise


# --- Sequential executor ----------------------------------------------


def test_sequential_executor_runs_tools_in_order_and_uses_scripted_answer() -> None:
    registry = build_travel_registry()
    plan_json = (
        '[{"id": "step1", "tool": "get_weather", "args": {"city": "Lisbon"}, "depends_on": []},'
        ' {"id": "step2", "tool": "search_attractions", "args": {"city": "Lisbon"}, "depends_on": []}]'
    )
    provider = MockProvider([plan_json, "Lisbon is lovely, go see the sights."])
    run = run_sequential(provider, "plan Lisbon", registry)

    assert [r.step_id for r in run.results] == ["step1", "step2"]
    assert run.results[0].output.startswith("Sunny and warm")
    assert run.final_answer == "Lisbon is lovely, go see the sights."
    assert len(provider.calls) == 2  # exactly one planner call, one solver call


def test_sequential_executor_marks_a_failed_step_not_ok() -> None:
    registry = build_travel_registry()
    plan_json = '[{"id": "step1", "tool": "book_hotel", "args": {"city": "Paris", "nights": 2}, "depends_on": []}]'
    provider = MockProvider([plan_json, "final answer"])
    run = run_sequential(provider, "book Paris", registry)

    assert run.results[0].ok is False
    assert run.results[0].output.startswith("ERROR:")


def test_sequential_executor_substitutes_dependency_output() -> None:
    registry = build_travel_registry()
    plan_json = (
        '[{"id": "step1", "tool": "get_weather", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "step2", "tool": "search_attractions", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "step3", "tool": "draft_itinerary",'
        '  "args": {"weather": "$step1", "attractions": "$step2"}, "depends_on": ["step1", "step2"]}]'
    )
    provider = MockProvider([plan_json, "final"])
    run = run_sequential(provider, "plan Paris", registry)
    itinerary = run.results[2].output
    assert "Mild and cloudy" in itinerary
    assert "Louvre Museum" in itinerary


# --- DAG executor --------------------------------------------------------


def test_dag_executor_groups_dependent_step_into_a_later_wave() -> None:
    registry = build_travel_registry()
    plan_json = (
        '[{"id": "a", "tool": "get_weather", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "b", "tool": "search_attractions", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "c", "tool": "draft_itinerary",'
        '  "args": {"weather": "$a", "attractions": "$b"}, "depends_on": ["a", "b"]}]'
    )
    provider = MockProvider([plan_json])
    run = run_dag(provider, "plan Paris", registry)

    assert sorted(run.waves[0]) == ["a", "b"]
    assert run.waves[1] == ["c"]
    # c must have received both upstream outputs, not the raw placeholders
    assert "Mild and cloudy" in run.results["c"].output
    assert "Louvre Museum" in run.results["c"].output


def test_dag_executor_marks_a_failed_step_not_ok() -> None:
    registry = build_travel_registry()
    plan_json = '[{"id": "a", "tool": "book_hotel", "args": {"city": "Paris", "nights": 1}, "depends_on": []}]'
    provider = MockProvider([plan_json])
    run = run_dag(provider, "book Paris", registry)

    assert run.results["a"].ok is False
    assert run.results["a"].output.startswith("ERROR:")


def test_dag_executor_dispatches_independent_steps_concurrently() -> None:
    log: list[str] = []
    lock = threading.Lock()

    def slow_a() -> str:
        with lock:
            log.append("start_a")
        time.sleep(0.05)
        with lock:
            log.append("end_a")
        return "A done"

    def slow_b() -> str:
        with lock:
            log.append("start_b")
        time.sleep(0.05)
        with lock:
            log.append("end_b")
        return "B done"

    registry = ToolRegistry()
    registry.register(Tool(name="slow_a", description="d", parameters={"type": "object", "properties": {}}, fn=slow_a))
    registry.register(Tool(name="slow_b", description="d", parameters={"type": "object", "properties": {}}, fn=slow_b))

    plan_json = '[{"id": "a", "tool": "slow_a", "args": {}, "depends_on": []}, {"id": "b", "tool": "slow_b", "args": {}, "depends_on": []}]'
    provider = MockProvider([plan_json])
    run_dag(provider, "go", registry)

    # both steps must have started before either finished
    assert log.index("start_a") < log.index("end_b")
    assert log.index("start_b") < log.index("end_a")


# --- Replanning ------------------------------------------------------------


def test_replanning_recovers_after_a_scripted_step_failure() -> None:
    registry = build_travel_registry()
    plan_json = '[{"id": "step1", "tool": "book_hotel", "args": {"city": "Paris", "nights": 2}, "depends_on": []}]'
    revised_json = '[{"id": "step2", "tool": "book_hotel", "args": {"city": "Lyon", "nights": 2}, "depends_on": []}]'
    provider = MockProvider([plan_json, revised_json])

    run = run_with_replanning(provider, "book Paris", registry)

    assert run.replans == 1
    assert len(run.results) == 1
    assert run.results[0].step_id == "step2"
    assert "Booked 2 night(s) in Lyon" in run.results[0].output
    # the replanner call carried the failure reason and the (empty) completed summary
    replan_call_messages = provider.calls[1]["messages"]
    assert "No rooms available in Paris" in replan_call_messages[0].content


def test_replanning_cap_halts_a_step_that_always_fails() -> None:
    def always_fail() -> str:
        raise RuntimeError("permanently unavailable")

    registry = ToolRegistry()
    registry.register(
        Tool(name="always_fail", description="d", parameters={"type": "object", "properties": {}}, fn=always_fail)
    )
    plan_json = '[{"id": "s1", "tool": "always_fail", "args": {}, "depends_on": []}]'
    replan_json = '[{"id": "s2", "tool": "always_fail", "args": {}, "depends_on": []}]'
    provider = MockProvider([plan_json, replan_json])

    with pytest.raises(ReplanBudgetExceeded):
        run_with_replanning(provider, "goal", registry, max_replans=1)

    # exactly the planner call plus one replan call; the loop did not spin forever
    assert len(provider.calls) == 2


def test_replanning_triggers_on_an_invalidating_observation_not_only_a_failure() -> None:
    def scout_weather() -> str:
        return "Storm warning: heavy winds expected"

    def outdoor_tour() -> str:
        return "walked the outdoor market"  # should never run: plan gets replaced

    def indoor_museum() -> str:
        return "visited the National Museum"

    registry = ToolRegistry()
    registry.register(Tool(name="scout_weather", description="d", parameters={"type": "object", "properties": {}}, fn=scout_weather))
    registry.register(Tool(name="outdoor_tour", description="d", parameters={"type": "object", "properties": {}}, fn=outdoor_tour))
    registry.register(Tool(name="indoor_museum", description="d", parameters={"type": "object", "properties": {}}, fn=indoor_museum))

    plan_json = (
        '[{"id": "s1", "tool": "scout_weather", "args": {}, "depends_on": []},'
        ' {"id": "s2", "tool": "outdoor_tour", "args": {}, "depends_on": []}]'
    )
    revised_json = '[{"id": "s3", "tool": "indoor_museum", "args": {}, "depends_on": []}]'
    provider = MockProvider([plan_json, revised_json])

    run = run_with_replanning(provider, "plan a day out", registry)

    assert run.replans == 1
    step_ids = [r.step_id for r in run.results]
    assert step_ids == ["s1", "s3"]  # the invalidating step is kept; s2 never ran
    assert all(r.step_id != "s2" for r in run.results)


# --- ReWOO -------------------------------------------------------------


def test_rewoo_uses_exactly_two_model_calls_regardless_of_tool_count() -> None:
    registry = build_travel_registry()
    blueprint_json = (
        '[{"id": "E1", "tool": "get_weather", "args": {"city": "Lisbon"}, "depends_on": []},'
        ' {"id": "E2", "tool": "search_attractions", "args": {"city": "Lisbon"}, "depends_on": []},'
        ' {"id": "E3", "tool": "estimate_hotel_cost", "args": {"city": "Lisbon", "nights": 4}, "depends_on": []}]'
    )
    provider = MockProvider([blueprint_json, "final synthesis"])
    run = run_rewoo(provider, "gather Lisbon evidence", registry)

    assert run.model_calls == 2
    assert len(provider.calls) == 2
    assert len(run.evidence) == 3
    assert run.final_answer == "final synthesis"


def test_rewoo_marks_a_failed_evidence_step_not_ok() -> None:
    registry = build_travel_registry()
    blueprint_json = '[{"id": "E1", "tool": "book_hotel", "args": {"city": "Paris", "nights": 1}, "depends_on": []}]'
    provider = MockProvider([blueprint_json, "final"])
    run = run_rewoo(provider, "book Paris", registry)

    assert run.evidence[0].ok is False
    assert run.evidence[0].output.startswith("ERROR:")


def test_rewoo_substitutes_hash_prefixed_evidence_placeholders() -> None:
    registry = build_travel_registry()
    blueprint_json = (
        '[{"id": "E1", "tool": "get_weather", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "E2", "tool": "search_attractions", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "E3", "tool": "draft_itinerary",'
        '  "args": {"weather": "#E1", "attractions": "#E2"}, "depends_on": ["E1", "E2"]}]'
    )
    provider = MockProvider([blueprint_json, "final"])
    run = run_rewoo(provider, "plan Paris", registry)
    itinerary = run.evidence[2].output
    assert "Mild and cloudy" in itinerary
    assert "Louvre Museum" in itinerary


# --- ReAct baseline vs. plan-then-execute -----------------------------


def test_react_baseline_uses_more_model_calls_than_plan_then_execute() -> None:
    registry = build_travel_registry()
    goal = "What's the weather and attractions in Lisbon?"

    seq_plan_json = (
        '[{"id": "step1", "tool": "get_weather", "args": {"city": "Lisbon"}, "depends_on": []},'
        ' {"id": "step2", "tool": "search_attractions", "args": {"city": "Lisbon"}, "depends_on": []}]'
    )
    seq_provider = MockProvider([seq_plan_json, "Lisbon final answer"])
    seq_run = run_sequential(seq_provider, goal, registry)

    react_script = [
        {"tool": "get_weather", "args": {"city": "Lisbon"}},
        {"tool": "search_attractions", "args": {"city": "Lisbon"}},
        "Lisbon final answer",
    ]
    react_provider = MockProvider(react_script)
    react_run = run_react(react_provider, goal, registry)

    assert len(react_provider.calls) > len(seq_provider.calls)
    assert len(seq_provider.calls) == 2
    assert len(react_provider.calls) == 3
    # both gathered the same underlying facts from the same tools
    assert any("Sunny and warm" in r.output for r in seq_run.results)
    assert any(m.role == "tool" and "Sunny and warm" in m.content for m in react_run.transcript)


def test_react_baseline_raises_when_max_steps_is_exceeded() -> None:
    registry = build_travel_registry()
    # a script that never stops calling tools
    script = [{"tool": "get_weather", "args": {"city": "Paris"}}] * 5
    provider = MockProvider(script)
    with pytest.raises(RuntimeError):
        run_react(provider, "goal", registry, max_steps=3)


# --- Plan-and-Solve --------------------------------------------------------


def test_plan_and_solve_makes_a_single_model_call_with_no_tools() -> None:
    provider = MockProvider(["Plan: divide then subtract. Solve: 12 - 5 = 7. Answer: 7."])
    run = run_plan_and_solve(provider, "How many boxes are left?", plus=True)
    assert run.response.startswith("Plan:")
    assert len(provider.calls) == 1
    assert provider.calls[0]["tools"] is None


# --- Todo-list in-context planning --------------------------------------


def test_todo_list_offers_write_todos_and_ends_with_everything_done() -> None:
    registry = build_travel_registry()
    script = [
        {"tool": "write_todos", "args": {"items": [{"id": "t1", "text": "check weather", "status": "in_progress"}]}},
        {"tool": "get_weather", "args": {"city": "Lyon"}},
        {"tool": "write_todos", "args": {"items": [{"id": "t1", "text": "check weather", "status": "done"}]}},
        "Lyon looks fine.",
    ]
    provider = MockProvider(script)
    run = run_todo_list(provider, "plan Lyon", registry)

    assert "write_todos" in {t["name"] for t in provider.calls[0]["tools"]}
    assert run.state.todos[0].status == "done"
    assert run.final_answer == "Lyon looks fine."
    assert run.model_calls == 4


# --- Context offload / resumability -------------------------------------


def test_offload_fresh_run_calls_planner_once_and_checkpoints(tmp_path: Path) -> None:
    registry = build_travel_registry()
    plan_json = '[{"id": "step1", "tool": "get_weather", "args": {"city": "Lyon"}, "depends_on": []}]'
    provider = MockProvider([plan_json])
    state_path = tmp_path / "state.json"

    run = run_with_offload(provider, "plan Lyon", registry, state_path)

    assert run.resumed is False
    assert run.planner_calls == 1
    assert state_path.exists()
    assert "step1" in run.results


def test_offload_resumed_run_never_calls_the_planner(tmp_path: Path) -> None:
    registry = build_travel_registry()
    plan = Plan(
        goal="plan Lyon",
        steps=[
            Step(id="step1", tool="get_weather", args={"city": "Lyon"}, depends_on=[]),
            Step(id="step2", tool="estimate_hotel_cost", args={"city": "Lyon", "nights": 2}, depends_on=[]),
        ],
    )
    state_path = tmp_path / "state.json"
    save_state(state_path, plan, {"step1": StepResult(step_id="step1", output="Partly cloudy, high of 20C")})

    provider = MockProvider([])  # would raise MockScriptExhausted if ever called
    run = run_with_offload(provider, plan.goal, registry, state_path)

    assert run.resumed is True
    assert run.planner_calls == 0
    assert len(provider.calls) == 0
    assert run.results["step1"].output == "Partly cloudy, high of 20C"
    assert "night(s)" in run.results["step2"].output

    reloaded_plan, reloaded_results = load_state(state_path)
    assert reloaded_plan.goal == "plan Lyon"
    assert set(reloaded_results) == {"step1", "step2"}


def test_offload_resume_reexecutes_a_failed_step_instead_of_skipping_it(tmp_path: Path) -> None:
    # Start at 1 to represent the pre-crash attempt that already failed and
    # produced the checkpointed "ERROR: ..." result below; the resumed run
    # should make exactly one more call, which succeeds.
    attempts = {"count": 1}

    def flaky_lookup() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient outage")
        return "sunny and clear"

    def echo_weather(weather: str) -> str:
        return f"itinerary uses: {weather}"

    registry = ToolRegistry()
    registry.register(
        Tool(name="flaky_lookup", description="d", parameters={"type": "object", "properties": {}}, fn=flaky_lookup)
    )
    registry.register(
        Tool(
            name="echo_weather",
            description="d",
            parameters={
                "type": "object",
                "properties": {"weather": {"type": "string"}},
                "required": ["weather"],
            },
            fn=echo_weather,
        )
    )
    plan = Plan(
        goal="plan with a flaky step",
        steps=[
            Step(id="step1", tool="flaky_lookup", args={}, depends_on=[]),
            Step(id="step2", tool="echo_weather", args={"weather": "$step1"}, depends_on=["step1"]),
        ],
    )
    state_path = tmp_path / "state.json"
    # Simulate a crash that happened right after step1's failed attempt was
    # checkpointed, before step2 ever ran.
    save_state(
        state_path,
        plan,
        {"step1": StepResult(step_id="step1", output="ERROR: transient outage", ok=False)},
    )

    provider = MockProvider([])  # resume never calls the planner
    run = run_with_offload(provider, plan.goal, registry, state_path)

    assert attempts["count"] == 2  # step1 was re-executed exactly once on resume
    assert run.results["step1"].ok is True
    assert run.results["step1"].output == "sunny and clear"
    # the retried output, not the stale error text, must flow into step2
    assert run.results["step2"].output == "itinerary uses: sunny and clear"

    reloaded_plan, reloaded_results = load_state(state_path)
    assert reloaded_results["step1"].ok is True


# --- Subagent-per-subtask -------------------------------------------------


def test_subagent_executor_keeps_parent_context_to_compact_strings() -> None:
    registry = build_travel_registry()
    plan_json = (
        '[{"id": "step1", "tool": "get_weather", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "step2", "tool": "search_attractions", "args": {"city": "Paris"}, "depends_on": []}]'
    )
    parent_provider = MockProvider([plan_json])
    child_scripts = {
        "step1": [
            {"tool": "get_weather", "args": {"city": "Paris"}},
            "Paris weather summary.",
        ],
        "step2": [
            {"tool": "search_attractions", "args": {"city": "Paris"}},
            "Paris attractions summary.",
        ],
    }

    def subagent_provider_for(step: Step) -> MockProvider:
        return MockProvider(child_scripts[step.id])

    run = run_with_subagents(parent_provider, "plan Paris", registry, subagent_provider_for)

    assert len(parent_provider.calls) == 1  # only the plan; child turns never reach the parent's provider
    assert run.parent_context == ["step1: Paris weather summary.", "step2: Paris attractions summary."]
    assert all(report.child_message_count >= 2 for report in run.reports)


# --- Shared plan helpers -------------------------------------------------


def test_substitute_args_replaces_only_known_placeholders() -> None:
    results = {"a": StepResult(step_id="a", output="sunny")}
    args = {"city": "Paris", "note": "weather is $a today", "nested": {"inner": "$missing stays"}}
    substituted = substitute_args(args, results)
    assert substituted["note"] == "weather is sunny today"
    assert substituted["nested"]["inner"] == "$missing stays"
    assert substituted["city"] == "Paris"


def test_substitute_args_does_not_corrupt_a_prefix_id_collision() -> None:
    # "step1" is a string-prefix of "step10"; a naive str.replace would
    # substitute step1's output inside the "$step10" placeholder and leave
    # a stray "0" behind.
    results = {
        "step1": StepResult(step_id="step1", output="ONE"),
        "step10": StepResult(step_id="step10", output="TEN"),
    }
    args = {"solo": "$step10", "both": "$step1 then $step10"}
    substituted = substitute_args(args, results)
    assert substituted["solo"] == "TEN"
    assert substituted["both"] == "ONE then TEN"


def test_substitute_args_does_not_corrupt_hash_prefixed_id_collision() -> None:
    # Same collision, but for ReWOO's "#E1" / "#E10" notation.
    results = {
        "E1": StepResult(step_id="E1", output="EV1"),
        "E10": StepResult(step_id="E10", output="EV10"),
    }
    args = {"solo": "#E10", "both": "#E1 then #E10"}
    substituted = substitute_args(args, results, prefix="#")
    assert substituted["solo"] == "EV10"
    assert substituted["both"] == "EV1 then EV10"


def test_topological_waves_orders_a_diamond_dependency() -> None:
    steps = [
        Step(id="c", tool="t", args={}, depends_on=["a", "b"]),
        Step(id="a", tool="t", args={}, depends_on=[]),
        Step(id="b", tool="t", args={}, depends_on=["a"]),
    ]
    waves = topological_waves(steps)
    assert [s.id for s in waves[0]] == ["a"]
    assert [s.id for s in waves[1]] == ["b"]
    assert [s.id for s in waves[2]] == ["c"]
