"""Tests for the multi-agent orchestration pattern.

Deterministic and offline: every test drives `MockProvider` scripts through
the pattern's own modules, with no network call and no API key. Test names
that reference a MAST failure category (arXiv:2503.13657) tie a specific
assertion back to that category rather than relying on one generic
error-isolation test.
"""

from __future__ import annotations

import pytest

from agentic_patterns import MockProvider, Provider

from patterns.multi_agent import (
    agent_card,
    aggregation,
    debate,
    economics,
    failure_attribution,
    group_chat,
    handoff,
    hierarchical,
    maker_checker,
    magentic,
    supervisor,
)
from patterns.multi_agent.state import SharedState, TraceEntry
from patterns.multi_agent.worker import Subtask, Worker, WorkerResult, dispatch_parallel, run_worker


class _RaisingProvider(Provider):
    """A `Provider` whose `complete()` always raises, for error-isolation tests."""

    def complete(self, messages, *, tools=None, system=None, temperature=0.0, max_tokens=1024):
        raise RuntimeError("simulated provider failure")


# --- worker.py: Subtask, WorkerResult, run_worker, dispatch_parallel -------


def test_subtask_to_prompt_includes_objective_format_and_boundaries() -> None:
    subtask = Subtask(
        id="s1", role="analyst", objective="Summarize X", output_format="one sentence",
        boundaries=["Do not mention Y"],
    )
    prompt = subtask.to_prompt()
    assert "Summarize X" in prompt
    assert "one sentence" in prompt
    assert "Do not mention Y" in prompt


def test_run_worker_returns_ok_result_with_no_tools() -> None:
    subtask = Subtask("s1", "analyst", "Summarize X", "one sentence")
    worker = Worker("analyst", "You are an analyst.", MockProvider(script=["X is short for Xylophone Corp."]))
    result = run_worker(worker, subtask)
    assert result.status == "ok"
    assert result.subtask_id == "s1"
    assert result.content == "X is short for Xylophone Corp."


def test_run_worker_isolates_a_raised_exception() -> None:
    """MAST: task verification. A failing worker must not crash the run; it
    must return a labeled error result a caller can check before trusting it.
    """
    subtask = Subtask("s1", "analyst", "Summarize X", "one sentence")
    worker = Worker("analyst", "You are an analyst.", _RaisingProvider())
    result = run_worker(worker, subtask)
    assert result.status == "error"
    assert "simulated provider failure" in result.content


def test_dispatch_parallel_invokes_every_independent_worker() -> None:
    subtasks = [Subtask(f"s{i}", "analyst", f"Task {i}", "one sentence") for i in range(3)]
    workers = [Worker("analyst", "You are an analyst.", MockProvider(script=[f"answer {i}"])) for i in range(3)]
    results = dispatch_parallel(list(zip(workers, subtasks)))
    assert len(results) == 3
    assert [r.subtask_id for r in results] == ["s0", "s1", "s2"]


def test_dispatch_parallel_preserves_input_order_not_completion_order() -> None:
    subtasks = [Subtask("a", "r", "obj a", "fmt"), Subtask("b", "r", "obj b", "fmt")]
    workers = [Worker("r", "sys", MockProvider(script=["reply a"])), Worker("r", "sys", MockProvider(script=["reply b"]))]
    results = dispatch_parallel(list(zip(workers, subtasks)))
    assert [r.subtask_id for r in results] == ["a", "b"]


def test_dispatch_parallel_isolates_one_bad_worker_from_the_rest() -> None:
    subtasks = [Subtask("good", "r", "ok task", "fmt"), Subtask("bad", "r", "bad task", "fmt")]
    workers = [Worker("r", "sys", MockProvider(script=["fine"])), Worker("r", "sys", _RaisingProvider())]
    results = dispatch_parallel(list(zip(workers, subtasks)))
    by_id = {r.subtask_id: r for r in results}
    assert by_id["good"].status == "ok"
    assert by_id["bad"].status == "error"


# --- state.py: single-writer rule, checkpoint/resume, trace ---------------


def test_write_result_rejects_non_supervisor_writer() -> None:
    """The single-writer rule: only SharedState.WRITER_ROLE may write."""
    state = SharedState(goal="g")
    with pytest.raises(PermissionError):
        state.write_result("market_researcher", "market", "some worker proposal")


def test_write_result_succeeds_for_the_writer_role() -> None:
    state = SharedState(goal="g")
    state.write_result(SharedState.WRITER_ROLE, "final_report", "the answer")
    assert state.results["final_report"] == "the answer"
    assert "final_report" in state.completed_subtask_ids


def test_checkpoint_resume_round_trips_completed_work() -> None:
    state = SharedState(goal="g")
    state.write_result(SharedState.WRITER_ROLE, "market", "market finding")
    resumed = SharedState.resume(state.checkpoint())
    assert resumed.completed_subtask_ids == {"market"}
    assert resumed.results["market"] == "market finding"
    assert resumed.trace[-1].action == "resume"


def test_trace_records_events_in_sequence() -> None:
    state = SharedState(goal="g")
    state.record("supervisor", "decompose", "2 subtasks")
    state.write_result(SharedState.WRITER_ROLE, "k", "v")
    assert [e.seq for e in state.trace] == [1, 2]
    assert state.trace[0].action == "decompose"
    assert state.trace[1].action == "write_result"


# --- supervisor.py: canonical decompose -> dispatch -> synthesize ---------


def test_decompose_parses_delegate_subtasks_tool_call() -> None:
    state, results = supervisor.run_supervisor_demo()
    subtask_ids = {r.subtask_id for r in results}
    assert subtask_ids == {"market", "tech", "risk"}


def test_decompose_raises_without_a_delegate_subtasks_call() -> None:
    provider = MockProvider(script=["I decline to use a tool."])
    with pytest.raises(ValueError):
        supervisor.decompose(provider, "some goal")


def test_supervisor_demo_writes_every_worker_proposal_and_the_final_report() -> None:
    state, results = supervisor.run_supervisor_demo()
    assert set(state.results.keys()) == {"market", "tech", "risk", "final_report"}
    assert len(results) == 3
    # The write order in the trace follows collection order: workers first, synthesis last.
    write_keys = [e.detail.split(":")[0] for e in state.trace if e.action == "write_result"]
    assert write_keys[-1] == "final_report"


def test_resume_demo_skips_the_already_completed_subtask() -> None:
    resumed, assignments = supervisor.run_resume_demo()
    dispatched_ids = [subtask.id for _, subtask in assignments]
    assert "market" not in dispatched_ids
    assert set(dispatched_ids) == {"tech", "risk"}
    assert resumed.completed_subtask_ids == {"market"}


# --- aggregation.py: majority vote and model synthesis ---------------------


def test_majority_vote_picks_the_agreed_answer_over_the_lone_dissent() -> None:
    results = [
        WorkerResult("v1", "reviewer_a", "ok", "yes"),
        WorkerResult("v2", "reviewer_b", "ok", "yes"),
        WorkerResult("v3", "reviewer_c", "ok", "no"),
    ]
    vote = aggregation.majority_vote(results)
    assert vote.winner == "yes"
    assert vote.counts == {"yes": 2, "no": 1}
    assert vote.unanimous is False


def test_majority_vote_ignores_errored_workers() -> None:
    """MAST: task verification. An errored worker must not silently count
    as a valid vote; it should be excluded from the tally, not misread.
    """
    results = [
        WorkerResult("v1", "reviewer_a", "ok", "yes"),
        WorkerResult("v2", "reviewer_b", "error", "ERROR: timeout"),
    ]
    vote = aggregation.majority_vote(results)
    assert vote.winner == "yes"
    assert vote.counts == {"yes": 1}


def test_majority_vote_raises_with_no_ok_results() -> None:
    results = [WorkerResult("v1", "reviewer_a", "error", "ERROR: timeout")]
    with pytest.raises(ValueError):
        aggregation.majority_vote(results)


def test_model_synthesize_sends_every_finding_to_the_provider() -> None:
    results = [
        WorkerResult("t1", "timeline_analyst", "ok", "started at 14:02"),
        WorkerResult("t2", "root_cause_analyst", "ok", "connection pool exhausted"),
    ]
    provider = MockProvider(script=["combined summary"])
    summary = aggregation.model_synthesize(provider, results, goal="postmortem", system="sys")
    assert summary == "combined summary"
    sent = provider.calls[0]["messages"][0].content
    assert "timeline_analyst" in sent and "started at 14:02" in sent
    assert "root_cause_analyst" in sent and "connection pool exhausted" in sent


def test_run_majority_vote_demo_matches_the_scripted_two_to_one_split() -> None:
    vote = aggregation.run_majority_vote_demo()
    assert vote.winner == "yes"
    assert vote.counts["yes"] == 2


# --- handoff.py: A2A-style lifecycle, handoff vs. subagent -----------------


def test_handoff_transfers_control_and_payload_reaches_specialist_intact() -> None:
    triage = MockProvider(script=["ROUTE: billing_specialist\nREASON: duplicate charge"])
    specialist = MockProvider(script=["Refund issued for the duplicate charge."])
    task = handoff.run_handoff_demo(triage, specialist)
    assert task.to_agent == "billing_specialist"
    assert task.status == "completed"
    assert task.payload == "Refund issued for the duplicate charge."
    assert [h.split(":")[0] for h in task.history] == ["in_progress", "completed"]


def test_handoff_does_not_call_the_triage_agent_again() -> None:
    """A true handoff transfers control permanently; the triage provider is
    called exactly once, never resumed after the specialist takes over.
    """
    triage = MockProvider(script=["ROUTE: billing_specialist\nREASON: duplicate charge"])
    specialist = MockProvider(script=["Refund issued."])
    handoff.run_handoff_demo(triage, specialist)
    assert len(triage.calls) == 1


def test_subagent_variant_returns_control_to_the_parent() -> None:
    """Unlike handoff, the parent provider is called both before and after
    the child runs, since control returns to it when the child completes.
    """
    parent = MockProvider(script=["QUESTION: what changed?", "Final note using the child's answer."])
    child = MockProvider(script=["Latency regression was fixed."])
    task, final_answer = handoff.run_subagent_demo(parent, child)
    assert task.status == "completed"
    assert len(parent.calls) == 2
    assert final_answer == "Final note using the child's answer."


# --- group_chat.py: chat manager picks speakers, turn cap guards loops -----


def test_group_chat_stops_when_the_manager_says_stop() -> None:
    result = group_chat.run_group_chat_demo()
    assert result.stop_reason == "manager_stopped"
    assert [t.speaker for t in result.turns] == ["engineer", "skeptic", "product_manager"]


def test_group_chat_cap_stops_a_manager_that_never_says_stop() -> None:
    """MAST: inter-agent misalignment. Two agents keep restating their own
    position instead of converging; the hard turn cap is what actually ends
    the run, not the manager's judgment.
    """
    result = group_chat.run_group_chat_cap_demo()
    assert result.stop_reason == "max_turns"
    assert len(result.turns) == 4


def test_group_chat_raises_on_an_unknown_participant_name() -> None:
    manager = MockProvider(script=["NEXT: nobody"])
    participants = {"engineer": MockProvider(script=["hello"])}
    with pytest.raises(ValueError):
        group_chat.run_group_chat(manager, participants, "topic")


def test_group_chat_placeholder_only_appears_before_the_first_turn() -> None:
    """The "(no turns yet)" placeholder must not linger once real turns exist,
    or every later prompt would misrepresent an active discussion as empty.
    """
    manager = MockProvider(script=["NEXT: engineer", "NEXT: engineer", "STOP"])
    engineer = MockProvider(script=["first point", "second point"])
    result = group_chat.run_group_chat(manager, {"engineer": engineer}, "topic")

    assert len(result.turns) == 2
    manager_prompts = [call["messages"][0].content for call in manager.calls]
    assert "(no turns yet)" in manager_prompts[0]
    assert "(no turns yet)" not in manager_prompts[1]
    assert "(no turns yet)" not in manager_prompts[2]
    assert "first point" in manager_prompts[1]

    engineer_prompts = [call["messages"][0].content for call in engineer.calls]
    assert "(no turns yet)" in engineer_prompts[0]
    assert "(no turns yet)" not in engineer_prompts[1]
    assert "first point" in engineer_prompts[1]


# --- debate.py: converge, or fall back to majority at the round cap -------


def test_debate_converges_before_the_round_cap() -> None:
    result = debate.run_debate_convergence_demo()
    assert result.stop_reason == "converged"
    assert result.final_answer == "0.05"
    assert len(result.rounds) == 2  # would be 3 without early convergence


def test_debate_falls_back_to_majority_at_the_round_cap() -> None:
    result = debate.run_debate_fallback_demo()
    assert result.stop_reason == "max_rounds"
    assert result.final_answer == "Go"
    assert len(result.rounds) == 2


# --- maker_checker.py: iteration cap and its fallback -----------------------


def test_maker_checker_runs_exactly_three_maker_turns_then_approves() -> None:
    result = maker_checker.run_maker_checker_demo()
    assert len(result.attempts) == 3
    assert result.approved is True
    assert result.stop_reason == "approved"
    assert result.checks[0].passed is False
    assert result.checks[1].passed is False
    assert result.checks[2].passed is True


def test_maker_checker_cap_returns_the_fallback_not_the_rejected_attempt() -> None:
    result = maker_checker.run_cap_demo()
    assert result.stop_reason == "cap_reached"
    assert result.approved is False
    assert result.final_output.startswith("Escalated to legal")
    assert result.final_output not in result.attempts


def test_maker_checker_stops_at_a_cap_below_the_turns_needed() -> None:
    maker = MockProvider(script=["draft 1", "draft 2"])
    checker = MockProvider(script=["RESULT: FAIL\nFEEDBACK: not there yet", "RESULT: FAIL\nFEEDBACK: still not there"])
    result = maker_checker.run_maker_checker(maker, checker, "task", max_attempts=2, fallback="use fallback")
    assert result.stop_reason == "cap_reached"
    assert result.final_output == "use fallback"
    assert len(result.attempts) == 2


# --- hierarchical.py: supervisor of supervisors -----------------------------


def test_hierarchical_demo_nests_two_teams_under_one_top_supervisor() -> None:
    state, team_results = hierarchical.run_hierarchical_demo()
    assert set(team_results.keys()) == {"frontend_lead", "backend_lead"}
    assert len(team_results["frontend_lead"].worker_results) == 2
    assert len(team_results["backend_lead"].worker_results) == 2
    assert set(state.results.keys()) == {"frontend_lead", "backend_lead", "final_report"}


def test_hierarchical_leads_do_not_write_shared_state_directly() -> None:
    """Only the top supervisor writes; a lead's `TeamResult` is a proposal.
    Confirmed by construction: `run_mid_supervisor` never receives the
    `SharedState` object, so it has no way to call `write_result`.
    """
    import inspect

    params = inspect.signature(hierarchical.run_mid_supervisor).parameters
    assert "state" not in params and "shared_state" not in params


# --- corrections: normalize_answer shared by aggregation and debate --------


def test_majority_vote_normalizes_case_before_tallying() -> None:
    """Correction: exact-string comparison would split "Yes" and "yes" into
    two separate one-vote entries instead of tallying them as agreement.
    """
    results = [
        WorkerResult("v1", "reviewer_a", "ok", "Yes"),
        WorkerResult("v2", "reviewer_b", "ok", "yes"),
        WorkerResult("v3", "reviewer_c", "ok", "no"),
    ]
    vote = aggregation.majority_vote(results)
    assert vote.counts == {"yes": 2, "no": 1}
    assert vote.winner == "Yes"


def test_debate_converges_on_differently_formatted_equivalent_answers() -> None:
    """Correction: "$0.05" and "0.05" mean the same answer; without
    `normalize_answer` this would not converge until the round cap.
    """
    agents = {
        "agent_a": MockProvider(script=["The ball costs the leftover amount. ANSWER: $0.05"]),
        "agent_b": MockProvider(script=["Solving the algebra gives ANSWER: 0.05"]),
    }
    result = debate.run_debate(agents, "How much does the ball cost?", max_rounds=3)
    assert result.stop_reason == "converged"
    assert len(result.rounds) == 1


# --- failure_attribution.py: MAST taxonomy and the three attribution strategies


def test_mast_modes_table_has_fourteen_modes_in_three_categories() -> None:
    assert len(failure_attribution.MAST_MODES) == 14
    ids = [m.id for m in failure_attribution.MAST_MODES.values()]
    assert len(set(ids)) == 14
    categories = {m.category for m in failure_attribution.MAST_MODES.values()}
    assert categories == {
        failure_attribution.CATEGORY_SPECIFICATION,
        failure_attribution.CATEGORY_INTER_AGENT,
        failure_attribution.CATEGORY_VERIFICATION,
    }


def test_attribute_all_at_once_parses_verdict_and_resolves_category_from_table() -> None:
    steps = [TraceEntry(1, "supervisor", "decompose"), TraceEntry(2, "tech_researcher", "produce")]
    provider = MockProvider(script=["AGENT: tech_researcher\nSTEP: 2\nMODE: FM-2.3"])
    attribution = failure_attribution.attribute_all_at_once(provider, "goal", steps)
    assert attribution.agent == "tech_researcher"
    assert attribution.step == 2
    assert attribution.mode_id == "FM-2.3"
    assert attribution.category == failure_attribution.CATEGORY_INTER_AGENT
    assert attribution.strategy == "all_at_once"


def test_attribute_step_by_step_stops_at_the_first_yes() -> None:
    steps = [TraceEntry(i, f"agent_{i}", "produce") for i in range(1, 6)]
    provider = MockProvider(script=["VERDICT: NO", "VERDICT: NO", "VERDICT: YES\nMODE: FM-2.3"])
    attribution = failure_attribution.attribute_step_by_step(provider, "goal", steps)
    assert len(provider.calls) == 3
    assert attribution.step == 3
    assert attribution.agent == "agent_3"


def test_attribute_binary_search_call_count_is_logarithmic_not_linear() -> None:
    steps = [TraceEntry(i, f"agent_{i}", "produce") for i in range(1, 9)]
    provider = MockProvider(
        script=["HALF: second\nMODE: FM-1.1", "HALF: first\nMODE: FM-1.1", "HALF: second\nMODE: FM-1.1"]
    )
    attribution = failure_attribution.attribute_binary_search(provider, "goal", steps)
    assert len(provider.calls) == 3
    assert len(provider.calls) < len(steps)
    assert attribution.step == 6


def test_attribute_rejects_a_mode_id_not_in_the_table() -> None:
    steps = [TraceEntry(1, "supervisor", "decompose")]
    provider = MockProvider(script=["AGENT: supervisor\nSTEP: 1\nMODE: FM-9.9"])
    with pytest.raises(ValueError):
        failure_attribution.attribute_all_at_once(provider, "goal", steps)


def test_failure_attribution_demo_all_three_strategies_name_the_same_agent() -> None:
    result = failure_attribution.run_failure_attribution_demo()
    agents = {a.agent for a in result.values()}
    assert agents == {"market_researcher"}
    steps_named = {a.step for a in result.values()}
    assert len(steps_named) > 1  # they differ on the exact step


# --- economics.py: token accounting and context-isolation comparison ------


def test_economics_multiple_is_greater_than_one() -> None:
    report = economics.run_economics_demo()
    assert report.multiple > 1.0


def test_economics_worker_peaks_smaller_than_single_threaded_peak() -> None:
    report = economics.run_economics_demo()
    assert all(peak < report.single_threaded_peak_context for peak in report.worker_peak_contexts.values())


def test_economics_call_count_matches_tracked_provider_calls() -> None:
    single_provider = MockProvider(script=["market notes", "tech notes", "risk notes", "final report"])
    _, single_tracked = economics.run_single_threaded(supervisor.GOAL, single_provider)
    _, supervisor_tracked, worker_trackeds = economics.run_supervised(supervisor.GOAL)
    report = economics.compute_report(single_tracked, supervisor_tracked, worker_trackeds)
    assert report.single_threaded_call_count == len(single_tracked.calls) == 4
    assert report.supervised_call_count == len(supervisor_tracked.calls) + sum(len(t.calls) for t in worker_trackeds.values())


def test_economics_report_is_deterministic() -> None:
    report_1 = economics.run_economics_demo()
    report_2 = economics.run_economics_demo()
    assert report_1 == report_2


def test_economics_supervised_total_is_additive() -> None:
    report = economics.run_economics_demo()
    assert report.supervised_tokens == report.supervisor_tokens + sum(report.worker_tokens.values())


# --- magentic.py: dual-ledger orchestrator, stall counter, replan ---------


def test_magentic_happy_path_has_zero_replans() -> None:
    orchestrator = MockProvider(
        script=[
            "FACTS:\n- known fact\nGUESSES:\n- a guess\nPLAN:\n- do the one step",
            "DONE: yes\nPROGRESS: yes\nNEXT_AGENT: none\nNEXT_INSTRUCTION: The answer is 42.",
        ]
    )
    result = magentic.run_magentic(orchestrator, {}, "answer the question")
    assert result.replans == 0
    assert result.stop_reason == "completed"
    assert result.answer == "The answer is 42."


def test_magentic_stall_then_recover_replans_exactly_once() -> None:
    orchestrator = MockProvider(
        script=[
            "FACTS:\n- fact\nGUESSES:\n- guess\nPLAN:\n- try approach A",
            "DONE: no\nPROGRESS: no\nNEXT_AGENT: worker_a\nNEXT_INSTRUCTION: try A",
            "DONE: no\nPROGRESS: no\nNEXT_AGENT: worker_a\nNEXT_INSTRUCTION: try B",
            "FACTS:\n- fact\n- A and B both failed\nGUESSES:\n- try C instead\nPLAN:\n- try approach C",
            "DONE: yes\nPROGRESS: yes\nNEXT_AGENT: worker_a\nNEXT_INSTRUCTION: found it via C",
        ]
    )
    worker = Worker("worker_a", "sys", MockProvider(script=["A failed", "B failed"]))
    result = magentic.run_magentic(orchestrator, {"worker_a": worker}, "find it", stall_threshold=2)
    assert result.replans == 1
    assert result.stop_reason == "completed"
    assert result.answer == "found it via C"


def test_magentic_stall_counter_resets_on_progress() -> None:
    """no, yes, no should never trip a threshold-2 stall, since a `PROGRESS:
    yes` step resets the counter back to zero instead of it accumulating.
    """
    orchestrator = MockProvider(
        script=[
            "FACTS:\n- fact\nGUESSES:\n- guess\nPLAN:\n- steps",
            "DONE: no\nPROGRESS: no\nNEXT_AGENT: worker_a\nNEXT_INSTRUCTION: step A",
            "DONE: no\nPROGRESS: yes\nNEXT_AGENT: worker_a\nNEXT_INSTRUCTION: step B",
            "DONE: no\nPROGRESS: no\nNEXT_AGENT: worker_a\nNEXT_INSTRUCTION: step C",
            "DONE: yes\nPROGRESS: yes\nNEXT_AGENT: worker_a\nNEXT_INSTRUCTION: done",
        ]
    )
    worker = Worker("worker_a", "sys", MockProvider(script=["a", "b", "c"]))
    result = magentic.run_magentic(orchestrator, {"worker_a": worker}, "goal", stall_threshold=2)
    assert result.replans == 0
    assert result.stop_reason == "completed"


def test_magentic_replan_cap_returns_fallback_not_a_false_success() -> None:
    orchestrator = MockProvider(
        script=[
            "FACTS:\n- fact\nGUESSES:\n- guess\nPLAN:\n- steps",
            "DONE: no\nPROGRESS: no\nNEXT_AGENT: worker_a\nNEXT_INSTRUCTION: step A",
            "FACTS:\n- fact\nGUESSES:\n- guess 2\nPLAN:\n- steps 2",
            "DONE: no\nPROGRESS: no\nNEXT_AGENT: worker_a\nNEXT_INSTRUCTION: step B",
        ]
    )
    worker = Worker("worker_a", "sys", MockProvider(script=["a", "b"]))
    result = magentic.run_magentic(orchestrator, {"worker_a": worker}, "goal", stall_threshold=1, max_replans=1)
    assert result.stop_reason == "replan_cap"
    assert result.answer == magentic.FALLBACK_MESSAGE
    assert result.replans == 1


def test_magentic_replan_rewrites_the_plan() -> None:
    result = magentic.run_magentic_demo()
    assert result.ledger.plan == ["Ask calendar_bot for the room field on the 3pm design review event"]
    assert result.ledger.plan != ["Check Room A", "Check Room B"]


# --- agent_card.py: A2A capability discovery and card-based delegation ----


def _sample_registry() -> agent_card.Registry:
    registry = agent_card.Registry()
    registry.register(agent_card.AgentCard("billing_agent", "Handles refunds.", [agent_card.Skill("billing", ["refund", "invoice", "charge"])]))
    registry.register(agent_card.AgentCard("export_agent", "Exports data.", [agent_card.Skill("export", ["export", "csv"])]))
    return registry


def test_agent_card_select_picks_the_best_skill_match() -> None:
    registry = _sample_registry()
    selection = agent_card.select(registry, {"refund", "invoice", "charge"})
    assert selection.card.name == "billing_agent"
    assert selection.score == 3


def test_agent_card_select_raises_when_no_card_is_capable() -> None:
    registry = _sample_registry()
    with pytest.raises(agent_card.NoCapableAgentError):
        agent_card.select(registry, {"weather", "forecast"})


def test_agent_card_tie_break_is_deterministic_by_name_with_no_model_call() -> None:
    registry = agent_card.Registry()
    registry.register(agent_card.AgentCard("zeta", "d", [agent_card.Skill("s", ["x"])]))
    registry.register(agent_card.AgentCard("alpha", "d", [agent_card.Skill("s", ["x"])]))
    selection = agent_card.select(registry, {"x"})
    assert selection.card.name == "alpha"
    assert selection.tie_broken_by_llm is False


def test_agent_card_tie_break_uses_the_llm_when_provided() -> None:
    registry = agent_card.Registry()
    registry.register(agent_card.AgentCard("zeta", "d", [agent_card.Skill("s", ["x"])]))
    registry.register(agent_card.AgentCard("alpha", "d", [agent_card.Skill("s", ["x"])]))
    tie_break_provider = MockProvider(script=["zeta"])
    selection = agent_card.select(registry, {"x"}, tie_break_provider=tie_break_provider)
    assert selection.card.name == "zeta"
    assert selection.tie_broken_by_llm is True
    assert len(tie_break_provider.calls) == 1


def test_agent_card_delegate_runs_the_full_a2a_lifecycle() -> None:
    registry = _sample_registry()
    selection = agent_card.select(registry, {"refund", "invoice", "charge"})
    provider = MockProvider(script=["Refund issued."])
    task = agent_card.delegate("coordinator", selection, "refund my invoice", provider)
    assert task.status == "completed"
    assert [h.split(":")[0] for h in task.history] == ["in_progress", "completed"]
    assert task.payload == "Refund issued."
