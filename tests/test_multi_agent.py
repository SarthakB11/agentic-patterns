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

from patterns.multi_agent import aggregation, debate, group_chat, handoff, hierarchical, maker_checker, supervisor
from patterns.multi_agent.state import SharedState
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
