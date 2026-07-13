"""Multi-agent orchestration: a supervisor decomposes, delegates, and synthesizes.

Multi-agent orchestration splits a task across several agents that each
hold a narrow role, and coordinates them so their combined output solves
what one agent handles poorly. The default shape is a supervisor
(orchestrator) that decomposes a goal, dispatches scoped subtasks to
worker agents, and synthesizes their returns; other shapes move control
between agents directly (handoff), share one conversation (group chat,
debate), or check and revise work before accepting it (maker-checker).

This demo runs every sub-variant end to end, entirely offline against
`MockProvider` with scripted, coherent conversations, no network call and
no API key:

1. Supervisor / orchestrator-worker (star topology): decompose a research
   goal into subtasks, dispatch workers in parallel, synthesize a final
   report, with a durable-execution resume demo.
2. Aggregation (concurrent fan-out / fan-in): majority vote on a
   classification vote, and model synthesis on narrative findings.
3. Handoff / routing / triage, and the subagent variant where control
   returns to the parent instead of transferring permanently.
4. Group chat / roundtable: a chat manager picks the next speaker, with a
   turn-cap guard against a manager that never says stop.
5. Debate / society of minds: agents converge on a shared answer, or fall
   back to a majority tally when a round cap is reached first.
6. Maker-checker / generator-critic loop: a checker fails work twice before
   approving it, and a capped run that falls back to escalation.
7. Hierarchical teams (supervisor of supervisors): two team leads each
   synthesize their own sub-team, and a top supervisor synthesizes both.
8. Failure attribution: the MAST taxonomy applied to a broken run, with all
   three attribution strategies (All-at-Once, Step-by-Step, Binary-Search).
9. Economics: the same goal run single-threaded and through the supervisor
   fan-out, with the actual token multiple and context-isolation numbers.
10. Magentic dual-ledger orchestrator: a plan that stalls, replans, and
    finishes, driven by a Task Ledger and a Progress Ledger.
11. Agent Card discovery: a coordinator picks an un-hardcoded delegate by
    matching a task against registered A2A-style capability cards.

Run it from the repository root:

    python -m patterns.multi_agent.main

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key in the environment) to run the same code against
a real model instead of the mock. No source change is required; every demo
function builds its providers through `agentic_patterns.get_provider`.
"""

from __future__ import annotations

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
from agentic_patterns import get_provider


def main() -> None:
    """Run every multi-agent sub-variant demo and print a readable transcript."""
    print("MULTI-AGENT ORCHESTRATION PATTERN: supervisor, workers, and their cousins\n")

    _run_supervisor_section()
    _run_aggregation_section()
    _run_handoff_section()
    _run_group_chat_section()
    _run_debate_section()
    _run_maker_checker_section()
    _run_hierarchical_section()
    _run_failure_attribution_section()
    _run_economics_section()
    _run_magentic_section()
    _run_agent_card_section()

    print("All sub-variants completed without exhausting their scripts.")


def _run_supervisor_section() -> None:
    print("=== 1. Supervisor / orchestrator-worker (star topology) ===")
    state, results = supervisor.run_supervisor_demo()
    print(f"goal: {state.goal}")
    for r in results:
        print(f"  worker {r.role} ({r.subtask_id}): {r.content}")
    print(f"final report: {state.results['final_report']}")
    print("trace:")
    print("\n".join(f"  {line}" for line in state.format_trace().splitlines()))
    print()

    print("--- 1b. Resume from a checkpoint (durable execution) ---")
    resumed, assignments = supervisor.run_resume_demo()
    print(f"completed before resume: {sorted(resumed.completed_subtask_ids)}")
    print(f"workers dispatched on resume: {[s.id for _, s in assignments]}")
    assert "market" not in [s.id for _, s in assignments], "resume must not redispatch a completed subtask"
    print("market subtask was skipped: no worker for it was ever built, so its provider was never called")
    print()


def _run_aggregation_section() -> None:
    print("=== 2. Fan-out / fan-in aggregation: majority vote and model synthesis ===")
    vote = aggregation.run_majority_vote_demo()
    print(f"votes: {vote.counts} -> winner: {vote.winner!r} (unanimous={vote.unanimous})")

    results, summary = aggregation.run_model_synthesis_demo()
    for r in results:
        print(f"  finding ({r.role}): {r.content}")
    print(f"synthesized summary: {summary}")
    print()


def _run_handoff_section() -> None:
    print("=== 3. Handoff / routing / triage, and the subagent variant ===")
    triage_provider = get_provider(script=["ROUTE: billing_specialist\nREASON: duplicate charge is a billing issue"])
    billing_provider = get_provider(
        script=["We found the duplicate charge from the 3rd and issued a refund of $24.00; it will "
                "post within 3-5 business days."]
    )
    handoff_task = handoff.run_handoff_demo(triage_provider, billing_provider)
    print(f"handoff task -> to_agent={handoff_task.to_agent}, status={handoff_task.status}")
    print(f"  history: {handoff_task.history}")
    print(f"  resolution: {handoff_task.payload}")

    parent_provider = get_provider(
        script=[
            "QUESTION: What changed in today's deploy?",
            "Deploy note: fixed the checkout latency regression from the connection-pool change.",
        ]
    )
    child_provider = get_provider(script=["Today's deploy fixed the checkout latency regression."])
    subagent_task, final_answer = handoff.run_subagent_demo(parent_provider, child_provider)
    print(f"subagent task -> status={subagent_task.status}, history={subagent_task.history}")
    print(f"  parent's final answer after control returned: {final_answer}")
    print()


def _run_group_chat_section() -> None:
    print("=== 4. Group chat / roundtable ===")
    result = group_chat.run_group_chat_demo()
    for turn in result.turns:
        print(f"  {turn.speaker}: {turn.content}")
    print(f"stop_reason={result.stop_reason}")

    cap_result = group_chat.run_group_chat_cap_demo()
    print(f"--- cap guard: {len(cap_result.turns)} turns, stop_reason={cap_result.stop_reason} ---")
    print()


def _run_debate_section() -> None:
    print("=== 5. Debate / society of minds ===")
    converged = debate.run_debate_convergence_demo()
    for round_ in converged.rounds:
        print(f"  round {round_.index}: {round_.positions}")
    print(f"final_answer={converged.final_answer!r}, stop_reason={converged.stop_reason}")

    fallback = debate.run_debate_fallback_demo()
    print(f"--- no consensus within cap: final round {fallback.rounds[-1].positions} ---")
    print(f"fallback answer={fallback.final_answer!r}, stop_reason={fallback.stop_reason}")
    print()


def _run_maker_checker_section() -> None:
    print("=== 6. Maker-checker / generator-critic loop ===")
    result = maker_checker.run_maker_checker_demo()
    print(f"attempts made: {len(result.attempts)}")
    for i, (attempt, check) in enumerate(zip(result.attempts, result.checks), start=1):
        print(f"  attempt {i}: {attempt}")
        print(f"    checker: passed={check.passed}, feedback={check.feedback}")
    print(f"approved={result.approved}, final_output={result.final_output}")

    cap_result = maker_checker.run_cap_demo()
    print(f"--- cap reached without approval: stop_reason={cap_result.stop_reason} ---")
    print(f"fallback final_output: {cap_result.final_output}")
    print()


def _run_hierarchical_section() -> None:
    print("=== 7. Hierarchical teams (supervisor of supervisors) ===")
    state, team_results = hierarchical.run_hierarchical_demo()
    for name, team_result in team_results.items():
        print(f"  {name}: {team_result.summary}")
        for r in team_result.worker_results:
            print(f"    - {r.role}: {r.content}")
    print(f"top supervisor's final verdict: {state.results['final_report']}")
    print()


def _run_failure_attribution_section() -> None:
    print("=== 8. Failure attribution (MAST taxonomy) ===")
    attributions = failure_attribution.run_failure_attribution_demo()
    for strategy in ("all_at_once", "step_by_step", "binary_search"):
        a = attributions[strategy]
        print(f"  {strategy}: agent={a.agent} step={a.step} mode={a.mode_id} ({a.category})")
    agents_named = {a.agent for a in attributions.values()}
    steps_named = {a.step for a in attributions.values()}
    print(f"all strategies agree on the agent ({agents_named.pop()}); they differ on the step: {sorted(steps_named)}")
    print()


def _run_economics_section() -> None:
    print("=== 9. Economics: single-threaded vs. supervisor fan-out ===")
    report = economics.run_economics_demo()
    print(f"single-threaded: {report.single_threaded_tokens} tokens across {report.single_threaded_call_count} calls, "
          f"peak context {report.single_threaded_peak_context} tokens")
    print(f"supervised:      {report.supervised_tokens} tokens across {report.supervised_call_count} calls, "
          f"peak context {report.supervised_peak_context} tokens")
    print(f"token multiple: {report.multiple:.2f}x; worker peak contexts: {report.worker_peak_contexts}")
    print("the fan-out costs more tokens but caps every worker's context to its own subtask, not the whole job")
    print()


def _run_magentic_section() -> None:
    print("=== 10. Magentic dual-ledger orchestrator (stall + replan) ===")
    result = magentic.run_magentic_demo()
    for line in result.transcript:
        print(f"  {line}")
    print(f"replans={result.replans}, stop_reason={result.stop_reason}")
    print(f"answer: {result.answer}")
    print()


def _run_agent_card_section() -> None:
    print("=== 11. Agent Card discovery (A2A capability matching) ===")
    selection, delegated, no_capable_error = agent_card.run_agent_card_demo()
    print(f"discovered: {selection.card.name} (score={selection.score}) for a refund request")
    print(f"  delegation -> status={delegated.status}, resolution={delegated.payload}")
    print(f"unmatched task correctly found no capable agent: {no_capable_error}")
    print()


if __name__ == "__main__":
    main()
