"""Hierarchical teams: a supervisor of supervisors.

The star topology scales past one supervisor's tracking capacity by
nesting it: a top supervisor delegates to mid-level supervisors ("leads"),
each of which owns its own sub-team of workers, dispatches them with the
same fan-out mechanics as a flat supervisor, and reports one team-level
summary up. The top supervisor never sees an individual worker's output,
only each lead's synthesized summary, mirroring an org chart of managers
and specialists.

The single-writer rule still applies at the top: leads return `TeamResult`
proposals, and only the top supervisor (the sole holder of
`SharedState.WRITER_ROLE`) writes anything into `SharedState`. A lead
synthesizing its own team's results is not a shared-state write; it is a
proposal, the same as a plain worker's return value.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from agentic_patterns import Provider, get_provider

from patterns.multi_agent import aggregation
from patterns.multi_agent.state import SharedState
from patterns.multi_agent.worker import Subtask, Worker, WorkerResult, dispatch_parallel

TOP_SYNTHESIS_SYSTEM = (
    "You are the top-level supervisor combining team leads' summaries into one launch "
    "readiness verdict. Name the true blocker if one exists."
)


@dataclass
class TeamResult:
    """A mid-level supervisor's proposal to the top supervisor.

    Attributes:
        team: Name of the team (the lead's role).
        summary: The lead's synthesized summary of its team's worker results.
        worker_results: The raw results the lead synthesized from.
    """

    team: str
    summary: str
    worker_results: list[WorkerResult]


def run_mid_supervisor(
    team: str,
    lead_provider: Provider,
    assignments: list[tuple[Worker, Subtask]],
    *,
    goal: str,
    synthesis_system: str,
) -> TeamResult:
    """Dispatch a lead's own sub-team and synthesize their results into a team summary.

    Args:
        team: Name of this team, used as the returned `TeamResult.team`.
        lead_provider: The lead's own provider, used only for the
            synthesis call; the lead does not run its own worker loop.
        assignments: (worker, subtask) pairs owned by this team.
        goal: The top-level goal, for framing the synthesis prompt.
        synthesis_system: System prompt for the lead's synthesis call.
    """
    results = dispatch_parallel(assignments)
    summary = aggregation.model_synthesize(lead_provider, results, goal=goal, system=synthesis_system)
    return TeamResult(team=team, summary=summary, worker_results=results)


def run_hierarchical_demo() -> tuple[SharedState, dict[str, TeamResult]]:
    """Run a two-team hierarchy reporting up to one top supervisor.

    A frontend lead owns a UI engineer and a QA engineer; a backend lead
    owns an API engineer and an infra engineer. Both leads run their own
    team in parallel, synthesize a team summary, and report up; the top
    supervisor synthesizes both team summaries into one launch verdict and
    is the only agent that writes to `SharedState`.
    """
    goal = "Prepare the Q3 launch readiness report for the mobile redesign."
    state = SharedState(goal=goal)
    state.record("supervisor", "decompose", "delegating to 2 team leads: frontend_lead, backend_lead")

    frontend_assignments = [
        (
            Worker("ui_engineer", "You are a UI engineer reporting status.", get_provider(
                script=["The onboarding flow UI is feature-complete and passed design review; "
                        "only the tablet breakpoint still needs polish."]
            )),
            Subtask("ui_status", "ui_engineer", "Report the status of the new onboarding flow UI", "one sentence"),
        ),
        (
            Worker("qa_engineer", "You are a QA engineer reporting status.", get_provider(
                script=["There are zero P0 bugs and one P1 bug (login button unresponsive on iOS 16) still open."]
            )),
            Subtask("qa_status", "qa_engineer", "Report open P0/P1 bugs blocking launch", "one sentence"),
        ),
    ]
    backend_assignments = [
        (
            Worker("api_engineer", "You are an API engineer reporting status.", get_provider(
                script=["All three new profile endpoints are deployed to staging and passing contract "
                        "tests; production rollout is gated on the infra change."]
            )),
            Subtask("api_status", "api_engineer", "Report the status of the new profile API endpoints", "one sentence"),
        ),
        (
            Worker("infra_engineer", "You are an infra engineer reporting status.", get_provider(
                script=["The user-profile migration finished on the read replicas but the primary "
                        "migration is still queued, currently the launch's critical path."]
            )),
            Subtask("infra_status", "infra_engineer", "Report the status of the database migration", "one sentence"),
        ),
    ]

    frontend_lead = get_provider(
        script=["Frontend is launch-ready pending the tablet breakpoint polish and a fix for the "
                "one open iOS 16 P1 bug."]
    )
    backend_lead = get_provider(
        script=["Backend API work is done, but launch is blocked on the primary database migration, "
                "which is the critical path for this release."]
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        frontend_future = pool.submit(
            run_mid_supervisor, "frontend_lead", frontend_lead, frontend_assignments, goal=goal,
            synthesis_system="You lead the frontend team. Summarize your team's status in one sentence.",
        )
        backend_future = pool.submit(
            run_mid_supervisor, "backend_lead", backend_lead, backend_assignments, goal=goal,
            synthesis_system="You lead the backend team. Summarize your team's status in one sentence.",
        )
        team_results = {"frontend_lead": frontend_future.result(), "backend_lead": backend_future.result()}

    for name, team_result in team_results.items():
        state.set_status(name, "done")
        state.write_result(SharedState.WRITER_ROLE, name, team_result.summary)

    top_provider = get_provider(
        script=["Mobile redesign is launch-ready on the frontend pending a minor iOS bug fix, but is "
                "blocked overall on the backend's primary database migration, which is the true "
                "critical path; target launch for the day after that migration completes."]
    )
    proposals = [
        WorkerResult(subtask_id=name, role=name, status="ok", content=result.summary)
        for name, result in team_results.items()
    ]
    final_verdict = aggregation.model_synthesize(top_provider, proposals, goal=goal, system=TOP_SYNTHESIS_SYSTEM)
    state.write_result(SharedState.WRITER_ROLE, "final_report", final_verdict)

    return state, team_results
