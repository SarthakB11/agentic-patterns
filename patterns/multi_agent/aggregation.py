"""Fan-in aggregation strategies for the concurrent / parallel variant.

`worker.dispatch_parallel` fans a subtask out to several independent
workers. This module provides two ways to fan the results back in, per the
brief: majority vote for classification-shaped outputs, and model synthesis
for narrative outputs that need reconciling rather than counting.

Both strategies take a list of `WorkerResult` in caller-supplied order (the
deterministic order `dispatch_parallel` returns) and both are agnostic to
who produced each result, so they compose with any fan-out, not only the
supervisor demo in `supervisor.py`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from agentic_patterns import Message, Provider, get_provider

from patterns.multi_agent.worker import Subtask, Worker, WorkerResult, dispatch_parallel


def normalize_answer(text: str) -> str:
    """Normalize an answer for equality comparison across agents.

    Lowercases and strips surrounding whitespace and common punctuation or
    currency symbols, so answers that mean the same thing but differ in
    casing or formatting ("Yes" vs "yes", "$0.05" vs "0.05") compare equal.
    Correction: the folder previously compared answers by exact string after
    only a whitespace strip, which would have split votes or blocked debate
    convergence on cosmetically different but semantically identical
    answers. Used by `majority_vote` and by `debate.run_debate`.
    """
    return text.strip().strip("$€£.,!?:;\"'").strip().lower()


@dataclass
class VoteResult:
    """The outcome of a majority-vote aggregation.

    Attributes:
        winner: The vote text (original casing) that received the most votes.
        counts: Every distinct normalized vote mapped to how many workers cast it.
        unanimous: True if every worker cast the same vote once normalized.
    """

    winner: str
    counts: dict[str, int]
    unanimous: bool


def majority_vote(results: list[WorkerResult]) -> VoteResult:
    """Aggregate classification-shaped worker outputs by majority vote.

    Each result's `content` is treated as one vote, tallied by its
    `normalize_answer` form so cosmetically different but equivalent votes
    ("Yes" and "yes") are not split. Ties are broken by the order votes were
    cast (the order of `results`), so the outcome is deterministic rather
    than depending on dict iteration order.

    Args:
        results: Worker outputs to tally. Only "ok" results vote; a worker
            that errored contributes no vote, so its failure does not skew
            the count toward whichever label happens to be first.

    Raises:
        ValueError: If no worker produced an "ok" result to vote with.
    """
    votes = [r.content.strip() for r in results if r.status == "ok"]
    if not votes:
        raise ValueError("majority_vote received no ok results to tally")
    counts = Counter(normalize_answer(v) for v in votes)
    top_count = max(counts.values())
    # Break ties by first-cast order rather than Counter's insertion-derived order.
    winner = next(v for v in votes if counts[normalize_answer(v)] == top_count)
    return VoteResult(winner=winner, counts=dict(counts), unanimous=len(counts) == 1)


def model_synthesize(provider: Provider, results: list[WorkerResult], *, goal: str, system: str) -> str:
    """Aggregate narrative worker outputs into one answer with a model call.

    Unlike `majority_vote`, this does not count anything: it hands every
    worker's finding to a model and asks it to reconcile them into a single
    coherent answer. Use this when outputs are prose, not a small label set.

    Args:
        provider: The provider that performs the synthesis call. Typically
            the supervisor's own provider, scripted with the synthesis turn
            as its next expected call.
        results: Worker outputs to reconcile, in the order they should be
            presented; only "ok" results are included; errored workers are
            noted by role so a reader can see coverage gaps.
        goal: The overall goal the synthesis should serve.
        system: System prompt for the synthesis call.
    """
    lines = [f"Goal: {goal}", "", "Worker findings:"]
    for r in results:
        if r.status == "ok":
            lines.append(f"- {r.role} ({r.subtask_id}): {r.content}")
        else:
            lines.append(f"- {r.role} ({r.subtask_id}): [no finding, worker failed: {r.content}]")
    completion = provider.complete([Message.user("\n".join(lines))], system=system)
    return completion.content


# --- demos -------------------------------------------------------------


def run_majority_vote_demo() -> VoteResult:
    """Three reviewers vote yes/no on shipping a PR; two agree, one differs.

    Demonstrates majority-vote fan-in: independent workers each answer the
    same classification question from their own scoped view, and the
    aggregation step picks the agreed answer without a model call.
    """
    subtask = Subtask(
        id="ship_vote",
        role="reviewer",
        objective="Vote yes or no: should PR #482 (rate limiter rewrite) ship today?",
        output_format="A single word: yes or no.",
        boundaries=["Answer with exactly one word, no explanation."],
    )
    reviewers = [
        Worker("reviewer_a", "You are a careful code reviewer.", get_provider(script=["yes"])),
        Worker("reviewer_b", "You are a careful code reviewer.", get_provider(script=["yes"])),
        Worker("reviewer_c", "You are a careful code reviewer.", get_provider(script=["no"])),
    ]
    results = dispatch_parallel([(w, subtask) for w in reviewers])
    return majority_vote(results)


def run_model_synthesis_demo() -> tuple[list[WorkerResult], str]:
    """Three specialists each summarize one angle of an incident; a model reconciles them.

    Demonstrates model-synthesis fan-in on prose outputs that a vote cannot
    meaningfully tally: the synthesis call reads every finding and writes
    one coherent incident summary.
    """
    goal = "Summarize the checkout-latency incident for the postmortem doc."
    subtasks = {
        "timeline": Subtask(
            "timeline", "timeline_analyst", "Reconstruct when the incident started and was resolved",
            "1-2 sentences", ["Use timestamps only, no root cause speculation"],
        ),
        "cause": Subtask(
            "cause", "root_cause_analyst", "Identify the technical root cause",
            "1-2 sentences", ["State the cause, not the fix"],
        ),
        "impact": Subtask(
            "impact", "impact_analyst", "Quantify user-facing impact",
            "1 sentence", ["Use the numbers from monitoring only"],
        ),
    }
    scripts = {
        "timeline": "Checkout latency began climbing at 14:02 UTC and returned to baseline at 14:41 UTC after a rollback.",
        "cause": "A new connection-pool size of 5 (down from 50) was deployed to the payments service, exhausting pooled connections under normal load.",
        "impact": "Roughly 8% of checkout attempts during the window saw latency above 3 seconds, per the checkout-latency dashboard.",
    }
    workers = [
        (Worker(role, f"You are a {role} for an incident postmortem.", get_provider(script=[scripts[sid]])), subtasks[sid])
        for sid, role in [("timeline", "timeline_analyst"), ("cause", "root_cause_analyst"), ("impact", "impact_analyst")]
    ]
    results = dispatch_parallel(workers)
    synthesis_provider = get_provider(
        script=[
            "Checkout latency spiked from 14:02 to 14:41 UTC, affecting about 8% of checkout "
            "attempts with response times over 3 seconds, after a deploy shrank the payments "
            "service's connection pool from 50 to 5 and exhausted it under normal load. Rollback "
            "resolved the incident; the follow-up is a pool-size floor in the deploy config."
        ]
    )
    summary = model_synthesize(
        synthesis_provider,
        results,
        goal=goal,
        system="You write concise, factual incident postmortem summaries from analyst findings.",
    )
    return results, summary
