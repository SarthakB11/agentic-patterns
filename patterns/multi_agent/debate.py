"""Debate / society of minds: agents read each other's reasoning and revise.

Several agents answer the same question independently, then each round
every agent sees every other agent's prior answer and reasoning and may
revise its own. The loop stops when all agents converge on the same answer,
or falls back to a majority tally of the final round when a round cap is
hit without full agreement. Unlike group chat (aimed at coverage and
ideation), debate is aimed at correctness: Du et al. showed this raises
factual accuracy and math reasoning versus a single model
(arXiv:2305.14325), because a wrong first answer gets a chance to be
challenged by a right one before it becomes the final answer.

Convergence is checked with `aggregation.normalize_answer`, not exact
string equality: two agents who both mean "yes" but write "Yes" and "yes",
or "$0.05" and "0.05", now agree.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, get_provider

from patterns.multi_agent.aggregation import normalize_answer

DEBATER_SYSTEM = (
    "You are one of several agents debating a question. Reason briefly, then end your "
    "reply with a line 'ANSWER: <value>' giving your current answer."
)


@dataclass
class DebateRound:
    """One round of a debate: every agent's position that round.

    Attributes:
        index: 1-based round number.
        positions: Agent name mapped to the ANSWER value it gave this round.
        replies: Agent name mapped to its full reasoning text this round.
    """

    index: int
    positions: dict[str, str]
    replies: dict[str, str]


@dataclass
class DebateResult:
    """The outcome of a debate run.

    Attributes:
        rounds: Every round that ran, in order.
        final_answer: The converged answer, or the majority-fallback answer.
        stop_reason: "converged" if every agent agreed, or "max_rounds" if
            the cap was hit with the roster still split.
    """

    rounds: list[DebateRound] = field(default_factory=list)
    final_answer: str = ""
    stop_reason: str = "max_rounds"


def _extract_answer(text: str) -> str:
    """Pull the value after the last 'ANSWER:' marker out of a debater's reply.

    Searches the whole text rather than requiring a dedicated line, since a
    debater's reasoning and its answer are often one short paragraph rather
    than separate lines.
    """
    marker = "ANSWER:"
    idx = text.upper().rfind(marker)
    if idx == -1:
        return text.strip()
    return text[idx + len(marker):].strip()


def run_debate(agents: dict[str, Provider], question: str, *, max_rounds: int = 3) -> DebateResult:
    """Run a multi-round debate to convergence or a round cap.

    Args:
        agents: Agent name mapped to that agent's provider, scripted with
            one reply per round it participates in.
        question: The question every agent debates.
        max_rounds: Hard cap on rounds. When reached without every agent
            agreeing, the result falls back to a majority tally of the
            final round's positions instead of looping further.
    """
    rounds: list[DebateRound] = []
    prior_replies: dict[str, str] = {}

    for round_index in range(1, max_rounds + 1):
        positions: dict[str, str] = {}
        replies: dict[str, str] = {}
        for name, provider in agents.items():
            if round_index == 1:
                prompt = f"Question: {question}"
            else:
                others = "\n".join(f"- {other}: {reply}" for other, reply in prior_replies.items() if other != name)
                prompt = f"Question: {question}\n\nOther agents' round {round_index - 1} answers:\n{others}"
            reply = provider.complete([Message.user(prompt)], system=DEBATER_SYSTEM).content
            replies[name] = reply
            positions[name] = _extract_answer(reply)
        rounds.append(DebateRound(round_index, positions, replies))
        prior_replies = replies

        # Compare by normalize_answer, not exact string, so "Yes" and "yes" (or
        # "0.05" and "$0.05") count as convergence rather than staying split.
        if len({normalize_answer(v) for v in positions.values()}) == 1:
            return DebateResult(rounds=rounds, final_answer=next(iter(positions.values())), stop_reason="converged")

    final_positions = list(rounds[-1].positions.values())
    tally = Counter(normalize_answer(v) for v in final_positions)
    top_count = max(tally.values())
    # Keep the original casing of the first position matching the winning normalized form.
    fallback = next(v for v in final_positions if tally[normalize_answer(v)] == top_count)
    return DebateResult(rounds=rounds, final_answer=fallback, stop_reason="max_rounds")


# --- demos -------------------------------------------------------------


def run_debate_convergence_demo() -> DebateResult:
    """The classic bat-and-ball problem: a wrong intuitive answer meets a correct one.

    Agent A starts with the common wrong answer ($0.10, ignoring that "$1.00
    more" is relative to the ball's price). Agent B starts correct ($0.05).
    In round two, A reads B's algebra, redoes its own, and converges.
    """
    agents = {
        "agent_a": get_provider(
            script=[
                "The bat costs $1.00 more, so if the total is $1.10, the ball is the leftover "
                "amount. ANSWER: 0.10",
                "Rereading agent_b's algebra: ball + (ball + 1.00) = 1.10, so 2*ball = 0.10 and "
                "ball = 0.05. My first answer ignored that the $1.00 is added to the ball's "
                "price, not subtracted from the total. ANSWER: 0.05",
            ]
        ),
        "agent_b": get_provider(
            script=[
                "Let ball = x. Bat = x + 1.00. Total: x + (x + 1.00) = 1.10, so 2x = 0.10, "
                "x = 0.05. ANSWER: 0.05",
                "My algebra from round one still holds: x + (x + 1.00) = 1.10 gives x = 0.05. "
                "ANSWER: 0.05",
            ]
        ),
    }
    return run_debate(agents, "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?")


def run_debate_fallback_demo() -> DebateResult:
    """Three agents debate a judgment call and never fully agree within the cap.

    Two prefer Go, one prefers Python, in both rounds; the cap is reached
    with the roster still split 2-1, so the result falls back to a majority
    tally rather than debating indefinitely.
    """
    agents = {
        "agent_a": get_provider(
            script=["Go's lower memory footprint fits a high-throughput microservice. ANSWER: Go"] * 2
        ),
        "agent_b": get_provider(
            script=["Python's ecosystem gets us shipping faster and this service is not CPU-bound. ANSWER: Python"] * 2
        ),
        "agent_c": get_provider(
            script=["Go's static typing and low latency matter more here than iteration speed. ANSWER: Go"] * 2
        ),
    }
    return run_debate(agents, "Should the new notification microservice be written in Go or Python?", max_rounds=2)
