"""Agent-as-judge: trajectory evaluation.

Final-answer judging only sees the last message; it cannot tell a
well-grounded answer from a confident-sounding guess. A trajectory judge
instead grades the agent's whole reasoning trace, the sequence of actions
and observations, checking whether the process actually supports the final
answer and not only whether the final answer reads plausibly
(arXiv:2410.10934, "Agent-as-a-Judge"). This module contrasts the two
directly: a shortcut trajectory that skips verification produces a final
answer that a final-answer-only judge would pass, and a trajectory judge
that catches the missing step and fails it.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, Provider, get_provider
from patterns.evaluation.eval_set import get_case
from patterns.evaluation.pointwise import build_pointwise_judge
from patterns.evaluation.verdict import Verdict, parse_pointwise_verdict

_TRAJECTORY_SYSTEM = (
    "You grade an agent's whole trajectory for a task, not just its final "
    "answer. Check whether each action was necessary and whether the "
    "observations actually support the final answer, rather than whether "
    "the final answer merely reads plausibly. Reason briefly step by step, "
    "then end with a SCORE line (0-10) and a VERDICT line (pass only if "
    "the process actually supports the final answer, else fail)."
)


@dataclass
class TrajectoryStep:
    """One step of an agent's reasoning trace.

    Attributes:
        action: What the agent did, e.g. a tool call.
        observation: What the agent learned as a result.
    """

    action: str
    observation: str


def run_trajectory_judge(
    provider: Provider, goal: str, steps: list[TrajectoryStep], final_answer: str
) -> Verdict:
    """Grade a full agent trajectory against its goal.

    Args:
        provider: The model that plays the judge.
        goal: The task the agent was asked to accomplish.
        steps: The agent's action/observation trace, in order.
        final_answer: The agent's final answer or resolution.
    """
    numbered = "\n".join(
        f"{i}. action: {s.action}\n   observation: {s.observation}" for i, s in enumerate(steps, start=1)
    )
    prompt = f"Goal:\n{goal}\n\nTrajectory:\n{numbered}\n\nFinal answer:\n{final_answer}"
    completion = provider.complete([Message.user(prompt)], system=_TRAJECTORY_SYSTEM)
    return parse_pointwise_verdict(completion.content)


def run_trajectory_grounded_demo(provider: Provider | None = None) -> Verdict:
    """Grade a trajectory that verifies the order and damage before refunding.

    Args:
        provider: Judge provider. Defaults to a `MockProvider` scripted to
            pass, since every step is necessary and the final answer
            follows from the observations.
    """
    case = get_case("refund_investigation")
    steps = [
        TrajectoryStep("look up order 48213", "order found: status=delivered, item=headphones"),
        TrajectoryStep("request photo of the damage", "customer provided a photo showing a cracked casing"),
        TrajectoryStep(
            "check refund policy for damaged items",
            "policy allows a refund for damaged items within 30 days, no receipt required if the order is on file",
        ),
        TrajectoryStep("issue the refund", "refund of $49.99 issued to the original payment method"),
    ]
    final_answer = (
        "Verified order 48213 was delivered and damaged (photo confirmed); "
        "refund of $49.99 issued per policy."
    )
    if provider is None:
        provider = get_provider(
            script=[
                "Each step is necessary: the order lookup confirms the item "
                "and delivery, the photo request confirms the damage claim, "
                "the policy check justifies eligibility, and only then is "
                "the refund issued. The final answer follows directly from "
                "the observations.\nSCORE: 9\nVERDICT: pass"
            ]
        )
    return run_trajectory_judge(provider, case.input, steps, final_answer)


def run_trajectory_shortcut_demo(provider: Provider | None = None) -> Verdict:
    """Grade a trajectory that refunds without verifying the order or damage.

    Args:
        provider: Judge provider. Defaults to a `MockProvider` scripted to
            fail, since the refund was issued with no order lookup and no
            damage confirmation, even though the final answer text alone
            reads confidently.
    """
    case = get_case("refund_investigation")
    steps = [TrajectoryStep("issue a refund for order 48213", "refund of $49.99 issued to the original payment method")]
    final_answer = "Refund of $49.99 issued for order 48213."
    if provider is None:
        provider = get_provider(
            script=[
                "The refund was issued with no order lookup confirming order "
                "48213 exists or was delivered, and no verification of the "
                "damage claim. The final answer reads confidently but "
                "nothing in the trajectory supports it.\nSCORE: 3\nVERDICT: fail"
            ]
        )
    return run_trajectory_judge(provider, case.input, steps, final_answer)


def run_final_answer_only_comparison(provider: Provider | None = None) -> Verdict:
    """Grade the shortcut demo's final answer alone, with no trajectory shown.

    Args:
        provider: Judge provider. Defaults to a `MockProvider` scripted to
            pass, since the text "Refund of $49.99 issued for order 48213"
            reads as complete and confident with nothing to contradict it.
            This is the case the trajectory judge exists to catch: the same
            final answer that a final-answer-only judge passes fails once
            the process behind it is examined.
    """
    case = get_case("refund_investigation")
    final_answer = "Refund of $49.99 issued for order 48213."
    if provider is None:
        provider = get_provider(
            script=["States the order id and refund amount clearly and reads "
                    "as a complete resolution.\nSCORE: 8\nVERDICT: pass"]
        )
    judge = build_pointwise_judge(provider, reference_mode=False)
    return judge(case, final_answer)
