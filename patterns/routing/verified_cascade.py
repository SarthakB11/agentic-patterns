"""Sub-module: model-judge cascade with three-tier escalation and abstention.

`cascade.quality_check` is a length-plus-hedge heuristic, honestly flagged
there as a stand-in. The faithful FrugalGPT shape (Chen, Zaharia, Zou,
arXiv:2305.05176: a router, a quality estimator, and a stop judge) and the
2025 cascade papers (Fanconi and van der Schaar, arXiv:2506.11887; Zellinger,
Liu, Thomson, arXiv:2502.09054) use a model to estimate whether an answer is
correct, and escalate on a low verdict rather than a text heuristic. This
module builds that: cheap tier, judged; if rejected, strong tier, judged
again; if still rejected, abstain to a human rather than return a
low-confidence answer as if it were certain. This is `escalation.py`'s human
route reached by uncertainty instead of by a sensitive-topic flag.

Territory note: the judge here is used only as an escalation trigger, one
ACCEPT/DEFER call. Judge design and reliability (bias, position effects,
rubric calibration, meta-evaluation) belong to `patterns/evaluation/`; this
module intentionally keeps the judge thin and does not build that machinery.

Every judge call is charged: `metadata["provider_calls"]` on the returned
decision counts every `provider.complete()` the run consumed (an answer plus
a judge call per tier attempted), so a caller can see that verification is
not free, the correction `cascade.quality_check` notes about its own,
zero-cost heuristic.
"""

from __future__ import annotations

from agentic_patterns import Message, Provider, get_provider

from patterns.routing import cascade
from patterns.routing.registry import RouteDecision

_JUDGE_SYSTEM = (
    "You are a strict verifier. Given a question and a candidate answer, reply with "
    "exactly one word: ACCEPT if the answer is correct and complete, or DEFER if it "
    "is not confident, incomplete, or wrong."
)


def _judge(question: str, answer: str, provider: Provider) -> bool:
    """Ask the judge model to ACCEPT or DEFER on `answer`; True means accepted."""
    prompt = f"Question: {question}\nCandidate answer: {answer}\nVerdict (ACCEPT or DEFER):"
    verdict = provider.complete([Message.user(prompt)], system=_JUDGE_SYSTEM).content
    return "accept" in verdict.strip().lower()


def run_verified_cascade(question: str, provider: Provider) -> RouteDecision:
    """Run cheap -> strong -> human, each hop gated by a scripted judge verdict.

    Args:
        question: The question to answer.
        provider: Scripted, in order: the cheap answer, the cheap-answer
            judge verdict, and only if the judge defers, the strong answer
            and its judge verdict.

    Returns:
        A `RouteDecision` on "cheap" (attempts=1) if the cheap answer is
        accepted, "strong" (attempts=2, escalated) if the strong answer is
        accepted after the cheap one is rejected, or "human" (attempts=3,
        abstained) if both are rejected. `metadata` records both verdicts
        (as available), the answering tier, and `provider_calls`.
    """
    calls_before = len(provider.calls)
    cheap_answer = provider.complete([Message.user(question)], system=cascade._CHEAP_SYSTEM).content
    if _judge(question, cheap_answer, provider):
        return RouteDecision(
            route="cheap", score=1.0, method="verified_cascade", attempts=1,
            metadata={
                "answer": cheap_answer, "cheap_verdict": "accept", "escalated": False,
                "abstained": False, "provider_calls": len(provider.calls) - calls_before,
            },
        )

    strong_answer = provider.complete([Message.user(question)], system=cascade._STRONG_SYSTEM).content
    if _judge(question, strong_answer, provider):
        return RouteDecision(
            route="strong", score=1.0, method="verified_cascade", attempts=2,
            metadata={
                "cheap_answer": cheap_answer, "answer": strong_answer, "cheap_verdict": "defer",
                "strong_verdict": "accept", "escalated": True, "abstained": False,
                "provider_calls": len(provider.calls) - calls_before,
            },
        )

    return RouteDecision(
        route="human", score=0.0, method="verified_cascade", attempts=3,
        metadata={
            "cheap_answer": cheap_answer, "strong_answer": strong_answer, "cheap_verdict": "defer",
            "strong_verdict": "defer", "escalated": True, "abstained": True,
            "provider_calls": len(provider.calls) - calls_before,
        },
    )


def run_verified_cascade_demo() -> tuple[RouteDecision, RouteDecision, RouteDecision]:
    """Run one accept-on-cheap, one defer-then-accept-on-strong, and one full abstention.

    Returns:
        An (accepted, escalated, abstained) triple.
    """
    accept_provider = get_provider(
        script=[
            "The refund was processed on the 3rd and will post to your card in 5 to 7 business days.",
            "ACCEPT",
        ]
    )
    accepted = run_verified_cascade("When will my refund post?", accept_provider)

    escalate_provider = get_provider(
        script=[
            "I believe the total is around $400, but I am not fully certain of the exact figure.",
            "DEFER",
            "Break-even price = (fixed cost / volume) + variable cost per unit = $21.00 per unit.",
            "ACCEPT",
        ]
    )
    escalated = run_verified_cascade(
        "Derive the break-even price given $120,000 fixed cost, 8,000 units, and $6 variable cost per unit.",
        escalate_provider,
    )

    abstain_provider = get_provider(
        script=[
            "The asset purchase structure is probably better, though I have not compared the tax basis in detail.",
            "DEFER",
            (
                "Comparing the two structures in full requires jurisdiction-specific tax counsel; "
                "I can outline general considerations but cannot recommend one without that context."
            ),
            "DEFER",
        ]
    )
    abstained = run_verified_cascade(
        "Compare the tax implications of the two acquisition structures and recommend one.", abstain_provider
    )

    return accepted, escalated, abstained
