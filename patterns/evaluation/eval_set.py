"""The eval set: a versioned collection of input cases.

Every scorer in this pattern grades output against cases defined here, so
the eval set is the one place that changes when the task under test changes.
Each `EvalCase` carries an input, an optional reference answer (for
reference-based scoring), and an optional expected property string consumed
by the exact evaluators in `exact.py`. The whole pattern demo runs against a
single scenario: a subscription-product support bot, so the same cases can
be reused across exact checks, semantic similarity, and both LLM-judge
styles without inventing a new domain per module.

`EVAL_SET_VERSION` is bumped whenever a case's input or reference changes,
so a stored baseline (see `regression.py`) can be checked against the
version it was computed on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

EVAL_SET_VERSION = "2026.07.1"


@dataclass(frozen=True)
class EvalCase:
    """One input case in the eval set.

    Attributes:
        id: Stable identifier for the case. Used to look up baselines and to
            report per-case failures.
        input: The prompt or task given to the system under test.
        reference: A gold answer to compare against, for reference-based
            scoring. None for genuinely open-ended cases.
        expected_property: A machine-checkable property string consumed by
            an exact evaluator, e.g. "regex:\\b48213\\b" or
            "json_schema:order_status". None if no exact evaluator applies.
        tags: Free-form labels for slicing metrics, e.g. ["billing"].
    """

    id: str
    input: str
    reference: str | None = None
    expected_property: str | None = None
    tags: list[str] = field(default_factory=list)


EVAL_SET: list[EvalCase] = [
    EvalCase(
        id="refund_policy",
        input="What is your refund policy?",
        reference=(
            "Refunds are available within 30 days of purchase if you have "
            "your receipt or order number."
        ),
        tags=["billing", "open_ended"],
    ),
    EvalCase(
        id="order_status_lookup",
        input="What is the status of order 48213?",
        expected_property=r"regex:\b48213\b",
        tags=["billing", "exact"],
    ),
    EvalCase(
        id="order_extraction",
        input="Extract the order id and status from: Order 48213 shipped on July 2.",
        expected_property="json_schema:order_status",
        tags=["extraction", "exact"],
    ),
    EvalCase(
        id="cancel_subscription",
        input="How do I cancel my subscription?",
        reference=(
            "Go to Account Settings, select Subscription, and click Cancel. "
            "Access continues until the end of the current billing period."
        ),
        tags=["billing", "open_ended"],
    ),
    EvalCase(
        id="refund_investigation",
        input="Customer says order 48213 arrived damaged and wants a refund. Resolve it.",
        reference="Verify the order, confirm damage, and issue the refund.",
        tags=["billing", "trajectory"],
    ),
]


def get_case(case_id: str) -> EvalCase:
    """Look up a case by id.

    Raises:
        KeyError: If no case with that id exists in `EVAL_SET`.
    """
    for case in EVAL_SET:
        if case.id == case_id:
            return case
    known = ", ".join(c.id for c in EVAL_SET)
    raise KeyError(f"Unknown eval case {case_id!r}. Known cases: {known}")
