"""Groundedness guard: checks whether an answer's claims are supported by context.

The brief calls this a semantic judgment, "often done with an LLM-as-judge."
For a deterministic, offline teaching example this module uses a token-
overlap heuristic instead: split the answer into sentence-level claims, and
call a claim grounded when enough of its distinctive words also appear in
the retrieved context. This is weaker than a real judge (it cannot catch a
claim that paraphrases the context into something false), but it is
reproducible and needs no model call, and the same `GroundednessGuard`
shape swaps in a provider-backed judge without changing its callers: pass a
`score_fn` that calls a model instead of the default heuristic.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from patterns.guardrails.core import GuardResult, OnFail

_STOPWORDS = frozenset(
    "a an the is are was were be been being this that these those it its of to in on for "
    "and or but with as at by from into your you our we they he she i".split()
)


def _distinctive_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _split_claims(answer: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class ClaimCheck:
    """Whether one claim from the answer is supported by the context.

    Attributes:
        claim: The claim's text, as split from the answer.
        grounded: Whether enough of the claim's distinctive words appear in
            the context.
        overlap: The fraction of the claim's distinctive words found in the
            context, in [0.0, 1.0]. 1.0 when the claim has no distinctive
            words to check.
    """

    claim: str
    grounded: bool
    overlap: float


def _default_score(claim: str, context: str) -> float:
    """Heuristic overlap score: fraction of the claim's distinctive tokens found in context."""
    claim_tokens = _distinctive_tokens(claim)
    if not claim_tokens:
        return 1.0
    context_tokens = _distinctive_tokens(context)
    found = claim_tokens & context_tokens
    return len(found) / len(claim_tokens)


@dataclass
class GroundednessGuard:
    """Flags claims in an answer that the provided context does not support.

    Attributes:
        name: Guard name.
        context: The retrieved or provided context the answer must be
            grounded in.
        threshold: Minimum overlap score for a claim to count as grounded.
        score_fn: Scoring function, `(claim, context) -> float in [0, 1]`.
            Defaults to the token-overlap heuristic; pass a provider-backed
            judge to replace it without touching the guard's callers.
        on_fail: Action when any claim is ungrounded. Defaults to
            `OnFail.REFRAIN`.
    """

    name: str = "groundedness"
    context: str = ""
    threshold: float = 0.4
    score_fn: Callable[[str, str], float] = field(default=_default_score, repr=False)
    on_fail: OnFail = OnFail.REFRAIN

    def check(self, value: str) -> GuardResult:
        # `score_fn` is called once per claim and the result reused for both
        # `grounded` and `overlap`. With the default pure heuristic, calling
        # it twice only wastes work; with a provider-backed judge swapped in
        # (which this guard's docstring invites), it would double the model
        # calls and risk the two values disagreeing if the judge is not
        # perfectly deterministic.
        claims = []
        for c in _split_claims(value):
            overlap = self.score_fn(c, self.context)
            claims.append(ClaimCheck(claim=c, grounded=overlap >= self.threshold, overlap=overlap))
        ungrounded = [c for c in claims if not c.grounded]
        if not ungrounded:
            return GuardResult(passed=True, action=OnFail.NOOP, value=claims)
        preview = "; ".join(c.claim for c in ungrounded)
        return GuardResult(
            passed=False,
            action=self.on_fail,
            value=claims,
            message=f"{len(ungrounded)} unsupported claim(s): {preview}",
        )


_DEMO_CONTEXT = (
    "Our refund policy allows returns within 30 days of delivery for a full refund to the "
    "original payment method. Store credit is issued instantly; card refunds take 5-7 business days."
)
_DEMO_GROUNDED_ANSWER = (
    "You can return items within 30 days of delivery for a full refund to the original payment method."
)
_DEMO_UNGROUNDED_ANSWER = (
    "You can return items within 30 days of delivery for a full refund. We will also overnight a free "
    "replacement gift with every purchase, no questions asked."
)


def run_groundedness_demo() -> tuple[GuardResult, GuardResult]:
    """Check one grounded and one partly fabricated answer against a fixed context.

    Returns:
        The `GuardResult` for the grounded answer, then for the ungrounded
        one, so a caller can assert on both.
    """
    guard = GroundednessGuard(context=_DEMO_CONTEXT, threshold=0.4)

    print("=== Groundedness guard: claims checked against retrieved context ===")
    print(f"context: {_DEMO_CONTEXT}")

    good = guard.check(_DEMO_GROUNDED_ANSWER)
    print(f"answer:  {_DEMO_GROUNDED_ANSWER}")
    print(f"  passed={good.passed}")

    bad = guard.check(_DEMO_UNGROUNDED_ANSWER)
    print(f"answer:  {_DEMO_UNGROUNDED_ANSWER}")
    print(f"  passed={bad.passed}: {bad.message}")

    return good, bad
