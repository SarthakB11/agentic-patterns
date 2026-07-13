"""Shared engine: parsing a judge's free-form text into a structured verdict.

Every LLM-judge variant in this pattern (pointwise, pairwise, ensemble,
trajectory) asks the model for the same shape of structured output and
reuses the parsers here, so the parsing logic and its malformed-output
fallback are written and tested once instead of once per module. Judges
occasionally wrap the answer in prose even when asked for a fixed format, so
every parser here degrades to a safe, clearly-marked fallback rather than
raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SCORE_RE = re.compile(r"SCORE:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_VERDICT_RE = re.compile(r"VERDICT:\s*(pass|fail)", re.IGNORECASE)
_WINNER_RE = re.compile(r"WINNER:\s*(a|b|tie)", re.IGNORECASE)


@dataclass
class Verdict:
    """A judge's structured judgment of one output.

    Attributes:
        score: A numeric score if the judge's text contained a SCORE line,
            else None.
        passed: True/False if the judge's text contained a VERDICT line,
            else None.
        reasoning: The judge's free-form reasoning, kept for audit and for
            feeding into a jury or trajectory report.
        raw: The judge's full, unparsed response text.
        malformed: True if neither a SCORE nor a VERDICT line could be
            found, meaning the fallback values were used.
    """

    score: float | None
    passed: bool | None
    reasoning: str
    raw: str
    malformed: bool = False


def parse_pointwise_verdict(text: str) -> Verdict:
    """Parse a pointwise judge's response into a `Verdict`.

    Looks for a `SCORE: <number>` line and a `VERDICT: pass|fail` line
    anywhere in the text, so chain-of-thought reasoning can precede them.
    If neither is found, the response is treated as malformed and a safe
    fallback is returned: `passed=False`, `score=None`, so a downstream
    regression gate fails closed rather than silently passing on garbage.

    Args:
        text: The judge's raw response text.
    """
    score_match = _SCORE_RE.search(text)
    verdict_match = _VERDICT_RE.search(text)
    score = float(score_match.group(1)) if score_match else None
    passed = verdict_match.group(1).lower() == "pass" if verdict_match else None

    if score_match is None and verdict_match is None:
        return Verdict(score=None, passed=False, reasoning=text.strip(), raw=text, malformed=True)

    return Verdict(score=score, passed=passed, reasoning=text.strip(), raw=text)


@dataclass
class PairwiseVerdict:
    """A judge's choice between two candidates in one presentation order.

    Attributes:
        winner: "a", "b", or "tie", in terms of the order the judge actually
            saw (slot A, slot B), not the caller's candidate labels.
        reasoning: The judge's free-form reasoning.
        raw: The judge's full, unparsed response text.
        malformed: True if no WINNER line could be found. Falls back to
            "tie" rather than guessing a winner from unstructured text.
    """

    winner: str
    reasoning: str
    raw: str
    malformed: bool = False


def parse_pairwise_verdict(text: str) -> PairwiseVerdict:
    """Parse a pairwise judge's response into a `PairwiseVerdict`.

    Looks for a `WINNER: a|b|tie` line. Falls back to "tie" on malformed
    output, since a judge that cannot be parsed should not be trusted to
    prefer either candidate.

    Args:
        text: The judge's raw response text.
    """
    match = _WINNER_RE.search(text)
    if match is None:
        return PairwiseVerdict(winner="tie", reasoning=text.strip(), raw=text, malformed=True)
    return PairwiseVerdict(winner=match.group(1).lower(), reasoning=text.strip(), raw=text)
