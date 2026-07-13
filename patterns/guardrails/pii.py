"""PII detection, masking, and redaction.

A specialized input guard (mask before the model ever sees raw personal
data) and a specialized output guard (redact before a response reaches a
user) built on the same regex-based detector. Masking is reversible: it
returns a placeholder map so the original values can be restored after
generation, for example to fill a real order lookup with the customer's
actual email once the model's reasoning is done. Redaction is not
reversible; it is meant for a response that leaves the system.

Detected categories: email, phone number, credit card number, and a
US-style identity number (SSN). Real systems reach for a dedicated library
such as Microsoft Presidio (used by openai-guardrails-python's PII check);
this module uses stdlib regex only, which covers the common formats well
enough for a deterministic teaching example.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from patterns.guardrails.core import GuardResult, OnFail

_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}
# Order matters: SSN and CARD both match digit runs, so check the more
# specific hyphenated SSN shape before the looser CARD shape swallows it.
_ORDERED_CATEGORIES = ("EMAIL", "SSN", "CARD", "PHONE")


@dataclass
class PIIMatch:
    """One detected span of personal data.

    Attributes:
        category: One of "EMAIL", "PHONE", "CARD", "SSN".
        original: The exact substring that matched.
        placeholder: The token it was replaced with, e.g. "[PII_EMAIL_1]".
    """

    category: str
    original: str
    placeholder: str


def detect_pii(text: str) -> list[PIIMatch]:
    """Find every PII span in `text`, in left-to-right order, without overlaps."""
    matches: list[PIIMatch] = []
    claimed: list[tuple[int, int]] = []
    counters: dict[str, int] = {}

    for category in _ORDERED_CATEGORIES:
        for m in _PATTERNS[category].finditer(text):
            span = m.span()
            if any(span[0] < end and start < span[1] for start, end in claimed):
                continue
            claimed.append(span)
            counters[category] = counters.get(category, 0) + 1
            placeholder = f"[PII_{category}_{counters[category]}]"
            matches.append(PIIMatch(category=category, original=m.group(0), placeholder=placeholder))
    matches.sort(key=lambda pm: text.index(pm.original))
    return matches


def mask_pii(text: str) -> tuple[str, dict[str, str]]:
    """Replace every detected PII span with a placeholder token.

    Returns:
        A tuple of the masked text and a placeholder map from token to the
        original value it replaced, so the substitution can be reversed
        with `unmask_pii`.
    """
    matches = detect_pii(text)
    masked = text
    placeholder_map: dict[str, str] = {}
    for match in matches:
        masked = masked.replace(match.original, match.placeholder, 1)
        placeholder_map[match.placeholder] = match.original
    return masked, placeholder_map


def unmask_pii(text: str, placeholder_map: dict[str, str]) -> str:
    """Restore original PII values from a placeholder map produced by `mask_pii`."""
    restored = text
    for placeholder, original in placeholder_map.items():
        restored = restored.replace(placeholder, original)
    return restored


def redact_pii(text: str) -> str:
    """Replace every detected PII span with "[REDACTED]", with no way back.

    Used on output that leaves the system, where there is no legitimate
    later need to recover the original value.
    """
    matches = detect_pii(text)
    redacted = text
    for match in matches:
        redacted = redacted.replace(match.original, "[REDACTED]", 1)
    return redacted


@dataclass
class PIIMaskGuard:
    """Input guard: masks PII in a request before it reaches the model.

    Always reports `passed=False, action=OnFail.FIX` when it finds
    anything to mask, since masking is itself the deterministic fix; a
    request with no PII passes untouched.
    """

    name: str = "pii_mask"
    placeholder_map: dict[str, str] = field(default_factory=dict, repr=False)

    def check(self, value: str) -> GuardResult:
        masked, placeholder_map = mask_pii(value)
        if not placeholder_map:
            return GuardResult(passed=True, action=OnFail.NOOP, value=value)
        self.placeholder_map.update(placeholder_map)
        categories = sorted({p.split("_")[1] for p in placeholder_map})
        return GuardResult(
            passed=False,
            action=OnFail.FIX,
            value=masked,
            message=f"masked {len(placeholder_map)} PII value(s): {', '.join(categories)}",
        )


@dataclass
class PIIRedactGuard:
    """Output guard: redacts PII in a response before it reaches the user."""

    name: str = "pii_redact"

    def check(self, value: str) -> GuardResult:
        matches = detect_pii(value)
        if not matches:
            return GuardResult(passed=True, action=OnFail.NOOP, value=value)
        redacted = redact_pii(value)
        categories = sorted({m.category for m in matches})
        return GuardResult(
            passed=False,
            action=OnFail.FIX,
            value=redacted,
            message=f"redacted {len(matches)} PII value(s): {', '.join(categories)}",
        )
