"""Input guards: pre-model checks on the raw request.

Three small, deterministic guards, cheapest first, matching the brief's
"order guards by cost" rule so a regex rejection never waits on a model
call:

- `PromptInjectionGuard`: keyword and regex detection of common jailbreak
  and instruction-override phrasing (OWASP LLM01).
- `TopicalAllowlistGuard`: keeps the conversation inside a fixed set of
  subjects a support bot is allowed to discuss.
- `LengthGuard`: rejects an input longer than a configured character limit.

None of these call a model. They exist to demonstrate that the cheapest
checks run before anything expensive does, and that a confirmed injection
can escalate all the way to `OnFail.TRIPWIRE` when the match is unambiguous.

`PromptInjectionGuard` in particular should not be read as *the* defense.
It is a fixed pattern set, and Zhan et al., "Adaptive Attacks Break
Defenses Against Indirect Prompt Injection Attacks on LLM Agents"
(arXiv:2503.00061), bypassed eight published defenses of this class with
adaptively phrased attacks, holding attack-success-rate above 50 percent.
Treat it as a cheap first filter and an audit signal, not the guarantee:
the guarantee that survives adaptive phrasing lives in `dual_llm.py`'s
capability layer and `policy_engine.py`'s declarative policy, neither of
which depends on recognizing what an attack looks like.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from patterns.guardrails.core import GuardResult, OnFail

_INJECTION_PATTERNS = (
    re.compile(r"ignore (all|any|the) (previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"disregard (your|the) (system prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"you are now (in )?(dan|developer|jailbreak) mode", re.IGNORECASE),
    re.compile(r"reveal (your|the) (system prompt|hidden instructions)", re.IGNORECASE),
    re.compile(r"pretend (you are|to be) an? (ai|assistant) with no (rules|restrictions)", re.IGNORECASE),
)


@dataclass
class PromptInjectionGuard:
    """Flags text that tries to override the system prompt or persona.

    Attributes:
        name: Guard name, used in the decision log.
        on_fail: Action to take on a match. Defaults to `OnFail.TRIPWIRE`
            since an unambiguous instruction-override attempt has no safe
            partial handling: the whole request is refused, not repaired.
    """

    name: str = "prompt_injection"
    on_fail: OnFail = OnFail.TRIPWIRE

    def check(self, value: str) -> GuardResult:
        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(value)
            if match:
                return GuardResult(
                    passed=False,
                    action=self.on_fail,
                    value=value,
                    message="input matched a known instruction-override pattern",
                )
        return GuardResult(passed=True, action=OnFail.NOOP, value=value)


@dataclass
class TopicalAllowlistGuard:
    """Keeps a conversation inside an allowed set of subjects.

    Attributes:
        name: Guard name.
        allowed_keywords: Keywords that mark an input as on-topic. An input
            matching none of them is treated as off-topic.
        blocked_keywords: Keywords that mark an input as off-topic even if
            an allowed keyword also appears, e.g. requests for advice
            outside the bot's remit.
        on_fail: Action to take when the input is off-topic. Defaults to
            `OnFail.REFRAIN`, returning a safe fallback rather than raising.
    """

    name: str = "topical_allowlist"
    allowed_keywords: tuple[str, ...] = field(
        default_factory=lambda: ("order", "refund", "shipping", "account", "invoice", "billing", "subscription")
    )
    blocked_keywords: tuple[str, ...] = field(
        default_factory=lambda: ("medical advice", "legal advice", "diagnose", "lawsuit", "prescription")
    )
    on_fail: OnFail = OnFail.REFRAIN

    def check(self, value: str) -> GuardResult:
        lowered = value.lower()
        for blocked in self.blocked_keywords:
            if blocked in lowered:
                return GuardResult(
                    passed=False,
                    action=self.on_fail,
                    value=value,
                    message=f"input falls outside the supported topics (matched {blocked!r})",
                )
        if any(kw in lowered for kw in self.allowed_keywords):
            return GuardResult(passed=True, action=OnFail.NOOP, value=value)
        return GuardResult(
            passed=False,
            action=self.on_fail,
            value=value,
            message="input does not match any supported topic",
        )


@dataclass
class LengthGuard:
    """Rejects an input longer than `max_chars`.

    Attributes:
        name: Guard name.
        max_chars: Maximum allowed input length, in characters.
        on_fail: Action to take when the input is too long. Defaults to
            `OnFail.FIX`, truncating deterministically rather than
            rejecting the whole request.
    """

    name: str = "length"
    max_chars: int = 2000
    on_fail: OnFail = OnFail.FIX

    def check(self, value: str) -> GuardResult:
        if len(value) <= self.max_chars:
            return GuardResult(passed=True, action=OnFail.NOOP, value=value)
        if self.on_fail == OnFail.FIX:
            return GuardResult(
                passed=False,
                action=OnFail.FIX,
                value=value[: self.max_chars],
                message=f"input truncated from {len(value)} to {self.max_chars} characters",
            )
        return GuardResult(
            passed=False,
            action=self.on_fail,
            value=value,
            message=f"input length {len(value)} exceeds the {self.max_chars} character limit",
        )
