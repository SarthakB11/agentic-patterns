"""Shared engine every guardrail variant builds on.

Three pieces live here, kept independent of any one guard or provider:

- `OnFail`, the vocabulary of actions a guard can take when a value does not
  pass, drawn from Guardrails AI's validator on-fail actions plus a
  `TRIPWIRE` action for the pipeline-wide abort introduced by OpenAI's
  open-source guardrails library (2025).
- `Guard`, a protocol every guard module implements: a pure function from a
  value to a `GuardResult`. Pure means the same input always yields the same
  decision, which is what keeps guards unit-testable in isolation and
  composable into a pipeline.
- `run_guard`, the fail-closed wrapper that calls a guard, records the
  decision in a `DecisionLog`, and turns an internal guard exception into a
  failing result rather than letting it silently pass raw output through.

Every guard in this pattern is a small dataclass holding its own
configuration with a `check` method; none of them call a model or perform
the side effect they are gating, per the brief's rule that guards must stay
side-effect free.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class OnFail(enum.Enum):
    """What a pipeline should do when a guard's check does not pass.

    Values:
        NOOP: Log the failure and continue with the original value.
        EXCEPTION: Raise `GuardViolation` immediately.
        FIX: The guard already computed a deterministic replacement value;
            use it and continue.
        FILTER: Drop the offending value (a chunk, a field) rather than
            passing anything through.
        REFRAIN: Do not use the value at all; the caller should return a
            safe fallback instead.
        RETRY: The value is repairable; feed the guard's message back to
            the model and ask it to try again, up to a bounded budget.
        TRIPWIRE: Abort the entire pipeline run immediately, not just this
            one field. Reserved for high-severity hits, e.g. a confirmed
            prompt injection or a blocked tool call with no safe partial
            result. Modeled on the tripwire in openai-guardrails-python.
    """

    NOOP = "noop"
    EXCEPTION = "exception"
    FIX = "fix"
    FILTER = "filter"
    REFRAIN = "refrain"
    RETRY = "retry"
    TRIPWIRE = "tripwire"


@dataclass
class GuardResult:
    """The outcome of running one guard against one value.

    Attributes:
        passed: Whether the value satisfied the guard.
        action: What to do given the outcome. Meaningful even when
            `passed` is True (usually `OnFail.NOOP`), since a guard that
            fixes a value on the way through still reports `OnFail.FIX`.
        value: The value to use going forward: the original value when
            unchanged, or the guard's deterministic replacement when
            `action` is `OnFail.FIX`.
        message: Human-readable explanation, written to the decision log
            and safe to show in an audit trail. Kept generic enough not to
            teach an attacker which exact rule fired, per the brief's rule
            against leaking guard internals.
        requires_approval: True when a guard would otherwise pass or fail
            outright but instead wants a human decision before the value is
            used. Only the pre-tool guard sets this.
    """

    passed: bool
    action: OnFail
    value: Any
    message: str = ""
    requires_approval: bool = False


@runtime_checkable
class Guard(Protocol):
    """Interface every guard in this pattern implements.

    A guard is a pure function from a value to a `GuardResult`: it reads
    `value` (and whatever configuration it was built with) and returns a
    judgment, without mutating anything or performing the action it is
    gating.
    """

    name: str

    def check(self, value: Any) -> GuardResult:
        """Judge `value` and return a `GuardResult`."""
        ...


class GuardViolation(Exception):
    """Raised when a guard's `OnFail.EXCEPTION` action fires."""

    def __init__(self, guard_name: str, result: GuardResult) -> None:
        super().__init__(f"guard {guard_name!r} raised: {result.message}")
        self.guard_name = guard_name
        self.result = result


class Tripwire(Exception):
    """Raised when a guard's `OnFail.TRIPWIRE` action fires.

    Distinct from `GuardViolation` so callers can catch it separately: a
    tripwire means abort the whole run, not just reject one field.
    """

    def __init__(self, guard_name: str, result: GuardResult) -> None:
        super().__init__(f"tripwire from guard {guard_name!r}: {result.message}")
        self.guard_name = guard_name
        self.result = result


@dataclass
class DecisionLogEntry:
    """One recorded guard decision, in the order it happened.

    Attributes:
        guard_name: Which guard produced this decision.
        passed: Whether the value passed.
        action: The `OnFail` action taken.
        message: The guard's explanation.
    """

    guard_name: str
    passed: bool
    action: OnFail
    message: str


@dataclass
class DecisionLog:
    """An ordered, append-only record of every guard decision in a run.

    Every guard call in this pattern is logged, passing or failing, so a
    run can be audited after the fact even when nothing was blocked.
    """

    entries: list[DecisionLogEntry] = field(default_factory=list)

    def record(self, guard_name: str, result: GuardResult) -> None:
        """Append one decision to the log."""
        self.entries.append(DecisionLogEntry(guard_name, result.passed, result.action, result.message))

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def render(self) -> str:
        """Render the log as one line per decision, for a transcript."""
        lines = []
        for e in self.entries:
            status = "pass" if e.passed else "fail"
            lines.append(f"  [{status}] {e.guard_name}: action={e.action.value} {e.message}".rstrip())
        return "\n".join(lines)


def run_guard(guard: Guard, value: Any, log: DecisionLog) -> GuardResult:
    """Run one guard fail-closed and record its decision.

    Fail-closed means: if the guard itself raises, that is treated as a
    failing result with `OnFail.EXCEPTION`, never as an implicit pass. This
    is what protects the pipeline from a guard that has a bug, and it is
    why guards must stay pure: `run_guard` on the same `(guard, value)` pair
    always logs the same decision.

    Args:
        guard: The guard to run.
        value: The value to check.
        log: The decision log to append this call's outcome to.

    Returns:
        The guard's `GuardResult`, or a synthesized failing result if the
        guard raised internally.

    Raises:
        GuardViolation: If the result's action is `OnFail.EXCEPTION`.
        Tripwire: If the result's action is `OnFail.TRIPWIRE`.
    """
    try:
        result = guard.check(value)
    except Exception as exc:  # noqa: BLE001 - fail-closed: any guard error is a failure
        result = GuardResult(passed=False, action=OnFail.EXCEPTION, value=value, message=f"guard raised: {exc}")

    log.record(guard.name, result)

    if not result.passed:
        if result.action == OnFail.EXCEPTION:
            raise GuardViolation(guard.name, result)
        if result.action == OnFail.TRIPWIRE:
            raise Tripwire(guard.name, result)
    return result
