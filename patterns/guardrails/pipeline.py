"""The GuardedAgent pipeline: input guards, the model call, output guards.

This is the canonical control flow from the brief, steps 2 through 9,
minus retrieval and pre-tool guards, which apply inside a RAG or tool loop
rather than around a single completion and get their own demos in
`retrieval_guard.py` and `pretool_guard.py`.

`run_guarded` is the validate-retry-repair loop:

1. Run input guards in order. A `FIX` guard transforms the value and the
   pipeline continues; a failing guard with any other action stops the
   pipeline before the model is ever called and returns the safe fallback.
2. Call the model.
3. Run output guards over the response. A `FIX` guard transforms the value
   in place. A `RETRY` failure feeds the guard's message back to the model
   as a new turn and calls it again, bounded by `max_retries`. A `REFRAIN`
   failure stops immediately and returns the fallback, never the unsafe
   value. `EXCEPTION` and `TRIPWIRE` propagate out of `run_guard` itself. A
   `NOOP` or `FILTER` failure is logged and the pipeline moves on to the
   next guard with the guard's own value (unchanged, for `NOOP`), exactly
   as `run_guard` reports it, but that guard's failure is remembered for
   the round.
4. Once every output guard in the round has run: if every one of them
   passed or was `FIX`ed, return the validated value. If any guard failed
   with `NOOP` or `FILTER` and was never fixed, the round cannot be called
   validated, so the pipeline returns the safe fallback with stop_reason
   `"output_guard_failed"` instead of the unresolved value.

Fail-closed applies at every exit: the only way this function returns
`passed=True` is that every guard in `output_guards` reported a pass or a
deterministic fix on the same round. Exhausting the retry budget, or
finishing a round with an unresolved `NOOP`/`FILTER` failure, always
returns the fallback, never the last unvalidated response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_patterns import Message, Provider

from patterns.guardrails.core import DecisionLog, Guard, OnFail, run_guard


@dataclass
class PipelineResult:
    """The outcome of one `run_guarded` call.

    Attributes:
        passed: Whether a validated value was produced.
        value: The final value: the validated response when `passed` is
            True, otherwise the fallback value.
        retries: How many reask round trips were used.
        stop_reason: One of "validated", "input_blocked", "refrained",
            "retries_exhausted", or "output_guard_failed" (an output guard
            failed with `OnFail.NOOP` or `OnFail.FILTER`, an action that
            neither fixes the value nor stops the pipeline outright, so the
            round finished without every guard reporting a pass or a fix).
        log: The full decision log for this run, for audit.
    """

    passed: bool
    value: Any
    retries: int
    stop_reason: str
    log: DecisionLog


def run_guarded(
    provider: Provider,
    user_input: str,
    *,
    system: str | None = None,
    input_guards: list[Guard] | None = None,
    output_guards: list[Guard] | None = None,
    max_retries: int = 2,
    fallback: Any = "I can't help with that request.",
) -> PipelineResult:
    """Run the full guarded pipeline for one user turn.

    Args:
        provider: The model to call.
        user_input: The raw user request.
        system: System prompt for the model call.
        input_guards: Guards run on `user_input` before the model is
            called, in order.
        output_guards: Guards run on the model's response, in order, with
            reask on a `RETRY` failure.
        max_retries: Maximum number of reask round trips.
        fallback: Value returned when the pipeline cannot produce a
            validated response, either because an input guard blocked the
            request or the retry budget was exhausted.
    """
    log = DecisionLog()
    input_guards = input_guards or []
    output_guards = output_guards or []

    value: Any = user_input
    for guard in input_guards:
        result = run_guard(guard, value, log)
        if result.action == OnFail.FIX:
            value = result.value
            continue
        if not result.passed:
            return PipelineResult(passed=False, value=fallback, retries=0, stop_reason="input_blocked", log=log)

    messages = [Message.user(value)]
    completion = provider.complete(messages, system=system)

    retries = 0
    while True:
        current: Any = completion.content
        needs_retry = False
        retry_reason = ""
        unresolved_failure = False

        for guard in output_guards:
            result = run_guard(guard, current, log)
            if not result.passed and result.action == OnFail.RETRY:
                needs_retry = True
                retry_reason = result.message
                break
            if not result.passed and result.action == OnFail.REFRAIN:
                return PipelineResult(passed=False, value=fallback, retries=retries, stop_reason="refrained", log=log)
            if not result.passed and result.action != OnFail.FIX:
                # NOOP or FILTER: log the failure and continue with the
                # guard's value, but a round containing this cannot be
                # reported as a full pass.
                unresolved_failure = True
            # FIX, NOOP, FILTER, or an outright pass: adopt the guard's
            # (possibly transformed) value and move on to the next guard.
            current = result.value

        if not needs_retry:
            if unresolved_failure:
                return PipelineResult(
                    passed=False, value=fallback, retries=retries, stop_reason="output_guard_failed", log=log
                )
            return PipelineResult(passed=True, value=current, retries=retries, stop_reason="validated", log=log)

        if retries >= max_retries:
            return PipelineResult(
                passed=False, value=fallback, retries=retries, stop_reason="retries_exhausted", log=log
            )

        retries += 1
        messages.append(Message.assistant(completion.content))
        messages.append(
            Message.user(f"Your previous response was invalid: {retry_reason}. Respond again, corrected.")
        )
        completion = provider.complete(messages, system=system)
