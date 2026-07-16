"""Sub-module: human-initiated take-over during a running task (real-time monitoring).

Dhanorkar, Passi, and Vorvoreanu, "Human oversight of agentic systems in
practice" (arXiv:2606.05391), interviewed 17 developers and found four
emergent oversight forms: a priori control, co-planning, real-time
monitoring, and post hoc review. This folder covers three of them
(`policy_guard`/`risk_tier.py`/`risk_classifier.py` as a priori control,
`plan_review.py` as co-planning, `post_hoc.py` as post hoc review) and
lacked the fourth until now: a human interrupting and taking over a task
while it runs. `plan_review.py` approves the whole plan before any step
runs; `resume.py` is an agent-initiated durable pause at one gate. Neither
models a human deciding, mid-run, to stop the loop, edit the remaining
steps, or inject a new instruction, then let it continue. This is
Magentic-UI's "co-tasking" mechanism (Mozannar et al., arXiv:2507.22358):
pause execution at any point, adjust upcoming steps or intervene
directly, then resume.

The interrupt here is external and time-based, scheduled by position in
the step sequence rather than triggered by the agent classifying its own
action. The scripted schedule (fire before step index i, with a fixed
take-over) is fully deterministic offline; no model call is required for
the step loop itself, though the proposed steps may come from any
provider.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from agentic_patterns import ToolCall, ToolRegistry
from patterns.human_in_the_loop.fake_tools import build_extended_ops_registry
from patterns.human_in_the_loop.gate import AuditLog, AuditRecord, GateOutcome, UnauthorizedDecisionError

_VALID_TAKEOVER_KINDS = frozenset({"edit_tail", "inject", "abort"})


@dataclass
class InterruptSchedule:
    """A scripted interrupt: fires before a given step index, with a take-over action.

    Attributes:
        fire_before_index: 0-based count of steps already executed when the
            interrupt fires. None means it never fires.
        kind: One of "edit_tail", "inject", "abort", or an unrecognized
            value, to exercise the fail-closed path.
        replacement_tail: Replacement steps for "edit_tail".
        injected_step: The step to insert for "inject".
        reason: Reviewer's reason, recorded to the audit log.
        reviewer: Reviewer identity.
    """

    fire_before_index: int | None
    kind: str = "abort"
    replacement_tail: list[ToolCall] | None = None
    injected_step: ToolCall | None = None
    reason: str = ""
    reviewer: str = "unspecified"


@dataclass
class InterruptRunResult:
    """Outcome of a step loop that may have been taken over mid-run.

    Attributes:
        outcomes: One `GateOutcome` per step actually executed, in order.
        intervened_before_index: The step count the interrupt fired at, or
            None if it never fired.
        aborted: True if the take-over was an abort.
        stop_reason: Reason recorded for an abort, else empty.
    """

    outcomes: list[GateOutcome]
    intervened_before_index: int | None
    aborted: bool
    stop_reason: str


def run_interruptible_steps(
    steps: list[ToolCall],
    registry: ToolRegistry,
    audit_log: AuditLog,
    *,
    plan_id: str,
    schedule: InterruptSchedule,
    clock: Callable[[], float] = time.time,
) -> InterruptRunResult:
    """Run a step list, checking for a human interrupt before each step.

    Args:
        steps: Proposed steps, in execution order.
        registry: Where each step executes.
        audit_log: Append-only log; one entry per executed step, plus one
            for the take-over itself when the interrupt fires.
        plan_id: Identifier used as the audit record's request id.
        schedule: The scripted interrupt: when it fires and what the
            reviewer does about it. Fires at most once.
        clock: Timestamp source.

    Returns:
        The outcomes actually executed and how the run ended.

    Raises:
        UnauthorizedDecisionError: If the interrupt fires with an
            unrecognized take-over kind. No further step executes.
    """
    outcomes: list[GateOutcome] = []
    remaining = list(steps)
    executed_count = 0
    fired = False

    while remaining:
        if not fired and schedule.fire_before_index == executed_count:
            fired = True
            now = clock()
            if schedule.kind not in _VALID_TAKEOVER_KINDS:
                audit_log.append(AuditRecord(
                    plan_id, "interrupt", {}, "invalid_takeover", None,
                    schedule.reviewer, schedule.reason or "unrecognized take-over kind", now, now,
                ))
                raise UnauthorizedDecisionError(
                    f"no further step executed: take-over kind {schedule.kind!r} for plan {plan_id!r} was unrecognized"
                )
            if schedule.kind == "abort":
                reason = schedule.reason or "run aborted mid-execution"
                audit_log.append(AuditRecord(
                    plan_id, "interrupt", {}, "aborted", None, schedule.reviewer, reason, now, now,
                ))
                return InterruptRunResult(outcomes, executed_count, True, reason)
            if schedule.kind == "edit_tail":
                remaining = list(schedule.replacement_tail or [])
                audit_log.append(AuditRecord(
                    plan_id, "interrupt", {}, "edited_tail", None,
                    schedule.reviewer, schedule.reason or "remaining steps replaced", now, now,
                ))
            elif schedule.kind == "inject":
                if schedule.injected_step is not None:
                    remaining = [schedule.injected_step, *remaining]
                audit_log.append(AuditRecord(
                    plan_id, "interrupt", {}, "injected_step", None,
                    schedule.reviewer, schedule.reason or "a step was inserted", now, now,
                ))

        step = remaining.pop(0)
        result = registry.execute(step)
        now = clock()
        audit_log.append(AuditRecord(
            plan_id, step.name, step.arguments, "executed", step.arguments, "agent_loop", "step executed", now, now,
        ))
        outcomes.append(GateOutcome(kind="executed", tool_result=result, final_arguments=step.arguments))
        executed_count += 1

    return InterruptRunResult(outcomes, schedule.fire_before_index if fired else None, False, "")


def _cancellation_then_refunds(customer_id: str) -> list[ToolCall]:
    """A three-step plan: cancel, then two refund steps, the demos' shared shape."""
    return [
        ToolCall(id="step-1", name="cancel_subscription", arguments={
            "customer_id": customer_id, "reason": "customer requested cancellation",
        }),
        ToolCall(id="step-2", name="send_refund", arguments={
            "customer_id": customer_id, "amount_usd": 40.00, "reason": "prorated refund, first estimate",
        }),
        ToolCall(id="step-3", name="send_refund", arguments={
            "customer_id": customer_id, "amount_usd": 5.00, "reason": "loyalty credit",
        }),
    ]


def run_no_interrupt_demo() -> InterruptRunResult:
    """A scheduled-but-never-fired interrupt runs every step unchanged."""
    registry, _ledger = build_extended_ops_registry()
    audit_log = AuditLog()
    schedule = InterruptSchedule(fire_before_index=None)
    return run_interruptible_steps(
        _cancellation_then_refunds("c-01"), registry, audit_log, plan_id="plan-01", schedule=schedule
    )


def run_edit_mid_run_demo() -> tuple[InterruptRunResult, list]:
    """A reviewer corrects the remaining steps after step 1 already ran."""
    registry, ledger = build_extended_ops_registry()
    audit_log = AuditLog()
    corrected_tail = [ToolCall(id="step-2-corrected", name="send_refund", arguments={
        "customer_id": "c-02", "amount_usd": 22.50, "reason": "prorated refund, recalculated from the actual cycle",
    })]
    schedule = InterruptSchedule(
        fire_before_index=1, kind="edit_tail", replacement_tail=corrected_tail,
        reviewer="ops-lead-dana", reason="the estimate was wrong, correcting before it sends",
    )
    result = run_interruptible_steps(
        _cancellation_then_refunds("c-02"), registry, audit_log, plan_id="plan-02", schedule=schedule
    )
    return result, ledger


def run_inject_demo() -> tuple[InterruptRunResult, list]:
    """A reviewer inserts a tier lookup before the next step continues."""
    registry, ledger = build_extended_ops_registry()
    audit_log = AuditLog()
    injected = ToolCall(id="step-injected", name="lookup_customer_tier", arguments={"customer_id": "c-03"})
    schedule = InterruptSchedule(
        fire_before_index=1, kind="inject", injected_step=injected,
        reviewer="ops-lead-dana", reason="confirm loyalty tier before the refund amount is finalized",
    )
    result = run_interruptible_steps(
        _cancellation_then_refunds("c-03"), registry, audit_log, plan_id="plan-03", schedule=schedule
    )
    return result, ledger


def run_abort_demo() -> tuple[InterruptRunResult, list]:
    """A reviewer aborts after step 1; the executed prefix stands, nothing else runs."""
    registry, ledger = build_extended_ops_registry()
    audit_log = AuditLog()
    schedule = InterruptSchedule(
        fire_before_index=1, kind="abort", reviewer="ops-lead-dana",
        reason="customer disputed the cancellation itself; hold the refunds",
    )
    result = run_interruptible_steps(
        _cancellation_then_refunds("c-04"), registry, audit_log, plan_id="plan-04", schedule=schedule
    )
    return result, ledger


def run_malformed_takeover_demo() -> tuple[InterruptRunResult | None, str | None]:
    """An unrecognized take-over kind stops the run instead of continuing silently."""
    registry, _ledger = build_extended_ops_registry()
    audit_log = AuditLog()
    schedule = InterruptSchedule(fire_before_index=1, kind="pause_forever", reviewer="ops-lead-dana")
    try:
        result = run_interruptible_steps(
            _cancellation_then_refunds("c-05"), registry, audit_log, plan_id="plan-05", schedule=schedule
        )
        return result, None
    except UnauthorizedDecisionError as exc:
        return None, str(exc)
