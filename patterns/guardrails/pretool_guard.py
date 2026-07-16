"""Execution (pre-tool) guard: validates a tool call before it runs.

This is deterministic code, not a model judgment, matching how the Claude
Agent SDK's `PreToolUse` hook and the OpenAI Agents SDK's tool guardrails
work: a plain function inspects the tool name and arguments and decides
allow, block, or pause for a human, before the tool's side effect happens.
Validation itself never performs the action it is gating; `execute_guarded`
only calls the registry after the guard has passed or a human has approved.

`PreToolGuard` checks two things per tool, from a static policy:

- the tool name against an allowlist,
- each constrained argument against a numeric range.

A third case, an argument over a soft "approval threshold", does not fail
outright: it sets `requires_approval=True` on the result and routes to a
human-approval callback, matching the brief's "may block the call or route
it to human approval" and the OpenAI Agents SDK's `needsApproval` pattern.
Input validation (allowlist, ranges) always runs before the approval pause,
never after, so an invalid call is never presented to a human as if it were
merely borderline.

This static policy shape (a fixed dict of ranges and thresholds, written
once in Python) is the simplest point on the privilege-control design
space, not the one the 2025 research settled on. Shi et al.'s Progent
(arXiv:2504.11703) makes the policy declarative data instead: an ordered
rule list a task can generate and narrow as it learns more, with privilege
expansion always gated on a human. See `policy_engine.py` for that general,
programmable shape; this module stays as the hard-coded starting point it
generalizes from.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from agentic_patterns import ToolCall, ToolRegistry
from patterns.guardrails.core import DecisionLog, GuardResult, OnFail, run_guard


@dataclass
class ToolPolicy:
    """Static policy for one allowlisted tool.

    Attributes:
        arg_ranges: Argument name to an inclusive `(min, max)` range. An
            argument outside its range blocks the call outright.
        approval_over: Argument name to a threshold above which the call is
            valid but requires human approval before it runs.
    """

    arg_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    approval_over: dict[str, float] = field(default_factory=dict)


@dataclass
class PreToolGuard:
    """Validates a `ToolCall` against a per-tool allowlist and policy.

    Attributes:
        name: Guard name.
        policies: Tool name to `ToolPolicy`. A tool not present here is not
            on the allowlist and is always blocked.
        on_fail: Action when the tool is not allowlisted or an argument is
            out of range. Defaults to `OnFail.REFRAIN`.
    """

    name: str = "pretool_guard"
    policies: dict[str, ToolPolicy] = field(default_factory=dict)
    on_fail: OnFail = OnFail.REFRAIN

    def check(self, value: ToolCall) -> GuardResult:
        policy = self.policies.get(value.name)
        if policy is None:
            return GuardResult(
                passed=False,
                action=self.on_fail,
                value=value,
                message=f"tool {value.name!r} is not on the allowlist",
            )
        for arg, (lo, hi) in policy.arg_ranges.items():
            if arg in value.arguments:
                arg_value = value.arguments[arg]
                if not (lo <= arg_value <= hi):
                    return GuardResult(
                        passed=False,
                        action=self.on_fail,
                        value=value,
                        message=f"argument {arg}={arg_value!r} outside allowed range [{lo}, {hi}]",
                    )
        for arg, threshold in policy.approval_over.items():
            if arg in value.arguments and value.arguments[arg] > threshold:
                return GuardResult(
                    passed=False,
                    action=OnFail.REFRAIN,
                    value=value,
                    message=f"argument {arg}={value.arguments[arg]!r} exceeds auto-approve threshold {threshold}",
                    requires_approval=True,
                )
        return GuardResult(passed=True, action=OnFail.NOOP, value=value)


def execute_guarded(
    call: ToolCall,
    guard: PreToolGuard,
    registry: ToolRegistry,
    log: DecisionLog,
    *,
    human_approve: Callable[[ToolCall, str], bool] | None = None,
) -> str:
    """Validate `call`, then execute it only if the guard clears it.

    Args:
        call: The tool call a model requested.
        guard: The pre-tool guard to validate against.
        registry: Where to execute the call once it is cleared.
        log: Decision log to record the guard's decision into.
        human_approve: Called with `(call, guard_message)` when the guard
            flags `requires_approval`. Returns True to approve execution,
            False to deny it. Required when any policy sets
            `approval_over`; omitted for demos with no such policy.

    Returns:
        The tool's observation string on success, or a "BLOCKED: ..."
        string when the guard (or a human) denies the call. Never raises
        for a denial: like `ToolRegistry.execute`, a blocked call is an
        observation the caller can react to, not a crash.
    """
    result = run_guard(guard, call, log)

    if result.passed:
        return registry.execute(call)

    if result.requires_approval:
        if human_approve is None:
            return f"BLOCKED: {result.message} (no human-approval channel configured)"
        approved = human_approve(call, result.message)
        if approved:
            return registry.execute(call)
        return f"BLOCKED: human denied approval ({result.message})"

    return f"BLOCKED: {result.message}"


def _demo_registry() -> ToolRegistry:
    """Build a small support-bot tool registry for the demo."""
    registry = ToolRegistry()

    def lookup_order(order_id: str) -> str:
        return f"order {order_id}: shipped, arriving in 2 business days"

    def issue_refund(order_id: str, amount: float) -> str:
        return f"refunded ${amount:.2f} to order {order_id}"

    def delete_account(user_id: str) -> str:
        return f"account {user_id} deleted"

    registry.tool(
        description="Look up an order's status by id.",
        parameters={"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    )(lookup_order)

    registry.tool(
        description="Issue a refund for an order.",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["order_id", "amount"],
        },
    )(issue_refund)

    registry.tool(
        description="Permanently delete a user account.",
        parameters={"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
    )(delete_account)

    return registry


def run_pretool_guard_demo() -> list[str]:
    """Run five tool calls through the guard: a disallowed tool, an out-of-range
    argument, an approved over-threshold refund, a denied one, and a clean call.

    `delete_account` is a real, registered tool; it is blocked here because
    the guard's policy never allowlists it, which is the point: registration
    in the tool registry and authorization to call it are two separate
    concerns, and this guard only handles the second one.

    Returns:
        One observation string per call, in the order the calls were made.
    """
    registry = _demo_registry()
    guard = PreToolGuard(
        policies={
            "lookup_order": ToolPolicy(),
            "issue_refund": ToolPolicy(arg_ranges={"amount": (0, 1000)}, approval_over={"amount": 500}),
        }
    )
    log = DecisionLog()
    calls = [
        ToolCall(id="call_1", name="delete_account", arguments={"user_id": "u_42"}),
        ToolCall(id="call_2", name="issue_refund", arguments={"order_id": "ORD-9", "amount": 5000}),
        ToolCall(id="call_3", name="issue_refund", arguments={"order_id": "ORD-10", "amount": 750}),
        ToolCall(id="call_4", name="issue_refund", arguments={"order_id": "ORD-11", "amount": 900}),
        ToolCall(id="call_5", name="lookup_order", arguments={"order_id": "ORD-12"}),
    ]
    approvals = {"call_3": True, "call_4": False}

    print("=== Pre-tool guard: validating tool calls before they execute ===")
    results: list[str] = []
    for call in calls:
        human_approve = (lambda c, msg, _id=call.id: approvals[_id]) if call.id in approvals else None
        observation = execute_guarded(call, guard, registry, log, human_approve=human_approve)
        print(f"  {call.name}({call.arguments}) -> {observation}")
        results.append(observation)
    print(log.render())

    return results
