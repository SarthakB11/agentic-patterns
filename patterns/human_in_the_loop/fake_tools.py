"""Shared scenario: a support agent that can issue refunds.

Every variant module reuses this one side-effecting fake tool so a reader
sees the same scenario across the whole pattern and so approve versus
reject is observable through the ledger's own state, not through mocked
network calls. `send_refund` is the "sending money" example the research
brief names as a canonical case for gating: irreversible, and worth a
second pair of eyes above a threshold.
"""

from __future__ import annotations

from typing import Any

from agentic_patterns import ToolRegistry

REFUND_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "customer_id": {"type": "string", "description": "Customer account id."},
        "amount_usd": {"type": "number", "description": "Refund amount in US dollars."},
        "reason": {"type": "string", "description": "Why the refund is being issued."},
    },
    "required": ["customer_id", "amount_usd", "reason"],
}


def _record_refund(ledger: list[dict[str, Any]], customer_id: str, amount_usd: float, reason: str) -> str:
    """Append a refund entry to `ledger` and return the confirmation message.

    Shared by every registry builder that exposes `send_refund`, so the
    ledger schema (including the `type` discriminator) stays identical no
    matter which registry produced the entry.

    Args:
        ledger: The ledger list to append the refund entry to.
        customer_id: Customer account id.
        amount_usd: Refund amount in US dollars.
        reason: Why the refund is being issued.

    Returns:
        A human-readable confirmation string.
    """
    ledger.append({"type": "refund", "customer_id": customer_id, "amount_usd": amount_usd, "reason": reason})
    return f"refund of ${amount_usd:.2f} sent to {customer_id} ({reason})"


def build_refund_registry() -> tuple[ToolRegistry, list[dict[str, Any]]]:
    """Build a `ToolRegistry` with one side-effecting `send_refund` tool.

    Returns:
        A tuple of the registry and the ledger list the tool appends to.
        The ledger is the observable proof that an action actually ran:
        an approved gate grows it, a rejected one leaves it untouched.
        Each entry has a `type` key ("refund") alongside `customer_id`,
        `amount_usd`, and `reason`.
    """
    ledger: list[dict[str, Any]] = []
    registry = ToolRegistry()

    @registry.tool(
        description="Send a refund payment to a customer. Irreversible once sent.",
        parameters=REFUND_TOOL_PARAMETERS,
    )
    def send_refund(customer_id: str, amount_usd: float, reason: str) -> str:
        return _record_refund(ledger, customer_id, amount_usd, reason)

    return registry, ledger


def build_support_ops_registry() -> tuple[ToolRegistry, list[dict[str, Any]]]:
    """Build a registry with `send_refund` and `cancel_subscription`, sharing one ledger.

    Delegates to `build_refund_registry` for the `send_refund` tool and its
    ledger, then adds `cancel_subscription` onto the same registry and
    ledger so both tools log entries with a matching schema. Used by the
    plan-review variant, where a plan is a short sequence of different
    action types reviewed together before any of them run.
    """
    registry, ledger = build_refund_registry()

    @registry.tool(
        description="Cancel a customer's subscription, effective immediately.",
        parameters={
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["customer_id", "reason"],
        },
    )
    def cancel_subscription(customer_id: str, reason: str) -> str:
        ledger.append({"type": "cancellation", "customer_id": customer_id, "reason": reason})
        return f"subscription for {customer_id} canceled ({reason})"

    return registry, ledger


def build_reversal_registry(ledger: list[dict[str, Any]]) -> ToolRegistry:
    """Build a registry with a `reverse_refund` tool that undoes the last refund.

    Used by the post-hoc review variant, where an action already executed
    and a reviewer later decides it must be rolled back.

    Args:
        ledger: The same ledger list `send_refund` appended to, so a
            reversal can pop the matching entry back off.
    """
    registry = ToolRegistry()

    @registry.tool(
        description="Reverse the most recent refund sent to a customer.",
        parameters={
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    )
    def reverse_refund(customer_id: str) -> str:
        for i in range(len(ledger) - 1, -1, -1):
            if ledger[i]["customer_id"] == customer_id:
                entry = ledger.pop(i)
                return f"reversed refund of ${entry['amount_usd']:.2f} to {customer_id}"
        return f"ERROR: no refund found for {customer_id}"

    return registry
