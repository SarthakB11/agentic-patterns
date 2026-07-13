"""Write actions: gated behind confirmation, with an elicitation step.

Read-only tools are safe to run on a model's say-so; a write action (send an
email, issue a refund, update a record) is not, and the brief calls this out
as the highest-risk tool category. `send_refund_email` below is gated by a
`confirmed` flag the tool itself enforces: called with `confirmed=False`
(the model's default, since it cannot be trusted to set this itself) it
raises, and that raise becomes an ordinary error observation.

Getting past the gate uses `elicit_confirmation`, a stand-in for MCP's
`elicitation/create` primitive: a mid-loop, out-of-band question to the
human with an accept, decline, or cancel answer, turned into a message the
model sees on its next turn. This is the protocol-level version of the
brief's "gate behind confirmation": the human, not the model, decides
whether the write action proceeds.
"""

from __future__ import annotations

from agentic_patterns import Message, ToolRegistry, get_provider, scripted_tool_call

from patterns.tool_use.schema import auto_tool
from patterns.tool_use.loop import ToolLoopResult, run_tool_loop

SYSTEM_PROMPT = (
    "You are an ops assistant. send_refund_email is a write action: it must "
    "be called with confirmed=True, and confirmed may only be set to True "
    "after the human has explicitly confirmed the request in the "
    "conversation. Never set confirmed=True on your own judgment."
)

WRITE_ACTION_ALLOWLIST = {"send_refund_email"}


def build_write_registry() -> ToolRegistry:
    """Build a registry containing the one gated write-action tool."""
    registry = ToolRegistry()

    @auto_tool(registry)
    def send_refund_email(to: str, subject: str, body: str, confirmed: bool = False) -> str:
        """Send a refund confirmation email. A write action: requires human confirmation.

        Args:
            to: Recipient email address.
            subject: Email subject line.
            body: Email body text.
            confirmed: Must be True, set only after the human confirmed.
        """
        if not confirmed:
            raise PermissionError("refund email blocked: requires confirmed=True after human confirmation")
        return f"sent to {to}: subject={subject!r}"

    return registry


def elicit_confirmation(prompt: str, decision: str) -> Message:
    """Simulate an elicitation/create round trip and turn the answer into a user message.

    In a real system this pauses the loop and asks a human out of band. The
    demos here simulate that pause explicitly by calling `run_tool_loop`
    twice: once up to the blocked call, then again after this function's
    message has been appended to the history.

    Args:
        prompt: The yes/no question shown to the human.
        decision: "accept", "decline", or "cancel".

    Returns:
        A user `Message` carrying the human's decision back into the
        conversation.

    Raises:
        ValueError: If `decision` is not a recognized elicitation outcome.
    """
    replies = {
        "accept": "Yes, go ahead and send it.",
        "decline": "No, don't send that.",
        "cancel": "Never mind, cancel this request.",
    }
    if decision not in replies:
        raise ValueError(f"unknown elicitation decision {decision!r}")
    return Message.user(f"[confirmation requested: {prompt}] {replies[decision]}")


def demo_write_action_accepted() -> ToolLoopResult:
    """The model attempts the write action, is blocked, elicits confirmation, then succeeds."""
    registry = build_write_registry()
    provider = get_provider(
        script=[
            scripted_tool_call(
                "send_refund_email",
                {
                    "to": "priya@example.com",
                    "subject": "Refund processed",
                    "body": "Your refund for ORD-1001 has been processed.",
                    "confirmed": False,
                },
            ),
            scripted_tool_call(
                "send_refund_email",
                {
                    "to": "priya@example.com",
                    "subject": "Refund processed",
                    "body": "Your refund for ORD-1001 has been processed.",
                    "confirmed": True,
                },
            ),
            "The refund email has been sent to priya@example.com confirming the refund for order ORD-1001.",
        ]
    )
    messages = [Message.user("Email priya@example.com to say the refund for ORD-1001 was processed.")]

    blocked = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=1)
    blocked_call = blocked.rounds[0].calls[0]

    confirmation = elicit_confirmation(
        f"Send refund email to {blocked_call.call.arguments['to']}?", decision="accept"
    )
    history_with_confirmation = blocked.history + [confirmation]
    finished = run_tool_loop(
        provider, registry, history_with_confirmation, system=SYSTEM_PROMPT, max_iterations=2
    )

    retried_call = finished.rounds[0].calls[0]

    print("=== 8a. Write action: blocked, elicited, confirmed, executed ===")
    print(f"user:  {messages[0].content}")
    print(f"  blocked:  {blocked_call.observation} (outcome={blocked_call.outcome})")
    print(f"  elicitation: {confirmation.content}")
    print(f"  retried:  {retried_call.observation} (outcome={retried_call.outcome})")
    print(f"final: {finished.final_answer}")
    print()
    return finished


def demo_write_action_declined() -> None:
    """The human declines; the app stops without ever retrying the call."""
    registry = build_write_registry()
    provider = get_provider(
        script=[
            scripted_tool_call(
                "send_refund_email",
                {"to": "sam@example.com", "subject": "Refund processed", "body": "...", "confirmed": False},
            )
        ]
    )
    messages = [Message.user("Email sam@example.com about the refund on ORD-1002.")]

    blocked = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=1)
    confirmation = elicit_confirmation("Send refund email to sam@example.com?", decision="decline")

    print("=== 8b. Write action: elicitation declined, request stopped ===")
    print(f"user:  {messages[0].content}")
    print(f"  blocked: {blocked.rounds[0].calls[0].observation}")
    print(f"  elicitation: {confirmation.content}")
    print("  decision: decline -> not retried, no email sent")
    print()


if __name__ == "__main__":
    demo_write_action_accepted()
    demo_write_action_declined()
