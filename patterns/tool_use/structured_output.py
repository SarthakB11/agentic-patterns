"""Structured-output-as-tool: extraction with no side effect.

A degenerate case of forced tool choice. A single tool is offered, its
schema describes the shape of the answer rather than an action, no
underlying system is touched, and the call's arguments are the final
answer. This is the standard way to get typed extraction or classification
out of a model that only has a text-completion and a tool-calling
interface: one completion, one call, done.
"""

from __future__ import annotations

from typing import Any

from agentic_patterns import Message, ToolRegistry, get_provider, scripted_tool_call

from patterns.tool_use.loop import validate_arguments
from patterns.tool_use.schema import auto_tool

SYSTEM_PROMPT = (
    "Extract the requested structured data from the user's message by "
    "calling record_contact with the fields filled in. Do not respond in "
    "plain text."
)


def build_extraction_registry() -> ToolRegistry:
    """Build a registry with one tool whose only purpose is to shape the answer."""
    registry = ToolRegistry()

    @auto_tool(registry)
    def record_contact(name: str, email: str, phone: str) -> dict[str, str]:
        """Record an extracted contact's name, email, and phone number.

        Args:
            name: Full name as written in the source text.
            email: Email address as written in the source text.
            phone: Phone number as written in the source text.
        """
        # No side effect: this tool's only job is to give the model a typed
        # shape to fill in. The return value doubles as the validated answer.
        return {"name": name, "email": email, "phone": phone}

    return registry


def extract_structured(text: str) -> dict[str, Any]:
    """Run one forced-tool completion and return its arguments as the answer.

    Unlike `run_tool_loop`, this never asks the model a second time: once
    the single forced tool has been called with schema-valid arguments,
    there is nothing left to reason about, so a second round trip would only
    cost tokens for no benefit.

    Args:
        text: Free text to extract a contact record from.

    Returns:
        The validated `{"name": ..., "email": ..., "phone": ...}` mapping.

    Raises:
        ValueError: If the model's call has no tool call, or its arguments
            fail schema validation.
    """
    registry = build_extraction_registry()
    tool = registry.get("record_contact")
    provider = get_provider(
        script=[
            scripted_tool_call(
                "record_contact",
                {"name": "Dana Alvarez", "email": "dana@example.com", "phone": "+1-555-0199"},
            )
        ]
    )
    messages = [Message.user(text)]

    completion = provider.complete(messages, tools=registry.specs(), system=SYSTEM_PROMPT)
    if not completion.tool_calls:
        raise ValueError("expected a forced record_contact call, got plain text")

    call = completion.tool_calls[0]
    errors = validate_arguments(tool.parameters, call.arguments)
    if errors:
        raise ValueError(f"extraction failed schema validation: {'; '.join(errors)}")

    return dict(call.arguments)


def demo_structured_output() -> dict[str, Any]:
    """Extract a contact record from free text with one forced-tool call."""
    text = "Reach out to Dana Alvarez, dana@example.com, +1-555-0199, about the renewal."
    result = extract_structured(text)

    print("=== 5. Structured-output-as-tool (forced extraction, no side effect) ===")
    print(f"source: {text}")
    print(f"extracted: {result}")
    print()
    return result


if __name__ == "__main__":
    demo_structured_output()
