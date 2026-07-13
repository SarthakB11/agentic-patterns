"""Core data types shared by every agentic pattern in this repo.

These are the provider-neutral shapes that flow between an agent loop, a
`Provider`, and a `ToolRegistry`. Every pattern example builds on these three
types plus `Completion`, so they stay intentionally small and dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation requested by a model.

    Attributes:
        id: Provider-assigned identifier for this call. Used to match the
            corresponding tool result back to the request.
        name: Name of the tool to invoke.
        arguments: Parsed keyword arguments for the tool call.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A single turn in a conversation.

    Attributes:
        role: One of "system", "user", "assistant", "tool".
        content: Text content of the turn. Empty string when a turn is pure
            tool calls with no accompanying text.
        tool_calls: Tool calls requested by an assistant turn. Empty for all
            other roles.
        tool_call_id: For role="tool", the id of the ToolCall this message is
            the result of. None otherwise.
    """

    role: str
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    @classmethod
    def system(cls, text: str) -> Message:
        """Build a system message."""
        return cls(role="system", content=text)

    @classmethod
    def user(cls, text: str) -> Message:
        """Build a user message."""
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, text: str, tool_calls: list[ToolCall] | None = None) -> Message:
        """Build an assistant message, optionally carrying tool calls."""
        return cls(role="assistant", content=text, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        """Build a tool-result message that answers a prior ToolCall."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


@dataclass
class Completion:
    """A model's response to a `Provider.complete()` call.

    Attributes:
        content: Text the model produced. Empty when the model only made
            tool calls.
        tool_calls: Tool calls the model requested, if any.
        stop_reason: One of "stop", "tool_use", "length".
        raw: The raw provider response, kept for debugging. Never relied on
            by pattern code.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "stop"
    raw: Any = None
