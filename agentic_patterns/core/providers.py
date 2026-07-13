"""Model providers: a scriptable mock for offline examples, plus thin HTTP
clients for OpenAI-compatible APIs and Anthropic.

Every provider implements the same `Provider.complete()` signature, so
pattern examples are written once and can run offline (`MockProvider`) or
against a real API by swapping providers via `core.config.get_provider`. The
wire-format conversions (`to_openai_messages`, `to_anthropic_payload`, etc.)
are plain functions so they can be unit tested with no network call and no
installed `httpx`.
"""

from __future__ import annotations

import abc
import json
import os
from collections.abc import Sequence
from typing import Any

from agentic_patterns.core.types import Completion, Message, ToolCall


class Provider(abc.ABC):
    """Interface every model provider implements."""

    @abc.abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        """Send a conversation to the model and return its response.

        Args:
            messages: Conversation turns, not including the system prompt.
            tools: Provider-neutral tool specs, as returned by `ToolRegistry.specs()`.
            system: System prompt, kept separate so each provider can place
                it wherever its API expects.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        raise NotImplementedError


class MockScriptExhausted(RuntimeError):
    """Raised when a `MockProvider` receives more calls than it was scripted for."""


def scripted_tool_call(name: str, arguments: dict[str, Any], call_id: str | None = None) -> Completion:
    """Build a Completion representing a single tool call.

    A convenience for script authors who want an explicit `Completion`
    instead of the `{"tool": ..., "args": ...}` shorthand.

    Args:
        name: Tool name to call.
        arguments: Arguments to pass to the tool.
        call_id: Provider-style call id. Defaults to "call_1"; pass an
            explicit id when a script needs multiple distinguishable calls.
    """
    return Completion(
        tool_calls=[ToolCall(id=call_id or "call_1", name=name, arguments=arguments)],
        stop_reason="tool_use",
    )


class MockProvider(Provider):
    """A `Provider` that replays a fixed script of responses.

    Every example in this repo runs against `MockProvider` by default, so it
    can be cloned and run with no API key. Each `complete()` call pops the
    next scripted turn and records the call for test assertions.
    """

    def __init__(self, script: Sequence[Completion | str | dict[str, Any]]) -> None:
        """Build a mock provider from a script of turns.

        Each entry is normalized to a `Completion`: a `str` becomes
        `Completion(content=str)`; a `dict` shorthand `{"tool": name, "args":
        {...}}` becomes a Completion with one ToolCall, auto-assigned id
        "call_1", "call_2", ... in the order shorthand entries appear; any
        other `dict` is passed to `Completion(**d)`; a `Completion` is used
        as-is.
        """
        self._script: list[Completion] = []
        tool_call_count = 0
        for entry in script:
            if isinstance(entry, Completion):
                self._script.append(entry)
            elif isinstance(entry, str):
                self._script.append(Completion(content=entry))
            elif isinstance(entry, dict):
                if "tool" in entry:
                    tool_call_count += 1
                    call = ToolCall(
                        id=f"call_{tool_call_count}",
                        name=entry["tool"],
                        arguments=entry.get("args", {}),
                    )
                    self._script.append(Completion(tool_calls=[call], stop_reason="tool_use"))
                else:
                    self._script.append(Completion(**entry))
            else:
                raise TypeError(f"Unsupported script entry type: {type(entry)!r}")
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        if self._index >= len(self._script):
            raise MockScriptExhausted(
                f"Mock script exhausted after {len(self._script)} scripted turn(s); "
                "the example made more LLM calls than expected."
            )
        completion = self._script[self._index]
        self._index += 1
        return completion


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert conversation turns to the OpenAI chat-completions message shape.

    Does not include a system message; callers that have a `system` string
    prepend `{"role": "system", "content": system}` themselves.
    """
    result: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            result.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        elif m.role == "tool":
            result.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        else:
            result.append({"role": m.role, "content": m.content})
    return result


def to_openai_tools(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert provider-neutral tool specs to the OpenAI function-calling shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in specs
    ]


_OPENAI_STOP_REASONS = {"stop": "stop", "tool_calls": "tool_use", "length": "length"}


def parse_openai_response(data: dict[str, Any]) -> Completion:
    """Parse a raw OpenAI chat-completions response into a `Completion`."""
    choice = data["choices"][0]
    message = choice["message"]
    tool_calls = [
        ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=json.loads(tc["function"]["arguments"]))
        for tc in message.get("tool_calls") or []
    ]
    stop_reason = _OPENAI_STOP_REASONS.get(choice.get("finish_reason", "stop"), "stop")
    return Completion(content=message.get("content") or "", tool_calls=tool_calls, stop_reason=stop_reason, raw=data)


def to_anthropic_tools(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert provider-neutral tool specs to the Anthropic tool shape."""
    return [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]} for s in specs]


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert conversation turns to the Anthropic messages shape.

    Assistant tool calls become `tool_use` content blocks; tool results
    become `tool_result` blocks inside a `user` message. Consecutive tool
    results are merged into a single user message, matching Anthropic's
    expectation that all results for one turn arrive together.
    """
    result: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue  # system is a top-level payload field, not a message
        if m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            result.append({"role": "assistant", "content": blocks if blocks else m.content})
        elif m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            last = result[-1] if result else None
            if last and last["role"] == "user" and isinstance(last["content"], list) and last["content"][0].get("type") == "tool_result":
                last["content"].append(block)
            else:
                result.append({"role": "user", "content": [block]})
        else:
            result.append({"role": "user", "content": m.content})
    return result


def to_anthropic_payload(
    messages: list[Message],
    tools: list[dict[str, Any]] | None,
    system: str | None,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Build the request body for `POST /v1/messages`."""
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": _to_anthropic_messages(messages),
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = to_anthropic_tools(tools)
    return payload


_ANTHROPIC_STOP_REASONS = {"end_turn": "stop", "tool_use": "tool_use", "max_tokens": "length"}


def parse_anthropic_response(data: dict[str, Any]) -> Completion:
    """Parse a raw Anthropic Messages API response into a `Completion`."""
    content_text = ""
    tool_calls: list[ToolCall] = []
    for block in data.get("content", []):
        if block["type"] == "text":
            content_text += block["text"]
        elif block["type"] == "tool_use":
            tool_calls.append(ToolCall(id=block["id"], name=block["name"], arguments=block.get("input", {})))
    stop_reason = _ANTHROPIC_STOP_REASONS.get(data.get("stop_reason", "end_turn"), "stop")
    return Completion(content=content_text, tool_calls=tool_calls, stop_reason=stop_reason, raw=data)


def _require_httpx() -> Any:
    """Import httpx lazily, with a clear error if the extra isn't installed."""
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            'This provider requires httpx. Install with: pip install "agentic-patterns[providers]"'
        ) from exc
    return httpx


class OpenAICompatibleProvider(Provider):
    """A `Provider` for OpenAI and OpenAI-compatible chat-completions APIs."""

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None) -> None:
        self._httpx = _require_httpx()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        payload_messages: list[dict[str, Any]] = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        payload_messages.extend(to_openai_messages(messages))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
        response = self._httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        return parse_openai_response(response.json())


class AnthropicProvider(Provider):
    """A `Provider` for the Anthropic Messages API."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self._httpx = _require_httpx()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Completion:
        payload = to_anthropic_payload(messages, tools, system, self.model, temperature, max_tokens)
        response = self._httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        return parse_anthropic_response(response.json())
