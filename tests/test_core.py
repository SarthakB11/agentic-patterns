"""Tests for the shared agentic-patterns core.

Deterministic and offline: no network calls, and no dependency on httpx
being installed. The provider/embedder-selection tests use monkeypatch to
control environment variables rather than mutating the real environment.
"""

from __future__ import annotations

import pytest

from agentic_patterns.core.config import get_embedder, get_provider
from agentic_patterns.core.embeddings import HashEmbedder, cosine_similarity
from agentic_patterns.core.providers import (
    Completion,
    MockProvider,
    MockScriptExhausted,
    ToolCall,
    parse_anthropic_response,
    parse_openai_response,
    scripted_tool_call,
    to_anthropic_payload,
    to_openai_messages,
)
from agentic_patterns.core.tools import Tool, ToolRegistry
from agentic_patterns.core.types import Message

# --- MockProvider ---------------------------------------------------------


def test_mock_provider_str_entry_becomes_text_completion() -> None:
    provider = MockProvider(["hello there"])
    result = provider.complete([Message.user("hi")])
    assert result.content == "hello there"
    assert result.stop_reason == "stop"
    assert result.tool_calls == []


def test_mock_provider_dict_shorthand_becomes_tool_call() -> None:
    provider = MockProvider([{"tool": "search", "args": {"query": "cats"}}])
    result = provider.complete([Message.user("look up cats")])
    assert result.stop_reason == "tool_use"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0] == ToolCall(id="call_1", name="search", arguments={"query": "cats"})


def test_mock_provider_shorthand_ids_increment_by_position() -> None:
    provider = MockProvider(
        [
            {"tool": "a", "args": {}},
            "some text in between",
            {"tool": "b", "args": {}},
        ]
    )
    first = provider.complete([Message.user("go")])
    provider.complete([Message.user("go")])
    third = provider.complete([Message.user("go")])
    assert first.tool_calls[0].id == "call_1"
    assert third.tool_calls[0].id == "call_2"


def test_mock_provider_explicit_completion_entry() -> None:
    completion = Completion(content="explicit", stop_reason="length")
    provider = MockProvider([completion])
    result = provider.complete([Message.user("hi")])
    assert result is completion


def test_mock_provider_records_calls() -> None:
    provider = MockProvider(["ok"])
    tools = [{"name": "t", "description": "d", "parameters": {}}]
    provider.complete([Message.user("hi")], tools=tools, system="be nice")
    assert len(provider.calls) == 1
    assert provider.calls[0]["tools"] == tools
    assert provider.calls[0]["system"] == "be nice"
    assert provider.calls[0]["messages"][0].content == "hi"


def test_mock_provider_exhaustion_raises() -> None:
    provider = MockProvider(["only one turn"])
    provider.complete([Message.user("hi")])
    with pytest.raises(MockScriptExhausted):
        provider.complete([Message.user("again")])


def test_scripted_tool_call_helper() -> None:
    completion = scripted_tool_call("lookup", {"x": 1}, call_id="call_9")
    assert completion.stop_reason == "tool_use"
    assert completion.tool_calls == [ToolCall(id="call_9", name="lookup", arguments={"x": 1})]


# --- ToolRegistry -----------------------------------------------------------


def test_tool_registry_register_and_specs() -> None:
    registry = ToolRegistry()
    registry.register(Tool(name="add", description="Add two numbers", parameters={"type": "object"}, fn=lambda a, b: a + b))
    specs = registry.specs()
    assert specs == [{"name": "add", "description": "Add two numbers", "parameters": {"type": "object"}}]


def test_tool_registry_decorator_form() -> None:
    registry = ToolRegistry()

    @registry.tool(description="Multiply two numbers", parameters={"type": "object"})
    def multiply(a: int, b: int) -> int:
        return a * b

    tool = registry.get("multiply")
    assert tool.description == "Multiply two numbers"
    assert tool.fn(2, 3) == 6


def test_tool_registry_execute_happy_path() -> None:
    registry = ToolRegistry()
    registry.register(Tool(name="add", description="", parameters={}, fn=lambda a, b: a + b))
    result = registry.execute(ToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3}))
    assert result == "5"


def test_tool_registry_execute_catches_exceptions() -> None:
    def boom() -> None:
        raise ValueError("bad input")

    registry = ToolRegistry()
    registry.register(Tool(name="boom", description="", parameters={}, fn=boom))
    result = registry.execute(ToolCall(id="call_1", name="boom", arguments={}))
    assert result == "ERROR: bad input"


def test_tool_registry_get_unknown_lists_known_names() -> None:
    registry = ToolRegistry()
    registry.register(Tool(name="add", description="", parameters={}, fn=lambda: None))
    registry.register(Tool(name="sub", description="", parameters={}, fn=lambda: None))
    with pytest.raises(KeyError) as exc_info:
        registry.get("missing")
    message = str(exc_info.value)
    assert "add" in message
    assert "sub" in message


# --- HashEmbedder -------------------------------------------------------


def test_hash_embedder_deterministic() -> None:
    embedder = HashEmbedder()
    a = embedder.embed(["the cat sat on the mat"])[0]
    b = embedder.embed(["the cat sat on the mat"])[0]
    assert a == b


def test_hash_embedder_similarity_reflects_token_overlap() -> None:
    embedder = HashEmbedder()
    query, close, far = embedder.embed(
        ["the cat sat", "the cat sat on the mat", "quarterly finance report"]
    )
    assert cosine_similarity(query, close) > cosine_similarity(query, far)


def test_hash_embedder_vectors_are_unit_norm() -> None:
    embedder = HashEmbedder()
    vector = embedder.embed(["some text with several tokens"])[0]
    norm = sum(v * v for v in vector) ** 0.5
    assert norm == pytest.approx(1.0)


def test_hash_embedder_empty_text_is_zero_vector() -> None:
    embedder = HashEmbedder()
    vector = embedder.embed([""])[0]
    assert vector == [0.0] * 256


def test_hash_embedder_respects_dim() -> None:
    embedder = HashEmbedder(dim=16)
    vector = embedder.embed(["some text"])[0]
    assert len(vector) == 16


# --- cosine_similarity ------------------------------------------------------


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# --- Format converters ----------------------------------------------------


def _sample_conversation() -> list[Message]:
    call = ToolCall(id="call_1", name="search", arguments={"query": "cats"})
    return [
        Message.user("find cats"),
        Message.assistant("", tool_calls=[call]),
        Message.tool("call_1", "found 3 cats"),
    ]


def test_to_openai_messages_shapes_tool_calls_and_results() -> None:
    result = to_openai_messages(_sample_conversation())
    assistant_msg = result[1]
    assert assistant_msg["role"] == "assistant"
    assert isinstance(assistant_msg["tool_calls"][0]["function"]["arguments"], str)
    import json

    assert json.loads(assistant_msg["tool_calls"][0]["function"]["arguments"]) == {"query": "cats"}

    tool_msg = result[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "found 3 cats"


def test_to_anthropic_payload_shapes_tool_use_and_results() -> None:
    payload = to_anthropic_payload(
        _sample_conversation(), tools=None, system="be helpful", model="claude-opus-4-8",
        temperature=0.0, max_tokens=512,
    )
    assert payload["system"] == "be helpful"
    assert all(m["role"] != "system" for m in payload["messages"])

    assistant_msg = payload["messages"][1]
    assert assistant_msg["role"] == "assistant"
    tool_use_blocks = [b for b in assistant_msg["content"] if b["type"] == "tool_use"]
    assert tool_use_blocks == [{"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "cats"}}]

    tool_result_msg = payload["messages"][2]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "found 3 cats"}
    ]


def test_parse_openai_response_text() -> None:
    data = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi there"}}]}
    completion = parse_openai_response(data)
    assert completion.content == "hi there"
    assert completion.stop_reason == "stop"
    assert completion.tool_calls == []


def test_parse_openai_response_tool_call() -> None:
    data = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"query": "cats"}'}}
                    ],
                },
            }
        ]
    }
    completion = parse_openai_response(data)
    assert completion.stop_reason == "tool_use"
    assert completion.tool_calls == [ToolCall(id="call_1", name="search", arguments={"query": "cats"})]


def test_parse_anthropic_response_text() -> None:
    data = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "hi there"}]}
    completion = parse_anthropic_response(data)
    assert completion.content == "hi there"
    assert completion.stop_reason == "stop"


def test_parse_anthropic_response_tool_use() -> None:
    data = {
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "let me check"},
            {"type": "tool_use", "id": "call_1", "name": "search", "input": {"query": "cats"}},
        ],
    }
    completion = parse_anthropic_response(data)
    assert completion.content == "let me check"
    assert completion.stop_reason == "tool_use"
    assert completion.tool_calls == [ToolCall(id="call_1", name="search", arguments={"query": "cats"})]


def test_parse_anthropic_response_max_tokens() -> None:
    data = {"stop_reason": "max_tokens", "content": [{"type": "text", "text": "cut off"}]}
    completion = parse_anthropic_response(data)
    assert completion.stop_reason == "length"


# --- get_provider / get_embedder -------------------------------------------


def test_get_provider_defaults_to_mock() -> None:
    provider = get_provider(script=["hi"])
    assert isinstance(provider, MockProvider)


def test_get_provider_mock_without_script_raises() -> None:
    with pytest.raises(ValueError):
        get_provider(name="mock")


def test_get_provider_unknown_name_raises() -> None:
    with pytest.raises(ValueError):
        get_provider(script=["hi"], name="not-a-real-provider")


def test_get_provider_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_PATTERNS_PROVIDER", "mock")
    provider = get_provider(script=["hi"])
    assert isinstance(provider, MockProvider)

    monkeypatch.setenv("AGENTIC_PATTERNS_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError):
        get_provider(script=["hi"])


def test_get_embedder_defaults_to_hash() -> None:
    assert isinstance(get_embedder(), HashEmbedder)


def test_get_embedder_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_PATTERNS_EMBEDDER", "hash")
    assert isinstance(get_embedder(), HashEmbedder)

    monkeypatch.setenv("AGENTIC_PATTERNS_EMBEDDER", "not-a-real-embedder")
    with pytest.raises(ValueError):
        get_embedder()


def test_mock_provider_snapshots_messages_per_call() -> None:
    """calls[i] must record what was sent on call i, even if the caller
    keeps appending to the same history list afterward, as agent loops do."""
    provider = MockProvider(["first", "second"])
    history = [Message.user("question")]
    provider.complete(history)
    history.append(Message.assistant("first"))
    provider.complete(history)

    assert len(provider.calls[0]["messages"]) == 1
    assert len(provider.calls[1]["messages"]) == 2


def test_completion_and_message_carry_a_reasoning_channel() -> None:
    """The reasoning channel is opaque text, defaulting to empty."""
    c = Completion(content="answer", reasoning="I checked the observation first.")
    assert c.reasoning == "I checked the observation first."
    m = Message.assistant("answer", reasoning=c.reasoning)
    assert m.reasoning == c.reasoning
    assert Message.assistant("plain").reasoning == ""


def test_parse_anthropic_response_captures_thinking_blocks() -> None:
    data = {
        "content": [
            {"type": "thinking", "thinking": "The capital question needs one lookup."},
            {"type": "text", "text": "Paris"},
        ],
        "stop_reason": "end_turn",
    }
    completion = parse_anthropic_response(data)
    assert completion.reasoning == "The capital question needs one lookup."
    assert completion.content == "Paris"


def test_parse_openai_response_captures_reasoning_content() -> None:
    data = {
        "choices": [
            {
                "message": {"content": "Paris", "reasoning_content": "One lookup suffices."},
                "finish_reason": "stop",
            }
        ]
    }
    completion = parse_openai_response(data)
    assert completion.reasoning == "One lookup suffices."
    assert completion.content == "Paris"
