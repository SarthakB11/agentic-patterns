"""Tests for the MCP pattern.

Deterministic and offline: no network call and no API key anywhere. Some
tests spawn the real `patterns.mcp.server` subprocess over stdio, which is
still offline and fully deterministic; those tests always shut the
subprocess down in a `finally` block. Model calls go through `MockProvider`
scripts, asserted against `MockProvider.calls` where the test cares what
was sent to the model.
"""

from __future__ import annotations

import pytest

from agentic_patterns import Message, MockProvider, ToolRegistry, scripted_tool_call

from patterns.mcp import bridge, http_transport, jsonrpc, multi_server, sampling, server, server_data
from patterns.mcp.client import MCPClient, MCPProtocolError
from patterns.mcp.transport import TransportTimeoutError

# --- jsonrpc codec -----------------------------------------------------


def test_build_request_and_decode_line_roundtrip() -> None:
    request = jsonrpc.build_request("id-1", "tools/call", {"name": "add", "arguments": {"a": 1, "b": 2}})
    line = jsonrpc.encode_line(request)
    assert line.endswith("\n")
    decoded = jsonrpc.decode_line(line)
    assert decoded == request


def test_decode_line_rejects_embedded_newline() -> None:
    # A literal, unescaped newline inside a JSON string value is invalid JSON
    # (control characters must be escaped as \n), which is exactly the case
    # the newline-delimited framing on stdio must guard against.
    with pytest.raises(jsonrpc.JSONRPCDecodeError):
        jsonrpc.decode_line('{"jsonrpc": "2.0", "id": 1, "method": "x", "params": {"note": "line one\nline two"}}')


def test_decode_line_rejects_missing_jsonrpc_field() -> None:
    with pytest.raises(jsonrpc.JSONRPCDecodeError):
        jsonrpc.decode_line('{"id": 1, "method": "tools/list"}')


def test_message_shape_helpers() -> None:
    request = jsonrpc.build_request(1, "tools/list")
    notification = jsonrpc.build_notification("notifications/initialized")
    response = jsonrpc.build_response(1, {"tools": []})
    assert jsonrpc.is_request(request) and not jsonrpc.is_notification(request) and not jsonrpc.is_response(request)
    assert jsonrpc.is_notification(notification) and not jsonrpc.is_request(notification)
    assert jsonrpc.is_response(response) and not jsonrpc.is_request(response)


# --- server dispatch (in-process, no subprocess) ------------------------


def test_handle_message_rejects_tools_list_before_initialize() -> None:
    state = server.ServerState()
    response = server.handle_message(state, jsonrpc.build_request(1, "tools/list"))
    assert response is not None
    assert response["error"]["code"] == jsonrpc.INVALID_REQUEST


def test_handle_message_initialize_then_tools_list() -> None:
    state = server.ServerState()
    init = server.handle_message(state, jsonrpc.build_request(1, "initialize", {"protocolVersion": server.PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}))
    assert init["result"]["protocolVersion"] == server.PROTOCOL_VERSION
    assert server.handle_message(state, jsonrpc.build_notification("notifications/initialized")) is None
    listing = server.handle_message(state, jsonrpc.build_request(2, "tools/list"))
    names = {t["name"] for t in listing["result"]["tools"]}
    assert names == {"add", "divide", "summarize_note"}


def test_handle_message_unknown_tool_is_protocol_error() -> None:
    state = server.ServerState(initialized=True, ready=True)
    response = server.handle_message(state, jsonrpc.build_request(1, "tools/call", {"name": "multiply", "arguments": {}}))
    assert response["error"]["code"] == jsonrpc.METHOD_NOT_FOUND


def test_handle_message_divide_by_zero_is_isError_not_protocol_error() -> None:
    state = server.ServerState(initialized=True, ready=True)
    response = server.handle_message(state, jsonrpc.build_request(1, "tools/call", {"name": "divide", "arguments": {"a": 10, "b": 0}}))
    assert "error" not in response
    assert response["result"]["isError"] is True
    assert "zero" in response["result"]["content"][0]["text"]


def test_handle_message_bad_arguments_is_isError_not_protocol_error() -> None:
    """SEP-1303: invalid tool input comes back as isError, not a JSON-RPC error."""
    state = server.ServerState(initialized=True, ready=True)
    response = server.handle_message(state, jsonrpc.build_request(1, "tools/call", {"name": "add", "arguments": {"a": "x"}}))
    assert "error" not in response
    assert response["result"]["isError"] is True


def test_handle_message_resource_read_missing_uri() -> None:
    state = server.ServerState(initialized=True, ready=True)
    response = server.handle_message(state, jsonrpc.build_request(1, "resources/read", {"uri": "note://missing"}))
    assert response["error"]["code"] == jsonrpc.RESOURCE_NOT_FOUND


def test_handle_message_prompts_get_unknown_prompt() -> None:
    state = server.ServerState(initialized=True, ready=True)
    response = server.handle_message(state, jsonrpc.build_request(1, "prompts/get", {"name": "nope"}))
    assert response["error"]["code"] == jsonrpc.INVALID_PARAMS


def test_server_data_read_resource_text_and_blob() -> None:
    text = server_data.read_resource("note://todo")
    assert text[0]["text"] == server_data.NOTES["todo"]
    blob = server_data.read_resource("asset://logo")
    assert "blob" in blob[0] and blob[0]["mimeType"] == "image/png"


# --- client, real server subprocess --------------------------------------


def _connected_client(**kwargs: object) -> MCPClient:
    client = MCPClient(**kwargs)
    client.connect()
    client.initialize()
    client.notify_initialized()
    return client


def test_client_handshake_negotiates_version_and_capabilities() -> None:
    client = MCPClient()
    client.connect()
    try:
        result = client.initialize()
        assert result["protocolVersion"] == server.PROTOCOL_VERSION
        assert client.negotiated_version == server.PROTOCOL_VERSION
        assert set(client.server_capabilities) == {"tools", "resources", "prompts"}
        client.notify_initialized()
    finally:
        client.shutdown()


def test_client_call_before_handshake_raises() -> None:
    client = MCPClient()
    client.connect()
    try:
        with pytest.raises(RuntimeError, match="handshake not complete"):
            client.list_tools()
    finally:
        client.shutdown()


def test_client_tools_call_success() -> None:
    client = _connected_client()
    try:
        result = client.call_tool("add", {"a": 12, "b": 30})
        assert result["isError"] is False
        assert result["content"][0]["text"] == "42"
    finally:
        client.shutdown()


def test_client_tools_call_isError_result() -> None:
    client = _connected_client()
    try:
        result = client.call_tool("divide", {"a": 1, "b": 0})
        assert result["isError"] is True
    finally:
        client.shutdown()


def test_client_unknown_tool_raises_protocol_error() -> None:
    client = _connected_client()
    try:
        with pytest.raises(MCPProtocolError) as excinfo:
            client.call_tool("multiply", {"a": 1, "b": 2})
        assert excinfo.value.code == jsonrpc.METHOD_NOT_FOUND
    finally:
        client.shutdown()


def test_client_resources_list_read_text_blob_and_missing() -> None:
    client = _connected_client()
    try:
        uris = {r["uri"] for r in client.list_resources()}
        assert uris == {"note://todo", "asset://logo"}
        assert "milk" in client.read_resource("note://todo")[0]["text"].lower()
        assert "blob" in client.read_resource("asset://logo")[0]
        with pytest.raises(MCPProtocolError) as excinfo:
            client.read_resource("note://missing")
        assert excinfo.value.code == jsonrpc.RESOURCE_NOT_FOUND
    finally:
        client.shutdown()


def test_client_prompts_list_and_get() -> None:
    client = _connected_client()
    try:
        names = {p["name"] for p in client.list_prompts()}
        assert names == {"summarize_notes"}
        filled = client.get_prompt("summarize_notes", {"tone": "formal"})
        assert "formal" in filled["messages"][0]["content"]["text"]
    finally:
        client.shutdown()


def test_client_shutdown_terminates_subprocess() -> None:
    client = MCPClient()
    client.connect()
    transport = client._transport  # noqa: SLF001 - test-only introspection of process liveness
    assert transport.is_running()
    client.initialize()
    client.notify_initialized()
    client.shutdown()
    assert not transport.is_running()


def test_client_call_tool_timeout_raises() -> None:
    client = _connected_client()
    try:
        with pytest.raises(TransportTimeoutError):
            client.call_tool("add", {"a": 1, "b": 1}, timeout=0.0)
    finally:
        client.shutdown()


# --- bridge: MCP tools through the core ToolRegistry ----------------------


def test_bridge_registers_tools_and_provider_receives_schemas() -> None:
    client = _connected_client()
    try:
        registry = ToolRegistry()
        names = bridge.register_mcp_tools(registry, client)
        assert set(names) == {"add", "divide", "summarize_note"}

        provider = MockProvider([scripted_tool_call("add", {"a": 2, "b": 2}), "4"])
        messages = [Message.user("what is 2 + 2?")]
        completion = provider.complete(messages, tools=registry.specs())
        assert completion.tool_calls[0].name == "add"
        observation = registry.execute(completion.tool_calls[0])
        assert observation == "4"

        sent_tool_names = {t["name"] for t in provider.calls[0]["tools"]}
        assert sent_tool_names == {"add", "divide", "summarize_note"}
    finally:
        client.shutdown()


def test_bridge_isError_result_becomes_error_prefixed_observation() -> None:
    client = _connected_client()
    try:
        registry = ToolRegistry()
        bridge.register_mcp_tools(registry, client)
        call = scripted_tool_call("divide", {"a": 1, "b": 0}).tool_calls[0]
        observation = registry.execute(call)
        assert observation.startswith("ERROR:")
    finally:
        client.shutdown()


def test_run_host_loop_demo_produces_two_answers() -> None:
    answers = bridge.run_host_loop_demo()
    assert len(answers) == 2
    assert "42" in answers[0]
    assert "divid" in answers[1].lower() or "zero" in answers[1].lower()


# --- multi-server host ----------------------------------------------------


def test_multi_server_collision_namespaces_both_sides() -> None:
    host = multi_server.MultiServerHost()
    host.add_server("alpha", MCPClient(client_name="alpha"))
    host.add_server("beta", MCPClient(client_name="beta"))
    try:
        names = host.merged_names()
        assert "add" not in names  # bare name is gone once it collided
        assert "alpha.add" in names and "beta.add" in names
        assert host.route_of("alpha.add") == "alpha"
        assert host.route_of("beta.add") == "beta"
    finally:
        host.shutdown()


def test_multi_server_routes_call_to_correct_connection() -> None:
    host = multi_server.MultiServerHost()
    host.add_server("alpha", MCPClient(client_name="alpha"))
    host.add_server("beta", MCPClient(client_name="beta"))
    try:
        registry = ToolRegistry()
        host.register_into(registry)
        call = scripted_tool_call("beta.add", {"a": 7, "b": 8}).tool_calls[0]
        assert registry.execute(call) == "15"
    finally:
        host.shutdown()


def test_run_multi_server_demo() -> None:
    merged_names, final_answer = multi_server.run_multi_server_demo()
    assert "alpha.add" in merged_names and "beta.add" in merged_names
    assert "9" in final_answer and "101" in final_answer


# --- sampling ---------------------------------------------------------


def test_sampling_round_trip_with_capability_offered() -> None:
    accepted, refused = sampling.run_sampling_demo()
    assert accepted["isError"] is False
    assert "milk" in accepted["content"][0]["text"].lower() or "errands" in accepted["content"][0]["text"].lower()
    assert refused["isError"] is True
    assert "sampling" in refused["content"][0]["text"].lower()


def test_sampling_handler_shapes_result_like_createMessageResult() -> None:
    handler = sampling.make_sampling_handler()
    params = {"messages": [{"role": "user", "content": {"type": "text", "text": "Summarize: buy milk."}}]}
    result = handler(params)
    assert result["role"] == "assistant"
    assert result["content"]["type"] == "text"
    assert result["stopReason"] == "endTurn"


# --- HTTP transport variant ------------------------------------------------


def test_http_transport_round_trip() -> None:
    result = http_transport.run_http_transport_demo()
    assert result["server_info"]["name"] == "agentic-patterns-mcp-server"
    assert set(result["tool_names"]) == {"add", "divide", "summarize_note"}
    assert result["add_result"]["content"][0]["text"] == "5"


def test_http_transport_unknown_tool_returns_error_body() -> None:
    server_obj, thread, base_url = http_transport.start_http_server()
    try:
        transport = http_transport.HTTPClientTransport(base_url)
        transport.post(
            jsonrpc.build_request(transport.next_id(), "initialize", {"protocolVersion": server.PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})
        )
        response = transport.post(jsonrpc.build_request(transport.next_id(), "tools/call", {"name": "nope", "arguments": {}}))
        assert response is not None
        assert response["error"]["code"] == jsonrpc.METHOD_NOT_FOUND
    finally:
        http_transport.stop_http_server(server_obj, thread)
