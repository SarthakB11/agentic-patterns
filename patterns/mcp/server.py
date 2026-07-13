"""A small, standalone MCP server, run as a child process over stdio.

Run directly it serves the stdio transport (`python -m patterns.mcp.server`,
which is how `client.py` spawns it). `handle_message` is factored out as a
transport-agnostic function so `http_transport.py` can drive the exact same
dispatch logic over loopback HTTP instead. The tools, resources, and
prompts themselves live in `server_data.py`; this module is only the
protocol mechanics: handshake, capability advertisement, and method
dispatch.

Bad tool arguments come back as `isError: true` tool results, not JSON-RPC
protocol errors, per SEP-1303's clarification: a model can read and react to
a tool error, but a client cannot show a raw protocol error to a model in
any useful way. An unknown tool name or unknown prompt name is a genuine
protocol fault (the target does not exist at all) and does come back as a
JSON-RPC error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from patterns.mcp import jsonrpc, server_data
from patterns.mcp.transport import StdioServerTransport

PROTOCOL_VERSION = "2025-11-25"

SERVER_CAPABILITIES: dict[str, Any] = {
    "tools": {"listChanged": False},
    "resources": {"listChanged": False, "subscribe": False},
    "prompts": {"listChanged": False},
}


@dataclass
class ServerState:
    """Per-connection state a stdio or HTTP server handler tracks.

    Attributes:
        initialized: Set once `initialize` has been answered.
        ready: Set once `notifications/initialized` has arrived.
        client_capabilities: The capabilities object the client sent in
            `initialize`, used to gate server behavior such as sampling.
        client_info: The client's declared name and version.
        next_request_id: Counter for server-initiated requests (sampling).
    """

    initialized: bool = False
    ready: bool = False
    client_capabilities: dict[str, Any] = field(default_factory=dict)
    client_info: dict[str, Any] = field(default_factory=dict)
    next_request_id: int = 0

    def new_request_id(self) -> str:
        self.next_request_id += 1
        return f"srv-{self.next_request_id}"


def handle_message(state: ServerState, message: dict[str, Any], transport: StdioServerTransport | None = None) -> dict[str, Any] | None:
    """Dispatch one decoded JSON-RPC message and return the response, if any.

    Returns `None` for notifications, which never get a reply. Enforces
    handshake ordering server-side: nothing but `initialize` is answered
    until it has completed.
    """
    method = message.get("method")
    msg_id = message.get("id")
    is_notification = "id" not in message

    if method == "initialize":
        params = message.get("params", {})
        state.client_capabilities = params.get("capabilities", {})
        state.client_info = params.get("clientInfo", {})
        state.initialized = True
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": SERVER_CAPABILITIES,
            "serverInfo": {"name": "agentic-patterns-mcp-server", "version": "0.1.0"},
        }
        return jsonrpc.build_response(msg_id, result)

    if method == "notifications/initialized":
        state.ready = True
        return None

    if not state.initialized:
        if is_notification:
            return None
        return jsonrpc.build_error(msg_id, jsonrpc.INVALID_REQUEST, "server not initialized: call initialize first")

    if method == "tools/list":
        return jsonrpc.build_response(msg_id, {"tools": [t.spec() for t in server_data.TOOLS.values()]})

    if method == "tools/call":
        params = message.get("params", {})
        name = params.get("name")
        tool = server_data.TOOLS.get(name)
        if tool is None:
            return jsonrpc.build_error(msg_id, jsonrpc.METHOD_NOT_FOUND, f"unknown tool: {name!r}")
        content, is_error = tool.handler(params.get("arguments", {}), state, transport)
        return jsonrpc.build_response(msg_id, {"content": content, "isError": is_error})

    if method == "resources/list":
        return jsonrpc.build_response(msg_id, {"resources": server_data.RESOURCE_LIST})

    if method == "resources/read":
        uri = message.get("params", {}).get("uri", "")
        try:
            contents = server_data.read_resource(uri)
        except KeyError:
            return jsonrpc.build_error(msg_id, jsonrpc.RESOURCE_NOT_FOUND, f"resource not found: {uri}")
        return jsonrpc.build_response(msg_id, {"contents": contents})

    if method == "prompts/list":
        return jsonrpc.build_response(msg_id, {"prompts": server_data.PROMPT_LIST})

    if method == "prompts/get":
        params = message.get("params", {})
        try:
            result = server_data.get_prompt(params.get("name", ""), params.get("arguments", {}))
        except KeyError:
            return jsonrpc.build_error(msg_id, jsonrpc.INVALID_PARAMS, f"unknown prompt: {params.get('name')!r}")
        return jsonrpc.build_response(msg_id, result)

    if is_notification:
        return None
    return jsonrpc.build_error(msg_id, jsonrpc.METHOD_NOT_FOUND, f"unknown method: {method!r}")


def serve_stdio() -> None:
    """Run the server's read-dispatch-write loop over stdio until stdin closes."""
    transport = StdioServerTransport()
    state = ServerState()
    while True:
        try:
            message = transport.read_message()
        except jsonrpc.JSONRPCDecodeError as exc:
            transport.write_message(jsonrpc.build_error(None, jsonrpc.PARSE_ERROR, str(exc)))
            continue
        if message is None:
            break
        response = handle_message(state, message, transport)
        if response is not None:
            transport.write_message(response)


if __name__ == "__main__":
    serve_stdio()
