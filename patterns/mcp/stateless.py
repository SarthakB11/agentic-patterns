"""The 2026-07-28 release candidate's stateless core: no handshake, `_meta` per request.

Every other module in this pattern is built on the `initialize` /
`notifications/initialized` handshake: `client.py` sends `initialize`,
waits, sends the follow-up notification, and gates every later call on
`self._ready` and the capabilities remembered from that one exchange.
`server.py` mirrors this with a per-connection `ServerState` that remembers
`client_capabilities` for as long as the connection lives.

The 2026-07-28 release candidate (locked May 21, 2026; final targeted for
July 28, 2026, so it is still an RC as of this writing) deletes all of
that. SEP-2575 removes `initialize` and `notifications/initialized`;
SEP-2567 removes the `Mcp-Session-Id` header. What replaces the
connection-time exchange is `_meta`: protocol version, client info, and
client capabilities now ride on every single request instead of being
negotiated once and remembered. `handle_stateless` is a pure function of
one message for exactly this reason: it never reads or writes any object
that outlives the call, so any request can land on any server instance and
a horizontal deployment needs no sticky routing and no shared session
store. That property is the entire reason SEP-2567 exists, and it is
directly demonstrable offline by routing a fixed request sequence
round-robin across two independent `StatelessServer` instances and
checking both answer correctly, which `run_stateless_demo` does.

The RC's `server/discover` method (the server's capability object up
front, since there is no handshake response left to carry it) is
implemented here as a plain request. It is RC-provisional: this repository
could not independently confirm the method name against a shipped spec
page, only against the RC blog post's description of the stateless
model's needs, so treat the name as illustrative of the shape rather than
a verified wire contract.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from patterns.mcp import jsonrpc
from patterns.mcp.client import MCPProtocolError

STATELESS_PROTOCOL_VERSION = "2026-07-28"

STATELESS_SERVER_CAPABILITIES: dict[str, Any] = {"tools": {"listChanged": False}}

_HANDSHAKE_METHODS = {"initialize", "notifications/initialized"}

ToolHandler = Callable[[dict[str, Any], set[str]], tuple[list[dict[str, Any]], bool]]


def _tool_add(arguments: dict[str, Any], capabilities: set[str]) -> tuple[list[dict[str, Any]], bool]:
    try:
        return [{"type": "text", "text": str(arguments["a"] + arguments["b"])}], False
    except (KeyError, TypeError) as exc:
        return [{"type": "text", "text": f"invalid arguments for add: {exc}"}], True


def _tool_echo_if_sampling(arguments: dict[str, Any], capabilities: set[str]) -> tuple[list[dict[str, Any]], bool]:
    if "sampling" not in capabilities:
        return [{"type": "text", "text": "client did not offer the sampling capability in _meta; refusing"}], True
    return [{"type": "text", "text": f"echo: {arguments.get('text', '')}"}], False


_STATELESS_TOOLS: dict[str, tuple[dict[str, Any], ToolHandler]] = {
    "add": (
        {
            "name": "add",
            "description": "Add two numbers and return the sum.",
            "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
        },
        _tool_add,
    ),
    "echo_if_sampling": (
        {
            "name": "echo_if_sampling",
            "description": "Echo text back, but only if the caller's _meta.capabilities offers sampling.",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        },
        _tool_echo_if_sampling,
    ),
}


def handle_stateless(message: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch one message with no reference to any prior message.

    Every fact a stateful `ServerState` would have remembered instead comes
    out of `params["_meta"]` on this one message: `protocolVersion`,
    `clientInfo`, and `capabilities`. There is no `self` to hold a
    connection's history because there is no connection, only requests.

    Args:
        message: A decoded JSON-RPC request or notification. Must carry
            `params._meta.protocolVersion` for anything except the removed
            handshake methods, which are rejected outright.

    Returns:
        The JSON-RPC response, or `None` for a notification.
    """
    method = message.get("method")
    msg_id = message.get("id")
    is_notification = "id" not in message
    params = message.get("params", {}) or {}
    meta = params.get("_meta", {}) or {}

    if method in _HANDSHAKE_METHODS:
        if is_notification:
            return None
        return jsonrpc.build_error(
            msg_id,
            jsonrpc.METHOD_NOT_FOUND,
            f"{method!r} is not served under the stateless {STATELESS_PROTOCOL_VERSION} core; "
            "protocol version, client info, and capabilities travel in _meta on every request instead",
        )

    protocol_version = meta.get("protocolVersion")
    if protocol_version is None or protocol_version < STATELESS_PROTOCOL_VERSION:
        if is_notification:
            return None
        return jsonrpc.build_error(
            msg_id,
            jsonrpc.INVALID_REQUEST,
            f"missing or stale _meta.protocolVersion: {protocol_version!r}; "
            "the stateless core has no handshake to fall back on",
        )

    if method == "server/discover":
        return jsonrpc.build_response(
            msg_id,
            {
                "capabilities": STATELESS_SERVER_CAPABILITIES,
                "serverInfo": {"name": "agentic-patterns-stateless-server", "version": "0.1.0"},
            },
        )

    if method == "tools/list":
        return jsonrpc.build_response(msg_id, {"tools": [spec for spec, _ in _STATELESS_TOOLS.values()]})

    if method == "tools/call":
        name = params.get("name")
        entry = _STATELESS_TOOLS.get(name)
        if entry is None:
            return jsonrpc.build_error(msg_id, jsonrpc.METHOD_NOT_FOUND, f"unknown tool: {name!r}")
        capabilities = set(meta.get("capabilities", {}) or {})
        content, is_error = entry[1](params.get("arguments", {}), capabilities)
        return jsonrpc.build_response(msg_id, {"content": content, "isError": is_error})

    if is_notification:
        return None
    return jsonrpc.build_error(msg_id, jsonrpc.METHOD_NOT_FOUND, f"unknown method: {method!r}")


@dataclass
class StatelessServer:
    """One stateless server instance.

    Attributes:
        instance_id: A label distinguishing this instance in demo output.
        calls_served: A counter kept only for narration; `handle_stateless`
            never reads it, which is the point: nothing this class owns
            affects how a request is answered.
    """

    instance_id: str
    calls_served: int = 0

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch through the shared pure function and count the call."""
        response = handle_stateless(message)
        self.calls_served += 1
        return response


class StatelessClient:
    """A client that attaches `_meta` to every request and never sends a handshake.

    Args:
        servers: The pool of `StatelessServer` instances this client routes
            requests across, round-robin, in the order given.
        capabilities: Capabilities attached to `_meta` on every request.
    """

    def __init__(
        self,
        servers: list[StatelessServer],
        *,
        client_name: str = "agentic-patterns-stateless-client",
        client_version: str = "0.1.0",
        capabilities: dict[str, Any] | None = None,
    ) -> None:
        self.servers = servers
        self.client_name = client_name
        self.client_version = client_version
        self.capabilities = capabilities or {}
        self._ids = itertools.count(1)
        self._router = itertools.cycle(range(len(servers)))

    def _meta(self) -> dict[str, Any]:
        return {
            "protocolVersion": STATELESS_PROTOCOL_VERSION,
            "clientInfo": {"name": self.client_name, "version": self.client_version},
            "capabilities": self.capabilities,
        }

    def _send(self, method: str, params: dict[str, Any] | None = None) -> tuple[Any, str]:
        server = self.servers[next(self._router)]
        full_params = dict(params or {})
        full_params["_meta"] = self._meta()
        request = jsonrpc.build_request(f"stl-{next(self._ids)}", method, full_params)
        response = server.handle(request)
        if response is None:
            raise RuntimeError(f"{method!r} produced no response")
        if "error" in response:
            err = response["error"]
            raise MCPProtocolError(err["code"], err["message"])
        return response["result"], server.instance_id

    def call_tool(self, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Call a tool, routed round-robin. Returns `(result, instance_id)`."""
        return self._send("tools/call", {"name": name, "arguments": arguments})


def run_stateless_demo() -> dict[str, Any]:
    """Demonstrate no-handshake dispatch, instance independence, and the `_meta` capability gate.

    Two independent `StatelessServer` instances share no state; a client
    routes two `add` calls round-robin across them and both succeed, which
    is only possible because neither instance remembers anything from a
    prior request. A third call exercises the `_meta.capabilities` gate:
    the same tool succeeds when `sampling` is offered and is refused when a
    second client omits it, with no stored state distinguishing the two.

    Returns:
        A dict of demo outcomes, keyed for `main.py` to print.
    """
    server_a = StatelessServer("instance-a")
    server_b = StatelessServer("instance-b")

    sampling_client = StatelessClient([server_a, server_b], capabilities={"sampling": {}})
    add_1 = sampling_client.call_tool("add", {"a": 12, "b": 30})
    add_2 = sampling_client.call_tool("add", {"a": 100, "b": 1})
    gated_ok, _ = sampling_client.call_tool("echo_if_sampling", {"text": "hello"})

    plain_client = StatelessClient([server_a, server_b], capabilities={})
    gated_refused, _ = plain_client.call_tool("echo_if_sampling", {"text": "hello"})

    handshake_gone = handle_stateless(
        jsonrpc.build_request(
            "x",
            "initialize",
            {"_meta": {"protocolVersion": STATELESS_PROTOCOL_VERSION, "clientInfo": {}, "capabilities": {}}},
        )
    )
    missing_meta = handle_stateless(jsonrpc.build_request("y", "tools/call", {"name": "add", "arguments": {"a": 1, "b": 1}}))

    return {
        "add_1": add_1,
        "add_2": add_2,
        "gated_ok": gated_ok,
        "gated_refused": gated_refused,
        "handshake_gone_code": handshake_gone["error"]["code"],
        "missing_meta_code": missing_meta["error"]["code"],
    }
