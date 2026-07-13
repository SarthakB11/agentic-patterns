"""A small, honest MCP client: the host-side half of the stdio connection.

`MCPClient` spawns a server subprocess, runs the `initialize` handshake,
gates every later call on the capabilities the server actually negotiated,
and exposes `tools/*`, `resources/*`, and `prompts/*` as plain Python
methods. It also handles the one reverse-direction case this pattern
implements: while waiting for a `tools/call` response, a server may send a
`sampling/createMessage` request back over the same connection, which
`call_tool` answers through an optional `on_sampling_request` callback
before continuing to wait for the original response.

What is deliberately not implemented: request pipelining (this client
issues one request and waits before sending the next), `notifications/cancelled`
delivery, and `roots/list` / elicitation. See the package README for the
full list of omissions and why.
"""

from __future__ import annotations

import itertools
import sys
import time
from collections.abc import Callable
from typing import Any

from patterns.mcp import jsonrpc
from patterns.mcp.server import PROTOCOL_VERSION
from patterns.mcp.transport import StdioClientTransport, TransportClosedError, TransportTimeoutError

_DEFAULT_SERVER_COMMAND = [sys.executable, "-m", "patterns.mcp.server"]

SamplingHandler = Callable[[dict[str, Any]], dict[str, Any]]


class MCPProtocolError(Exception):
    """A JSON-RPC `error` response: a protocol fault, not a tool-level failure.

    Attributes:
        code: The JSON-RPC error code, e.g. `jsonrpc.METHOD_NOT_FOUND`.
        message: The server's error message.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class MCPClient:
    """Client-side connection to one MCP server over stdio."""

    def __init__(
        self,
        command: list[str] | None = None,
        *,
        client_name: str = "agentic-patterns-mcp-client",
        client_version: str = "0.1.0",
        supports_sampling: bool = False,
    ) -> None:
        self._command = command or _DEFAULT_SERVER_COMMAND
        self._client_name = client_name
        self._client_version = client_version
        self._supports_sampling = supports_sampling
        self._transport: StdioClientTransport | None = None
        self._ids = itertools.count(1)
        self._ready = False
        self.server_capabilities: dict[str, Any] = {}
        self.server_info: dict[str, Any] = {}
        self.negotiated_version: str | None = None

    def connect(self) -> None:
        """Spawn the server subprocess. Does not perform the handshake yet."""
        self._transport = StdioClientTransport(self._command)

    def _transport_or_raise(self) -> StdioClientTransport:
        if self._transport is None:
            raise RuntimeError("not connected: call connect() first")
        return self._transport

    def _next_id(self) -> str:
        return f"cli-{next(self._ids)}"

    def initialize(self, timeout: float = 5.0) -> dict[str, Any]:
        """Send `initialize` and return the server's result.

        Does not by itself unlock the operation phase; call
        `notify_initialized()` afterward, matching the spec's two-step
        handshake.
        """
        transport = self._transport_or_raise()
        capabilities: dict[str, Any] = {}
        if self._supports_sampling:
            capabilities["sampling"] = {}
        request = jsonrpc.build_request(
            self._next_id(),
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": capabilities,
                "clientInfo": {"name": self._client_name, "version": self._client_version},
            },
        )
        transport.send(request)
        response = transport.receive(timeout)
        if "error" in response:
            err = response["error"]
            raise MCPProtocolError(err["code"], err["message"])
        result = response["result"]
        self.server_capabilities = result.get("capabilities", {})
        self.server_info = result.get("serverInfo", {})
        self.negotiated_version = result.get("protocolVersion")
        return result

    def notify_initialized(self) -> None:
        """Send `notifications/initialized`, unlocking the operation phase."""
        transport = self._transport_or_raise()
        transport.send(jsonrpc.build_notification("notifications/initialized"))
        self._ready = True

    def _require_ready(self) -> StdioClientTransport:
        if not self._ready:
            raise RuntimeError(
                "handshake not complete: call initialize() then notify_initialized() before making requests"
            )
        return self._transport_or_raise()

    def _require_capability(self, key: str) -> None:
        self._require_ready()
        if key not in self.server_capabilities:
            raise RuntimeError(f"server did not negotiate the {key!r} capability")

    def _request(self, method: str, params: dict[str, Any] | None, timeout: float) -> Any:
        transport = self._require_ready()
        request = jsonrpc.build_request(self._next_id(), method, params)
        transport.send(request)
        response = transport.receive(timeout)
        if "error" in response:
            err = response["error"]
            raise MCPProtocolError(err["code"], err["message"])
        return response["result"]

    def list_tools(self, timeout: float = 5.0) -> list[dict[str, Any]]:
        """Return the server's tool definitions. Requires the `tools` capability."""
        self._require_capability("tools")
        return self._request("tools/list", None, timeout)["tools"]

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_sampling_request: SamplingHandler | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Call a tool and return its result: `{"content": [...], "isError": bool}`.

        If the server sends a `sampling/createMessage` request back before
        answering, this loop handles it with `on_sampling_request` (raising
        if none was given) and then keeps waiting for the original response,
        so one `timeout` budget covers the whole exchange.

        Raises:
            MCPProtocolError: The tool name is unknown, or another
                protocol-level fault occurred. A failed tool *execution*
                does not raise; it comes back as `isError: true`.
        """
        self._require_capability("tools")
        transport = self._transport_or_raise()
        call_id = self._next_id()
        transport.send(jsonrpc.build_request(call_id, "tools/call", {"name": name, "arguments": arguments}))

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransportTimeoutError(f"no response for tools/call {name!r} within {timeout}s")
            message = transport.receive(remaining)

            if jsonrpc.is_request(message):
                self._handle_server_request(message, on_sampling_request, transport)
                continue
            if jsonrpc.is_notification(message):
                continue  # progress/log notifications are not modeled; ignore and keep waiting
            if message.get("id") == call_id:
                if "error" in message:
                    err = message["error"]
                    raise MCPProtocolError(err["code"], err["message"])
                return message["result"]
            # a response to a stale or unrelated id; not expected in this single-flight client

    def _handle_server_request(
        self, message: dict[str, Any], on_sampling_request: SamplingHandler | None, transport: StdioClientTransport
    ) -> None:
        """Answer a server-initiated request (currently only `sampling/createMessage`)."""
        method = message["method"]
        if method == "sampling/createMessage" and self._supports_sampling and on_sampling_request is not None:
            result = on_sampling_request(message.get("params", {}))
            transport.send(jsonrpc.build_response(message["id"], result))
            return
        transport.send(
            jsonrpc.build_error(message["id"], jsonrpc.METHOD_NOT_FOUND, f"client cannot handle {method!r}")
        )

    def list_resources(self, timeout: float = 5.0) -> list[dict[str, Any]]:
        """Return the server's resource listing. Requires the `resources` capability."""
        self._require_capability("resources")
        return self._request("resources/list", None, timeout)["resources"]

    def read_resource(self, uri: str, timeout: float = 5.0) -> list[dict[str, Any]]:
        """Read one resource by URI. Requires the `resources` capability.

        Raises:
            MCPProtocolError: With `code == jsonrpc.RESOURCE_NOT_FOUND` if
                `uri` is not known to the server.
        """
        self._require_capability("resources")
        return self._request("resources/read", {"uri": uri}, timeout)["contents"]

    def list_prompts(self, timeout: float = 5.0) -> list[dict[str, Any]]:
        """Return the server's prompt listing. Requires the `prompts` capability."""
        self._require_capability("prompts")
        return self._request("prompts/list", None, timeout)["prompts"]

    def get_prompt(self, name: str, arguments: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
        """Fill a prompt template. Requires the `prompts` capability."""
        self._require_capability("prompts")
        return self._request("prompts/get", {"name": name, "arguments": arguments or {}}, timeout)

    def shutdown(self) -> None:
        """Close the connection and terminate the server subprocess cleanly."""
        if self._transport is not None:
            self._transport.close()
        self._ready = False


__all__ = ["MCPClient", "MCPProtocolError", "TransportTimeoutError", "TransportClosedError"]
