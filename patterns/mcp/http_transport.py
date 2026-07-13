"""HTTP transport variant: the same JSON-RPC semantics, framed over loopback HTTP.

The spec's real "Streamable HTTP" transport adds sessions
(`Mcp-Session-Id`), resumable streams, and Server-Sent Events for
server-to-client pushes. None of that is implemented here. What this module
shows is the point the brief makes about transports: protocol semantics
(the same `initialize`, `tools/list`, `tools/call` messages, the same
`handle_message` dispatch from `server.py`) are identical to stdio, and only
the framing changes. Each JSON-RPC message becomes one HTTP POST with a
JSON body; a request gets a `200` with the JSON-RPC response, a
notification gets a bare `202`. No sessions, no streaming, no reverse
direction, so `summarize_note`'s sampling round trip is not reachable over
this transport (see `server.py`, which returns `isError: true` for it when
`transport is None`).
"""

from __future__ import annotations

import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from patterns.mcp import jsonrpc
from patterns.mcp.server import PROTOCOL_VERSION, ServerState, handle_message


class _MCPHTTPServer(HTTPServer):
    """An `HTTPServer` carrying the one shared `ServerState` for its lifetime."""

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _Handler)
        self.state = ServerState()


class _Handler(BaseHTTPRequestHandler):
    server: _MCPHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass  # keep the demo transcript free of per-request access logs

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            message = jsonrpc.decode_line(raw.decode("utf-8"))
        except jsonrpc.JSONRPCDecodeError as exc:
            self._write_json(jsonrpc.build_error(None, jsonrpc.PARSE_ERROR, str(exc)), 400)
            return
        response = handle_message(self.server.state, message, transport=None)
        if response is None:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._write_json(response, 200)

    def _write_json(self, obj: dict[str, Any], status: int) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_http_server() -> tuple[_MCPHTTPServer, threading.Thread, str]:
    """Start the MCP-over-HTTP server on an OS-assigned loopback port."""
    server = _MCPHTTPServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/mcp"
    return server, thread, base_url


def stop_http_server(server: _MCPHTTPServer, thread: threading.Thread) -> None:
    """Stop the server and join its thread, for clean shutdown."""
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


class HTTPClientTransport:
    """Client side of the loopback HTTP transport: one POST per JSON-RPC message."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self._ids = itertools.count(1)

    def next_id(self) -> str:
        return f"http-{next(self._ids)}"

    def post(self, message: dict[str, Any], timeout: float = 5.0) -> dict[str, Any] | None:
        """POST one JSON-RPC message and return the decoded response, or `None` for a `202`."""
        import urllib.request

        body = json.dumps(message).encode("utf-8")
        request = urllib.request.Request(
            self.base_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - loopback only
            status = resp.status
            raw = resp.read()
        if status == 202 or not raw:
            return None
        return json.loads(raw.decode("utf-8"))


def run_http_transport_demo() -> dict[str, Any]:
    """Run one handshake plus one tool call over the loopback HTTP transport.

    Returns:
        A dict with the negotiated `server_info`, the discovered tool
        names, and the result of calling `add(2, 3)`.
    """
    server, thread, base_url = start_http_server()
    try:
        transport = HTTPClientTransport(base_url)
        init_response = transport.post(
            jsonrpc.build_request(
                transport.next_id(),
                "initialize",
                {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "http-demo-client", "version": "0.1.0"}},
            )
        )
        assert init_response is not None
        transport.post(jsonrpc.build_notification("notifications/initialized"))

        list_response = transport.post(jsonrpc.build_request(transport.next_id(), "tools/list", None))
        assert list_response is not None

        call_response = transport.post(
            jsonrpc.build_request(transport.next_id(), "tools/call", {"name": "add", "arguments": {"a": 2, "b": 3}})
        )
        assert call_response is not None

        return {
            "server_info": init_response["result"]["serverInfo"],
            "tool_names": [t["name"] for t in list_response["result"]["tools"]],
            "add_result": call_response["result"],
        }
    finally:
        stop_http_server(server, thread)
