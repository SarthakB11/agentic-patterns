"""JSON-RPC 2.0 message codec: the wire format every MCP message rides on.

MCP messages are JSON-RPC 2.0 objects, one per line, UTF-8 encoded. A
request carries an `id` and expects a response; a notification carries no
`id` and expects none; a response echoes the request's `id` with either
`result` or `error`. This module builds and parses those three shapes and
rejects anything that does not conform, independent of which transport
(stdio, HTTP) moves the bytes.

Error codes below -32000 are reserved by the JSON-RPC spec for standard
protocol faults. MCP servers use the -32000 to -32099 range for their own
implementation-defined errors; `RESOURCE_NOT_FOUND` is one such code, used
by this implementation's `resources/read` for a missing URI.
"""

from __future__ import annotations

import json
from typing import Any

JSONRPC_VERSION = "2.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
RESOURCE_NOT_FOUND = -32002


class JSONRPCDecodeError(ValueError):
    """Raised when a line of input is not a well-formed JSON-RPC 2.0 message."""


def build_request(request_id: str | int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a JSON-RPC request object, which expects a matching response."""
    message: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def build_notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a JSON-RPC notification object, which carries no `id` and expects no reply."""
    message: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
    if params is not None:
        message["params"] = params
    return message


def build_response(request_id: str | int, result: Any) -> dict[str, Any]:
    """Build a JSON-RPC success response, echoing the request's `id`."""
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}


def build_error(request_id: str | int | None, code: int, message: str, data: Any = None) -> dict[str, Any]:
    """Build a JSON-RPC error response.

    `request_id` is `None` when the failure happened before an `id` could be
    read from the malformed input (per the JSON-RPC spec's `Parse error`
    case); every other error echoes the request's `id`.
    """
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}


def encode_line(message: dict[str, Any]) -> str:
    """Serialize a message to one newline-terminated line of JSON.

    `json.dumps` escapes control characters (including newlines) inside
    string values by default, so a well-formed message can never produce an
    embedded raw newline in its own right; this function exists mainly to
    keep the "one JSON value per line" framing rule in one place.
    """
    return json.dumps(message, separators=(",", ":")) + "\n"


def decode_line(line: str) -> dict[str, Any]:
    """Parse one line of input into a validated JSON-RPC message dict.

    Raises:
        JSONRPCDecodeError: If the line is not valid JSON, is not a JSON
            object, is missing the `jsonrpc` field, or has the wrong
            `jsonrpc` version. A line containing a literal (unescaped)
            newline inside a string is invalid JSON by the JSON spec itself,
            so `json.loads` already rejects it; this function surfaces that
            as a `JSONRPCDecodeError` rather than a bare `json` exception.
    """
    stripped = line.strip("\n")
    if not stripped.strip():
        raise JSONRPCDecodeError("empty line is not a JSON-RPC message")
    try:
        message = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise JSONRPCDecodeError(f"line is not valid JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise JSONRPCDecodeError(f"JSON-RPC message must be an object, got {type(message).__name__}")
    if message.get("jsonrpc") != JSONRPC_VERSION:
        raise JSONRPCDecodeError(f"unsupported or missing jsonrpc version: {message.get('jsonrpc')!r}")
    return message


def is_request(message: dict[str, Any]) -> bool:
    """True if `message` is a request (has both `method` and `id`)."""
    return "method" in message and "id" in message


def is_notification(message: dict[str, Any]) -> bool:
    """True if `message` is a notification (has `method`, no `id`)."""
    return "method" in message and "id" not in message


def is_response(message: dict[str, Any]) -> bool:
    """True if `message` is a response (has `id`, no `method`; `result` xor `error`)."""
    return "method" not in message and "id" in message
