"""Tool, resource, and prompt content for `server.py`.

Kept separate from the dispatch loop in `server.py` so that module stays a
short, readable statement of the protocol mechanics. Everything here is
plain deterministic Python: three tools (`add`, `divide`, and
`summarize_note`, the one that reaches back to the client via sampling),
one text resource, one binary resource, and one prompt template.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from patterns.mcp import jsonrpc

if TYPE_CHECKING:
    from patterns.mcp.server import ServerState
    from patterns.mcp.transport import StdioServerTransport

NOTES: dict[str, str] = {"todo": "Buy milk. Call Grandma. Finish the MCP demo before Friday."}

ToolHandler = Callable[[dict[str, Any], "ServerState", "StdioServerTransport | None"], tuple[list[dict[str, Any]], bool]]


@dataclass
class ToolDef:
    """A server-side tool: its schema plus the function that runs it."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "inputSchema": self.input_schema}


def _handle_add(arguments: dict[str, Any], state: "ServerState", transport: "StdioServerTransport | None") -> tuple[list[dict[str, Any]], bool]:
    try:
        result = arguments["a"] + arguments["b"]
    except (KeyError, TypeError) as exc:
        return [{"type": "text", "text": f"invalid arguments for add: {exc}"}], True
    return [{"type": "text", "text": str(result)}], False


def _handle_divide(arguments: dict[str, Any], state: "ServerState", transport: "StdioServerTransport | None") -> tuple[list[dict[str, Any]], bool]:
    try:
        a, b = arguments["a"], arguments["b"]
    except KeyError as exc:
        return [{"type": "text", "text": f"missing argument: {exc}"}], True
    if b == 0:
        return [{"type": "text", "text": "cannot divide by zero"}], True
    return [{"type": "text", "text": str(a / b)}], False


def _await_matching_response(transport: "StdioServerTransport", request_id: str, max_messages: int = 10) -> dict[str, Any] | None:
    """Block-read from the client until a response with `request_id` arrives."""
    for _ in range(max_messages):
        message = transport.read_message()
        if message is None:
            return None
        if message.get("id") == request_id and "method" not in message:
            return message
    return None


def _handle_summarize_note(arguments: dict[str, Any], state: "ServerState", transport: "StdioServerTransport | None") -> tuple[list[dict[str, Any]], bool]:
    if "sampling" not in state.client_capabilities:
        return [{"type": "text", "text": "client did not offer the sampling capability; cannot summarize"}], True
    note_text = NOTES.get(arguments.get("note_id", ""))
    if note_text is None:
        return [{"type": "text", "text": f"unknown note id: {arguments.get('note_id')!r}"}], True
    if transport is None:
        return [{"type": "text", "text": "sampling needs a duplex transport; unavailable over this connection"}], True

    request = jsonrpc.build_request(
        state.new_request_id(),
        "sampling/createMessage",
        {
            "messages": [
                {"role": "user", "content": {"type": "text", "text": f"Summarize this note in one short sentence: {note_text}"}}
            ],
            "maxTokens": 200,
        },
    )
    transport.write_message(request)
    reply = _await_matching_response(transport, request["id"])
    if reply is None or "error" in reply:
        return [{"type": "text", "text": "sampling request failed or was refused by the client"}], True
    return [reply["result"]["content"]], False


TOOLS: dict[str, ToolDef] = {
    "add": ToolDef(
        name="add",
        description="Add two numbers and return the sum.",
        input_schema={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
        handler=_handle_add,
    ),
    "divide": ToolDef(
        name="divide",
        description="Divide a by b and return the quotient.",
        input_schema={"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
        handler=_handle_divide,
    ),
    "summarize_note": ToolDef(
        name="summarize_note",
        description="Summarize a stored note by id, using the client's model via sampling.",
        input_schema={"type": "object", "properties": {"note_id": {"type": "string"}}, "required": ["note_id"]},
        handler=_handle_summarize_note,
    ),
}

RESOURCE_LIST: list[dict[str, Any]] = [
    {"uri": "note://todo", "name": "todo notes", "description": "A short todo list.", "mimeType": "text/plain"},
    {"uri": "asset://logo", "name": "logo", "description": "A tiny placeholder logo image.", "mimeType": "image/png"},
]

PROMPT_LIST: list[dict[str, Any]] = [
    {
        "name": "summarize_notes",
        "description": "Summarize the todo notes for a status update.",
        "arguments": [{"name": "tone", "description": "Tone of the summary, e.g. 'formal' or 'casual'.", "required": False}],
    }
]


def read_resource(uri: str) -> list[dict[str, Any]]:
    """Return `resources/read` contents for `uri`, or raise `KeyError` if unknown."""
    if uri == "note://todo":
        return [{"uri": uri, "mimeType": "text/plain", "text": NOTES["todo"]}]
    if uri == "asset://logo":
        raw = b"\x89PNG\r\n\x1a\n" + b"tiny-placeholder-logo-bytes"
        return [{"uri": uri, "mimeType": "image/png", "blob": base64.b64encode(raw).decode("ascii")}]
    raise KeyError(uri)


def get_prompt(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Fill and return the named prompt, or raise `KeyError` if unknown."""
    if name != "summarize_notes":
        raise KeyError(name)
    tone = arguments.get("tone", "neutral")
    text = f"In a {tone} tone, write a one-sentence status update from these notes: {NOTES['todo']}"
    return {
        "description": "Summarize the todo notes for a status update.",
        "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
    }
