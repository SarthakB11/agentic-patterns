"""Sampling: a server borrows the host's model instead of holding its own key.

Every other tool in `server.py` is a plain deterministic function. This
module drives the one exception: `summarize_note`, which needs a model to
do its job. Rather than the server calling an LLM API directly, it sends a
`sampling/createMessage` request back to the *client* over the same stdio
connection and waits for the client to run it and answer. That keeps API
keys and model choice on the host side, where a human can review or block
the call, matching the spec's guidance that sampling requests should route
through human approval.

Two demo runs: one client offers the `sampling` capability and answers the
request with a scripted `MockProvider` call, showing the round trip
succeed; the other omits the capability, showing the server's server-side
capability gate refuse the tool before any sampling request is even sent.
"""

from __future__ import annotations

from typing import Any

from agentic_patterns import Message, get_provider

from patterns.mcp.client import MCPClient

SUMMARY_TEXT = "Errands and one project deadline: milk, a call to Grandma, and the MCP demo due Friday."


def make_sampling_handler() -> Any:
    """Build an `on_sampling_request` callback backed by a scripted `MockProvider`.

    The callback receives the server's `sampling/createMessage` params (a
    `messages` array in MCP's content-block shape), converts the one user
    message into this repo's `Message` type, and asks a one-turn scripted
    `MockProvider` to answer it. The result is shaped back into MCP's
    `CreateMessageResult`: a role, a single text content block, the model
    name that answered, and a stop reason.
    """
    provider = get_provider(script=[SUMMARY_TEXT])

    def handle(params: dict[str, Any]) -> dict[str, Any]:
        block = params["messages"][0]["content"]
        completion = provider.complete([Message.user(block["text"])])
        return {
            "role": "assistant",
            "content": {"type": "text", "text": completion.content},
            "model": "mock-model",
            "stopReason": "endTurn",
        }

    return handle


def run_sampling_demo() -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the sampling round trip once with the capability offered, once without.

    Returns:
        A tuple `(accepted_result, refused_result)`, each the raw
        `tools/call` result dict for `summarize_note`.
    """
    accepting_client = MCPClient(client_name="sampling-capable-client", supports_sampling=True)
    accepting_client.connect()
    accepting_client.initialize()
    accepting_client.notify_initialized()
    try:
        accepted = accepting_client.call_tool(
            "summarize_note", {"note_id": "todo"}, on_sampling_request=make_sampling_handler()
        )
    finally:
        accepting_client.shutdown()

    plain_client = MCPClient(client_name="sampling-unaware-client", supports_sampling=False)
    plain_client.connect()
    plain_client.initialize()
    plain_client.notify_initialized()
    try:
        refused = plain_client.call_tool("summarize_note", {"note_id": "todo"})
    finally:
        plain_client.shutdown()

    return accepted, refused
