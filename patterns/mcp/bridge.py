"""Bridge: expose a live MCP server's tools through the core `ToolRegistry`.

This is the piece that makes MCP useful to the rest of this repository: once
`register_mcp_tools` has run, the tools a real (subprocess) MCP server
offers are indistinguishable, from the agent loop's point of view, from any
locally-defined `Tool`. The same `ToolRegistry.execute` that every other
pattern in this repo uses drives a `tools/call` round trip to a separate
process instead of an in-process function call.

`run_host_loop_demo` is the canonical MCP control flow end to end: connect,
discover tools, hand their schemas to the model, execute what it selects,
feed the result back, repeat until the model has a final answer.
"""

from __future__ import annotations

import sys

from agentic_patterns import Message, ToolRegistry, get_provider, scripted_tool_call
from agentic_patterns.core.tools import Tool
from patterns.mcp.client import MCPClient

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to arithmetic tools over MCP. Use them when asked to compute something."
)


def register_mcp_tools(registry: ToolRegistry, client: MCPClient) -> list[str]:
    """Discover a connected client's tools and register each as a `Tool`.

    Each registered tool's `fn` calls `client.call_tool` and flattens the
    MCP result into the plain string `ToolRegistry.execute` expects: text
    content joined together, prefixed with "ERROR: " when the server set
    `isError: true`, mirroring the convention `ToolRegistry.execute` already
    uses for a tool that raised locally.

    Returns:
        The names of the tools registered, in discovery order.
    """
    names: list[str] = []
    for spec in client.list_tools():
        registry.register(
            Tool(
                name=spec["name"],
                description=spec["description"],
                parameters=spec["inputSchema"],
                fn=make_bridge_fn(client, spec["name"]),
            )
        )
        names.append(spec["name"])
    return names


def make_bridge_fn(client: MCPClient, tool_name: str):
    """Build a `Tool.fn` callable that calls `tool_name` on `client` over MCP."""

    def call(**arguments: object) -> str:
        result = client.call_tool(tool_name, arguments)
        text = " ".join(block.get("text", "") for block in result["content"])
        return f"ERROR: {text}" if result.get("isError") else text

    return call


def run_host_loop_demo() -> list[str]:
    """Run a small host loop against a real MCP server, entirely offline.

    Connects to a fresh server subprocess, registers its tools, then drives
    two scripted turns: one that calls `add` successfully, and one that
    calls `divide` by zero and has the model explain the failure using the
    `isError` result. No network call happens anywhere; the only subprocess
    involved is the local MCP server, and the model is `MockProvider`.

    Returns:
        The two final answers the model produced, in order.
    """
    client = MCPClient()
    client.connect()
    client.initialize()
    client.notify_initialized()
    try:
        registry = ToolRegistry()
        register_mcp_tools(registry, client)

        provider = get_provider(
            script=[
                scripted_tool_call("add", {"a": 12, "b": 30}),
                "12 + 30 = 42.",
                scripted_tool_call("divide", {"a": 10, "b": 0}),
                "I can't compute that: dividing by zero is undefined, so I can't give you a quotient for 10 / 0.",
            ]
        )

        answers: list[str] = []
        for question in ("What is 12 + 30?", "What is 10 divided by 0?"):
            messages = [Message.user(question)]
            completion = provider.complete(messages, tools=registry.specs(), system=SYSTEM_PROMPT)
            while completion.tool_calls:
                messages.append(Message.assistant(completion.content, completion.tool_calls))
                for call in completion.tool_calls:
                    observation = registry.execute(call)
                    messages.append(Message.tool(call.id, observation))
                completion = provider.complete(messages, tools=registry.specs(), system=SYSTEM_PROMPT)
            answers.append(completion.content)
        return answers
    finally:
        client.shutdown()


if __name__ == "__main__":
    for answer in run_host_loop_demo():
        print(answer, file=sys.stderr)
