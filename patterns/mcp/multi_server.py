"""Multi-server host: one agent, several MCP servers, one merged tool namespace.

A real host rarely talks to just one server. `MultiServerHost` connects to
several at once, merges their `tools/list` results into a single namespace,
and routes each `call` to the connection that actually owns the tool. Names
that only one server offers stay bare; names two or more servers offer
collide and get namespaced as `"<alias>.<name>"` for every colliding entry,
so the merged list never silently shadows one server's tool with another's.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, ToolRegistry, get_provider, scripted_tool_call
from agentic_patterns.core.tools import Tool
from patterns.mcp.bridge import make_bridge_fn
from patterns.mcp.client import MCPClient

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools from several MCP servers. "
    "Use them when asked to compute something."
)


@dataclass
class _Routed:
    alias: str
    client: MCPClient
    original_name: str
    spec: dict[str, object]


class MultiServerHost:
    """Aggregates several `MCPClient` connections into one tool namespace."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}
        self._routes: dict[str, _Routed] = {}

    def add_server(self, alias: str, client: MCPClient) -> None:
        """Connect and initialize `client`, then merge its tools under `alias`."""
        client.connect()
        client.initialize()
        client.notify_initialized()
        self._clients[alias] = client
        for spec in client.list_tools():
            self._merge_tool(alias, client, spec)

    def _merge_tool(self, alias: str, client: MCPClient, spec: dict[str, object]) -> None:
        name = str(spec["name"])
        if name in self._routes:
            existing = self._routes.pop(name)
            self._routes[f"{existing.alias}.{name}"] = existing
            self._routes[f"{alias}.{name}"] = _Routed(alias, client, name, spec)
        elif any(r.original_name == name for r in self._routes.values()):
            self._routes[f"{alias}.{name}"] = _Routed(alias, client, name, spec)
        else:
            self._routes[name] = _Routed(alias, client, name, spec)

    def merged_specs(self) -> list[dict[str, object]]:
        """Return tool specs under their merged (possibly namespaced) names."""
        specs = []
        for merged_name, routed in self._routes.items():
            spec = dict(routed.spec)
            spec["name"] = merged_name
            specs.append(spec)
        return specs

    def register_into(self, registry: ToolRegistry) -> None:
        """Register every merged tool into `registry`, routed to its owning server."""
        for spec in self.merged_specs():
            merged_name = str(spec["name"])
            routed = self._routes[merged_name]
            call = make_bridge_fn(routed.client, routed.original_name)
            input_schema = spec["inputSchema"]
            assert isinstance(input_schema, dict), f"{merged_name!r} inputSchema must be a JSON object"
            registry.register(
                Tool(
                    name=merged_name,
                    description=str(spec["description"]),
                    parameters=input_schema,
                    fn=call,
                )
            )

    def route_of(self, merged_name: str) -> str:
        """Return the alias that owns `merged_name`, for tests and logging."""
        return self._routes[merged_name].alias

    def merged_names(self) -> list[str]:
        """Return the merged (possibly namespaced) tool names, sorted."""
        return sorted(self._routes)

    def shutdown(self) -> None:
        """Shut down every connected server cleanly."""
        for client in self._clients.values():
            client.shutdown()


def run_multi_server_demo() -> tuple[list[str], str]:
    """Connect to two identical MCP servers and demonstrate collision routing.

    Both servers expose the same tool set (`add`, `divide`,
    `summarize_note`), so every tool name collides and every merged name
    ends up namespaced as `"alpha.add"`, `"beta.divide"`, and so on. The
    scripted model calls one tool on each server; both calls succeed and are
    routed to the right subprocess.

    Returns:
        A tuple of (the merged tool names, the model's final answer).
    """
    host = MultiServerHost()
    host.add_server("alpha", MCPClient(client_name="alpha-client"))
    host.add_server("beta", MCPClient(client_name="beta-client"))
    try:
        registry = ToolRegistry()
        host.register_into(registry)
        merged_names = host.merged_names()

        provider = get_provider(
            script=[
                scripted_tool_call("alpha.add", {"a": 4, "b": 5}, call_id="call_1"),
                scripted_tool_call("beta.add", {"a": 100, "b": 1}, call_id="call_2"),
                "alpha.add gave 9, and beta.add gave 101, confirming both servers answered independently.",
            ]
        )
        messages = [Message.user("Call add on both the alpha and beta servers with small numbers and compare.")]
        completion = provider.complete(messages, tools=registry.specs(), system=SYSTEM_PROMPT)
        while completion.tool_calls:
            messages.append(Message.assistant(completion.content, completion.tool_calls))
            for call in completion.tool_calls:
                observation = registry.execute(call)
                messages.append(Message.tool(call.id, observation))
            completion = provider.complete(messages, tools=registry.specs(), system=SYSTEM_PROMPT)
        return merged_names, completion.content
    finally:
        host.shutdown()
