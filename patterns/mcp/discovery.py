"""Registry-based server discovery.

Every connection everywhere else in this pattern hard-codes the launch
command: `client.py`'s `_DEFAULT_SERVER_COMMAND` is `[sys.executable, "-m",
"patterns.mcp.server"]`, chosen at import time, never discovered. A real
host does not know its servers in advance; it looks them up. The official
MCP Registry (`registry.modelcontextprotocol.io`) is live in preview for
exactly this, a REST API returning `server.json`-shaped records, and
Anthropic reported the ecosystem passing ten thousand public servers in a
January 2026 note.

This module stands in for a registry lookup with a small static document
shaped like `server.json` entries, so the demo stays offline and fully
deterministic: no network call replaces the registry's REST API, just a
plain Python list. `find_servers` filters it; `connect_discovered` builds a
real `MCPClient` from the selected record's launch command and runs the
normal connect-and-list flow against the real subprocess, proving the
discovered record is not just data, it actually works.

A client-facing `.well-known/mcp.json` server-card mechanism (SEP-2127 /
SEP-1649), letting a client probe an origin directly instead of consulting
a registry, is proposed but unshipped as of this revision and is not built
here; a live client today discovers servers by querying the registry's REST
API, not by origin probing.

Ranking or retrieving among hundreds of discovered tools (as opposed to
servers) is `patterns/tool_use/` territory; this module stops at connecting
to one selected server.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from patterns.mcp.client import MCPClient

REGISTRY: list[dict[str, Any]] = [
    {
        "name": "agentic-patterns/arithmetic-server",
        "description": "Three deterministic arithmetic and note tools over stdio.",
        "capabilities": ["tools", "resources", "prompts"],
        "launch": {"type": "stdio", "command": [sys.executable, "-m", "patterns.mcp.server"]},
    },
    {
        "name": "agentic-patterns/archive-server",
        "description": "A hypothetical resources-only archive server, listed but not launchable here.",
        "capabilities": ["resources"],
        "launch": {"type": "stdio", "command": ["python3", "-m", "hypothetical.archive_server"]},
    },
    {
        "name": "agentic-patterns/http-mirror",
        "description": "A hypothetical HTTP-transport mirror of the arithmetic server.",
        "capabilities": ["tools"],
        "launch": {"type": "http", "url": "https://mirror.example.invalid/mcp"},
    },
]

RegistryPredicate = Callable[[dict[str, Any]], bool]


def has_capability(capability: str) -> RegistryPredicate:
    """Build a predicate matching registry entries that declare `capability`."""

    def predicate(entry: dict[str, Any]) -> bool:
        return capability in entry.get("capabilities", [])

    return predicate


def named(name: str) -> RegistryPredicate:
    """Build a predicate matching a registry entry by exact name."""

    def predicate(entry: dict[str, Any]) -> bool:
        return entry.get("name") == name

    return predicate


def find_servers(registry: list[dict[str, Any]], predicate: RegistryPredicate) -> list[dict[str, Any]]:
    """Return the registry entries for which `predicate` is true, in registry order."""
    return [entry for entry in registry if predicate(entry)]


def connect_discovered(record: dict[str, Any]) -> MCPClient:
    """Build, connect, and hand-shake an `MCPClient` from a discovered registry record.

    Args:
        record: A `server.json`-shaped entry with a `launch` field.

    Returns:
        A connected, initialized `MCPClient`, ready for `list_tools()` and
        `call_tool()`. Caller is responsible for `shutdown()`.

    Raises:
        ValueError: `record`'s launch type is not `"stdio"`. This offline
            demo only spawns local subprocesses; a live client would also
            support an HTTP launch type by opening a Streamable HTTP
            connection instead.
    """
    launch = record["launch"]
    if launch["type"] != "stdio":
        raise ValueError(
            f"connect_discovered only supports stdio launches in this offline demo; "
            f"{record['name']!r} declares {launch['type']!r}"
        )
    client = MCPClient(command=launch["command"], client_name=f"discovery-client:{record['name']}")
    client.connect()
    client.initialize()
    client.notify_initialized()
    return client


def run_discovery_demo() -> dict[str, Any]:
    """Filter the static registry, then connect the one launchable match.

    Returns:
        A dict of discovery outcomes, keyed for `main.py` to print and
        `tests/test_mcp.py` to assert against.
    """
    tools_capable = find_servers(REGISTRY, has_capability("tools"))
    named_lookup = find_servers(REGISTRY, named("agentic-patterns/arithmetic-server"))
    empty_result = find_servers(REGISTRY, named("agentic-patterns/does-not-exist"))

    client = connect_discovered(named_lookup[0])
    try:
        tools = client.list_tools()
    finally:
        client.shutdown()

    return {
        "tools_capable_names": [entry["name"] for entry in tools_capable],
        "named_lookup_names": [entry["name"] for entry in named_lookup],
        "empty_result": empty_result,
        "connected_tool_names": sorted(t["name"] for t in tools),
    }
