"""A tiny in-memory knowledge base and tools shared by the ReAct demos.

Every variant module in this package answers questions against the same toy
world, using the same two tools (`search`, `lookup`). That lets a reader
compare how each control-flow variant handles an equivalent task instead of
re-deriving a new toy world for every module.
"""

from __future__ import annotations

from agentic_patterns import Tool, ToolRegistry

_FACTS: dict[str, str] = {
    "great wall": "The Great Wall is located in China.",
    "capital of china": "The capital of China is Beijing.",
    "eiffel tower": "The Eiffel Tower is located in Paris, France.",
    "population of beijing": "Beijing has a population of about 21.5 million people.",
}


def search(query: str) -> str:
    """Look up a query against the toy knowledge base.

    Args:
        query: A free-text search query.

    Returns:
        The matching fact, or a not-found message if nothing matches.
    """
    key = query.strip().lower()
    if key in _FACTS:
        return _FACTS[key]
    for fact_key, fact_value in _FACTS.items():
        if key and (key in fact_key or fact_key in key):
            return fact_value
    return f"No results found for '{query}'."


def lookup(term: str) -> str:
    """Look up one specific term, the way the original ReAct paper's Lookup[] action does.

    Args:
        term: The term to look up, mirroring an in-page Ctrl+F search.

    Returns:
        The same fact `search` would return, since this toy world has a
        single flat knowledge base rather than a source document to scan.
    """
    return search(term)


def build_registry() -> ToolRegistry:
    """Build a ToolRegistry with `search` and `lookup` for the demos."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="search",
            description="Search the knowledge base for a topic and return a short fact.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Topic to search for."}},
                "required": ["query"],
            },
            fn=search,
        )
    )
    registry.register(
        Tool(
            name="lookup",
            description="Look up one specific term, like a Ctrl+F search over a page.",
            parameters={
                "type": "object",
                "properties": {"term": {"type": "string", "description": "Term to look up."}},
                "required": ["term"],
            },
            fn=lookup,
        )
    )
    return registry
