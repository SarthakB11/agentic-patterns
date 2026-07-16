"""Memory-as-tools: expose store, retrieve, update, and delete as callable
tools instead of a fixed pre-retrieval step.

Every other module in this pattern retrieves memory for the model, before
the model ever sees a prompt (`assembler.assemble_context` runs retrieval,
then hands the model a finished prompt). Here the model decides, inside its
own loop, whether and when memory needs consulting, exactly like any other
tool call. Retrieval timing becomes a model decision instead of a fixed
pipeline step, at the cost of an extra round trip whenever the model
chooses to use a tool.
"""

from __future__ import annotations

from agentic_patterns import (
    Embedder,
    Message,
    Provider,
    Tool,
    ToolRegistry,
    get_embedder,
    get_provider,
    scripted_tool_call,
)
from patterns.memory.vector_store import VectorStore


def build_memory_toolset(store: VectorStore, embedder: Embedder, namespace: str) -> ToolRegistry:
    """Build a `ToolRegistry` exposing memory_store, memory_retrieve,
    memory_update, and memory_delete, all bound to one namespace.
    """
    registry = ToolRegistry()

    def memory_store(key: str, text: str) -> str:
        embedding = embedder.embed([text])[0]
        store.upsert(key, namespace, text, embedding)
        return f"stored under {key!r}"

    def memory_retrieve(query: str, top_k: int = 2) -> str:
        query_vec = embedder.embed([query])[0]
        hits = store.search(namespace, query_vec, top_k=top_k, min_similarity=0.1)
        if not hits:
            return "no matching memory"
        return "; ".join(f"{h.record.id}: {h.record.text}" for h in hits)

    def memory_update(key: str, text: str) -> str:
        if store.get(namespace, key) is None:
            return f"ERROR: no memory named {key!r} to update"
        return memory_store(key, text)

    def memory_delete(key: str) -> str:
        removed = store.delete(namespace, key)
        return f"deleted {key!r}" if removed else f"ERROR: no memory named {key!r}"

    registry.register(
        Tool(
            name="memory_store",
            description="Persist a fact for later sessions, under a short key.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}, "text": {"type": "string"}},
                "required": ["key", "text"],
            },
            fn=memory_store,
        )
    )
    registry.register(
        Tool(
            name="memory_retrieve",
            description="Search stored memory by similarity to a query.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query"],
            },
            fn=memory_retrieve,
        )
    )
    registry.register(
        Tool(
            name="memory_update",
            description="Overwrite an existing memory by key.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}, "text": {"type": "string"}},
                "required": ["key", "text"],
            },
            fn=memory_update,
        )
    )
    registry.register(
        Tool(
            name="memory_delete",
            description="Delete a memory by key.",
            parameters={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
            fn=memory_delete,
        )
    )
    return registry


def run_memory_tools_demo(provider: Provider | None = None) -> str:
    """The model chooses, through tool calls, to store a fact in one turn
    and retrieve it in a later turn, instead of memory being fetched for it
    up front.
    """
    if provider is None:
        provider = get_provider(
            script=[
                scripted_tool_call(
                    "memory_store", {"key": "coffee_pref", "text": "User drinks dark roast coffee only in the morning."}
                ),
                "Got it, I'll remember you drink dark roast coffee in the mornings.",
                scripted_tool_call("memory_retrieve", {"query": "What coffee does the user drink?"}),
                "You drink dark roast coffee in the mornings.",
            ]
        )
    embedder = get_embedder()
    store = VectorStore()
    registry = build_memory_toolset(store, embedder, namespace="user:alex")

    store_call = provider.complete([Message.user("I only drink dark roast coffee, and only in the morning.")])
    store_results = [registry.execute(c) for c in store_call.tool_calls]
    provider.complete([Message.user("(tool result: " + "; ".join(store_results) + ")")])

    retrieve_call = provider.complete([Message.user("What coffee do I drink?")])
    retrieve_results = [registry.execute(c) for c in retrieve_call.tool_calls]
    answer = provider.complete([Message.user("(tool result: " + "; ".join(retrieve_results) + ")")])
    return answer.content
