"""agentic-patterns: core agentic AI patterns as small, runnable, tested examples.

This top-level package re-exports the shared core (types, tools, providers,
embeddings, config) so pattern examples can `import agentic_patterns` rather
than reaching into `agentic_patterns.core` directly.
"""

from agentic_patterns.core import (
    AnthropicProvider,
    Completion,
    Embedder,
    HashEmbedder,
    Message,
    MockProvider,
    MockScriptExhausted,
    OpenAICompatibleProvider,
    OpenAIEmbedder,
    Provider,
    Tool,
    ToolCall,
    ToolRegistry,
    cosine_similarity,
    get_embedder,
    get_provider,
    scripted_tool_call,
)

__all__ = [
    "Message",
    "ToolCall",
    "Completion",
    "Tool",
    "ToolRegistry",
    "Provider",
    "MockProvider",
    "MockScriptExhausted",
    "scripted_tool_call",
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "Embedder",
    "HashEmbedder",
    "OpenAIEmbedder",
    "cosine_similarity",
    "get_provider",
    "get_embedder",
]
