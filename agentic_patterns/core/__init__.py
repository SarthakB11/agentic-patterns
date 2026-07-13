"""Public API surface for the shared agentic-patterns core.

Every pattern example imports from `agentic_patterns` (which re-exports this
module) rather than reaching into submodules directly, so the core's
internal layout can change without breaking examples.
"""

from agentic_patterns.core.config import get_embedder, get_provider
from agentic_patterns.core.embeddings import Embedder, HashEmbedder, OpenAIEmbedder, cosine_similarity
from agentic_patterns.core.providers import (
    AnthropicProvider,
    MockProvider,
    MockScriptExhausted,
    OpenAICompatibleProvider,
    Provider,
    scripted_tool_call,
)
from agentic_patterns.core.tools import Tool, ToolRegistry
from agentic_patterns.core.types import Completion, Message, ToolCall

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
