"""Provider and embedder selection from environment variables.

Every example calls `get_provider()` / `get_embedder()` rather than
constructing a provider directly, so the same example code runs offline
against `MockProvider` by default and switches to a real API by setting an
environment variable, with no code changes.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from agentic_patterns.core.embeddings import Embedder, HashEmbedder, OpenAIEmbedder
from agentic_patterns.core.providers import (
    AnthropicProvider,
    Completion,
    MockProvider,
    OpenAICompatibleProvider,
    Provider,
)

_PROVIDER_NAMES = ("mock", "openai", "anthropic")
_EMBEDDER_NAMES = ("hash", "openai")


def get_provider(
    script: Sequence[Completion | str | dict[str, Any]] | None = None,
    name: str | None = None,
    model: str | None = None,
) -> Provider:
    """Build a `Provider` selected by name.

    Args:
        script: Scripted turns for `MockProvider`. Required when the
            resolved provider is "mock"; ignored otherwise.
        name: Provider name, one of "mock", "openai", "anthropic". Defaults
            to the `AGENTIC_PATTERNS_PROVIDER` environment variable, or
            "mock" if that is also unset.
        model: Model name to pass to a real provider. Ignored for "mock".

    Raises:
        ValueError: If "mock" is selected with no script, or `name` is not
            one of the recognized provider names.
    """
    resolved = name or os.environ.get("AGENTIC_PATTERNS_PROVIDER", "mock")
    if resolved == "mock":
        if script is None:
            raise ValueError(
                "The mock provider requires a script. Pass script=[...] with the "
                "turns this example expects to send, or select a real provider "
                "with name='openai' / name='anthropic' or AGENTIC_PATTERNS_PROVIDER."
            )
        return MockProvider(script)
    if resolved == "openai":
        return OpenAICompatibleProvider(model=model)
    if resolved == "anthropic":
        return AnthropicProvider(model=model)
    raise ValueError(f"Unknown provider {resolved!r}. Valid providers: {', '.join(_PROVIDER_NAMES)}")


def get_embedder(name: str | None = None) -> Embedder:
    """Build an `Embedder` selected by name.

    Args:
        name: Embedder name, one of "hash", "openai". Defaults to the
            `AGENTIC_PATTERNS_EMBEDDER` environment variable, or "hash" if
            that is also unset.

    Raises:
        ValueError: If `name` is not one of the recognized embedder names.
    """
    resolved = name or os.environ.get("AGENTIC_PATTERNS_EMBEDDER", "hash")
    if resolved == "hash":
        return HashEmbedder()
    if resolved == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"Unknown embedder {resolved!r}. Valid embedders: {', '.join(_EMBEDDER_NAMES)}")
