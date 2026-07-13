"""Tool definitions and a small registry for executing them.

A `Tool` pairs a JSON Schema description (what the model sees) with a plain
Python callable (what actually runs). `ToolRegistry` collects tools, exposes
them in the provider-neutral spec shape, and executes calls the model makes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentic_patterns.core.types import ToolCall


@dataclass
class Tool:
    """A single callable tool, described for a model and runnable locally.

    Attributes:
        name: Unique tool name, as the model will refer to it.
        description: Natural-language description shown to the model.
        parameters: JSON Schema for the arguments object the model must
            produce, e.g. {"type": "object", "properties": {...}}.
        fn: The Python callable that performs the tool's work. Called with
            the parsed arguments as keyword arguments.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]


class ToolRegistry:
    """Collects tools and executes calls made against them."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry, keyed by its name."""
        self._tools[tool.name] = tool

    def tool(
        self, *, description: str, parameters: dict[str, Any]
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form of `register`. Infers the tool name from the function name.

        Example:
            @registry.tool(description="Add two numbers", parameters={...})
            def add(a: int, b: int) -> int:
                return a + b
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                Tool(name=fn.__name__, description=description, parameters=parameters, fn=fn)
            )
            return fn

        return decorator

    def specs(self) -> list[dict[str, Any]]:
        """Return provider-neutral tool specs for passing to `Provider.complete()`."""
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def get(self, name: str) -> Tool:
        """Look up a tool by name.

        Raises:
            KeyError: If no tool with that name is registered. The message
                lists the names that are registered, to make typos obvious.
        """
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(sorted(self._tools)) or "(none registered)"
            raise KeyError(f"Unknown tool {name!r}. Known tools: {known}") from None

    def execute(self, call: ToolCall) -> str:
        """Run a tool call and return its result as a string.

        Exceptions raised by the tool's `fn` are caught and turned into an
        "ERROR: ..." observation string rather than propagated. Agent loops
        treat tool results as observations the model reasons about, so a
        failed tool call should feed back into the loop like any other
        observation instead of crashing it.
        """
        try:
            tool = self.get(call.name)
            result = tool.fn(**call.arguments)
            return str(result)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            return f"ERROR: {exc}"
