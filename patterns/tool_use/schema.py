"""Schema autogeneration: derive a tool's description and JSON Schema from
its type hints and docstring instead of writing them by hand.

The core `ToolRegistry.tool()` decorator (agentic_patterns.core.tools) takes
an explicit `description` and `parameters` argument; it does not look at the
wrapped function's signature. `auto_tool` wraps that decorator: it reads the
function's type hints and a Google-style docstring and builds both
automatically, so a tool author writes ordinary typed Python once and gets a
model-facing schema for free, with no separate schema to keep in sync.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, get_type_hints

from agentic_patterns import ToolRegistry

_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_ARG_LINE_RE = re.compile(r"^\s*(\w+):\s*(.+)$")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into a one-line summary and an Args: map.

    Args:
        doc: The function's `__doc__`, or None.

    Returns:
        A (summary, arg_descriptions) pair. `summary` is the first non-blank
        line. `arg_descriptions` maps each parameter name to its description
        line under an "Args:" section; empty if there is no such section.
    """
    if not doc:
        return "", {}
    lines = [line.strip() for line in doc.strip().splitlines()]
    summary = lines[0] if lines else ""
    arg_descriptions: dict[str, str] = {}
    in_args = False
    for line in lines[1:]:
        if line.startswith("Args:"):
            in_args = True
            continue
        if in_args:
            if not line or line.endswith(":"):
                break
            match = _ARG_LINE_RE.match(line)
            if match:
                arg_descriptions[match.group(1)] = match.group(2)
    return summary, arg_descriptions


def schema_from_function(fn: Callable[..., Any]) -> dict[str, Any]:
    """Derive a tool description and JSON Schema from a function's signature and docstring.

    Args:
        fn: A function with full type hints and a Google-style docstring
            whose first line is a one-sentence summary and whose optional
            "Args:" section documents each parameter. Parameters with a
            default value are treated as optional; all others are required.

    Returns:
        {"description": str, "parameters": <JSON Schema object>}.
    """
    summary, arg_descriptions = _parse_docstring(fn.__doc__)
    hints = get_type_hints(fn)
    signature = inspect.signature(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        json_type = _TYPE_TO_JSON.get(hints.get(name, str), "string")
        prop: dict[str, Any] = {"type": json_type}
        if name in arg_descriptions:
            prop["description"] = arg_descriptions[name]
        properties[name] = prop
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "description": summary,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def auto_tool(registry: ToolRegistry) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a function as a tool, autogenerating its schema.

    Example:
        registry = ToolRegistry()

        @auto_tool(registry)
        def add(a: int, b: int) -> int:
            '''Add two integers.

            Args:
                a: First addend.
                b: Second addend.
            '''
            return a + b

        # registry.specs() now includes {"name": "add", "description":
        # "Add two integers.", "parameters": {"type": "object", ...}}
        # with no schema written by hand.

    Args:
        registry: The registry to register the derived tool into.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        derived = schema_from_function(fn)
        registry.tool(description=derived["description"], parameters=derived["parameters"])(fn)
        return fn

    return decorator
