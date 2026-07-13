"""Tool use / function calling: a model proposes structured calls, the app runs them.

Instead of only producing free text, the model reads a catalog of tools
(name, description, JSON Schema) and either answers directly or returns a
tool name plus typed arguments. The app parses that call, validates the
arguments, runs the real function, and feeds the result back so the model
can call more tools or produce a final answer. The model never executes
code itself; it only proposes calls, and the runtime executes them inside a
controlled boundary.

This demo runs eleven sub-variants end to end, entirely offline against
`MockProvider` with scripted, coherent conversations, no network call and no
API key:

1. Schema autogeneration: JSON Schema derived from type hints and docstring.
2. Single-shot tool use: one call, one final answer.
3. Sequential multi-step loop: a second call depends on the first's result.
4. Parallel tool calls: three independent calls in one turn, run concurrently.
5. Forced tool choice: none, required, and a named choice.
6. Structured-output-as-tool: a forced extraction call with no side effect.
7. Argument validation with self-repair on a structural error.
8. Guardrails: a tool error, an unknown tool, a retry cap, and an
   iteration cap, none of which crash the loop.
9. A write action gated behind confirmation, unlocked by a simulated
   elicitation step.
10. Programmatic tool calling (code-as-action): a multi-step plan run
    locally in one pass instead of one round trip per step.
11. Retrieval-based tool selection (tool search): the top few of ten tools
    offered to the model instead of the whole catalog.

It also prints conceptual notes on the two taxonomy entries with no
runnable demo: learned tool use (Toolformer) and neuro-symbolic routing
(MRKL).

Run it from the repository root:

    python -m patterns.tool_use.main

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key in the environment) to run the same code against
a real model instead of the mock. No source change is required; every demo
function builds its provider through `agentic_patterns.get_provider`.
"""

from __future__ import annotations

import inspect

from patterns.tool_use import (
    code_execution,
    concepts,
    forced_choice,
    guardrails,
    parallel,
    schema as schema_module,
    sequential,
    single_shot,
    structured_output,
    tool_search,
    validation,
    write_action,
)
from patterns.tool_use.catalog import build_registry


def _demo_schema_autogen() -> None:
    """Show a tool's schema derived from its type hints and docstring, no schema written by hand."""
    registry = build_registry()
    spec = registry.get("convert_currency")
    derived = schema_module.schema_from_function(spec.fn)

    print("=== 1. Schema autogeneration (decorator-based registry) ===")
    print(f"function: {spec.fn.__name__}{inspect.signature(spec.fn)}")
    print(f"derived description: {derived['description']}")
    print(f"derived parameters: {derived['parameters']}")
    print()


def main() -> None:
    """Run every tool_use sub-variant demo and print a readable transcript."""
    print("TOOL USE PATTERN: model proposes calls, app validates and executes them\n")

    _demo_schema_autogen()
    single_shot.demo_single_shot()
    sequential.demo_sequential()
    parallel.demo_parallel()
    forced_choice.demo_none()
    forced_choice.demo_required()
    forced_choice.demo_named()
    structured_output.demo_structured_output()
    validation.demo_structural_repair()
    guardrails.demo_tool_error()
    guardrails.demo_unknown_tool()
    guardrails.demo_retry_cap()
    guardrails.demo_iteration_cap()
    write_action.demo_write_action_accepted()
    write_action.demo_write_action_declined()
    code_execution.demo_code_execution()
    tool_search.demo_tool_search()
    concepts.print_concept_notes()

    print("All eleven sub-variants completed without exhausting their scripts.")


if __name__ == "__main__":
    main()
