"""Programmatic tool calling, linear form.

Instead of one round trip per tool call, the model emits a short program: a
list of steps, each naming a tool, its arguments, and where to store the
result, with later steps able to reference an earlier step's output. The
runtime executes the whole program locally against the real tools and sends
only the final summary back to the model, so intermediate observations never
re-enter the context window.

This is programmatic tool calling in its linear form, not the code-as-action
variant CodeAct (Wang et al., ICML 2024, arXiv:2402.01030) describes: the
step list has no loops and no branches, so a task like "look up ten orders
and return only the delayed ones" needs ten steps here, one per order, and
cannot express the filtering itself. That is a real limit, not a simplifying
detail: it is exactly what keeps this module from showing the context
collapse Anthropic's "Code execution with MCP" (Nov 2025) reports, a claimed
150K-to-2K-token reduction on a real workflow. `code_action.py` is the
Turing-complete form, a real `exec`-backed sandbox where the model writes
arbitrary Python with control flow; that module is where the collapse shows
up, since one action there can process many tool results and return only
the aggregate, matching what Anthropic's number depends on.
"""

from __future__ import annotations

from typing import Any

from agentic_patterns import Message, MockProvider, ToolCall, ToolRegistry, scripted_tool_call
from patterns.tool_use.catalog import SYSTEM_PROMPT, build_registry

PROGRAM_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + " When a request needs more than one tool, call run_program once with "
    "every step it needs instead of one call per step. A later step may "
    'reference an earlier one\'s result with "$step_name.field".'
)

_PROGRAM_TOOL_SPEC = {
    "name": "run_program",
    "description": (
        "Run a short sequence of tool calls locally in one pass. Each step "
        "names a registered tool, its arguments, and a name to save its "
        "result under; later steps may reference '$name.field' to read a "
        "prior step's parsed result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": "Ordered list of {call, args, save_as} steps to run.",
            }
        },
        "required": ["steps"],
    },
}


def _resolve(value: Any, results: dict[str, str]) -> Any:
    """Resolve a '$step_name.field' placeholder against prior step results.

    Prior results are plain `key=value key=value` observation strings (the
    same shape `catalog.lookup_order` returns), so resolution is a small
    parse rather than a real object lookup. Non-placeholder values pass
    through unchanged.
    """
    if not (isinstance(value, str) and value.startswith("$")):
        return value
    step_name, _, field = value[1:].partition(".")
    observation = results[step_name]
    parsed = dict(part.split("=", 1) for part in observation.split())
    return parsed[field]


def run_program(registry: ToolRegistry, steps: list[dict[str, Any]]) -> dict[str, str]:
    """Execute a programmatic-tool-calling plan locally, without a model round trip per step.

    Args:
        registry: Tools the plan is allowed to call.
        steps: Each step is {"call": tool_name, "args": {...}, "save_as": name}.
            Argument values starting with "$" are resolved against a prior
            step's saved result before the call runs.

    Returns:
        A mapping from each step's `save_as` name to its observation string.
    """
    results: dict[str, str] = {}
    for step in steps:
        save_as = step.get("save_as", step["call"])
        args = {key: _resolve(value, results) for key, value in step.get("args", {}).items()}
        call = ToolCall(id=save_as, name=step["call"], arguments=args)
        results[save_as] = registry.execute(call)
    return results


def demo_code_execution() -> dict[str, str]:
    """Run a two-step plan (order lookup, then customer email) in one program instead of two round trips.

    Contrast with `sequential.py`, which performs the same two-step lookup
    but pays one model round trip per step (3 total: call, call, final
    text). This module pays 2 total: one round trip to get the program,
    one to read its summarized result and answer.
    """
    registry = build_registry()
    provider = MockProvider(
        [
            scripted_tool_call(
                "run_program",
                {
                    "steps": [
                        {"call": "lookup_order", "args": {"order_id": "ORD-1002"}, "save_as": "order"},
                        {
                            "call": "get_customer_email",
                            "args": {"customer_id": "$order.customer_id"},
                            "save_as": "email",
                        },
                    ]
                },
            ),
            "Order ORD-1002 is processing; the customer's email on file is sam@example.com.",
        ]
    )
    messages = [Message.user("Check order ORD-1002's status and find the customer's email, in one pass.")]

    completion = provider.complete(messages, tools=[_PROGRAM_TOOL_SPEC], system=PROGRAM_SYSTEM_PROMPT)
    program_call = completion.tool_calls[0]
    results = run_program(registry, program_call.arguments["steps"])

    history = [
        *messages,
        Message.assistant(completion.content, completion.tool_calls),
        Message.tool(program_call.id, str(results)),
    ]
    final = provider.complete(history, tools=[_PROGRAM_TOOL_SPEC], system=PROGRAM_SYSTEM_PROMPT)

    print("=== 9. Programmatic tool calling / code-as-action ===")
    print(f"user:  {messages[0].content}")
    print(f"  program: {len(program_call.arguments['steps'])} steps run locally, no round trip between them")
    print(f"  step results: {results}")
    print(f"final: {final.content}")
    print(
        f"model round trips used: {len(provider.calls)} "
        "(vs 3 for the equivalent per-call loop in sequential.py)"
    )
    print()
    return results


if __name__ == "__main__":
    demo_code_execution()
