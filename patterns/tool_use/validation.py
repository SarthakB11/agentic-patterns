"""Argument validation and self-repair.

Proposed arguments are checked against the tool's JSON Schema before
anything runs. On failure, the error is returned as an observation instead
of an exception, and the loop continues so the model gets a chance to
correct the call; `run_tool_loop`'s `retry_limit` bounds how many such
repair turns a run will grant in total.

A production system increasingly prevents the structural half of this
problem before generation even finishes: OpenAI's `strict: true` and
Anthropic's structured-outputs beta compile a tool's JSON Schema into a
grammar that masks invalid tokens during decoding, so a wrong type or a
missing required field mostly cannot be produced in the first place.
`validate_arguments` and the repair turn below still matter for what
constrained decoding cannot catch: a structurally valid call with a wrong
value, such as a well-formed but unsupported currency code, which is a
semantic error caught by the tool's own logic rather than its schema.
"""

from __future__ import annotations

from agentic_patterns import Message, get_provider, scripted_tool_call

from patterns.tool_use.catalog import SYSTEM_PROMPT, build_registry
from patterns.tool_use.loop import ToolLoopResult, run_tool_loop


def demo_structural_repair() -> ToolLoopResult:
    """Repair a structurally invalid call: amount sent as a string, not a number.

    `convert_currency`'s schema declares `amount` as a number. The first
    call sends it as a string, which fails `validate_arguments` before the
    tool ever runs; the loop returns that as an error observation, and the
    scripted second call resends the same request with a numeric amount.
    """
    registry = build_registry()
    provider = get_provider(
        script=[
            scripted_tool_call("convert_currency", {"amount": "100", "from_currency": "USD", "to_currency": "EUR"}),
            scripted_tool_call("convert_currency", {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}),
            "100 USD converts to about 92.00 EUR at the demo rate.",
        ]
    )
    messages = [Message.user("Convert 100 USD to EUR.")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=3, retry_limit=1)

    print("=== 6. Argument validation with self-repair (structural error) ===")
    print(f"user:  {messages[0].content}")
    for round_record in result.rounds:
        for record in round_record.calls:
            print(f"  round {round_record.index} [{record.outcome}]: {record.call.arguments} -> {record.observation}")
    print(f"final: {result.final_answer}")
    print()
    return result


if __name__ == "__main__":
    demo_structural_repair()
