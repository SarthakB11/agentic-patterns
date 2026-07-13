"""Reusable engine for the tool-calling loop.

This module holds the pattern's mechanics, kept separate from any one
provider, tool catalog, or demo script: sending the conversation plus tool
specs to the model, validating proposed arguments against each tool's JSON
Schema, executing valid calls (concurrently, when a turn has more than one),
turning both validation failures and raised exceptions into observations the
model can react to, and enforcing a repair-attempt budget and an iteration
cap so a stuck model cannot loop forever.

Canonical control flow (see docs/research/tool_use.md): call the model,
stop if it returned plain text, otherwise validate and execute every
requested call, append each call and its observation to the history, and
repeat with the extended history until the model stops calling tools or the
iteration cap is hit.

Unknown-tool policy: a call naming a tool that is not registered becomes an
"ERROR: Unknown tool ..." observation and the loop continues, rather than
raising. That mirrors how a raised exception from a real tool is handled,
and keeps a single hallucinated name from crashing an otherwise-recoverable
run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from agentic_patterns import Message, Provider, Tool, ToolCall, ToolRegistry

_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """Validate tool call arguments against a JSON Schema object.

    A teaching-scale validator, not a full JSON Schema implementation: it
    checks that every required field is present, that no unexpected field
    was passed, and that each field's value matches its declared top-level
    type. That is enough to catch the hallucinated-field and wrong-type
    mistakes the brief calls out as the most common self-repair triggers.

    Args:
        schema: JSON Schema for the arguments object, e.g.
            {"type": "object", "properties": {...}, "required": [...]}.
        arguments: The parsed arguments a model proposed for one call.

    Returns:
        Human-readable error strings, one per problem found. Empty when the
        arguments are valid.
    """
    errors: list[str] = []
    properties = schema.get("properties", {})
    for field_name in schema.get("required", []):
        if field_name not in arguments:
            errors.append(f"missing required field '{field_name}'")
    for key, value in arguments.items():
        if key not in properties:
            errors.append(f"unexpected field '{key}'")
            continue
        expected = properties[key].get("type")
        py_type = _JSON_TYPE_MAP.get(expected)
        if py_type is None:
            continue
        if expected == "integer" and isinstance(value, bool):
            errors.append(f"field '{key}' expected type integer, got boolean")
        elif not isinstance(value, py_type):
            errors.append(f"field '{key}' expected type {expected}, got {type(value).__name__}")
    return errors


@dataclass
class CallRecord:
    """One tool call within a round and what happened when it was handled.

    Attributes:
        call: The `ToolCall` the model requested.
        observation: The string fed back to the model for this call.
        outcome: One of "ok", "tool_error", "unknown_tool",
            "repair_requested", or "validation_failed".
    """

    call: ToolCall
    observation: str
    outcome: str


@dataclass
class RoundRecord:
    """One round trip to the model: every call it made and how each resolved."""

    index: int
    calls: list[CallRecord] = field(default_factory=list)


@dataclass
class ToolLoopResult:
    """The outcome of a full tool-calling loop run.

    Attributes:
        final_answer: The model's closing text, empty if the loop stopped on
            the iteration cap instead of a text-only turn.
        rounds: One `RoundRecord` per round that involved tool calls.
        history: The full message history, including every tool result,
            useful for a caller that wants to continue the conversation
            (see `write_action.py`'s elicitation demo).
        stop_reason: "stop" (model returned plain text) or "max_iterations".
    """

    final_answer: str
    rounds: list[RoundRecord]
    history: list[Message]
    stop_reason: str


def run_tool_loop(
    provider: Provider,
    registry: ToolRegistry,
    messages: list[Message],
    *,
    system: str | None = None,
    offered_specs: list[dict[str, Any]] | None = None,
    max_iterations: int = 6,
    retry_limit: int = 2,
    validate: bool = True,
) -> ToolLoopResult:
    """Run the call, validate, execute, observe loop to a stop condition.

    Args:
        provider: The model to drive. `MockProvider` in every demo here.
        registry: Tools available for execution. Always authoritative for
            what can run, independent of `offered_specs`.
        messages: Conversation so far, not including the system prompt.
        system: System prompt passed straight through to the provider.
        offered_specs: Tool specs shown to the model this run. Defaults to
            `registry.specs()` (every registered tool). Pass a filtered or
            empty list to emulate a provider's `tool_choice` (see
            `forced_choice.py`) without changing what the registry can run.
        max_iterations: Maximum number of model round trips before the loop
            gives up and returns with stop_reason="max_iterations".
        retry_limit: Maximum number of invalid-argument repair turns granted
            across the whole run. Once exhausted, further invalid calls
            become a terminal validation-failed observation instead of
            another repair opportunity.
        validate: Whether to validate arguments against each tool's schema
            before executing. Demos that want to show a raw tool-execution
            error instead of a validation error pass False.

    Returns:
        A `ToolLoopResult` with the final answer (if any), the full
        round-by-round record, and why the loop stopped.
    """
    history = list(messages)
    rounds: list[RoundRecord] = []
    repairs_used = 0

    for round_index in range(1, max_iterations + 1):
        specs = registry.specs() if offered_specs is None else offered_specs
        completion = provider.complete(history, tools=specs, system=system)
        history.append(Message.assistant(completion.content, completion.tool_calls))

        if not completion.tool_calls:
            return ToolLoopResult(completion.content, rounds, history, "stop")

        order: list[str] = []
        resolved: dict[str, CallRecord] = {}
        pending: list[tuple[ToolCall, Tool]] = []

        for call in completion.tool_calls:
            order.append(call.id)
            try:
                tool = registry.get(call.name)
            except KeyError as exc:
                resolved[call.id] = CallRecord(call, f"ERROR: {exc}", "unknown_tool")
                continue

            if validate:
                errors = validate_arguments(tool.parameters, call.arguments)
                if errors:
                    if repairs_used >= retry_limit:
                        resolved[call.id] = CallRecord(
                            call,
                            "ERROR: invalid arguments and repair budget exhausted: " + "; ".join(errors),
                            "validation_failed",
                        )
                    else:
                        repairs_used += 1
                        resolved[call.id] = CallRecord(
                            call, "ERROR: invalid arguments: " + "; ".join(errors), "repair_requested"
                        )
                    continue

            pending.append((call, tool))

        if pending:
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                observations = list(pool.map(lambda pair: registry.execute(pair[0]), pending))
            for (call, _tool), observation in zip(pending, observations):
                outcome = "tool_error" if observation.startswith("ERROR:") else "ok"
                resolved[call.id] = CallRecord(call, observation, outcome)

        round_calls = [resolved[call_id] for call_id in order]
        for record in round_calls:
            history.append(Message.tool(record.call.id, record.observation))
        rounds.append(RoundRecord(round_index, round_calls))

    return ToolLoopResult("", rounds, history, "max_iterations")
