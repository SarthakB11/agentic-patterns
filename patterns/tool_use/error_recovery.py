"""Class-routed tool-failure recovery.

`guardrails.py` turns every failure into one undifferentiated "ERROR: ..."
observation, and `run_tool_loop`'s `retry_limit` is one blind retry budget
shared across every kind of problem. "When Tools Fail" / ToolMaze (Zhu et
al., arXiv:2606.05806) introduces a 2x2 taxonomy, explicit versus implicit
and transient versus permanent, and finds agents suffer systemic over-trust
in corrupted or empty outputs and get trapped in futile retry loops on a
call that will never succeed. PALADIN (Vuddanti et al., arXiv:2509.25238)
responds by retrieving a recovery action keyed to the failure's type at
inference time, lifting recovery from 32.8 to 89.7 percent against a
ToolBench baseline; this module borrows only that inference-time
keyed-lookup idea, not PALADIN's LoRA training on 50K recovery trajectories,
a training-time method that stays out of scope alongside `concepts.py`'s
existing note on Toolformer. ToolFailBench (Soni, arXiv:2607.04686)
separates Tool-Skip, Result-Ignore, Output-Fabrication and
Unnecessary-Tool-Use, the finding this module reflects most directly in its
"implicit" class: an empty or ignored result is a distinct failure from a
raised exception, not the same "ERROR:" shape.

Each failed call is classified into one of four classes and routed to its
own strategy instead of one shared retry budget: transient is retried up to
a per-tool budget, with each attempt recorded; permanent is substituted for
a sibling tool from a small alias map, or abstained if there is none;
malformed_args is handed to the existing `validate_arguments` repair path,
the one class `guardrails.py` already covers well; implicit (an empty
result) is never silently accepted as success, a "verify this" observation
is injected instead, directly answering ToolMaze's over-trust finding. The
per-tool transient budget is shared across the whole run, not reset per
call, which is ToolMaze's anti-futile-loop guard.

Classification needs the real exception type a call raised, so this module
calls each tool's function directly rather than through `ToolRegistry.execute`,
which stringifies every exception into one "ERROR: ..." observation before a
caller ever sees what kind of error it was.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentic_patterns import Message, Provider, Tool, ToolCall, ToolRegistry, get_provider, scripted_tool_call

from patterns.tool_use.catalog import SYSTEM_PROMPT, build_registry
from patterns.tool_use.loop import validate_arguments
from patterns.tool_use.schema import auto_tool


class TransientToolError(RuntimeError):
    """A tool failure expected to succeed on retry, e.g. a timeout or a dropped connection."""


class PermanentToolError(RuntimeError):
    """A tool failure that will not succeed on retry, e.g. a permanently unsupported input."""


def wrap_with_injected_failures(fn: Callable[..., Any], schedule: list[str]) -> Callable[..., Any]:
    """Wrap a tool function with a scripted, deterministic failure schedule.

    Args:
        fn: The real tool function to wrap.
        schedule: Outcomes consumed in order, one per call: "transient"
            raises `TransientToolError`, "permanent" raises
            `PermanentToolError`, "empty" returns "" without calling `fn`.
            Once exhausted (or for any other entry), calls `fn` normally.

    Returns:
        A wrapped function with the same calling convention as `fn`.
    """
    remaining = list(schedule)

    def wrapped(**kwargs: Any) -> Any:
        outcome = remaining.pop(0) if remaining else "ok"
        if outcome == "transient":
            raise TransientToolError(f"transient failure calling {fn.__name__}")
        if outcome == "permanent":
            raise PermanentToolError(f"permanent failure calling {fn.__name__}")
        if outcome == "empty":
            return ""
        return fn(**kwargs)

    return wrapped


def build_recovery_registry(schedules: dict[str, list[str]] | None = None) -> ToolRegistry:
    """Build the shared ops-assistant registry with scripted failures injected, plus two substitute tools.

    Args:
        schedules: Maps a catalog tool name to its injected failure
            schedule. Tools not named here behave normally.

    Returns:
        A registry with every catalog tool present, plus
        `get_weather_backup` and `lookup_order_backup`, lower-fidelity
        substitutes used on a permanent failure.
    """
    schedules = schedules or {}
    base = build_registry()
    registry = ToolRegistry()
    for spec in base.specs():
        tool = base.get(spec["name"])
        wrapped_fn = wrap_with_injected_failures(tool.fn, schedules.get(tool.name, []))
        registry.register(Tool(tool.name, tool.description, tool.parameters, wrapped_fn))

    @auto_tool(registry)
    def get_weather_backup(city: str) -> str:
        """Look up weather from a secondary source with lower precision.

        Args:
            city: City name, e.g. "Tokyo".
        """
        return f"{city}: backup source, live detail unavailable, assume seasonal average"

    @auto_tool(registry)
    def lookup_order_backup(order_id: str) -> str:
        """Look up an order's status from a secondary read-replica with no customer id.

        Args:
            order_id: Order identifier, e.g. "ORD-1001".
        """
        return f"{order_id}: backup source, status unknown, customer id unavailable"

    return registry


@dataclass(frozen=True)
class Classification:
    """The outcome class a call's execution was sorted into.

    Attributes:
        label: One of "ok", "transient", "permanent", "malformed_args", "implicit".
        detail: Human-readable detail, empty for "ok".
    """

    label: str
    detail: str = ""


def classify_failure(tool: Tool, arguments: dict[str, Any]) -> tuple[Classification, Any]:
    """Run one call directly against `tool.fn` and classify the outcome.

    Runs `validate_arguments` first, so a malformed call never reaches the
    tool at all, matching `run_tool_loop`'s validate-before-execute order.

    Args:
        tool: The tool to call.
        arguments: Arguments to pass, already resolved from the model's call.

    Returns:
        (classification, result): `result` is the tool's return value on
        "ok", otherwise None.
    """
    errors = validate_arguments(tool.parameters, arguments)
    if errors:
        return Classification("malformed_args", "; ".join(errors)), None
    try:
        result = tool.fn(**arguments)
    except TransientToolError as exc:
        return Classification("transient", str(exc)), None
    except PermanentToolError as exc:
        return Classification("permanent", str(exc)), None
    except Exception as exc:  # noqa: BLE001 - an unclassified exception still degrades gracefully
        return Classification("permanent", str(exc)), None
    if result == "" or result is None:
        return Classification("implicit", "empty result"), result
    return Classification("ok"), result


@dataclass
class RecoveryAttempt:
    """The record of one call's classification and recovery outcome.

    Attributes:
        call: The tool call this attempt resolves.
        classification: The class the failure (or success) was sorted into.
        strategy: Human-readable description of the strategy run.
        observation: The string fed back to the model for this call.
        succeeded: Whether the call ultimately produced a usable result.
    """

    call: ToolCall
    classification: str
    strategy: str
    observation: str
    succeeded: bool


def recover_call(
    registry: ToolRegistry,
    call: ToolCall,
    *,
    substitutes: dict[str, str],
    transient_budget: dict[str, int],
    max_transient_retries: int = 3,
) -> RecoveryAttempt:
    """Classify one call and route it to its class's recovery strategy.

    Args:
        registry: Tools the call and any substitute may run against.
        call: The tool call to resolve.
        substitutes: Maps a tool name to a sibling tool to try on a
            permanent failure.
        transient_budget: Mutable per-tool retry counters, shared across an
            entire run so a tool that keeps failing cannot be retried
            forever just because each call starts its own local counter.
        max_transient_retries: Retries granted per tool across the run.

    Returns:
        A `RecoveryAttempt` describing what was tried and whether it worked.
    """
    tool = registry.get(call.name)
    classification, result = classify_failure(tool, call.arguments)

    if classification.label == "ok":
        return RecoveryAttempt(call, "ok", "none", str(result), True)

    if classification.label == "malformed_args":
        return RecoveryAttempt(
            call,
            "malformed_args",
            "repair_requested",
            f"ERROR: invalid arguments: {classification.detail}",
            False,
        )

    if classification.label == "transient":
        attempts = 0
        while attempts < max_transient_retries and transient_budget.get(call.name, 0) < max_transient_retries:
            transient_budget[call.name] = transient_budget.get(call.name, 0) + 1
            attempts += 1
            classification, result = classify_failure(tool, call.arguments)
            if classification.label == "ok":
                return RecoveryAttempt(call, "transient", f"retried x{attempts}, succeeded", str(result), True)
            if classification.label != "transient":
                return RecoveryAttempt(
                    call,
                    classification.label,
                    "escalated_during_retry",
                    f"ERROR: {classification.label} failure surfaced during transient retry: {classification.detail}",
                    False,
                )
        return RecoveryAttempt(
            call,
            "transient",
            f"budget exhausted after {attempts} retries",
            f"ERROR: transient failure, retry budget exhausted for {call.name!r} after {attempts} retries",
            False,
        )

    if classification.label == "permanent":
        substitute_name = substitutes.get(call.name)
        if substitute_name is not None:
            substitute_tool = registry.get(substitute_name)
            sub_classification, sub_result = classify_failure(substitute_tool, call.arguments)
            if sub_classification.label == "ok":
                return RecoveryAttempt(
                    call, "permanent", f"substituted {substitute_name!r}", str(sub_result), True
                )
        return RecoveryAttempt(
            call,
            "permanent",
            "abstained, no working substitute",
            f"ERROR: permanent failure calling {call.name!r}: {classification.detail}",
            False,
        )

    # classification.label == "implicit"
    return RecoveryAttempt(
        call,
        "implicit",
        "verify_injected",
        f"VERIFY: {call.name!r} returned an empty result; confirm before treating this as an answer.",
        False,
    )


@dataclass
class RecoveryRound:
    """One round trip to the model, with every call's recovery attempt."""

    index: int
    attempts: list[RecoveryAttempt] = field(default_factory=list)


@dataclass
class RecoveryLoopResult:
    """The outcome of a full class-routed recovery loop run."""

    final_answer: str
    rounds: list[RecoveryRound]
    history: list[Message]
    stop_reason: str


def run_recovery_loop(
    provider: Provider,
    registry: ToolRegistry,
    messages: list[Message],
    *,
    system: str | None = None,
    substitutes: dict[str, str] | None = None,
    max_transient_retries: int = 3,
    max_iterations: int = 6,
) -> RecoveryLoopResult:
    """Run the call, classify, route, observe loop with class-routed failure recovery.

    Unlike `run_tool_loop`, a transient retry happens inside `recover_call`
    against the real tool, with no extra model round trip: the model asked
    for the call once, and app-level retry logic decides whether to try
    again, exactly as a production retry wrapper would.

    Args:
        provider: The model to drive.
        registry: Tools available for execution.
        messages: Conversation so far, not including the system prompt.
        system: System prompt passed straight through to the provider.
        substitutes: Maps a tool name to a sibling tool to try on a
            permanent failure. See `recover_call`.
        max_transient_retries: Retries granted per tool across the run.
        max_iterations: Maximum number of model round trips before giving up.

    Returns:
        A `RecoveryLoopResult` with the final answer (if any), the
        round-by-round recovery record, and why the loop stopped.
    """
    history = list(messages)
    rounds: list[RecoveryRound] = []
    transient_budget: dict[str, int] = {}
    substitutes = substitutes or {}

    for round_index in range(1, max_iterations + 1):
        completion = provider.complete(history, tools=registry.specs(), system=system)
        history.append(Message.assistant(completion.content, completion.tool_calls))

        if not completion.tool_calls:
            return RecoveryLoopResult(completion.content, rounds, history, "stop")

        round_attempts: list[RecoveryAttempt] = []
        for call in completion.tool_calls:
            attempt = recover_call(
                registry,
                call,
                substitutes=substitutes,
                transient_budget=transient_budget,
                max_transient_retries=max_transient_retries,
            )
            round_attempts.append(attempt)
            history.append(Message.tool(call.id, attempt.observation))
        rounds.append(RecoveryRound(round_index, round_attempts))

    return RecoveryLoopResult("", rounds, history, "max_iterations")


def demo_transient_retry() -> RecoveryLoopResult:
    """A transient failure recovers within budget: two failures then a success, all inside one model round trip."""
    registry = build_recovery_registry({"get_weather": ["transient", "transient"]})
    provider = get_provider(
        script=[
            scripted_tool_call("get_weather", {"city": "Tokyo"}),
            "Tokyo is 18C with light rain, after a couple of retried lookups.",
        ]
    )
    messages = [Message.user("What's the weather in Tokyo?")]

    result = run_recovery_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_transient_retries=3)
    attempt = result.rounds[0].attempts[0]

    print("=== 13a. Error recovery: transient failure retried within budget ===")
    print(f"user:  {messages[0].content}")
    print(f"  classification={attempt.classification} strategy={attempt.strategy} succeeded={attempt.succeeded}")
    print(f"  observation: {attempt.observation}")
    print(f"final: {result.final_answer}")
    print(f"model round trips used: {len(provider.calls)} (retries happened inside recover_call, not as new round trips)")
    print()
    return result


def demo_budget_stop() -> RecoveryLoopResult:
    """A tool that never stops failing transiently is stopped at the retry budget, not retried forever."""
    registry = build_recovery_registry({"get_weather": ["transient", "transient", "transient"]})
    provider = get_provider(
        script=[
            scripted_tool_call("get_weather", {"city": "Tokyo"}),
            "The weather lookup could not complete after repeated transient failures.",
        ]
    )
    messages = [Message.user("What's the weather in Tokyo?")]

    result = run_recovery_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_transient_retries=2)
    attempt = result.rounds[0].attempts[0]

    print("=== 13b. Error recovery: transient retry budget stops a futile loop ===")
    print(f"user:  {messages[0].content}")
    print(f"  classification={attempt.classification} strategy={attempt.strategy} succeeded={attempt.succeeded}")
    print(f"  observation: {attempt.observation}")
    print(f"final: {result.final_answer}")
    print()
    return result


def demo_all_classes() -> RecoveryLoopResult:
    """One run exercises all four failure classes, each reaching its own distinct strategy."""
    registry = build_recovery_registry(
        {
            "get_weather": ["transient"],
            "lookup_order": ["permanent"],
            "get_customer_email": ["empty"],
        }
    )
    provider = get_provider(
        script=[
            scripted_tool_call("convert_currency", {"amount": "100", "from_currency": "USD", "to_currency": "EUR"}),
            scripted_tool_call("get_weather", {"city": "Tokyo"}),
            scripted_tool_call("lookup_order", {"order_id": "ORD-1001"}),
            scripted_tool_call("get_customer_email", {"customer_id": "CUST-42"}),
            "Four lookups hit four different failure modes; here is what recovered and what did not.",
        ]
    )
    messages = [Message.user("Convert 100 USD to EUR, check Tokyo weather, order ORD-1001, and CUST-42's email.")]

    result = run_recovery_loop(
        provider,
        registry,
        messages,
        system=SYSTEM_PROMPT,
        substitutes={"lookup_order": "lookup_order_backup"},
        max_iterations=5,
    )

    print("=== 13c. Error recovery: all four classes routed in one run ===")
    print(f"user:  {messages[0].content}")
    for round_record in result.rounds:
        attempt = round_record.attempts[0]
        print(f"  {attempt.call.name}: classification={attempt.classification} strategy={attempt.strategy}")
    print(f"final: {result.final_answer}")
    print()
    return result


if __name__ == "__main__":
    demo_transient_retry()
    demo_budget_stop()
    demo_all_classes()
