"""Code-as-action: a real, Turing-complete action space instead of a fixed step list.

`code_execution.py` runs a fixed linear list of `{call, args, save_as}`
steps with `$step.field` string substitution and no control flow: it needs
one step per item and cannot express "look up these five orders and return
only the delayed ones" without five steps. CodeAct (Wang et al., ICML 2024,
arXiv:2402.01030) showed that letting the model emit real, executable code
that calls tools as plain functions beats both text and JSON action formats
by up to 20 percent success, because the action space is Turing-complete:
one action can loop, branch, and filter over many tool results and return
only the aggregate. Anthropic's "Code execution with MCP" (Nov 2025) turned
this into a first-party API primitive and reports collapsing a 150K-token
workflow to about 2K tokens (a vendor engineering claim, not a measured
benchmark) by keeping intermediate tool results inside the sandbox instead
of round-tripping every one of them through the model's context.

This module offers one `run_python` action whose argument is a code string,
executes it against a restricted namespace exposing each registered tool as
a plain callable, and feeds only the code's final `result` back to the
model. Every intermediate tool call the code makes stays inside the exec
call and never becomes a message in the conversation history, which is the
property behind Anthropic's token-collapse number and the exact thing the
step interpreter in `code_execution.py` cannot show.

The sandbox here is a teaching-scale stand-in, not a production one: a
restricted `__builtins__` dict and a plain `exec`, not a subprocess, a
container, or a real language runtime's sandboxing primitives. It is enough
to demonstrate the namespace-boundary guardrail (a name outside the
allowlist raises inside `exec`) without claiming to be secure against a
determined adversary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_patterns import Message, MockProvider, ToolCall, ToolRegistry, scripted_tool_call
from patterns.tool_use.catalog import SYSTEM_PROMPT
from patterns.tool_use.schema import auto_tool

CODE_ACTION_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + " When a request needs more than one tool call, or needs to filter, "
    "loop over, or branch on multiple results, call run_python once with a "
    "short Python program instead of one tool call per step. Call each "
    "registered tool as a plain function with keyword arguments and assign "
    "the answer to a variable named result."
)

_RUN_PYTHON_SPEC = {
    "name": "run_python",
    "description": (
        "Run a short Python program locally against the registered tools, "
        "exposed as plain callables. The program may use loops, "
        "conditionals, and comprehensions to process many tool results in "
        "one pass. Assign the final answer to a variable named `result`; "
        "only `result` is returned to the conversation, every intermediate "
        "tool call stays inside this one action."
    ),
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source to execute."}},
        "required": ["code"],
    },
}

_SAFE_BUILTINS: dict[str, Any] = {
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "sorted": sorted,
    "sum": sum,
    "min": min,
    "max": max,
    "any": any,
    "all": all,
    "list": list,
    "dict": dict,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "set": set,
    "zip": zip,
}


def build_sandbox_namespace(registry: ToolRegistry, call_counter: list[int]) -> dict[str, Any]:
    """Build a restricted exec namespace exposing each registered tool as a plain callable.

    Args:
        registry: Tools the sandboxed code may call by name.
        call_counter: A one-element list used as a mutable counter; each
            wrapped tool invocation increments `call_counter[0]`, so a
            caller can report how many tool calls ran inside one `exec`
            independent of how many round trips the model made.

    Returns:
        A namespace dict suitable for `exec(code, namespace)`: a restricted
        `__builtins__`, one callable per registered tool, and `result`
        pre-seeded to None.
    """
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "result": None}

    def make_wrapper(tool_name: str):
        def wrapper(**kwargs: Any) -> str:
            call_counter[0] += 1
            return registry.execute(ToolCall(id=f"sandbox_{call_counter[0]}", name=tool_name, arguments=kwargs))

        return wrapper

    for spec in registry.specs():
        namespace[spec["name"]] = make_wrapper(spec["name"])
    return namespace


def run_code_action_safe(registry: ToolRegistry, code: str) -> tuple[str, int]:
    """Execute a code action, catching any exception into an ERROR observation.

    Mirrors `ToolRegistry.execute`'s contract: an exception the code raises,
    including a `NameError` from touching a name outside the sandbox's
    namespace (the guardrail this module's point rests on), becomes an
    "ERROR: ..." string instead of propagating, so a faulty code action does
    not crash the loop any more than a faulty single tool call does
    elsewhere in this pattern.

    Args:
        registry: Tools the code may call by name.
        code: Python source assigning to `result`, calling registered tools
            as plain functions with keyword arguments.

    Returns:
        (observation, tool_calls_made): `str(result)` on success or an
        "ERROR: ..." string on failure, and how many tool wrapper
        invocations happened during the exec call, whether it succeeded or
        raised partway through.
    """
    call_counter = [0]
    namespace = build_sandbox_namespace(registry, call_counter)
    try:
        exec(code, namespace)  # noqa: S102 - the sandboxed namespace is the module's point
        return str(namespace.get("result")), call_counter[0]
    except Exception as exc:  # noqa: BLE001 - sandbox boundary: catch anything the code raises
        return f"ERROR: {exc}", call_counter[0]


def build_orders_registry() -> ToolRegistry:
    """Build a small order-status registry with enough delayed orders to demonstrate filtering results.

    Two of the five orders are "delayed", enough to show a code action
    filtering many tool results down to a few in one pass.
    """
    registry = ToolRegistry()
    statuses = {
        "ORD-2001": "shipped",
        "ORD-2002": "delayed",
        "ORD-2003": "processing",
        "ORD-2004": "delayed",
        "ORD-2005": "shipped",
    }

    @auto_tool(registry)
    def lookup_order_status(order_id: str) -> str:
        """Look up an order's shipping status.

        Args:
            order_id: Order identifier, e.g. "ORD-2001".
        """
        if order_id not in statuses:
            raise KeyError(f"no such order '{order_id}'")
        return f"{order_id}: {statuses[order_id]}"

    @auto_tool(registry)
    def escalate_order(order_id: str) -> str:
        """Flag a delayed order for manual escalation.

        Args:
            order_id: Order identifier, e.g. "ORD-2001".
        """
        return f"{order_id}: escalated to fulfillment ops"

    @auto_tool(registry)
    def mark_fulfilled(order_id: str) -> str:
        """Mark an order as fulfilled with no further action needed.

        Args:
            order_id: Order identifier, e.g. "ORD-2001".
        """
        return f"{order_id}: marked fulfilled, no action needed"

    return registry


@dataclass
class CodeActionResult:
    """The outcome of running one code action to completion, model-round-trip-accounted.

    Attributes:
        observation: The code action's result, as fed back to the model.
        tool_calls_made: How many tool wrapper invocations happened inside
            the sandboxed exec call.
        final_answer: The model's closing text after seeing the observation.
        history: The full message history, including the one tool observation.
        round_trips: Total `provider.complete` calls made across both the
            code-generation turn and the final-answer turn.
    """

    observation: str
    tool_calls_made: int
    final_answer: str
    history: list[Message]
    round_trips: int


def _run_one_action(registry: ToolRegistry, provider: MockProvider, messages: list[Message]) -> CodeActionResult:
    """Send one user turn, run the model's scripted run_python call, and return the full outcome.

    Takes `MockProvider` rather than the general `Provider` interface
    because it reports `provider.calls`, a scripted-transcript attribute
    that only the mock keeps; a real provider has no such record.
    """
    completion = provider.complete(messages, tools=[_RUN_PYTHON_SPEC], system=CODE_ACTION_SYSTEM_PROMPT)
    call = completion.tool_calls[0]
    observation, tool_calls_made = run_code_action_safe(registry, call.arguments["code"])

    history = [
        *messages,
        Message.assistant(completion.content, completion.tool_calls),
        Message.tool(call.id, observation),
    ]
    final = provider.complete(history, tools=[_RUN_PYTHON_SPEC], system=CODE_ACTION_SYSTEM_PROMPT)
    history.append(Message.assistant(final.content))
    return CodeActionResult(observation, tool_calls_made, final.content, history, len(provider.calls))


def demo_loop_filter() -> CodeActionResult:
    """Filter five orders down to the two delayed ones in one code action instead of five tool round trips."""
    registry = build_orders_registry()
    code = (
        "ids = ['ORD-2001', 'ORD-2002', 'ORD-2003', 'ORD-2004', 'ORD-2005']\n"
        "statuses = [lookup_order_status(order_id=i) for i in ids]\n"
        "result = [s for s in statuses if 'delayed' in s]\n"
    )
    provider = MockProvider(
        [
            scripted_tool_call("run_python", {"code": code}),
            "Two orders are delayed: ORD-2002 and ORD-2004.",
        ]
    )
    messages = [Message.user("Which of ORD-2001 through ORD-2005 are delayed?")]

    result = _run_one_action(registry, provider, messages)
    tool_messages = [m for m in result.history if m.role == "tool"]

    print("=== 14a. Code-as-action: loop and filter five results in one pass ===")
    print(f"user:  {messages[0].content}")
    print(f"  program ran ({result.tool_calls_made} tool call(s) inside the sandbox, one model round trip)")
    print(f"  result: {result.observation}")
    print(f"final: {result.final_answer}")
    print(f"message history holds {len(tool_messages)} tool observation(s) though 5 tool calls executed")
    print(f"model round trips used: {result.round_trips}")
    print()
    return result


def demo_branch() -> CodeActionResult:
    """A code action branches on a computed condition, calling a different tool per outcome."""
    registry = build_orders_registry()
    code = (
        "status = lookup_order_status(order_id='ORD-2002')\n"
        "if 'delayed' in status:\n"
        "    result = escalate_order(order_id='ORD-2002')\n"
        "else:\n"
        "    result = mark_fulfilled(order_id='ORD-2002')\n"
    )
    provider = MockProvider(
        [
            scripted_tool_call("run_python", {"code": code}),
            "ORD-2002 was delayed, so it has been escalated to fulfillment ops.",
        ]
    )
    messages = [Message.user("Handle ORD-2002: escalate if delayed, otherwise mark it fulfilled.")]

    result = _run_one_action(registry, provider, messages)

    print("=== 14b. Code-as-action: branch on a computed condition ===")
    print(f"user:  {messages[0].content}")
    print(f"  program ran: {result.observation}")
    print(f"final: {result.final_answer}")
    print()
    return result


def demo_sandbox_boundary() -> tuple[str, int]:
    """Code touching a name outside the sandbox raises inside exec and becomes an error observation, not a crash."""
    registry = build_orders_registry()
    code = "result = open('/etc/passwd').read()\n"

    observation, tool_calls_made = run_code_action_safe(registry, code)

    print("=== 14c. Code-as-action: sandbox boundary blocks an unavailable name ===")
    print(f"code: {code.strip()!r}")
    print(f"observation: {observation}")
    print(f"tool calls made before the boundary was hit: {tool_calls_made}")
    print()
    return observation, tool_calls_made


if __name__ == "__main__":
    demo_loop_filter()
    demo_branch()
    demo_sandbox_boundary()
