"""Forced and constrained tool choice.

Real APIs expose a `tool_choice` control: `auto` lets the model decide,
`required` forces some tool call, `none` forbids tool calls, and a named
choice forces one specific tool. The shared `Provider.complete()` contract
in this repo does not expose `tool_choice` directly, since it targets the
lowest common denominator across providers. This module shows the two
techniques that reproduce the same guarantee at the app layer, which is also
what a provider without native `tool_choice` support requires:

- `none` is enforced by simply not offering any tool specs that round
  (`offered_specs=[]`), so there is nothing for the model to call.
- `required` and a named choice are enforced by checking the result after
  the fact with `assert_tool_choice_satisfied`, and would raise
  `ToolChoiceViolation` if the model did not comply. A named choice is also
  narrowed on the way in, by offering only that one tool's spec, both to
  bias the model and to shrink the input.
"""

from __future__ import annotations

from agentic_patterns import Message, ToolCall, get_provider

from patterns.tool_use.catalog import SYSTEM_PROMPT, build_registry
from patterns.tool_use.loop import run_tool_loop


class ToolChoiceViolation(RuntimeError):
    """Raised when a completion does not honor a required tool_choice policy."""


def assert_tool_choice_satisfied(tool_calls: list[ToolCall], choice: str) -> None:
    """Check a completion's tool calls against a tool_choice policy.

    Args:
        tool_calls: Tool calls the model returned this turn (possibly none).
        choice: "auto" (no constraint), "required" (at least one call of any
            kind), or a specific tool name (at least one call to that tool).

    Raises:
        ToolChoiceViolation: If `choice` is not "auto" and the calls do not
            satisfy it.
    """
    if choice == "auto":
        return
    if choice == "required":
        if not tool_calls:
            raise ToolChoiceViolation("tool_choice=required but the model returned no tool call")
        return
    if not any(call.name == choice for call in tool_calls):
        raise ToolChoiceViolation(f"tool_choice={choice!r} but the model did not call it")


def demo_none() -> None:
    """tool_choice="none": no tools are offered, so the model must answer in text.

    Useful when a turn should never touch a side effect, e.g. answering a
    question about policy rather than data.
    """
    registry = build_registry()
    provider = get_provider(script=["2 + 2 is 4. No lookup needed for that one."])
    messages = [Message.user("Quick one, no need to look anything up: what's 2 + 2?")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, offered_specs=[], max_iterations=1)
    assert_tool_choice_satisfied([], "auto")  # nothing to check; no tools were even offered

    print('=== 4a. Forced tool choice: "none" (no tools offered) ===')
    print(f"user:  {messages[0].content}")
    print(f"final: {result.final_answer}")
    print(f"tools offered this turn: {provider.calls[0]['tools']}")
    print()


def demo_required() -> None:
    """tool_choice="required": the model must call something, verified after the fact."""
    registry = build_registry()
    provider = get_provider(
        script=[
            {"tool": "get_weather", "args": {"city": "Paris"}},
            "Conditions in Paris are 21C and clear, good weather for the trip.",
        ]
    )
    messages = [Message.user("Give me something useful about Paris right now.")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=2)
    first_round_calls = [record.call for record in result.rounds[0].calls]
    assert_tool_choice_satisfied(first_round_calls, "required")

    print('=== 4b. Forced tool choice: "required" (must call something) ===')
    print(f"user:  {messages[0].content}")
    print(f"policy satisfied: model called {first_round_calls[0].name}")
    print(f"final: {result.final_answer}")
    print()


def demo_named() -> None:
    """A named forced choice: only get_weather's spec is offered, narrowing both intent and input size."""
    registry = build_registry()
    provider = get_provider(
        script=[
            {"tool": "get_weather", "args": {"city": "San Francisco"}},
            "San Francisco is 16C with fog right now.",
        ]
    )
    messages = [Message.user("weather check")]
    named_spec = [spec for spec in registry.specs() if spec["name"] == "get_weather"]

    result = run_tool_loop(
        provider, registry, messages, system=SYSTEM_PROMPT, offered_specs=named_spec, max_iterations=2
    )
    first_round_calls = [record.call for record in result.rounds[0].calls]
    assert_tool_choice_satisfied(first_round_calls, "get_weather")

    print('=== 4c. Forced tool choice: named ("get_weather" only) ===')
    print(f"user:  {messages[0].content}")
    print(f"tools offered this turn: {[spec['name'] for spec in provider.calls[0]['tools']]}")
    print(f"final: {result.final_answer}")
    print()


if __name__ == "__main__":
    demo_none()
    demo_required()
    demo_named()
