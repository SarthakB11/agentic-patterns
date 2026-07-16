"""Sequential multi-step tool use: the agentic loop.

The model calls a tool, sees the observation, and decides its next call
from that result, repeating until it has enough to answer. Unlike
single-shot use, later calls depend on earlier ones: the second call here
cannot be formed until the first call's observation is known, which is the
defining property of this variant and the core loop ReAct shares.
"""

from __future__ import annotations

from agentic_patterns import Message, get_provider, scripted_tool_call
from patterns.tool_use.catalog import SYSTEM_PROMPT, build_registry
from patterns.tool_use.loop import ToolLoopResult, run_tool_loop


def demo_sequential() -> ToolLoopResult:
    """Run a two-step dependent chain: look up an order, then its customer's email.

    The customer id needed for the second call only exists inside the first
    call's observation, so the model must read that observation before it
    can form the second call's arguments.
    """
    registry = build_registry()
    provider = get_provider(
        script=[
            scripted_tool_call("lookup_order", {"order_id": "ORD-1001"}),
            scripted_tool_call("get_customer_email", {"customer_id": "CUST-42"}),
            "Order ORD-1001 has shipped. The customer's email on file is "
            "priya@example.com, so a shipping notice can go out to that address.",
        ]
    )
    messages = [Message.user("Has order ORD-1001 shipped, and what's the customer's email?")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=3)

    print("=== 2. Sequential multi-step loop (dependent calls) ===")
    print(f"user:  {messages[0].content}")
    for round_record in result.rounds:
        for record in round_record.calls:
            print(f"  step {round_record.index}: {record.call.name}({record.call.arguments})")
            print(f"           observation: {record.observation}")
    print(f"final: {result.final_answer}")
    print(f"stop_reason={result.stop_reason}, rounds={len(result.rounds)}")
    print()
    return result


if __name__ == "__main__":
    demo_sequential()
