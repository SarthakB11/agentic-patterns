"""Sub-module: handoff-style routing (transfer of control).

Every other module in this folder dispatches and returns: a classifier
picks a route, a handler answers, and the caller gets the answer back. This
module implements the other shape frameworks converged on in 2025, called a
handoff in the OpenAI Agents SDK, agent-initiated transfer in AutoGen, and a
swarm pattern in CrewAI: the triage model itself requests a transfer by
calling a tool named like `transfer_to_billing`, and once that transfer
happens the target agent owns the rest of the conversation. There is no
route back through the triage step; the caller's next message goes straight
to the sub-agent that took over.

Routing metadata is still recorded here even though the "classifier" is
just a tool call the provider chose to make: the `RouteDecision.metadata`
records which transfer tool fired, so a handoff is just as observable as an
explicit classification, even when the decision came from inside the
model rather than from a rule or embedding comparison the app wrote.
"""

from __future__ import annotations

from agentic_patterns import Message, Provider, get_provider, scripted_tool_call

from patterns.routing.registry import RouteDecision

_TRIAGE_SYSTEM = (
    "You triage customer messages for a support system. If the message is "
    "about billing, call transfer_to_billing. If it is a technical problem, "
    "call transfer_to_technical. Otherwise, answer directly yourself."
)

_SUB_AGENT_SYSTEMS = {
    "billing": (
        "You are the billing agent. You now own this conversation; answer "
        "the customer's billing question directly and completely."
    ),
    "technical": (
        "You are the technical support agent. You now own this "
        "conversation; answer the customer's technical question directly "
        "and completely."
    ),
}


def _transfer_tool_specs() -> list[dict[str, object]]:
    """Tool specs exposing each sub-agent as a `transfer_to_<name>` tool."""
    return [
        {
            "name": f"transfer_to_{name}",
            "description": f"Transfer this conversation to the {name} agent, who will take it from here.",
            "parameters": {"type": "object", "properties": {}},
        }
        for name in _SUB_AGENT_SYSTEMS
    ]


def run_handoff(user_text: str, triage_provider: Provider, sub_agent_providers: dict[str, Provider]) -> RouteDecision:
    """Run one turn of triage; on a transfer tool call, hand off to the sub-agent.

    Args:
        user_text: The customer's message.
        triage_provider: Scripted to either answer directly or call one
            `transfer_to_<name>` tool.
        sub_agent_providers: One `Provider` per sub-agent name in
            `_SUB_AGENT_SYSTEMS`, consulted only if triage transfers to it.

    Returns:
        A `RouteDecision` whose `route` is "triage" (no transfer happened,
        triage answered directly) or the sub-agent's name (a transfer
        happened); `metadata["answer"]` carries whichever agent's final
        text, and `metadata["transferred"]` records whether a handoff
        occurred.
    """
    completion = triage_provider.complete([Message.user(user_text)], tools=_transfer_tool_specs(), system=_TRIAGE_SYSTEM)

    if not completion.tool_calls:
        return RouteDecision(
            route="triage",
            score=None,
            method="handoff",
            metadata={"answer": completion.content, "transferred": False},
        )

    call = completion.tool_calls[0]
    target = call.name.removeprefix("transfer_to_")
    if target not in sub_agent_providers:
        raise ValueError(f"Triage requested an unknown transfer target: {call.name!r}")

    sub_provider = sub_agent_providers[target]
    sub_completion = sub_provider.complete([Message.user(user_text)], system=_SUB_AGENT_SYSTEMS[target])
    return RouteDecision(
        route=target,
        score=None,
        method="handoff",
        metadata={"answer": sub_completion.content, "transferred": True, "transfer_tool": call.name},
    )


def run_handoff_demo() -> tuple[RouteDecision, RouteDecision]:
    """Run one billing handoff and one direct triage answer, no transfer.

    Returns:
        A (handed_off, answered_directly) pair.
    """
    handoff_triage = get_provider(script=[scripted_tool_call("transfer_to_billing", {})])
    billing_agent = get_provider(
        script=["Your March invoice of $482.10 was paid successfully on the 1st; no action is needed on your end."]
    )
    handed_off = run_handoff(
        "Can you check whether my last invoice actually went through?",
        handoff_triage,
        {"billing": billing_agent, "technical": get_provider(script=[])},
    )

    direct_triage = get_provider(script=["We're open Monday through Friday, 9am to 6pm Eastern."])
    answered_directly = run_handoff(
        "What are your support hours?",
        direct_triage,
        {"billing": get_provider(script=[]), "technical": get_provider(script=[])},
    )

    return handed_off, answered_directly
