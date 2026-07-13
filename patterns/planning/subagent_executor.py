"""Subagent-per-subtask execution: an orchestrator spawns one child agent
per plan step instead of calling tools directly itself.

Each child gets its own isolated conversation and its own provider, does
whatever reasoning and tool calls its one subtask needs, and returns a
single compact result string. The parent orchestrator's context only ever
holds that one string per step, never the child's intermediate turns, which
is what keeps a long multi-step run from filling the parent's context
window. This mirrors Claude Agent SDK and LangGraph Deep Agents spawning
subagents with isolated context that each report back a compressed result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agentic_patterns import Message, Provider, ToolRegistry, get_provider

from patterns.planning.parser import parse_plan
from patterns.planning.plan import Plan, Step, StepResult, substitute_args
from patterns.planning.validator import validate_plan

PLANNER_SYSTEM = (
    "You are a trip-planning orchestrator. Respond with ONLY a JSON array of "
    "steps: id, tool, args, depends_on. Each step will be delegated to its "
    "own subagent."
)

SUBAGENT_SYSTEM = "You are a focused subagent with one job. Call the tool you were told to, then summarize the result in one sentence."

SubagentFactory = Callable[[Step], Provider]


@dataclass
class SubagentReport:
    """What the parent orchestrator learns about one delegated step.

    Attributes:
        step_id: The step this subagent handled.
        child_message_count: Length of the child's own conversation, shown
            only to make the point that the parent never sees those turns.
        result: The single compact result the child reported back.
    """

    step_id: str
    child_message_count: int
    result: StepResult


def run_subagent_step(step: Step, provider: Provider, registry: ToolRegistry, args: dict) -> SubagentReport:
    """Run one step inside its own isolated child conversation.

    Args:
        step: The step being delegated.
        provider: A provider scoped to this one child; the parent's provider
            never sees these turns.
        registry: Tools the child may call.
        args: The step's arguments, already substituted with upstream outputs.
    """
    messages: list[Message] = [
        Message.user(f"Complete this subtask: call {step.tool} with args {args}, then summarize the result.")
    ]
    completion = provider.complete(messages, tools=registry.specs(), system=SUBAGENT_SYSTEM)
    while completion.tool_calls:
        messages.append(Message.assistant(completion.content, tool_calls=completion.tool_calls))
        for call in completion.tool_calls:
            observation = registry.execute(call)
            messages.append(Message.tool(call.id, observation))
        completion = provider.complete(messages, tools=registry.specs(), system=SUBAGENT_SYSTEM)
    messages.append(Message.assistant(completion.content))
    result = StepResult(step_id=step.id, output=completion.content)
    return SubagentReport(step_id=step.id, child_message_count=len(messages), result=result)


@dataclass
class OrchestratorRun:
    """The outcome of a subagent-delegated plan-then-execute run.

    Attributes:
        plan: The plan the parent orchestrator produced.
        reports: One `SubagentReport` per step, in plan order.
        parent_context: The compact strings the parent's own context holds;
            this is what stays small no matter how much a child does.
    """

    plan: Plan
    reports: list[SubagentReport]
    parent_context: list[str]


def run_with_subagents(
    parent_provider: Provider, goal: str, registry: ToolRegistry, subagent_provider_for: SubagentFactory
) -> OrchestratorRun:
    """Plan once, then delegate every step to its own freshly built subagent.

    Args:
        parent_provider: Supplies the single orchestrator planning call.
        goal: The user's goal, sent to the orchestrator's planner.
        registry: Tools available to every subagent.
        subagent_provider_for: Builds a fresh, independently scripted
            provider for each step, standing in for spinning up a new child
            agent per subtask.
    """
    plan_completion = parent_provider.complete([Message.user(goal)], system=PLANNER_SYSTEM)
    plan = parse_plan(goal, plan_completion.content)
    validate_plan(plan, registry)

    results: dict[str, StepResult] = {}
    reports: list[SubagentReport] = []
    parent_context: list[str] = []
    for step in plan.steps:
        args = substitute_args(step.args, results)
        child_provider = subagent_provider_for(step)
        report = run_subagent_step(step, child_provider, registry, args)
        results[step.id] = report.result
        reports.append(report)
        parent_context.append(f"{step.id}: {report.result.output}")
    return OrchestratorRun(plan=plan, reports=reports, parent_context=parent_context)


def demo() -> None:
    """Delegate a two-step Paris plan to two isolated subagents and print the parent's view."""
    from patterns.planning.tools import build_travel_registry

    goal = "For a Paris trip, delegate the weather check and the attraction search to subagents."
    plan_json = (
        '[{"id": "step1", "tool": "get_weather", "args": {"city": "Paris"}, "depends_on": []},'
        ' {"id": "step2", "tool": "search_attractions", "args": {"city": "Paris"}, "depends_on": []}]'
    )
    parent_provider = get_provider(script=[plan_json])
    registry = build_travel_registry()

    child_scripts: dict[str, list] = {
        "step1": [
            {"tool": "get_weather", "args": {"city": "Paris"}},
            "Paris will be mild and cloudy with light rain, so pack a light jacket.",
        ],
        "step2": [
            {"tool": "search_attractions", "args": {"city": "Paris"}},
            "Top picks in Paris are the Louvre, the Eiffel Tower, and Montmartre.",
        ],
    }

    def subagent_provider_for(step: Step) -> Provider:
        return get_provider(script=child_scripts[step.id])

    print("=== Subagent-per-subtask execution ===")
    print(f"Goal: {goal}\n")
    run = run_with_subagents(parent_provider, goal, registry, subagent_provider_for)
    for report in run.reports:
        print(
            f"  {report.step_id}: child used {report.child_message_count} messages, "
            f"reported back -> {report.result.output}"
        )
    print(f"\nParent's own context ({len(run.parent_context)} compact entries, never the child turns):")
    for entry in run.parent_context:
        print(f"  {entry}")


if __name__ == "__main__":
    demo()
