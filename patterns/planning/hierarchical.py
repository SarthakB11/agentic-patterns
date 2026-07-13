"""Hierarchical decomposition: expand a compound step into a sub-plan on demand.

Every other variant in this folder produces a flat plan, one JSON array of
primitive tool calls. `todo_list.py`'s README line calling that "the
hierarchical idea of expanding work on demand" overstated it: rewriting a
flat checklist mid-run is not a nested decomposition, and `todo_list.py`
never expands one item into its own sub-plan with a depth and a node
budget. This module is the real thing. A top-level plan may name a step
"compound" (its `tool` is the reserved `expand` marker) instead of a real
tool; a compound step triggers its own sub-planner call, keyed to that
step's own sub-goal, and recurses into whatever sub-plan comes back until
every leaf names a real tool. Depth and total node count are capped: past
either cap, a compound step is left unmet rather than expanded, exactly
like ChatHTN falling back instead of looping when no method applies.

Like every wave-based executor in this folder, each sub-plan is validated
with `validate_plan` before its steps are grouped into `topological_waves`,
so a sub-plan step never silently drops out of a wave. The `expand` marker
itself is registered as a dummy tool purely so `validate_plan`'s
known-tool check accepts it; its `fn` is never called, because this module
intercepts every compound step before it would reach the registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, Tool, ToolCall, ToolRegistry, get_provider

from patterns.planning.parser import parse_plan
from patterns.planning.plan import Step, StepResult, is_error_observation, substitute_args, topological_waves
from patterns.planning.validator import validate_plan

EXPAND_TOOL = "expand"

PLANNER_SYSTEM = (
    "You are a trip-planning agent. Respond with ONLY a JSON array of steps: "
    "id, tool, args, depends_on. A step whose tool is \"expand\" is compound: "
    "its args.goal names a sub-goal to decompose later, not a real tool call."
)

SUBPLANNER_SYSTEM = (
    "You are a sub-planner. Given one compound step's own sub-goal, respond "
    "with ONLY a JSON array of its sub-steps: id, tool, args, depends_on. A "
    "sub-step's tool may itself be \"expand\" for a further compound step."
)


def _expand_marker_stub() -> str:
    raise NotImplementedError("'expand' is a compound-step marker; hierarchical.py intercepts it before execution")


def _registry_with_expand_marker(registry: ToolRegistry) -> ToolRegistry:
    """Return a copy of `registry` plus the reserved `expand` marker, so `validate_plan` accepts compound steps."""
    augmented = ToolRegistry()
    for spec in registry.specs():
        augmented.register(registry.get(spec["name"]))
    augmented.register(
        Tool(
            name=EXPAND_TOOL,
            description="Reserved marker for a compound step hierarchical.py expands; never executed directly.",
            parameters={"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]},
            fn=_expand_marker_stub,
        )
    )
    return augmented


@dataclass
class ExpansionNode:
    """One node in the expansion tree: either a primitive leaf or an expanded compound step.

    Attributes:
        step: The step this node wraps.
        depth: Nesting depth, 0 for a top-level step.
        children: Sub-steps this node expanded into, empty for a primitive
            leaf or an unmet compound step.
        result: The step's `StepResult`, a real tool result for a primitive
            leaf or a synthesis of its children's outputs for a compound step.
        unmet: True if this compound step hit the depth or node cap and was
            never expanded.
    """

    step: Step
    depth: int
    children: list[ExpansionNode] = field(default_factory=list)
    result: StepResult | None = None
    unmet: bool = False

    @property
    def primitive(self) -> bool:
        """True if this node names a real tool rather than a compound sub-goal."""
        return self.step.tool != EXPAND_TOOL


@dataclass
class HierarchicalRun:
    """The outcome of a recursive, expand-on-demand planning run.

    Attributes:
        nodes: The top-level expansion tree, one node per step the initial
            planner call produced.
        leaf_results: Every executed step's result (primitive leaves and
            synthesized compound steps alike), keyed by id.
        unmet_compound_ids: Compound step ids that hit the depth or node cap.
        node_count: Total nodes created, including the top-level ones.
        expansion_calls: Number of sub-planner calls made, one per expanded
            compound step; primitive steps never trigger one.
    """

    nodes: list[ExpansionNode]
    leaf_results: dict[str, StepResult]
    unmet_compound_ids: list[str]
    node_count: int
    expansion_calls: int


def _expand(
    node: ExpansionNode,
    provider: Provider,
    registry: ToolRegistry,
    results: dict[str, StepResult],
    unmet: list[str],
    budget: dict[str, int],
    max_depth: int,
    calls: list[int],
) -> None:
    """Resolve `node` in place: run its tool if primitive, else recurse into a sub-plan."""
    step = node.step
    if node.primitive:
        args = substitute_args(step.args, results)
        output = registry.execute(ToolCall(id=step.id, name=step.tool, arguments=args))
        result = StepResult(step_id=step.id, output=output, ok=not is_error_observation(output))
        results[step.id] = result
        node.result = result
        return

    if node.depth >= max_depth or budget["used"] >= budget["nodes"]:
        node.unmet = True
        unmet.append(step.id)
        node.result = StepResult(step_id=step.id, output="(unexpanded: depth or node budget reached)", ok=False)
        results[step.id] = node.result
        return

    sub_goal = step.args.get("goal", step.id)
    sub_completion = provider.complete([Message.user(f"Sub-goal: {sub_goal}")], system=SUBPLANNER_SYSTEM)
    calls[0] += 1
    sub_plan = parse_plan(str(sub_goal), sub_completion.content)
    validate_plan(sub_plan, registry)  # precondition before topological_waves, see module docstring

    child_nodes: list[ExpansionNode] = []
    for wave in topological_waves(sub_plan.steps):
        for sub_step in wave:
            if budget["used"] >= budget["nodes"]:
                node.unmet = True
                unmet.append(step.id)
                break
            budget["used"] += 1
            child = ExpansionNode(step=sub_step, depth=node.depth + 1)
            child_nodes.append(child)
            _expand(child, provider, registry, results, unmet, budget, max_depth, calls)
        if node.unmet:
            break

    node.children = child_nodes
    leaf_outputs = [c.result.output for c in child_nodes if c.result is not None]
    node.result = StepResult(step_id=step.id, output="; ".join(leaf_outputs), ok=not node.unmet)
    results[step.id] = node.result


def run_hierarchical(
    provider: Provider, goal: str, registry: ToolRegistry, max_depth: int = 3, max_nodes: int = 20
) -> HierarchicalRun:
    """Plan the top level, then recursively expand every compound step, capped.

    Args:
        provider: Supplies the top-level planner call and one call per
            expanded compound step.
        goal: The user's goal, sent to the top-level planner.
        registry: Tools available to primitive steps; also `validate_plan`'s
            allowlist, augmented with the `expand` marker.
        max_depth: Maximum nesting depth a compound step may expand into.
        max_nodes: Maximum total nodes (top-level and expanded) across the
            whole run.
    """
    augmented = _registry_with_expand_marker(registry)
    plan_completion = provider.complete([Message.user(goal)], system=PLANNER_SYSTEM)
    plan = parse_plan(goal, plan_completion.content)
    validate_plan(plan, augmented)

    results: dict[str, StepResult] = {}
    unmet: list[str] = []
    budget = {"nodes": max_nodes, "used": len(plan.steps)}
    calls = [0]

    top_nodes = [ExpansionNode(step=s, depth=0) for s in plan.steps]
    for node in top_nodes:
        _expand(node, provider, augmented, results, unmet, budget, max_depth, calls)

    return HierarchicalRun(
        nodes=top_nodes,
        leaf_results=results,
        unmet_compound_ids=unmet,
        node_count=budget["used"],
        expansion_calls=calls[0],
    )


def demo() -> None:
    """Expand one compound "plan the Lyon day" step into weather, attractions, and an itinerary."""
    from patterns.planning.tools import build_travel_registry

    goal = "Plan a day in Lyon."
    top_plan_json = '[{"id": "day", "tool": "expand", "args": {"goal": "plan the Lyon day"}, "depends_on": []}]'
    sub_plan_json = (
        '[{"id": "w", "tool": "get_weather", "args": {"city": "Lyon"}, "depends_on": []},'
        ' {"id": "a", "tool": "search_attractions", "args": {"city": "Lyon"}, "depends_on": []},'
        ' {"id": "i", "tool": "draft_itinerary",'
        '  "args": {"weather": "$w", "attractions": "$a"}, "depends_on": ["w", "a"]}]'
    )
    provider = get_provider(script=[top_plan_json, sub_plan_json])
    registry = build_travel_registry()

    print("=== Hierarchical decomposition (expand-on-demand) ===")
    print(f"Goal: {goal}")
    run = run_hierarchical(provider, goal, registry)
    top = run.nodes[0]
    print(f"Nodes: {run.node_count}, sub-planner calls: {run.expansion_calls}")
    print(f"Compound step {top.step.id!r} expanded into {len(top.children)} primitives, no tool call of its own:")
    for child in top.children:
        print(f"  {child.step.id}: {child.step.tool} -> {child.result.output}")
    print("Note: 'day' never called a tool; todo_list.py's flat checklist never nests like this.")


if __name__ == "__main__":
    demo()
