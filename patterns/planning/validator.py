"""Validates a parsed `Plan` before any step executes.

Three checks run, in order: every step's tool name is registered, every
dependency id refers to a real step in the same plan, and the dependency
graph is acyclic. Validation happens after parsing and before execution so a
malformed or unsafe plan is rejected up front, instead of failing mid-run or
running a tool call the plan never should have been allowed to make.
"""

from __future__ import annotations

from agentic_patterns import ToolRegistry

from patterns.planning.plan import Plan, topological_waves


class PlanValidationError(ValueError):
    """Raised when a plan fails validation."""


def validate_plan(plan: Plan, registry: ToolRegistry) -> None:
    """Validate `plan` against `registry`.

    Args:
        plan: The parsed plan to check.
        registry: The tool registry every step's `tool` must be a member of.
            This also doubles as the allowlist: a plan cannot invoke a tool
            that was never registered, however it was built from
            (potentially untrusted) model output.

    Raises:
        PlanValidationError: If a step names an unregistered tool, a step
            depends on an id that does not exist in the plan, or the
            dependency graph contains a cycle.
    """
    known_tools = {t["name"] for t in registry.specs()}
    for step in plan.steps:
        if step.tool not in known_tools:
            raise PlanValidationError(
                f"Step {step.id!r} references unknown tool {step.tool!r}. "
                f"Known tools: {', '.join(sorted(known_tools)) or '(none registered)'}"
            )

    step_ids = plan.step_ids()
    for step in plan.steps:
        dangling = [d for d in step.depends_on if d not in step_ids]
        if dangling:
            raise PlanValidationError(
                f"Step {step.id!r} depends on unknown step id(s): {', '.join(dangling)}"
            )

    waves = topological_waves(plan.steps)
    resolved_ids = {s.id for wave in waves for s in wave}
    if resolved_ids != step_ids:
        stuck = sorted(step_ids - resolved_ids)
        raise PlanValidationError(
            f"Plan has a dependency cycle involving step(s): {', '.join(stuck)}"
        )
