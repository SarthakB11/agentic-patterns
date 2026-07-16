"""Planning (plan-then-execute) pattern: turn a goal into an explicit plan,
then carry it out, instead of deciding one action at a time.

This module runs a guided tour of every variant implemented in this
package, in an order that builds from the simplest to the most structured:

1. Plan-and-Solve: one model call plans and solves in the same generation.
2. Classic plan-then-execute: a separate planner call, then a sequential executor.
3. DAG executor: a dependency graph, executed wave by wave with concurrent dispatch.
4. Replanning: a step fails, the replanner revises the remaining steps, capped.
5. Plan repair: localized blast-radius surgery instead of replanning from scratch.
6. LLM-Modulo: verify a plan against sound checkers, back-prompt, regenerate.
7. ReWOO: planner writes a full blueprint, workers gather evidence, solver answers.
8. ReAct baseline: the interleaved contrast, one model call per step.
9. Todo-list in-context planning: a self-rewritten plan held in agent state.
10. Hierarchical decomposition: expand a compound step into a sub-plan on demand.
11. Plan selection: generate several candidate plans, score them, execute the best.
12. Premortem: simulate a plan against tracked state before any real tool runs.
13. Context offload: checkpoint the plan and outputs to disk, resume after a restart.
14. Subagent-per-subtask: delegate each step to an isolated child conversation.

Every variant runs offline against `MockProvider` with a scripted, coherent
conversation; no network call and no API key are needed. Run it with:

    python -m patterns.planning.main

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key in the environment) to run the same code against
a real model instead; no code in this package special-cases the mock.
"""

from __future__ import annotations

from patterns.planning import (
    context_offload,
    dag_executor,
    hierarchical,
    modulo_loop,
    plan_and_solve,
    plan_repair,
    plan_selection,
    premortem,
    react_baseline,
    replanning,
    rewoo,
    sequential_executor,
    subagent_executor,
    todo_list,
)

_SECTIONS = (
    plan_and_solve,
    sequential_executor,
    dag_executor,
    replanning,
    plan_repair,
    modulo_loop,
    rewoo,
    react_baseline,
    todo_list,
    hierarchical,
    plan_selection,
    premortem,
    context_offload,
    subagent_executor,
)


def main() -> None:
    """Run every variant's demo in sequence, with a header between them."""
    for i, module in enumerate(_SECTIONS, start=1):
        print(f"\n{'#' * 70}")
        print(f"# {i}/{len(_SECTIONS)}: {module.__name__.rsplit('.', 1)[-1]}")
        print(f"{'#' * 70}")
        module.demo()
    print("\nAll planning variants ran successfully, offline, with no API key.")


if __name__ == "__main__":
    main()
