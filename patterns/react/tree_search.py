"""LATS-lite best-first search over ReAct trajectories.

Every other loop in this folder commits to one action per step and never
looks back. This module represents the run as a tree instead: each node
holds the trajectory that reaches it, an evaluator's score for how much
progress it represents, and whether it is terminal. A frontier of
non-terminal nodes is expanded best-first, so a low-scoring branch is simply
never re-expanded while a higher-scoring sibling is, which is what
backtracking looks like in a best-first search with no explicit undo step.

The value estimate that scores each node is itself a `Provider.complete()`
call, so under `MockProvider` it is scripted text like every other
completion in this repo, parsed to a number. This is the direct answer to
the earlier claim (now corrected in the README) that tree search needs a
value estimate a scripted mock cannot meaningfully produce: the "model" is
scripted, so the search it drives is fully deterministic.

This is LATS-lite, not full LATS: best-first expansion with backtracking
over a fixed node budget, no UCB selection, no rollout simulation, and no
value backpropagation. See `docs/research/react_deep.md` for why those were
left out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, ToolRegistry, get_provider

from patterns.react.native_loop import build_native_registry
from patterns.react.scratchpad import Step

TREE_SEARCH_SYSTEM_PROMPT = (
    "You are a research agent proposing the next single action toward a goal, "
    "given the trajectory so far. Call one tool, or call `finish` with your answer."
)

EVAL_SYSTEM_PROMPT = (
    "You are an evaluator. Given a goal and a candidate trajectory, respond with a "
    "single integer from 0 (no progress) to 10 (goal achieved) rating how close the "
    "trajectory is to answering the goal. Reply with the number only."
)

DEFAULT_FORCE_MESSAGE = "Stopped: no terminal trajectory found within the node budget."


@dataclass
class TreeNode:
    """One node in the search tree.

    Attributes:
        id: Deterministic id, assigned in creation order starting at 0 for the root.
        trajectory: The steps that reach this node from the root, in order.
        score: The evaluator's score for this node, 0 for the unscored root.
        terminal: True if this node's last step was a `finish` call.
        answer: The finish answer, set only when `terminal` is True.
    """

    id: int
    trajectory: list[Step]
    score: float = 0.0
    terminal: bool = False
    answer: str | None = None


@dataclass
class TreeSearchResult:
    """Outcome of a best-first tree search run.

    Attributes:
        answer: The winning terminal node's answer, a generated best-effort
            answer, or the canned force message, depending on how the run ended.
        best_node: The node the answer was drawn from.
        tree: Every node created during the search, in id (creation) order.
        expansion_order: Node ids in the order they were popped and expanded.
        nodes_expanded: Number of expansion rounds run.
        stopped_reason: One of "terminal_best", "budget_force", "budget_generate".
    """

    answer: str | None
    best_node: TreeNode
    tree: list[TreeNode]
    expansion_order: list[int]
    nodes_expanded: int
    stopped_reason: str


def _render_trajectory(goal: str, trajectory: list[Step]) -> str:
    """Render a goal plus a trajectory's steps as plain text for a provider call."""
    lines = [f"Goal: {goal}"]
    for step in trajectory:
        lines.append(f"Action: {step.action}[{step.action_input}] -> Observation: {step.observation}")
    return "\n".join(lines)


def _parse_score(text: str) -> float:
    """Parse an evaluator completion's content into a score, clamped to 0-10."""
    return max(0.0, min(10.0, float(text.strip())))


def _pop_best(frontier: list[TreeNode]) -> TreeNode:
    """Remove and return the highest-scoring node, ties broken by insertion order."""
    best_index = 0
    for i in range(1, len(frontier)):
        if frontier[i].score > frontier[best_index].score:
            best_index = i
    return frontier.pop(best_index)


def run_tree_search(
    provider: Provider,
    tools: ToolRegistry,
    goal: str,
    *,
    system_prompt: str = TREE_SEARCH_SYSTEM_PROMPT,
    eval_system_prompt: str = EVAL_SYSTEM_PROMPT,
    branching_factor: int = 2,
    node_budget: int = 6,
    max_frontier_width: int | None = None,
    on_budget_exhausted: str = "force",
    force_message: str = DEFAULT_FORCE_MESSAGE,
) -> TreeSearchResult:
    """Run best-first search over ReAct trajectories to completion.

    Args:
        provider: Model provider, called once per candidate action and once
            per candidate evaluation during each expansion.
        tools: Registry of tools a candidate action may invoke, including `finish`.
        goal: The question or task to solve.
        system_prompt: Instruction for the action-proposal calls.
        eval_system_prompt: Instruction for the evaluator calls.
        branching_factor: Number of candidate actions requested per expansion.
        node_budget: Maximum number of expansion rounds before stopping.
        max_frontier_width: If set, the frontier is pruned to its
            highest-scoring nodes after each expansion.
        on_budget_exhausted: "force" returns `force_message`; "generate" makes
            one further tool-free call asking for a best-effort answer from
            the highest-scored partial trajectory.
        force_message: The fixed message returned when `on_budget_exhausted="force"`.

    Returns:
        A TreeSearchResult describing the winning node and how the search ended.
    """
    root = TreeNode(id=0, trajectory=[])
    tree: list[TreeNode] = [root]
    frontier: list[TreeNode] = [root]
    terminal_nodes: list[TreeNode] = []
    expansion_order: list[int] = []
    next_id = 1
    expansions = 0

    while frontier and expansions < node_budget:
        node = _pop_best(frontier)
        expansions += 1
        expansion_order.append(node.id)

        # Step 4: ask for `branching_factor` candidate next actions, one call each.
        proposals = []
        for _ in range(branching_factor):
            prompt = _render_trajectory(goal, node.trajectory)
            completion = provider.complete([Message.user(prompt)], tools=tools.specs(), system=system_prompt)
            proposals.append(completion)

        # Step 5: execute each candidate's tool call and build a child node.
        children: list[TreeNode] = []
        for completion in proposals:
            call = completion.tool_calls[0]
            observation = tools.execute(call)
            step = Step(completion.content, call.name, str(call.arguments), observation=observation)
            terminal = call.name == "finish"
            child = TreeNode(next_id, [*node.trajectory, step], terminal=terminal, answer=observation if terminal else None)
            next_id += 1
            tree.append(child)
            children.append(child)

        # Step 6: score every child, terminal or not.
        for child in children:
            eval_prompt = _render_trajectory(goal, child.trajectory)
            score_completion = provider.complete([Message.user(eval_prompt)], system=eval_system_prompt)
            child.score = _parse_score(score_completion.content)

        # Steps 7-8: sort terminal children out; push the rest onto the frontier.
        for child in children:
            if child.terminal:
                terminal_nodes.append(child)
            else:
                frontier.append(child)
        if max_frontier_width is not None and len(frontier) > max_frontier_width:
            frontier.sort(key=lambda n: n.score, reverse=True)
            del frontier[max_frontier_width:]

        # Step 9: stop once a terminal node's score beats anything left to expand.
        if terminal_nodes:
            best_terminal = max(terminal_nodes, key=lambda n: n.score)
            frontier_best = max((n.score for n in frontier), default=float("-inf"))
            if best_terminal.score >= frontier_best:
                return TreeSearchResult(
                    best_terminal.answer, best_terminal, tree, expansion_order, expansions, "terminal_best"
                )

    # Step 10: budget exhausted (or frontier empty) with no decisive terminal.
    if terminal_nodes:
        best_terminal = max(terminal_nodes, key=lambda n: n.score)
        return TreeSearchResult(best_terminal.answer, best_terminal, tree, expansion_order, expansions, "terminal_best")
    best_partial = max(tree, key=lambda n: n.score)
    if on_budget_exhausted == "generate":
        prompt = (
            f"{_render_trajectory(goal, best_partial.trajectory)}\n"
            "You are out of search budget. Answer the goal as best you can from the trajectory above."
        )
        completion = provider.complete([Message.user(prompt)], system=system_prompt)
        return TreeSearchResult(completion.content, best_partial, tree, expansion_order, expansions, "budget_generate")
    if on_budget_exhausted == "force":
        return TreeSearchResult(force_message, best_partial, tree, expansion_order, expansions, "budget_force")
    raise ValueError(f"Unknown on_budget_exhausted policy: {on_budget_exhausted!r}")


def demo_tree_search() -> TreeSearchResult:
    """Search the same two-hop question as `native_loop.demo_native`, with a fork.

    The root forks into branch A (search "eiffel tower", a real but
    irrelevant fact that scores a misleadingly high 6) and branch B (search
    "great wall", the actually correct first hop, scored a more conservative
    5). Best-first expands A first. A's `finish` child is a confidently
    wrong answer, scored only 2, so the search does not stop there: B is
    still the best node left in the frontier. Expanding B reaches the
    correct `finish`, scored 9, which beats everything left and wins. This
    shows both a misleading initial score and the backtrack to the sibling
    that a bare "first terminal found" policy would have missed.
    """
    tools = build_native_registry()
    provider = get_provider(
        script=[
            # Root expansion: two candidate first moves.
            {"tool": "search", "args": {"query": "eiffel tower"}},  # branch A: real fact, wrong direction
            {"tool": "search", "args": {"query": "great wall"}},  # branch B: the correct first hop
            "6",  # score for A: looks like progress
            "5",  # score for B: correct but scored conservatively
            # Expand A (highest score): a confidently wrong finish, plus a dead-end sibling.
            {"tool": "search", "args": {"query": "population of beijing"}},  # irrelevant to A's own path
            {"tool": "finish", "args": {"answer": "Paris, France"}},  # wrong: answers about the Eiffel Tower
            "1",  # score for A's dead-end child
            "2",  # score for A's wrong finish: rejected, B is still better
            # Backtrack: expand B, which reaches the correct answer.
            {"tool": "finish", "args": {"answer": "Beijing"}},  # correct: capital of China
            {"tool": "search", "args": {"query": "eiffel tower"}},  # decoy, does not help this goal
            "9",  # score for B's finish: beats everything left in the frontier
            "3",  # score for B's decoy child
        ]
    )
    goal = "What is the capital of the country where the Great Wall is located?"
    return run_tree_search(provider, tools, goal, branching_factor=2, node_budget=5)
