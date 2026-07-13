"""Todo-list / in-context planning: the mainstream shape as of mid-2026.

Instead of a separate planner call producing a fixed structured `Plan`, a
single agent keeps a lightweight todo list in its own state and rewrites it
as it works, the way Claude Agent SDK's `TodoWrite` tool and LangGraph Deep
Agents' `write_todos` tool do. This sits between ReAct (no persisted plan at
all) and classic plan-then-execute (a fixed upfront plan): the list is
visible and inspectable at every step, but the agent can add, reorder, or
check off items mid-run instead of committing to the full structure before
the first tool call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agentic_patterns import Message, Provider, Tool, ToolRegistry

TodoStatus = Literal["pending", "in_progress", "done"]

_STATUS_MARKS = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]"}


@dataclass
class Todo:
    """One item on the agent's self-maintained todo list."""

    id: str
    text: str
    status: TodoStatus = "pending"


@dataclass
class TodoListState:
    """Mutable in-context plan state a single agent run holds and edits."""

    todos: list[Todo] = field(default_factory=list)

    def write_todos(self, items: list[dict[str, str]]) -> str:
        """Replace the todo list with `items` and return it rendered as text.

        This is the tool function itself: the model calls it with a full
        replacement list every time it wants to add, reorder, or check off
        an item, mirroring how `write_todos`-style tools work in practice.
        """
        self.todos = [
            Todo(id=i["id"], text=i["text"], status=i.get("status", "pending")) for i in items  # type: ignore[arg-type]
        ]
        return self.render()

    def render(self) -> str:
        """Render the current list as a checklist string."""
        if not self.todos:
            return "(empty)"
        return "\n".join(f"{_STATUS_MARKS[t.status]} {t.id}: {t.text}" for t in self.todos)


SYSTEM = (
    "You are a trip-planning agent. Keep your plan visible with the "
    "write_todos tool: call it whenever you add, start, or finish an item. "
    "Call other tools to do the actual work. Reply with plain text and no "
    "tool call once every todo is done."
)


def _write_todos_tool(state: TodoListState) -> Tool:
    """Build a `write_todos` tool bound to `state`."""
    return Tool(
        name="write_todos",
        description="Replace the current todo list with the given items.",
        parameters={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "done"],
                            },
                        },
                        "required": ["id", "text"],
                    },
                }
            },
            "required": ["items"],
        },
        fn=state.write_todos,
    )


@dataclass
class TodoRun:
    """The outcome of a todo-list in-context planning run.

    Attributes:
        state: The final `TodoListState`, for inspecting the last written list.
        transcript: The full conversation, including `write_todos` calls.
        final_answer: The model's final plain-text reply.
        model_calls: Total number of `provider.complete` calls made.
    """

    state: TodoListState
    transcript: list[Message]
    final_answer: str
    model_calls: int


def run_todo_list(provider: Provider, goal: str, registry: ToolRegistry, max_steps: int = 8) -> TodoRun:
    """Run an agent loop where the plan itself is a tool call, not a phase.

    Args:
        provider: Called once per loop iteration, same as the ReAct baseline.
        goal: The user's goal, sent as the first user message.
        registry: Work tools available alongside the injected `write_todos` tool.
        max_steps: Safety cap on loop iterations.

    Raises:
        RuntimeError: If the loop exceeds `max_steps` without a final answer.
    """
    state = TodoListState()
    work_registry = ToolRegistry()
    for spec in registry.specs():
        work_registry.register(registry.get(spec["name"]))
    work_registry.register(_write_todos_tool(state))

    messages: list[Message] = [Message.user(goal)]
    model_calls = 0
    for _ in range(max_steps):
        completion = provider.complete(messages, tools=work_registry.specs(), system=SYSTEM)
        model_calls += 1
        if not completion.tool_calls:
            messages.append(Message.assistant(completion.content))
            return TodoRun(state=state, transcript=messages, final_answer=completion.content, model_calls=model_calls)
        messages.append(Message.assistant(completion.content, tool_calls=completion.tool_calls))
        for call in completion.tool_calls:
            observation = work_registry.execute(call)
            messages.append(Message.tool(call.id, observation))
    raise RuntimeError(f"Todo-list loop exceeded max_steps={max_steps} without a final answer")


def demo() -> None:
    """Run the todo-list loop on a Lyon trip goal, showing the list get rewritten mid-run."""
    from agentic_patterns import get_provider
    from patterns.planning.tools import build_travel_registry

    goal = "Plan a day in Lyon: check the weather, list attractions, then draft an itinerary."
    labels = ["check weather", "list attractions", "draft itinerary"]

    def todos(*statuses: str) -> dict:
        items = [{"id": f"t{i + 1}", "text": labels[i], "status": s} for i, s in enumerate(statuses)]
        return {"tool": "write_todos", "args": {"items": items}}

    script = [
        todos("in_progress", "pending", "pending"),
        {"tool": "get_weather", "args": {"city": "Lyon"}},
        todos("done", "in_progress", "pending"),
        {"tool": "search_attractions", "args": {"city": "Lyon"}},
        {
            "tool": "draft_itinerary",
            "args": {
                "weather": "Partly cloudy, high of 20C",
                "attractions": "Basilica of Notre-Dame de Fourviere, Old Lyon, Traboules",
            },
        },
        todos("done", "done", "done"),
        "Lyon plan: partly cloudy at 20C, so visit the Basilica of "
        "Notre-Dame de Fourviere, walk Old Lyon, and duck into the traboules "
        "if it cools off.",
    ]
    provider = get_provider(script=script)
    registry = build_travel_registry()

    print("=== Todo-list in-context planning ===")
    print(f"Goal: {goal}\n")
    run = run_todo_list(provider, goal, registry)
    for message in run.transcript:
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                if call.name == "write_todos":
                    print("  [plan updated]")
                else:
                    print(f"  [call] {call.name}({call.arguments})")
        elif message.role == "tool":
            print(f"  [observation] {message.content}")
    print(f"\nFinal todo list:\n{run.state.render()}")
    print(f"\nFinal answer: {run.final_answer}")
    print(f"Total model calls: {run.model_calls}")


if __name__ == "__main__":
    demo()
