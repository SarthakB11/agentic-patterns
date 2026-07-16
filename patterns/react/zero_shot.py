"""Zero-shot text-parsing ReAct: same loop, no worked example in the prompt.

This reuses `text_loop.run_react` unchanged. The only difference from the
few-shot variant is the system prompt: it describes the Thought/Action
format and the tool list, but shows no example trajectory, leaning harder on
the model's instruction-following. Cheaper in prompt tokens, and more prone
to format drift on a weaker model.
"""

from __future__ import annotations

from agentic_patterns import get_provider
from patterns.react.text_loop import ReactResult, run_react
from patterns.react.world import build_registry

ZERO_SHOT_SYSTEM_PROMPT = """You are a research agent that answers questions by alternating between reasoning \
and tool calls.

Use exactly this format, one Thought/Action pair per reply:
Thought: <your reasoning about what to do next>
Action: <ToolName>[<argument>]

Available tools:
- search[query]: search the knowledge base for a topic.
- lookup[term]: look up one specific term, like Ctrl+F.
- Finish[answer]: give the final answer and stop.

No worked example is provided; follow the format exactly from the first reply."""


def demo_zero_shot() -> ReactResult:
    """Run a one-hop lookup through the zero-shot text-parsing loop.

    Question: "Where is the Eiffel Tower located?" A single search resolves
    it, showing the format holds up with no worked example in the prompt.
    """
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: I should search for the Eiffel Tower.\nAction: search[eiffel tower]",
            "Thought: The observation answers the question directly.\nAction: Finish[Paris, France]",
        ]
    )
    goal = "Where is the Eiffel Tower located?"
    return run_react(provider, tools, goal, system_prompt=ZERO_SHOT_SYSTEM_PROMPT)
