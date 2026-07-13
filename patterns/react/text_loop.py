"""Few-shot text-parsing ReAct: the canonical form from the original paper.

The model emits raw text containing a Thought and an `Action: Tool[args]`
line, the runtime parses it, executes the named tool, and appends the result
as an Observation. The system prompt carries one worked example trajectory
so the model has seen the format before it is asked to use it.

`zero_shot.py` reuses `run_react` below with no worked example in its system
prompt, showing the same loop holds up without one. `native_loop.py`
replaces the whole text grammar with the provider's structured tool-call API
and has no parser to break.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry, get_provider

from patterns.react.parser import ActionParseError, parse_action
from patterns.react.scratchpad import Scratchpad, Step
from patterns.react.world import build_registry

FEW_SHOT_SYSTEM_PROMPT = """You are a research agent that answers questions by alternating between reasoning and tool calls.

Use exactly this format, one Thought/Action pair per reply:
Thought: <your reasoning about what to do next>
Action: <ToolName>[<argument>]

Available tools:
- search[query]: search the knowledge base for a topic.
- lookup[term]: look up one specific term, like Ctrl+F.
- Finish[answer]: give the final answer and stop.

Example:
Question: What continent is the country that built the Colosseum on?
Thought: I need to find which country built the Colosseum.
Action: search[Colosseum]
Observation: The Colosseum was built in Italy.
Thought: Italy is on the continent of Europe, so that answers the question.
Action: Finish[Europe]

Now answer the real question the same way."""

DEFAULT_FORCE_MESSAGE = "Stopped: could not reach an answer within the iteration budget."


@dataclass
class ReactResult:
    """Outcome of a text-parsing ReAct run.

    Attributes:
        answer: The final answer text, or None if the loop stopped without one.
        scratchpad: The full Thought/Action/Observation history.
        steps_taken: Number of main-loop model calls made (excludes the extra
            tool-free call a "generate" stop makes).
        stopped_reason: One of "finish", "max_iterations_force",
            "max_iterations_generate", "loop_detected".
    """

    answer: str | None
    scratchpad: Scratchpad
    steps_taken: int
    stopped_reason: str


def _build_arguments(tools: ToolRegistry, tool_name: str, args_text: str) -> dict[str, object]:
    """Map a single bracketed `Tool[args]` string onto the tool's one parameter.

    Every tool wired to this loop must declare exactly one parameter, since
    the grammar carries exactly one positional string. This is a limitation
    of the text grammar, not of ReAct itself: `native_loop.py` and
    `tree_search.py` carry full structured argument objects and have no such
    restriction.

    Raises:
        KeyError: If `tool_name` is not registered.
        ValueError: If the tool does not declare exactly one parameter.
    """
    tool = tools.get(tool_name)
    props = tool.parameters.get("properties", {})
    if len(props) != 1:
        raise ValueError(
            f"Text-parsing ReAct only supports single-argument tools; "
            f"{tool_name!r} declares parameters {list(props)}"
        )
    (param_name,) = props.keys()
    return {param_name: args_text}


def _prompt(goal: str, scratchpad: Scratchpad) -> str:
    """Render the question plus the scratchpad so far, ending on 'Thought:'."""
    rendered = scratchpad.render()
    if rendered:
        return f"Question: {goal}\n{rendered}\nThought:"
    return f"Question: {goal}\nThought:"


def run_react(
    provider: Provider,
    tools: ToolRegistry,
    goal: str,
    *,
    system_prompt: str = FEW_SHOT_SYSTEM_PROMPT,
    max_iterations: int = 6,
    on_max_iterations: str = "force",
    force_message: str = DEFAULT_FORCE_MESSAGE,
) -> ReactResult:
    """Run the text-parsing ReAct loop to completion.

    Args:
        provider: Model provider to call each iteration.
        tools: Registry of tools the model may invoke by name.
        goal: The question or task to solve.
        system_prompt: Instruction plus, for few-shot, a worked example trajectory.
        max_iterations: Maximum Thought/Action/Observation steps before stopping.
        on_max_iterations: "force" returns `force_message` immediately;
            "generate" makes one further tool-free model call asking it to
            answer from the work done so far.
        force_message: The fixed message returned when `on_max_iterations="force"`.

    Returns:
        A ReactResult describing how the loop ended.
    """
    scratchpad = Scratchpad()
    for step_num in range(1, max_iterations + 1):
        completion = provider.complete([Message.user(_prompt(goal, scratchpad))], system=system_prompt)

        try:
            parsed = parse_action(completion.content)
        except ActionParseError:
            scratchpad.add(
                Step(
                    thought="",
                    action="(unparsed)",
                    action_input=completion.content,
                    observation="ERROR: could not parse an action. Use the format 'Action: Tool[args]'.",
                )
            )
            continue

        if parsed.is_finish:
            return ReactResult(
                answer=parsed.final_answer, scratchpad=scratchpad, steps_taken=step_num, stopped_reason="finish"
            )

        try:
            arguments = _build_arguments(tools, parsed.tool, parsed.args_text)
        except Exception as exc:  # unknown tool or a tool with the wrong argument shape
            observation = f"ERROR: {exc}"
        else:
            call = ToolCall(id=f"call_{step_num}", name=parsed.tool, arguments=arguments)
            observation = tools.execute(call)

        scratchpad.add(
            Step(thought=parsed.thought, action=parsed.tool, action_input=parsed.args_text, observation=observation)
        )
        if scratchpad.is_repeating():
            return ReactResult(answer=None, scratchpad=scratchpad, steps_taken=step_num, stopped_reason="loop_detected")

    return _stop_at_max_iterations(provider, goal, scratchpad, system_prompt, on_max_iterations, force_message, max_iterations)


def _stop_at_max_iterations(
    provider: Provider,
    goal: str,
    scratchpad: Scratchpad,
    system_prompt: str,
    on_max_iterations: str,
    force_message: str,
    max_iterations: int,
) -> ReactResult:
    """Apply the configured early-stop policy once the iteration cap is hit."""
    if on_max_iterations == "generate":
        prompt = (
            f"Question: {goal}\n{scratchpad.render()}\n"
            "You are out of tool-call budget. Answer the question as best you can "
            "from the work above. Reply with the answer text only, no Thought or Action."
        )
        completion = provider.complete([Message.user(prompt)], system=system_prompt)
        return ReactResult(
            answer=completion.content, scratchpad=scratchpad, steps_taken=max_iterations, stopped_reason="max_iterations_generate"
        )
    if on_max_iterations == "force":
        return ReactResult(
            answer=force_message, scratchpad=scratchpad, steps_taken=max_iterations, stopped_reason="max_iterations_force"
        )
    raise ValueError(f"Unknown on_max_iterations policy: {on_max_iterations!r}")


def demo_few_shot() -> ReactResult:
    """Run a two-hop lookup through the few-shot text-parsing loop.

    The model must search for the Great Wall's country, then search again
    for that country's capital, before it can finish.
    """
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: I need to find which country the Great Wall is located in.\nAction: search[Great Wall]",
            "Thought: Now I need the capital of that country.\nAction: search[capital of china]",
            "Thought: I have enough information to answer.\nAction: Finish[Beijing]",
        ]
    )
    goal = "What is the capital of the country where the Great Wall is located?"
    return run_react(provider, tools, goal, system_prompt=FEW_SHOT_SYSTEM_PROMPT)
