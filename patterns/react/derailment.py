"""Semantic derailment detection beyond exact-repeat, with a recovery nudge.

`native_loop._repeats_previous_call` and `scratchpad.Scratchpad.is_repeating`
only catch a turn that is byte-for-byte identical to the one right before
it. Zhu et al., "Where LLM Agents Fail and How They can Learn From Failures"
(arXiv:2509.25370), find that real long-horizon failure is usually one of a
few semantic patterns instead: oscillation between two or more actions,
no-progress (different actions that all land on the same unhelpful result),
or a run of tool errors. The three detectors below are pure functions over a
trajectory, no model call involved, so they run offline exactly as they
would in production.

Detection alone is not the whole fix. The first flag injects a recovery
nudge into the trajectory and gives the model one more chance instead of
stopping outright; only a second flag after the nudge ends the run with
`stopped_reason="derailed"`.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry, get_provider
from patterns.react.parser import ActionParseError, parse_action
from patterns.react.scratchpad import Scratchpad, Step
from patterns.react.text_loop import DEFAULT_FORCE_MESSAGE, FEW_SHOT_SYSTEM_PROMPT, _build_arguments, _prompt
from patterns.react.world import build_registry

RECOVERY_MESSAGE = (
    "You seem stuck: your last few actions have not made progress. Try a different "
    "tool, a broader or more specific query, or reconsider your approach."
)


def detect_oscillation(steps: list[Step], window: int = 4) -> bool:
    """True if the last `window` actions alternate in a period-2 or longer periodic pattern.

    Checks every period from 2 up to `window // 2`, so both a simple A, B,
    A, B swap and a longer cycle are caught, unlike the exact-repeat guard,
    which only sees two consecutive identical turns.
    """
    if len(steps) < window:
        return False
    tail = steps[-window:]
    keys = [(s.action, s.action_input) for s in tail]
    return any(
        all(keys[i] == keys[i - period] for i in range(period, window)) for period in range(2, window // 2 + 1)
    )


def detect_no_progress(steps: list[Step], window: int = 3) -> bool:
    """True if the last `window` steps used distinct actions but only repeated observations.

    Different queries that all come back with the same unhelpful result (for
    example, three different searches that all report "no results") signal
    the agent is trying variations without learning anything new, which an
    exact-repeat check cannot see since no single action repeats.
    """
    if len(steps) < window:
        return False
    tail = steps[-window:]
    distinct_actions = {(s.action, s.action_input) for s in tail}
    distinct_observations = {s.observation for s in tail}
    return len(distinct_actions) == window and len(distinct_observations) < window


def detect_error_storm(steps: list[Step], window: int = 2) -> bool:
    """True if the last `window` observations were all tool errors."""
    if len(steps) < window:
        return False
    tail = steps[-window:]
    return all(s.observation.startswith("ERROR") for s in tail)


def _first_detector(steps: list[Step]) -> str | None:
    """Run every detector in order and return the name of the first that fires."""
    if detect_oscillation(steps):
        return "oscillation"
    if detect_no_progress(steps):
        return "no_progress"
    if detect_error_storm(steps):
        return "error_storm"
    return None


@dataclass
class DerailmentResult:
    """Outcome of a text-parsing ReAct run guarded by derailment detection.

    Attributes:
        answer: The final answer, or None if the loop stopped without one.
        scratchpad: The full trajectory, including any recovery nudge.
        steps_taken: Number of main-loop model calls made.
        stopped_reason: One of "finish", "derailed", "max_iterations_force".
        detector_fired: Name of the detector that fired first, or None if none did.
        recovery_attempted: True if a recovery nudge was injected during the run.
    """

    answer: str | None
    scratchpad: Scratchpad
    steps_taken: int
    stopped_reason: str
    detector_fired: str | None
    recovery_attempted: bool


def run_react_with_derailment_recovery(
    provider: Provider,
    tools: ToolRegistry,
    goal: str,
    *,
    system_prompt: str = FEW_SHOT_SYSTEM_PROMPT,
    max_iterations: int = 8,
) -> DerailmentResult:
    """Run the text-parsing ReAct loop, nudging on the first derailment flag and stopping on the second.

    Args:
        provider: Model provider to call each iteration.
        tools: Registry of tools the model may invoke by name.
        goal: The question or task to solve.
        system_prompt: Instruction plus, for few-shot, a worked example trajectory.
        max_iterations: Maximum Thought/Action/Observation steps before stopping.

    Returns:
        A DerailmentResult describing how the loop ended, which detector
        fired, and whether recovery was attempted.
    """
    scratchpad = Scratchpad()
    nudged = False
    detector_name: str | None = None

    for step_num in range(1, max_iterations + 1):
        completion = provider.complete([Message.user(_prompt(goal, scratchpad))], system=system_prompt)
        try:
            parsed = parse_action(completion.content)
        except ActionParseError:
            error = "ERROR: could not parse an action. Use the format 'Action: Tool[args]'."
            scratchpad.add(Step("", "(unparsed)", completion.content, observation=error))
            continue

        if parsed.is_finish:
            return DerailmentResult(parsed.final_answer, scratchpad, step_num, "finish", detector_name, nudged)

        try:
            arguments = _build_arguments(tools, parsed.tool, parsed.args_text)
        except Exception as exc:  # unknown tool or a tool with the wrong argument shape
            observation = f"ERROR: {exc}"
        else:
            call = ToolCall(id=f"call_{step_num}", name=parsed.tool, arguments=arguments)
            observation = tools.execute(call)

        scratchpad.add(Step(parsed.thought, parsed.tool, parsed.args_text, observation=observation))

        fired = _first_detector(scratchpad.steps)
        if fired:
            if not nudged:
                scratchpad.add(Step("", "(recovery)", "", observation=RECOVERY_MESSAGE))
                nudged = True
                detector_name = fired
                continue
            return DerailmentResult(None, scratchpad, step_num, "derailed", fired, True)

    return DerailmentResult(
        DEFAULT_FORCE_MESSAGE, scratchpad, max_iterations, "max_iterations_force", detector_name, nudged
    )


def demo_derailment() -> DerailmentResult:
    """Oscillate between two searches, recover after the nudge, then finish.

    The first four steps alternate "monument" and "landmark", neither of
    which matches the knowledge base: a period-2 oscillation, which the
    exact-repeat guard would miss since no single turn repeats its
    immediate predecessor. The oscillation detector fires, a recovery nudge
    is injected, and the model responds with the correct query on its next turn.
    """
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: Search for the monument.\nAction: search[monument]",
            "Thought: Try landmark instead.\nAction: search[landmark]",
            "Thought: Back to monument.\nAction: search[monument]",
            "Thought: Landmark again.\nAction: search[landmark]",
            "Thought: Let me use the actual name instead.\nAction: search[eiffel tower]",
            "Thought: That answers it.\nAction: Finish[Paris, France]",
        ]
    )
    goal = "Where is the Eiffel Tower located?"
    return run_react_with_derailment_recovery(provider, tools, goal)
