"""Verify-before-finish: gate a Finish on a verifier before accepting it.

This intercepts the exact moment `text_loop.run_react` would return: before
handing the answer back, a verifier call is asked whether the trajectory
actually supports it. A rejection is appended as an observation and the loop
continues for a bounded number of extra cycles instead of accepting the
answer outright.

This is a different gap than `reflexion.py` covers. Reflexion only retries
when an episode stops *without* reaching Finish, a loop or the iteration cap;
an episode that calls Finish with a wrong answer is accepted as-is, since
from Reflexion's point of view the episode succeeded. This module is the
wrong-Finish gate: it checks every Finish before accepting it, whether or
not the episode would otherwise look successful.

As with `compaction.py`, the loop body below is a small, deliberate variant
of `run_react`'s loop rather than a call into it, since intercepting Finish
requires control `run_react` does not expose without changing its behavior.
Parsing and tool-argument handling are reused, not reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Message, Provider, ToolCall, ToolRegistry, get_provider
from patterns.react.parser import ActionParseError, parse_action
from patterns.react.scratchpad import Scratchpad, Step
from patterns.react.text_loop import DEFAULT_FORCE_MESSAGE, FEW_SHOT_SYSTEM_PROMPT, _build_arguments, _prompt
from patterns.react.world import build_registry

VERIFIER_SYSTEM_PROMPT = (
    "You are a verifier. Given a goal and a trajectory ending in a proposed answer, "
    "reply 'ACCEPT' if the trajectory's observations support the answer, or "
    "'REJECT: <reason>' if they do not."
)


@dataclass
class VerifyResult:
    """Outcome of a text-parsing ReAct run gated by a verify-before-finish check.

    Attributes:
        answer: The accepted answer, or None if the loop stopped without one.
        scratchpad: The full trajectory, including any rejected Finish attempts.
        steps_taken: Number of main-loop model calls made.
        stopped_reason: One of "finish", "verify_cap", "loop_detected",
            "max_iterations_force".
        first_try: True if the first Finish attempt was accepted with no rejection.
        verify_calls: Number of verifier calls made.
    """

    answer: str | None
    scratchpad: Scratchpad
    steps_taken: int
    stopped_reason: str
    first_try: bool
    verify_calls: int


def _verify_prompt(goal: str, scratchpad: Scratchpad, answer: str) -> str:
    """Render the verifier's prompt: the goal, the trajectory, and the proposed answer."""
    rendered = scratchpad.render()
    return f"Question: {goal}\n{rendered}\nProposed answer: {answer}\nIs this answer supported by the trajectory above?"


def run_react_with_verification(
    provider: Provider,
    tools: ToolRegistry,
    goal: str,
    *,
    system_prompt: str = FEW_SHOT_SYSTEM_PROMPT,
    verifier_system_prompt: str = VERIFIER_SYSTEM_PROMPT,
    max_iterations: int = 6,
    max_verify_cycles: int = 2,
) -> VerifyResult:
    """Run the text-parsing ReAct loop, verifying every Finish before accepting it.

    Args:
        provider: Model provider, used for loop steps and verifier calls alike.
        tools: Registry of tools the model may invoke by name.
        goal: The question or task to solve.
        system_prompt: Instruction plus, for few-shot, a worked example trajectory.
        verifier_system_prompt: Instruction for the verifier call.
        max_iterations: Maximum Thought/Action/Observation steps before stopping.
        max_verify_cycles: Maximum number of rejected Finish attempts before
            giving up rather than looping forever on a stubborn model.

    Returns:
        A VerifyResult describing how the loop ended and how many verify
        calls it took.
    """
    scratchpad = Scratchpad()
    verify_calls = 0
    reject_cycles = 0

    for step_num in range(1, max_iterations + 1):
        completion = provider.complete([Message.user(_prompt(goal, scratchpad))], system=system_prompt)
        try:
            parsed = parse_action(completion.content)
        except ActionParseError:
            error = "ERROR: could not parse an action. Use the format 'Action: Tool[args]'."
            scratchpad.add(Step("", "(unparsed)", completion.content, observation=error))
            continue

        if parsed.is_finish:
            assert parsed.final_answer is not None, "parse_action sets final_answer whenever is_finish is True"
            verify_calls += 1
            prompt = _verify_prompt(goal, scratchpad, parsed.final_answer)
            verdict = provider.complete([Message.user(prompt)], system=verifier_system_prompt).content.strip()
            if verdict.upper().startswith("ACCEPT"):
                return VerifyResult(
                    parsed.final_answer, scratchpad, step_num, "finish", reject_cycles == 0, verify_calls
                )
            reason = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
            reject_cycles += 1
            scratchpad.add(
                Step(parsed.thought, "Finish", parsed.final_answer, observation=f"verification failed: {reason}")
            )
            if reject_cycles >= max_verify_cycles:
                return VerifyResult(None, scratchpad, step_num, "verify_cap", False, verify_calls)
            continue

        try:
            arguments = _build_arguments(tools, parsed.tool, parsed.args_text)
        except Exception as exc:  # unknown tool or a tool with the wrong argument shape
            observation = f"ERROR: {exc}"
        else:
            call = ToolCall(id=f"call_{step_num}", name=parsed.tool, arguments=arguments)
            observation = tools.execute(call)

        scratchpad.add(Step(parsed.thought, parsed.tool, parsed.args_text, observation=observation))
        if scratchpad.is_repeating():
            return VerifyResult(None, scratchpad, step_num, "loop_detected", reject_cycles == 0, verify_calls)

    return VerifyResult(
        DEFAULT_FORCE_MESSAGE, scratchpad, max_iterations, "max_iterations_force", reject_cycles == 0, verify_calls
    )


def demo_verify() -> VerifyResult:
    """Reject a confidently wrong Finish, then accept the corrected one.

    The first Finish claims "Beijing" from an observation that only names
    China, not its capital. The verifier rejects it since the trajectory
    does not actually support that answer. The reject reason is threaded
    back as an observation, the model looks up the capital directly, and
    the second Finish is accepted.
    """
    tools = build_registry()
    provider = get_provider(
        script=[
            "Thought: The Great Wall is in China, so I will guess the capital.\nAction: search[great wall]",
            "Thought: That should be enough.\nAction: Finish[Beijing]",
            "REJECT: the trajectory only shows the Great Wall is in China, not that China's capital is Beijing",
            "Thought: Let me actually look up the capital.\nAction: search[capital of china]",
            "Thought: Now this is supported.\nAction: Finish[Beijing]",
            "ACCEPT",
        ]
    )
    goal = "What is the capital of the country where the Great Wall is located?"
    return run_react_with_verification(provider, tools, goal)
