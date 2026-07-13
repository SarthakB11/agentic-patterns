"""ReAct plus reflection (Reflexion-style retry).

After a failed episode the agent writes a short self-critique, which is
prepended to the system prompt as a lesson for the next attempt, then the
whole task is retried from scratch. This adds an outer loop over complete
ReAct episodes; each episode itself is an ordinary `text_loop.run_react`
call, so the retry logic here has no knowledge of tools or parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, ToolRegistry, get_provider

from patterns.react.text_loop import FEW_SHOT_SYSTEM_PROMPT, ReactResult, run_react
from patterns.react.world import build_registry

REFLECTION_SYSTEM_PROMPT = "You are a critic reviewing your own failed attempt at a task."


@dataclass
class ReflexionResult:
    """Outcome of a run with Reflexion-style retries across episodes.

    Attributes:
        final: The ReactResult of the last episode run.
        reflections: Self-critiques written after each failed episode, in order.
        episodes_run: Number of full ReAct episodes attempted.
    """

    final: ReactResult
    reflections: list[str] = field(default_factory=list)
    episodes_run: int = 0


def _reflection_prompt(goal: str, failed: ReactResult) -> str:
    """Ask the model to critique its own failed trajectory."""
    return (
        f"You attempted this question and did not reach a confident answer:\n"
        f"Question: {goal}\n\n"
        f"Your trajectory:\n{failed.scratchpad.render()}\n\n"
        "In one or two sentences, say what went wrong and what you should do "
        "differently on the next attempt."
    )


def run_with_reflexion(
    provider: Provider,
    tools: ToolRegistry,
    goal: str,
    *,
    system_prompt: str = FEW_SHOT_SYSTEM_PROMPT,
    max_episodes: int = 2,
    max_iterations_per_episode: int = 4,
) -> ReflexionResult:
    """Run ReAct episodes, reflecting and retrying after each failure.

    An episode "fails" when it stops without reaching Finish (loop detection
    or the per-episode iteration cap). After a failure, the model is asked to
    critique its own trajectory in a tool-free call, and the critique is
    prepended to the system prompt for the next episode. Stops on the first
    episode that reaches Finish, or after `max_episodes` attempts.

    Args:
        provider: Model provider shared across episodes and reflection calls.
        tools: Registry of tools available to every episode.
        goal: The question or task to solve.
        system_prompt: Base instruction for each episode, before any lessons are added.
        max_episodes: Maximum number of full ReAct episodes to attempt.
        max_iterations_per_episode: Iteration cap passed to each episode's `run_react` call.

    Returns:
        A ReflexionResult with the final episode's outcome and every reflection written.
    """
    reflections: list[str] = []
    result: ReactResult | None = None

    for episode in range(1, max_episodes + 1):
        episode_prompt = system_prompt
        if reflections:
            lessons = "\n".join(f"- {r}" for r in reflections)
            episode_prompt = f"{system_prompt}\n\nLessons from earlier attempts:\n{lessons}"

        result = run_react(
            provider,
            tools,
            goal,
            system_prompt=episode_prompt,
            max_iterations=max_iterations_per_episode,
            on_max_iterations="force",
        )
        if result.stopped_reason == "finish" or episode == max_episodes:
            return ReflexionResult(final=result, reflections=reflections, episodes_run=episode)

        critique = provider.complete(
            [Message.user(_reflection_prompt(goal, result))], system=REFLECTION_SYSTEM_PROMPT
        ).content
        reflections.append(critique)

    assert result is not None  # max_episodes >= 1 guarantees the loop body ran at least once
    return ReflexionResult(final=result, reflections=reflections, episodes_run=max_episodes)


def demo_reflexion() -> ReflexionResult:
    """Run a question that fails on the first attempt and succeeds after reflection.

    Episode 1 searches the vague term "monument", which matches nothing in
    the knowledge base, and repeats the identical failing search, tripping
    loop detection within its 2-step budget. The reflection call names the
    fix. Episode 2 searches the correct term and reaches Finish.
    """
    tools = build_registry()
    provider = get_provider(
        script=[
            # Episode 1: a query too vague to match anything, repeated once.
            "Thought: I will search for the monument in question.\nAction: search[monument]",
            "Thought: That did not help, let me try the same search again.\nAction: search[monument]",
            # Reflection: a tool-free critique call.
            "The query 'monument' was too vague to match the knowledge base; "
            "search for the landmark's full name, 'eiffel tower', instead.",
            # Episode 2: corrected query, reaches Finish.
            "Thought: I will search using the full name this time.\nAction: search[eiffel tower]",
            "Thought: The observation answers the question.\nAction: Finish[Paris, France]",
        ]
    )
    goal = "Where is the Eiffel Tower located?"
    return run_with_reflexion(provider, tools, goal, max_episodes=2, max_iterations_per_episode=2)
