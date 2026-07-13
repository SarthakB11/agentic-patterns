"""Scratchpad construction for the text-parsing ReAct loop.

The scratchpad accumulates Thought/Action/Observation steps and renders them
back to text, which is re-sent as part of the prompt on every iteration so
the model sees its own trajectory so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    """One Thought/Action/Observation triple.

    Attributes:
        thought: The model's reasoning for this step.
        action: Name of the tool invoked, or a sentinel like "(unparsed)" for
            a step that failed to parse.
        action_input: Raw argument text passed to the action.
        observation: The runtime's response to the action, possibly
            truncated by `Scratchpad.add`. Empty only for a step that has not
            been executed yet.
    """

    thought: str
    action: str
    action_input: str
    observation: str = ""


@dataclass
class Scratchpad:
    """Accumulates Thought/Action/Observation steps for one ReAct run.

    Attributes:
        steps: Steps recorded so far, in order.
        max_observation_chars: Observations longer than this are truncated
            before being stored, so a single verbose tool result cannot blow
            up the prompt on later iterations.
    """

    max_observation_chars: int = 500
    steps: list[Step] = field(default_factory=list)

    def add(self, step: Step) -> None:
        """Append a step, truncating a long observation to the size budget."""
        if len(step.observation) > self.max_observation_chars:
            step.observation = step.observation[: self.max_observation_chars] + "... [truncated]"
        self.steps.append(step)

    def render(self) -> str:
        """Render recorded steps as the Thought/Action/Observation text block."""
        lines: list[str] = []
        for step in self.steps:
            lines.append(f"Thought: {step.thought}")
            lines.append(f"Action: {step.action}[{step.action_input}]")
            if step.observation:
                lines.append(f"Observation: {step.observation}")
        return "\n".join(lines)

    def is_repeating(self, window: int = 2) -> bool:
        """Return True if the last `window` steps have identical action, input, and observation.

        Args:
            window: How many trailing steps must match for this to count as
                a repeat. Two consecutive identical steps is the default
                degenerate-loop signal.
        """
        if len(self.steps) < window:
            return False
        tail = self.steps[-window:]
        first = (tail[0].action, tail[0].action_input, tail[0].observation)
        return all((s.action, s.action_input, s.observation) == first for s in tail)
