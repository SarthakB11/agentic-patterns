"""Text parser for the `Thought: ... / Action: Tool[args]` ReAct grammar.

This is the brittle, hand-parsed half of the pattern that native tool calling
(`native_loop.py`) exists to replace. Kept as its own module so the grammar
and its failure mode can be unit tested independently of any loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*)\]", re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+)")


class ActionParseError(Exception):
    """Raised when model output does not contain a well-formed Action line."""


@dataclass
class ParsedAction:
    """The result of parsing one model turn.

    Attributes:
        thought: Free-text reasoning preceding the action, empty if absent.
        tool: Name of the tool named in the Action line, or "Finish".
        args_text: Raw text inside the Action's brackets, unparsed.
        is_finish: True if `tool == "Finish"`.
        final_answer: The answer text when `is_finish` is True, else None.
    """

    thought: str
    tool: str
    args_text: str
    is_finish: bool
    final_answer: str | None


def parse_action(text: str) -> ParsedAction:
    """Parse one model turn into a Thought and an Action.

    Expected shape, one Thought/Action pair per turn:
        Thought: <free text>
        Action: <ToolName>[<argument text>]

    Parsing is intentionally strict: a missing or malformed `Action:` line is
    a parse failure rather than a best-effort guess, so the runtime can turn
    it into a recoverable Observation instead of silently doing the wrong
    thing. The Thought line is optional; a turn with only an Action line
    still parses.

    Args:
        text: Raw text from the model for one turn.

    Returns:
        A ParsedAction describing the Thought and Action found.

    Raises:
        ActionParseError: If no `Action: Name[args]` line is found in `text`.
    """
    action_match = _ACTION_RE.search(text)
    if not action_match:
        raise ActionParseError(f"No 'Action: Name[args]' line found in model output: {text!r}")

    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""
    tool = action_match.group(1)
    args_text = action_match.group(2).strip()

    if tool == "Finish":
        return ParsedAction(thought=thought, tool=tool, args_text=args_text, is_finish=True, final_answer=args_text)
    return ParsedAction(thought=thought, tool=tool, args_text=args_text, is_finish=False, final_answer=None)
