"""A genuinely interactive `DecisionSource`, for real people at a terminal.

This module is the one place in the pattern that calls `input()`. It is
never imported by a demo that runs under the default `python -m
patterns.human_in_the_loop.main` invocation and never exercised by the
test suite, since both must stay non-interactive. It is reachable only
through `main.py`'s `--interactive` flag, which a human passes on purpose.

Every other decision source in this pattern is `ScriptedDecisionSource`
from `gate.py`; this class implements the same `DecisionSource` protocol
so gate logic never special-cases which one it is talking to.
"""

from __future__ import annotations

from patterns.human_in_the_loop.gate import Decision, ReviewRequest


class InteractiveDecisionSource:
    """A `DecisionSource` that prompts a real person with `input()`.

    Blocks on stdin until the reviewer types a decision. Intended for a
    human running the CLI directly, never for tests or CI.
    """

    def decide(self, request: ReviewRequest) -> Decision:
        """Prompt the terminal for a decision on one review request."""
        print(f"\n--- review request {request.id} ---")
        print(f"context: {request.context}")
        print(f"action:  {request.action.name}({request.action.arguments})")
        print("decisions: approve / edit / reject / respond")
        kind = input("decision: ").strip().lower()

        if kind == "edit":
            print("enter replacement arguments as key=value pairs, one per line, blank line to finish")
            arguments: dict[str, object] = {}
            while True:
                line = input("  ").strip()
                if not line:
                    break
                key, _, raw_value = line.partition("=")
                arguments[key.strip()] = _coerce(raw_value.strip())
            reason = input("reason: ").strip()
            return Decision(kind="edit", reviewer="terminal-user", reason=reason, arguments=arguments)

        if kind == "reject":
            reason = input("reason: ").strip()
            return Decision(kind="reject", reviewer="terminal-user", reason=reason)

        if kind == "respond":
            value = input("value to return: ").strip()
            return Decision(kind="respond", reviewer="terminal-user", value=value)

        if kind == "approve":
            reason = input("reason (optional): ").strip()
            return Decision(kind="approve", reviewer="terminal-user", reason=reason)

        # Anything else is passed through unrecognized on purpose: run_gate
        # fails closed on it rather than this class guessing what was meant.
        return Decision(kind=kind, reviewer="terminal-user")


def _coerce(raw_value: str) -> object:
    """Best-effort coercion of a typed-in value to int, float, or str."""
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    return raw_value
