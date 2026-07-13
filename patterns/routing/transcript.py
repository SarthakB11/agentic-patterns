"""Human-readable rendering of a `RouteDecision`.

Kept separate from routing logic so classifiers and dispatch stay free of
print formatting and can be reused wherever a transcript is not needed, for
example inside tests.
"""

from __future__ import annotations

from patterns.routing.registry import RouteDecision


# Deliberately duplicated (byte-for-byte) from patterns/reflection/transcript.py:
# pattern folders are self-contained by design and never import from each other.
def _snippet(text: str, width: int = 88) -> str:
    """Collapse whitespace and truncate `text` to `width` characters for display."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 1] + "..."


def format_decision(decision: RouteDecision, *, title: str, input_text: str | None = None) -> str:
    """Render a `RouteDecision` as a readable, indented block.

    Args:
        decision: The routing outcome to render.
        title: Heading for this block, e.g. the variant name.
        input_text: The input that was routed, shown if given.
    """
    lines = [f"=== {title} ==="]
    if input_text is not None:
        lines.append(f"input: {_snippet(input_text)}")
    score_text = "none" if decision.score is None else f"{decision.score:.3f}"
    lines.append(f"route: {decision.route}  (method={decision.method}, score={score_text}, attempts={decision.attempts})")
    for key, value in decision.metadata.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)
