"""Human-readable rendering of a `ReflectionResult`.

Kept separate from the loop mechanics so the loop stays free of print
formatting and can be reused wherever a transcript is not needed, for
example inside tests.
"""

from __future__ import annotations

from patterns.reflection.loop import ReflectionResult


def _snippet(text: str, width: int = 88) -> str:
    """Collapse whitespace and truncate `text` to `width` characters for display."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 1] + "..."


def format_transcript(result: ReflectionResult, *, title: str) -> str:
    """Render a `ReflectionResult` as a readable, indented transcript.

    Args:
        result: The loop outcome to render.
        title: Heading for this transcript block, e.g. the variant name.
    """
    lines = [f"=== {title} ===", f"initial draft: {_snippet(result.initial_draft)}"]
    for it in result.iterations:
        lines.append(f"-- round {it.index} --")
        lines.append(f"   draft:    {_snippet(it.draft)}")
        score_text = "none" if it.critique.score is None else f"{it.critique.score:g}"
        lines.append(f"   critique: score={score_text} approved={it.critique.approved}")
        lines.append(f"             {_snippet(it.critique.comments)}")
        lines.append(f"   decision: {it.note}")
    best_score_text = "none" if result.best_score is None else f"{result.best_score:g}"
    lines.append(f"stopped: {result.stop_reason} after {len(result.iterations)} round(s)")
    lines.append(f"best score: {best_score_text}")
    lines.append(f"best draft: {_snippet(result.best_draft)}")
    return "\n".join(lines)
