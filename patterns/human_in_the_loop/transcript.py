"""Small formatting helpers so `main.py` stays readable.

Pure string formatting only; no pattern logic lives here.
"""

from __future__ import annotations

from patterns.human_in_the_loop.gate import AuditLog, GateOutcome


def format_outcome(outcome: GateOutcome) -> str:
    """One-line summary of a gate outcome, for a transcript."""
    return f"[{outcome.kind}] {outcome.tool_result}"


def format_audit_log(audit_log: AuditLog, *, indent: str = "   ") -> str:
    """Render every record in an audit log as one line each, in order."""
    lines = []
    for record in audit_log.records:
        lines.append(
            f"{indent}audit: request={record.request_id} decision={record.decision_kind} "
            f"reviewer={record.reviewer} final_args={record.final_arguments} "
            f"latency={record.latency:.1f}s reason={record.reason!r}"
        )
    return "\n".join(lines)
