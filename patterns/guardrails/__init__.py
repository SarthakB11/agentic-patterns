"""Guardrails: checkpoints that inspect data crossing a trust boundary around a model.

This package implements the guardrails pattern and its major sub-variants:
input rails (prompt-injection detection, a topical allowlist, a length
limit), PII masking and redaction, a retrieval guard for RAG context, an
output schema validator, a moderation blocklist, a groundedness check, an
execution (pre-tool) guard with a human-approval branch, and a
Plan-Then-Execute architectural guard that removes indirect prompt
injection's path to a side effect rather than trying to detect it.

See `patterns/guardrails/README.md` for the full write-up and
`patterns/guardrails/main.py` for a runnable demo of every variant.
"""
