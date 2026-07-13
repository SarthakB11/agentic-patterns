"""Multi-agent orchestration: a supervisor decomposes, delegates, and synthesizes.

This package holds the shared mechanics (`state.py`, `worker.py`) plus one
small module per major sub-variant from the taxonomy (supervisor,
aggregation, handoff, group chat, debate, maker-checker, hierarchical), plus
the post-run and 2025-2026 additions: `failure_attribution.py` (MAST
taxonomy and attribution), `economics.py` (token cost accounting),
`magentic.py` (dual-ledger orchestrator with stall detection and replan),
and `agent_card.py` (A2A capability discovery). See
`patterns/multi_agent/README.md` for the full variant list and
`patterns/multi_agent/main.py` for the runnable demo.
"""
