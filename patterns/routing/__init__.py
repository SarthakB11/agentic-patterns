"""Routing pattern: classify an input, then dispatch it to a specialized handler.

This package implements the routing agentic pattern and its major
sub-variants: rule-based routing, semantic (embedding-similarity) routing,
LLM-classifier routing, cost/quality cascade and capability model
selection, fallback / resilience routing, human-escalation routing,
reasoning-mode routing, handoff-style (transfer-of-control) routing, a
router benchmark against random/oracle baselines, a continuous-score
threshold sweep, a model-judge verified cascade with abstention, and a
route-stability-under-perturbation check.

`registry.py` holds the shared route registry and routing metadata every
variant reads or returns; `transcript.py` renders a `RouteDecision`
readably. See `patterns/routing/README.md` for the full write-up and
`patterns/routing/main.py` for a runnable demo of every variant.
"""
