"""Human-in-the-loop pattern: pause an agent for a person to decide.

This package implements the human-in-the-loop (approval gate) agentic
pattern and its major sub-variants: the base approval gate (approve, edit,
reject, respond), risk-tiered gating, durable interrupt-and-resume,
escalation on confidence (synchronous and asynchronous), plan review,
post-hoc review with override, and batched review.

See `patterns/human_in_the_loop/README.md` for the full write-up and
`patterns/human_in_the_loop/main.py` for a runnable demo of every variant.
"""
