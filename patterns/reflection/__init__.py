"""Reflection pattern: generate, critique, refine, repeat.

This package implements the reflection (self-critique) agentic pattern and
its major sub-variants: single-model self-refinement, generator/critic
separation, rubric-based score-gated stopping, tool-grounded (verifier-gated)
critique, memory-augmented (Reflexion-style) reflection across attempts,
parallel specialist critics with an aggregation policy, self-consistent
judging of one noisy critic, a pre-critique revision gate plus a
diminishing-returns stop, and native reasoning self-critique benchmarked
against the explicit loop.

See `patterns/reflection/README.md` for the full write-up and
`patterns/reflection/main.py` for a runnable demo of every variant.
"""
