"""Measured benchmarks that run the pattern code against a real model.

Every benchmark reuses the pattern implementations under `patterns/` and the
core `Provider` abstraction, pointed at a real API (Gemini through its
OpenAI-compatible endpoint) instead of the scripted mock. The goal is one
honest number per pattern: does the technique this repo implements actually
change the outcome on a task with real ground truth.

Cost is treated as a hard constraint. See `harness.py` for the disk cache,
the per-run budget ceiling, and the free mock dry-run path that debugs a
benchmark before it spends anything.
"""
