"""Evaluation loops: eval sets, exact and LLM-judge scorers, and regression gates.

An evaluation loop turns "did this change make the system better or worse?"
into a repeatable, automatable answer: a versioned eval set of cases, one or
more scorers that grade a candidate output per case, and a regression gate
that compares a run against a baseline and returns pass or fail.
"""
