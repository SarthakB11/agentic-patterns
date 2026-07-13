"""Planning (plan-then-execute): turn a goal into an explicit plan, then run it.

This package holds a shared plan representation (`plan.py`, `parser.py`,
`validator.py`), a small shared tool domain (`tools.py`), and one module per
major planning variant from the research brief: Plan-and-Solve, classic
plan-then-execute, DAG execution with parallel dispatch, replanning, ReWOO,
a ReAct baseline for contrast, todo-list in-context planning, context
offload, and subagent delegation. Run `python -m patterns.planning.main`
for a guided tour of all of them.
"""
