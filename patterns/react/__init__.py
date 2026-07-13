"""ReAct: interleaved reasoning and acting.

This package implements the ReAct control pattern: at each step a model
emits a Thought, then an Action (a tool call), and the runtime returns an
Observation; the loop repeats until the model emits a terminal Finish or a
stop condition fires. See `patterns/react/main.py` for the runnable demo and
`README.md` for the full write-up.
"""
