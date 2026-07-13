"""Sub-module: tool-grounded critique, renamed verifier-gated action (CRITIC).

The critique step here calls no LLM at all. It runs a deterministic local
checker against the draft, the way CRITIC (Gou et al., ICLR 2024) grounds
critique in tool output instead of a model's opinion of its own reasoning.
This is the variant that most directly answers Huang et al.'s finding that
ungrounded self-correction does not reliably improve reasoning: the
feedback here is a test result, not an opinion.

Framed as verifier-gated action: the checker's pass or fail result is not
only the stop condition, it is also what authorizes the terminal action
(here, "cleared to merge"). A draft only reaches the gated action once it
actually passes, never on the model's say-so alone.

Scope note: this teaches tool-grounded stopping with a fixed, pre-written
checker, not CRITIC's full contribution. CRITIC's actual novelty is that
the model itself proposes which check or tool call verifies its claim (for
example, proposing a search query to verify a fact), which generalizes past
code to tasks where no test suite can be pre-authored. That
critic-proposes-the-check step is a candidate extension of this module, not
built here to avoid duplicating `tool_use`'s function-calling mechanics.
"""

from __future__ import annotations

import re
from typing import Any

from agentic_patterns import Provider, get_provider

from patterns.reflection.loop import Critique, ReflectionResult, run_reflection_loop
from patterns.reflection.prompting import make_generate, make_refine

_TASK = (
    "Write a Python function `is_palindrome(s)` that returns True if `s` "
    "reads the same forwards and backwards, ignoring letter case."
)

_GENERATOR_SYSTEM = (
    "You write small, correct Python functions. Reply with only a fenced "
    "python code block defining the requested function."
)

_TEST_CASES: list[tuple[str, bool]] = [
    ("racecar", True),
    ("Level", True),
    ("hello", False),
    ("", True),
]


def _extract_code(draft: str) -> str:
    """Strip a ```python fence from a draft, if present."""
    match = re.search(r"```(?:python)?\n(.*?)```", draft, re.DOTALL)
    return match.group(1) if match else draft


def _run_checks(code: str) -> tuple[bool, str]:
    """Execute candidate code and run the fixed test cases against it.

    Stands in for a real test runner or code interpreter tool: this is a
    teaching illustration. Only execute generated code like this inside a
    sandboxed, trusted pipeline; this demo's inputs are fixed scripted
    strings, never untrusted external input.

    Args:
        code: Python source expected to define `is_palindrome`.

    Returns:
        A (passed, detail) pair: whether every test case passed, and a
        message describing the first failure or confirming a full pass.
    """
    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)  # noqa: S102 - fixed scripted demo input only
    except Exception as exc:
        return False, f"code failed to execute: {exc}"

    fn = namespace.get("is_palindrome")
    if fn is None:
        return False, "no function named is_palindrome was defined"

    for arg, expected in _TEST_CASES:
        try:
            actual = fn(arg)
        except Exception as exc:
            return False, f"is_palindrome({arg!r}) raised {exc}"
        if actual != expected:
            return False, f"is_palindrome({arg!r}) returned {actual!r}, expected {expected!r}"

    return True, "all test cases passed"


def _checker_critique(draft: str) -> Critique:
    """Critique callable grounded entirely in `_run_checks`, no model call."""
    passed, detail = _run_checks(_extract_code(draft))
    if passed:
        return Critique(comments=f"verifier: {detail}", score=10.0, approved=True)
    return Critique(comments=f"verifier: {detail}", score=3.0, approved=False)


def _generator_script() -> list[str]:
    """Two turns: a case-sensitive draft, then a fixed, case-insensitive one."""
    draft_1 = "```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```"
    draft_2 = (
        "```python\ndef is_palindrome(s):\n"
        "    normalized = s.lower()\n"
        "    return normalized == normalized[::-1]\n```"
    )
    return [draft_1, draft_2]


def run_tool_grounded_demo(provider: Provider | None = None) -> tuple[ReflectionResult, str]:
    """Run a verifier-gated reflection loop and report the gated action.

    Args:
        provider: Drives generate and refine. Defaults to a `MockProvider`
            scripted with a buggy first draft and a fixed second draft.

    Returns:
        The loop result plus a message describing whether the verifier
        authorized the terminal action.
    """
    if provider is None:
        provider = get_provider(script=_generator_script())
    generate = make_generate(provider, _TASK, system=_GENERATOR_SYSTEM)
    refine = make_refine(provider, _TASK, system=_GENERATOR_SYSTEM)
    result = run_reflection_loop(generate, _checker_critique, refine, max_iterations=3)

    if result.stop_reason == "approved":
        action = "AUTHORIZED: verifier passed, code is cleared to merge."
    else:
        action = "BLOCKED: verifier did not pass within the iteration budget; do not merge."
    return result, action
