"""Benchmark: reflection on code generation, graded by real local unit tests.

Two variants generate a Python function from a natural-language spec and are
graded by running author-written unit tests against the candidate code in a
sandboxed subprocess. `one_shot` asks the model once and grades that answer.
`reflection` reuses `patterns.reflection.loop.run_reflection_loop`: it grades
the first draft, and if any test fails, feeds the failing cases back to the
model as a critique and asks for a fix, up to 2 refine rounds, re-grading
after each round. The critic here is the test runner itself (tool-grounded
critique, the same idea as `patterns/reflection/tool_grounded.py`), not a
second model call.

Ground truth: 14 small pure-function tasks, each with hand-authored
(args -> expected) cases the module author independently verified by
writing a standalone reference implementation per task and executing every
case against it before this file was written (129 cases total; see the
task list below). Every task and every expected value is stated in this
file, so the number is checkable line by line.

Difficulty: two earlier task sets (12, then 16 tasks) were both solved
perfectly on the first try by a small model (gemini-3.1-flash-lite), so
one_shot and reflection both scored 1.000 and reflection had nothing to
demonstrate. This set replaces every task with a harder, well-known problem
that has a documented history of tripping first attempts on a subtle case:
a minimal wildcard matcher (the "*" backtracking case), Levenshtein edit
distance, longest palindromic substring with an explicit earliest-tie
rule, a full float/int validator (leading/trailing whitespace, bare sign,
malformed exponent), an integer expression evaluator with real operator
precedence and parentheses, digit-string decoding with leading-zero rules,
2D spiral traversal on non-square matrices, integer-to-English-words with
teen and zero-group edge cases, next lexicographic permutation with the
descending-wraps-to-ascending case, case-sensitive run-length encoding,
word segmentation against a dictionary, Unix path simplification with
repeated slashes and root-level "..", C's `atoi` with 32-bit clamping, and
fraction-to-decimal with recurring-digit parenthesization. None of these
are tuned to make any one variant win; they are picked for edge cases a
fast/cheap model tends to skip on a first pass and then graded by the same
strict unit tests regardless of which variant produced the draft.

Safety: candidate code never runs in this process. Each grading pass writes
the candidate function plus a small test runner to a temp file and executes
it with `subprocess.run`, a 5 second timeout, and no shell, so a candidate
that hangs, imports something wild, or crashes cannot affect the benchmark
process. This is sandboxing hygiene for LLM-authored code, not a claim that
these specific pure-function tasks pose real risk.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_patterns import Provider
from benchmarks.harness import BenchProvider, BenchResult, finalize, live_provider, mock_provider
from patterns.reflection.loop import Critique, run_reflection_loop
from patterns.reflection.prompting import make_generate, make_refine

# Distinct system prompts per variant (legitimate difference: the reflection
# generator is told a fix pass may follow) also keep their first-draft cache
# keys from colliding, since `BenchProvider`'s disk cache is keyed on the
# full request including `system`. Two variants asking the identical prompt
# would otherwise collide and one would silently reuse the other's cached
# completion instead of consuming its own script/live turn.
_ONE_SHOT_SYSTEM = (
    "You write small, correct Python functions. Reply with only a fenced "
    "python code block defining the requested function. No explanation."
)
_REFLECTION_SYSTEM = (
    "You write small, correct Python functions in an iterative loop where a "
    "test runner may report failures for you to fix. Reply with only a "
    "fenced python code block defining the requested function. No explanation."
)

_TIMEOUT_S = 5.0
_MAX_REFINE_ROUNDS = 2


@dataclass(frozen=True)
class CodeTask:
    """One code-generation task with hand-authored ground truth.

    Attributes:
        task_id: Short unique identifier.
        spec: Natural-language spec given to the model as the task prompt.
        fn_name: Name of the function the spec asks for.
        cases: (args, expected) pairs. `args` is passed positionally.
    """

    task_id: str
    spec: str
    fn_name: str
    cases: list[tuple[tuple[Any, ...], Any]]


TASKS: list[CodeTask] = [
    CodeTask(
        "wildcard_match",
        "Write `wildcard_match(pattern: str, text: str) -> bool` implementing "
        "shell-style wildcard matching of the full `text` against `pattern`, "
        "where '?' in `pattern` matches any single character and '*' matches "
        "any sequence of characters, including the empty sequence. The match "
        "must cover the entire `text`, not just a prefix.",
        "wildcard_match",
        [
            (("a", "aa"), False),
            (("aa", "aa"), True),
            (("*", "aa"), True),
            (("a*", "aa"), True),
            (("?*", "ab"), True),
            (("?a", "cb"), False),
            (("*", ""), True),
            (("", "a"), False),
            (("*a*b*", "adceb"), True),
            (("a*c?b", "acdcb"), False),
            (("*abc???de*", "abcabczzzde"), True),
        ],
    ),
    CodeTask(
        "edit_distance",
        "Write `edit_distance(a: str, b: str) -> int` that returns the "
        "Levenshtein distance between `a` and `b`: the minimum number of "
        "single-character insertions, deletions, or substitutions needed to "
        "turn `a` into `b`.",
        "edit_distance",
        [
            (("horse", "ros"), 3),
            (("intention", "execution"), 5),
            (("", ""), 0),
            (("", "abc"), 3),
            (("abc", ""), 3),
            (("abc", "abc"), 0),
            (("kitten", "sitting"), 3),
            (("a", "b"), 1),
        ],
    ),
    CodeTask(
        "longest_palindromic_substring",
        "Write `longest_palindromic_substring(s: str) -> str` that returns "
        "the longest contiguous substring of `s` that reads the same "
        "forwards and backwards. If there are multiple substrings of the "
        "maximum length, return the one that starts earliest in `s`. An "
        "empty `s` returns an empty string.",
        "longest_palindromic_substring",
        [
            (("babad",), "bab"),
            (("cbbd",), "bb"),
            (("a",), "a"),
            (("",), ""),
            (("ac",), "a"),
            (("racecar",), "racecar"),
            (("aacabdkacaa",), "aca"),
            (("abb",), "bb"),
        ],
    ),
    CodeTask(
        "is_valid_number",
        "Write `is_valid_number(s: str) -> bool` that returns True if `s` is "
        "a valid representation of an integer or floating-point number, "
        "else False. A valid number has an optional leading '+' or '-' "
        "sign, then either digits, a decimal point with digits on at least "
        "one side (so '.1', '3.', and '3.1' are all valid but '.' alone is "
        "not), and an optional exponent part starting with 'e' or 'E' "
        "followed by an optional sign and at least one digit. No other "
        "characters, including leading or trailing whitespace, are allowed "
        "anywhere in `s`.",
        "is_valid_number",
        [
            (("0",), True),
            (("  0.1 ",), False),
            (("abc",), False),
            (("2e10",), True),
            (("-90E3",), True),
            (("+6e-1",), True),
            (("1a",), False),
            (("--6",), False),
            ((".1",), True),
            (("3.",), True),
            ((".",), False),
            (("4e+",), False),
        ],
    ),
    CodeTask(
        "basic_calculator",
        "Write `basic_calculator(s: str) -> int` that evaluates an "
        "arithmetic expression string of non-negative integers, the "
        "operators + - * /, parentheses, and spaces (which may appear "
        "anywhere and should be ignored), with standard operator "
        "precedence (* and / bind tighter than + and -), standard "
        "left-to-right associativity, and integer division that truncates "
        "toward zero.",
        "basic_calculator",
        [
            (("1 + 1",), 2),
            (("2-1 + 2",), 3),
            (("(1+(4+5+2)-3)+(6+8)",), 23),
            (("3+2*2",), 7),
            ((" 3/2 ",), 1),
            (("3+5 / 2",), 5),
            (("14-3/2",), 13),
            (("0-2*3",), -6),
            (("1*2-3/4+5*6",), 32),
            (("(1)",), 1),
            (("2*(5-3)",), 4),
        ],
    ),
    CodeTask(
        "decode_ways",
        "Write `decode_ways(s: str) -> int` that counts the number of ways "
        "to decode a string of digits `s` into letters, where 'A' maps to "
        "'1' through 'Z' maps to '26'. A decoding is invalid if any group "
        "of digits it uses starts with '0' (since no letter maps to '0' or "
        "to a two-digit code above '26'). An empty string or a string "
        "starting with '0' decodes zero ways.",
        "decode_ways",
        [
            (("12",), 2),
            (("226",), 3),
            (("0",), 0),
            (("06",), 0),
            (("10",), 1),
            (("27",), 1),
            (("100",), 0),
            (("11106",), 2),
            (("",), 0),
            (("1",), 1),
            (("30",), 0),
        ],
    ),
    CodeTask(
        "spiral_order",
        "Write `spiral_order(matrix: list[list[int]]) -> list[int]` that "
        "returns all elements of the 2D list `matrix` in clockwise spiral "
        "order, starting from the top-left corner. `matrix` may be empty or "
        "non-square (more rows than columns or vice versa).",
        "spiral_order",
        [
            (([[1, 2, 3], [4, 5, 6], [7, 8, 9]],), [1, 2, 3, 6, 9, 8, 7, 4, 5]),
            (
                ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],),
                [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7],
            ),
            (([[1]],), [1]),
            (([[1, 2, 3]],), [1, 2, 3]),
            (([[1], [2], [3]],), [1, 2, 3]),
            (([],), []),
            (([[1, 2], [3, 4]],), [1, 2, 4, 3]),
        ],
    ),
    CodeTask(
        "integer_to_words",
        "Write `integer_to_words(n: int) -> str` that converts an integer "
        "`n` from 0 to 2147483647 inclusive into its English words "
        "representation, using space-separated capitalized words (e.g. "
        "123 -> \"One Hundred Twenty Three\", 0 -> \"Zero\"), with no 'and' "
        "and no leading/trailing/double spaces, and no output for a zero "
        "group within a larger number (e.g. 1000000 -> \"One Million\", not "
        "\"One Million Zero Thousand\").",
        "integer_to_words",
        [
            ((0,), "Zero"),
            ((123,), "One Hundred Twenty Three"),
            ((12345,), "Twelve Thousand Three Hundred Forty Five"),
            (
                (1234567,),
                "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven",
            ),
            ((1000000000,), "One Billion"),
            ((100,), "One Hundred"),
            ((10,), "Ten"),
            ((13,), "Thirteen"),
            ((1000,), "One Thousand"),
            ((20,), "Twenty"),
            ((999999,), "Nine Hundred Ninety Nine Thousand Nine Hundred Ninety Nine"),
            ((500000,), "Five Hundred Thousand"),
        ],
    ),
    CodeTask(
        "next_permutation",
        "Write `next_permutation(nums: list[int]) -> list[int]` that "
        "returns a new list containing the lexicographically next greater "
        "permutation of `nums`'s elements, without modifying `nums` "
        "itself. If no greater permutation exists (the elements are in "
        "fully descending order), return the lowest possible order (fully "
        "ascending) instead. Duplicates in `nums` are allowed.",
        "next_permutation",
        [
            (([1, 2, 3],), [1, 3, 2]),
            (([3, 2, 1],), [1, 2, 3]),
            (([1, 1, 5],), [1, 5, 1]),
            (([1],), [1]),
            (([1, 5, 1],), [5, 1, 1]),
            (([2, 3, 1],), [3, 1, 2]),
            (([1, 3, 2],), [2, 1, 3]),
        ],
    ),
    CodeTask(
        "rle_compress",
        "Write `rle_compress(s: str) -> str` that run-length encodes `s` "
        "using the format \"<char><count>\" for each maximal run of a "
        "repeated character (character first, then its run length as a "
        "decimal number, e.g. \"aaabbc\" -> \"a3b2c1\"), where every run, "
        "including a single occurrence, gets an explicit count. Character "
        "comparison is case-sensitive, so \"aA\" is two separate runs, not "
        "one. An empty input returns an empty string.",
        "rle_compress",
        [
            (("aaabbc",), "a3b2c1"),
            (("abc",), "a1b1c1"),
            (("",), ""),
            (("aaaa",), "a4"),
            (("aAAa",), "a1A2a1"),
            (("a",), "a1"),
            (("a1a1",), "a111a111"),
        ],
    ),
    CodeTask(
        "word_break",
        "Write `word_break(s: str, words: list[str]) -> bool` that returns "
        "True if `s` can be segmented into a space-free sequence of one or "
        "more words drawn from `words`, where the same word may be reused "
        "any number of times. An empty `s` is trivially True.",
        "word_break",
        [
            (("leetcode", ["leet", "code"]), True),
            (("applepenapple", ["apple", "pen"]), True),
            (("catsandog", ["cats", "dog", "sand", "and", "cat"]), False),
            (("", ["a"]), True),
            (("a", []), False),
            (("aaaaaaa", ["aaaa", "aaa"]), True),
            (("cars", ["car", "ca", "rs"]), True),
        ],
    ),
    CodeTask(
        "simplify_unix_path",
        "Write `simplify_unix_path(path: str) -> str` that converts an "
        "absolute Unix-style file path (starting with '/') into its "
        "simplified canonical form: collapse repeated slashes, resolve "
        "single-dot segments ('.') by dropping them, resolve double-dot "
        "segments ('..') by popping the previous directory (or being a "
        "no-op at the root if there is nothing to pop), and drop any "
        "trailing slash. The result always starts with a single '/' and "
        "never ends with a trailing '/' unless the whole result is the "
        "root \"/\".",
        "simplify_unix_path",
        [
            (("/home/",), "/home"),
            (("/../",), "/"),
            (("/home//foo/",), "/home/foo"),
            (("/a/./b/../../c/",), "/c"),
            (("/a/../../b/../c//.//",), "/c"),
            (("/a//b////c/d//././/..",), "/a/b/c"),
            (("/",), "/"),
            (("/...",), "/..."),
        ],
    ),
    CodeTask(
        "atoi",
        "Write `atoi(s: str) -> int` implementing C's `atoi`: skip any "
        "leading spaces, then read an optional single '+' or '-' sign, "
        "then read as many following digits as form a contiguous run "
        "(stopping at the first non-digit), and convert that to an "
        "integer. If no digits are found after an optional sign, return 0. "
        "Clamp the result to the 32-bit signed integer range "
        "[-2147483648, 2147483647]: values outside that range saturate to "
        "the nearest bound instead of overflowing.",
        "atoi",
        [
            (("42",), 42),
            (("   -42",), -42),
            (("4193 with words",), 4193),
            (("words and 987",), 0),
            (("-91283472332",), -2147483648),
            (("91283472332",), 2147483647),
            (("+1",), 1),
            (("",), 0),
            (("+-2",), 0),
            (("2147483648",), 2147483647),
        ],
    ),
    CodeTask(
        "fraction_to_decimal",
        "Write `fraction_to_decimal(numerator: int, denominator: int) -> "
        "str` that returns the decimal string representation of "
        "`numerator / denominator`. If the fractional part is repeating, "
        "wrap only the first repeating block in parentheses (e.g. 1/6 -> "
        "\"0.1(6)\", 2/3 -> \"0.(6)\"). If there is no fractional part, "
        "return just the integer (e.g. 2/1 -> \"2\"). The overall sign is "
        "negative if exactly one of `numerator` or `denominator` is "
        "negative; `denominator` is never zero.",
        "fraction_to_decimal",
        [
            ((1, 2), "0.5"),
            ((2, 1), "2"),
            ((2, 3), "0.(6)"),
            ((4, 333), "0.(012)"),
            ((1, 5), "0.2"),
            ((-50, 8), "-6.25"),
            ((0, 5), "0"),
            ((1, 6), "0.1(6)"),
            ((-1, -2), "0.5"),
            ((7, -12), "-0.58(3)"),
        ],
    ),
]


def _extract_code(draft: str) -> str:
    """Strip a ```python fence from a draft, returning the raw draft if unfenced."""
    if "```" not in draft:
        return draft
    parts = draft.split("```")
    for part in parts[1::2]:
        return part[6:] if part.startswith("python\n") else part
    return draft


def _run_tests_in_subprocess(code: str, task: CodeTask) -> tuple[bool, list[str]]:
    """Run `task`'s cases against `code` in an isolated subprocess.

    Args:
        code: Candidate Python source, expected to define `task.fn_name`.
        task: The task whose cases grade the candidate.

    Returns:
        A (all_passed, failures) pair. `failures` holds one human-readable
        line per failing or erroring case, empty when all pass.
    """
    harness = (
        f"{code}\n\n"
        "import json, sys\n"
        f"_fn = {task.fn_name}\n"
        f"_cases = {task.cases!r}\n"
        "_failures = []\n"
        "for _args, _expected in _cases:\n"
        "    try:\n"
        "        _actual = _fn(*_args)\n"
        "    except Exception as exc:\n"
        "        _failures.append(f'{_args!r} raised {exc!r}')\n"
        "        continue\n"
        "    if _actual != _expected:\n"
        "        _failures.append(f'{_args!r} -> {_actual!r}, expected {_expected!r}')\n"
        "print(json.dumps(_failures))\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "candidate.py"
        script.write_text(harness)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return False, ["timed out"]
    if proc.returncode != 0:
        stderr_tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        return False, [f"process error (exit {proc.returncode}): {stderr_tail}"]
    try:
        import json as _json

        failures: list[str] = _json.loads(proc.stdout.strip() or "[]")
    except ValueError:
        return False, [f"could not parse test output: {proc.stdout!r}"]
    return not failures, failures


def _grade(draft: str, task: CodeTask) -> Critique:
    """Tool-grounded critique callable: the test runner is the critic."""
    passed, failures = _run_tests_in_subprocess(_extract_code(draft), task)
    if passed:
        return Critique(comments="all tests passed", score=10.0, approved=True)
    comments = "Failing test cases:\n" + "\n".join(f"- {task.fn_name}{f}" for f in failures)
    return Critique(comments=comments, score=0.0, approved=False)


def _run_one_shot(provider: Provider, task: CodeTask) -> tuple[bool, int]:
    """Generate once and grade it. Returns (passed, rounds_used=0)."""
    generate = make_generate(provider, task.spec, system=_ONE_SHOT_SYSTEM)
    draft = generate()
    passed, _ = _run_tests_in_subprocess(_extract_code(draft), task)
    return passed, 0


def _run_reflection(provider: Provider, task: CodeTask) -> tuple[bool, int]:
    """Generate, test, and refine up to `_MAX_REFINE_ROUNDS` times.

    Reuses `run_reflection_loop` with the local test runner as the critic
    (no second model call for critique), matching the tool-grounded variant
    in `patterns/reflection/tool_grounded.py`.

    Returns:
        (passed, refine_rounds_used). `refine_rounds_used` is the number of
        critique rounds that ran beyond the initial draft, i.e. how many
        refine calls the model produced.
    """
    generate = make_generate(provider, task.spec, system=_REFLECTION_SYSTEM)
    refine = make_refine(provider, task.spec, system=_REFLECTION_SYSTEM)
    critique = lambda draft: _grade(draft, task)  # noqa: E731
    result = run_reflection_loop(
        generate,
        critique,
        refine,
        max_iterations=_MAX_REFINE_ROUNDS + 1,
    )
    passed = result.stop_reason == "approved"
    # Every critique round is followed by a refine call except the one that
    # stops the loop, so refine calls issued = critique rounds run - 1.
    refine_calls = max(0, len(result.iterations) - 1)
    return passed, refine_calls


def _run(provider: BenchProvider) -> BenchResult:
    """Core benchmark logic shared by `run_mock` and `run_live`.

    Each variant gets its own `provider.complete()` call sequence, matching
    the scripted turns for `mock_provider` and simply issuing real calls for
    `live_provider`.
    """
    variant_results: dict[str, list[bool]] = {"one_shot": [], "reflection": []}
    refine_rounds: list[int] = []
    task_rows: list[dict[str, Any]] = []

    for task in TASKS:
        one_shot_passed, _ = _run_one_shot(provider, task)
        reflection_passed, rounds = _run_reflection(provider, task)
        variant_results["one_shot"].append(one_shot_passed)
        variant_results["reflection"].append(reflection_passed)
        refine_rounds.append(rounds)
        task_rows.append(
            {
                "id": task.task_id,
                "one_shot_passed": one_shot_passed,
                "reflection_passed": reflection_passed,
                "refine_rounds": rounds,
            }
        )

    n = len(TASKS)
    pass_rates = {variant: sum(outcomes) / n for variant, outcomes in variant_results.items()}
    mean_rounds = sum(refine_rounds) / n

    headline = (
        f"reflection passed {pass_rates['reflection']:.0%} of {n} code tasks vs "
        f"{pass_rates['one_shot']:.0%} for one_shot, using a mean of {mean_rounds:.2f} refine rounds."
    )

    return BenchResult(
        name="bench_reflection",
        model=provider.model,
        n=n,
        variants=pass_rates,
        headline=headline,
        detail={"mean_refine_rounds": mean_rounds, "max_refine_rounds": _MAX_REFINE_ROUNDS},
        tasks=task_rows,
    )


def run_mock() -> BenchResult:
    """Run both variants against a scripted `MockProvider` for a free smoke test.

    Scripts a mix of outcomes so both metrics compute a non-trivial,
    non-crashing pass rate for free: one_shot alternates a passing and a
    failing draft; reflection fixes a failing first draft within one refine
    round for every task except the last, which stays broken through all
    `_MAX_REFINE_ROUNDS` refine attempts to exercise the still-failing path.
    """
    script: list[str] = []
    for idx, task in enumerate(TASKS):
        good = _good_solution(task)
        bad = _bad_solution(task)
        # one_shot call: alternate a passing and a failing draft across tasks.
        script.append(good if idx % 2 == 0 else bad)
        if idx == len(TASKS) - 1:
            # reflection: never recovers, to exercise the still-failing path.
            # Each attempt gets a distinct trailing comment so the drafts
            # are not byte-identical; an identical refinement would trip the
            # loop's own no-change convergence guard before max_iterations.
            script.extend(f"{bad}  # attempt {i}" for i in range(_MAX_REFINE_ROUNDS + 1))
        else:
            # reflection: first draft fails, one refine call fixes it.
            script.append(bad)
            script.append(good)

    provider = mock_provider(script)
    result = _run(provider)
    result.headline = (
        f"[mock] reflection={result.variants['reflection']:.0%} one_shot={result.variants['one_shot']:.0%} "
        f"over n={result.n}, plumbing verified with zero cost."
    )
    return finalize(result, provider)


def run_live() -> BenchResult:
    """Run both variants against a real budgeted, cached Gemini provider."""
    provider = live_provider(model="gemini-3.1-flash-lite", budget_usd=0.5)
    result = _run(provider)
    return finalize(result, provider)


_GOOD_BODIES: dict[str, str] = {
    "wildcard_match": (
        "def wildcard_match(pattern, text):\n"
        " m,n=len(pattern),len(text)\n"
        " dp=[[False]*(n+1) for _ in range(m+1)]\n"
        " dp[0][0]=True\n"
        " for i in range(1,m+1):\n"
        "  if pattern[i-1]=='*':dp[i][0]=dp[i-1][0]\n"
        " for i in range(1,m+1):\n"
        "  for j in range(1,n+1):\n"
        "   pc=pattern[i-1]\n"
        "   if pc=='*':dp[i][j]=dp[i-1][j] or dp[i][j-1]\n"
        "   elif pc=='?' or pc==text[j-1]:dp[i][j]=dp[i-1][j-1]\n"
        "   else:dp[i][j]=False\n"
        " return dp[m][n]"
    ),
    "edit_distance": (
        "def edit_distance(a,b):\n"
        " m,n=len(a),len(b)\n"
        " dp=[[0]*(n+1) for _ in range(m+1)]\n"
        " for i in range(m+1):dp[i][0]=i\n"
        " for j in range(n+1):dp[0][j]=j\n"
        " for i in range(1,m+1):\n"
        "  for j in range(1,n+1):\n"
        "   if a[i-1]==b[j-1]:dp[i][j]=dp[i-1][j-1]\n"
        "   else:dp[i][j]=1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])\n"
        " return dp[m][n]"
    ),
    "longest_palindromic_substring": (
        "def longest_palindromic_substring(s):\n"
        " if not s:return ''\n"
        " start,best_len=0,1\n"
        " def expand(l,r):\n"
        "  while l>=0 and r<len(s) and s[l]==s[r]:l-=1;r+=1\n"
        "  return l+1,r-l-1\n"
        " for i in range(len(s)):\n"
        "  for l,r in ((i,i),(i,i+1)):\n"
        "   st,length=expand(l,r)\n"
        "   if length>best_len:start,best_len=st,length\n"
        " return s[start:start+best_len]"
    ),
    "is_valid_number": (
        "import re\n"
        "def is_valid_number(s):\n"
        " pat=re.compile(r'^[+-]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][+-]?\\d+)?$')\n"
        " return bool(pat.match(s))"
    ),
    "basic_calculator": (
        "def basic_calculator(s):\n"
        " def tokenize(expr):\n"
        "  toks=[];i=0\n"
        "  while i<len(expr):\n"
        "   c=expr[i]\n"
        "   if c.isspace():i+=1;continue\n"
        "   if c.isdigit():\n"
        "    j=i\n"
        "    while j<len(expr) and expr[j].isdigit():j+=1\n"
        "    toks.append(expr[i:j]);i=j\n"
        "   else:toks.append(c);i+=1\n"
        "  return toks\n"
        " tokens=tokenize(s)\n"
        " pos=[0]\n"
        " def peek():\n"
        "  return tokens[pos[0]] if pos[0]<len(tokens) else None\n"
        " def factor():\n"
        "  if peek()=='(':\n"
        "   pos[0]+=1;v=expr();pos[0]+=1\n"
        "   return v\n"
        "  if peek()=='-':\n"
        "   pos[0]+=1;return -factor()\n"
        "  if peek()=='+':\n"
        "   pos[0]+=1;return factor()\n"
        "  v=int(tokens[pos[0]]);pos[0]+=1\n"
        "  return v\n"
        " def term():\n"
        "  v=factor()\n"
        "  while peek() in ('*','/'):\n"
        "   op=tokens[pos[0]];pos[0]+=1\n"
        "   rhs=factor()\n"
        "   if op=='*':v=v*rhs\n"
        "   else:\n"
        "    q=abs(v)//abs(rhs)\n"
        "    v=-q if (v<0)!=(rhs<0) else q\n"
        "  return v\n"
        " def expr():\n"
        "  v=term()\n"
        "  while peek() in ('+','-'):\n"
        "   op=tokens[pos[0]];pos[0]+=1\n"
        "   rhs=term()\n"
        "   v=v+rhs if op=='+' else v-rhs\n"
        "  return v\n"
        " return expr()"
    ),
    "decode_ways": (
        "def decode_ways(s):\n"
        " if not s or s[0]=='0':return 0\n"
        " n=len(s)\n"
        " dp=[0]*(n+1)\n"
        " dp[0]=1\n"
        " dp[1]=1 if s[0]!='0' else 0\n"
        " for i in range(2,n+1):\n"
        "  one=s[i-1];two=s[i-2:i]\n"
        "  cnt=0\n"
        "  if one!='0':cnt+=dp[i-1]\n"
        "  if '10'<=two<='26':cnt+=dp[i-2]\n"
        "  dp[i]=cnt\n"
        " return dp[n]"
    ),
    "spiral_order": (
        "def spiral_order(matrix):\n"
        " if not matrix or not matrix[0]:return []\n"
        " result=[]\n"
        " top,bottom=0,len(matrix)-1\n"
        " left,right=0,len(matrix[0])-1\n"
        " while top<=bottom and left<=right:\n"
        "  for c in range(left,right+1):result.append(matrix[top][c])\n"
        "  top+=1\n"
        "  for r in range(top,bottom+1):result.append(matrix[r][right])\n"
        "  right-=1\n"
        "  if top<=bottom:\n"
        "   for c in range(right,left-1,-1):result.append(matrix[bottom][c])\n"
        "   bottom-=1\n"
        "  if left<=right:\n"
        "   for r in range(bottom,top-1,-1):result.append(matrix[r][left])\n"
        "   left+=1\n"
        " return result"
    ),
    "integer_to_words": (
        "_B=['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten',"
        "'Eleven','Twelve','Thirteen','Fourteen','Fifteen','Sixteen','Seventeen','Eighteen','Nineteen']\n"
        "_T=['','','Twenty','Thirty','Forty','Fifty','Sixty','Seventy','Eighty','Ninety']\n"
        "def _three(n):\n"
        " parts=[]\n"
        " if n>=100:parts.append(_B[n//100]);parts.append('Hundred');n%=100\n"
        " if n>=20:\n"
        "  parts.append(_T[n//10])\n"
        "  if n%10:parts.append(_B[n%10])\n"
        " elif n>0:parts.append(_B[n])\n"
        " return ' '.join(parts)\n"
        "def integer_to_words(n):\n"
        " if n==0:return 'Zero'\n"
        " groups=[(1000000000,'Billion'),(1000000,'Million'),(1000,'Thousand'),(1,'')]\n"
        " parts=[]\n"
        " for size,label in groups:\n"
        "  if n>=size:\n"
        "   count=n//size;n%=size\n"
        "   words=_three(count)\n"
        "   parts.append(f'{words} {label}'.strip())\n"
        " return ' '.join(parts)"
    ),
    "next_permutation": (
        "def next_permutation(nums):\n"
        " out=list(nums)\n"
        " n=len(out)\n"
        " i=n-2\n"
        " while i>=0 and out[i]>=out[i+1]:i-=1\n"
        " if i>=0:\n"
        "  j=n-1\n"
        "  while out[j]<=out[i]:j-=1\n"
        "  out[i],out[j]=out[j],out[i]\n"
        " out[i+1:]=reversed(out[i+1:])\n"
        " return out"
    ),
    "rle_compress": (
        "def rle_compress(s):\n"
        " if not s:return ''\n"
        " out,ch,n=[],s[0],1\n"
        " for c in s[1:]:\n"
        "  if c==ch:n+=1\n"
        "  else:out.append(f'{ch}{n}');ch,n=c,1\n"
        " out.append(f'{ch}{n}');return ''.join(out)"
    ),
    "word_break": (
        "def word_break(s,words):\n"
        " word_set=set(words)\n"
        " n=len(s)\n"
        " dp=[False]*(n+1)\n"
        " dp[0]=True\n"
        " for i in range(1,n+1):\n"
        "  for j in range(i):\n"
        "   if dp[j] and s[j:i] in word_set:dp[i]=True;break\n"
        " return dp[n]"
    ),
    "simplify_unix_path": (
        "def simplify_unix_path(path):\n"
        " parts=path.split('/')\n"
        " stack=[]\n"
        " for part in parts:\n"
        "  if part=='' or part=='.':continue\n"
        "  if part=='..':\n"
        "   if stack:stack.pop()\n"
        "  else:stack.append(part)\n"
        " return '/'+'/'.join(stack)"
    ),
    "atoi": (
        "def atoi(s):\n"
        " i,n=0,len(s)\n"
        " while i<n and s[i]==' ':i+=1\n"
        " sign=1\n"
        " if i<n and s[i] in '+-':\n"
        "  if s[i]=='-':sign=-1\n"
        "  i+=1\n"
        " start=i\n"
        " while i<n and s[i].isdigit():i+=1\n"
        " digits=s[start:i]\n"
        " value=int(digits) if digits else 0\n"
        " value*=sign\n"
        " lo,hi=-2**31,2**31-1\n"
        " if value<lo:return lo\n"
        " if value>hi:return hi\n"
        " return value"
    ),
    "fraction_to_decimal": (
        "def fraction_to_decimal(numerator,denominator):\n"
        " if numerator==0:return '0'\n"
        " result=[]\n"
        " if (numerator<0)!=(denominator<0):result.append('-')\n"
        " num,den=abs(numerator),abs(denominator)\n"
        " result.append(str(num//den))\n"
        " remainder=num%den\n"
        " if remainder==0:return ''.join(result)\n"
        " result.append('.')\n"
        " seen={}\n"
        " digits=[]\n"
        " while remainder!=0:\n"
        "  if remainder in seen:\n"
        "   idx=seen[remainder]\n"
        "   digits.insert(idx,'(')\n"
        "   digits.append(')')\n"
        "   break\n"
        "  seen[remainder]=len(digits)\n"
        "  remainder*=10\n"
        "  digits.append(str(remainder//den))\n"
        "  remainder%=den\n"
        " result.append(''.join(digits))\n"
        " return ''.join(result)"
    ),
}

# A shallow, plausible-but-wrong implementation per task (mock-only), each
# missing exactly the requirement the spec calls out as the trap: '*' does
# not match the empty sequence, substitution cost dropped from edit
# distance, only odd-length palindrome centers checked, no anchoring so
# trailing garbage validates, flat left-to-right eval with no operator
# precedence, the leading-zero gate on two-digit decode groups dropped,
# spiral bounds assume a square matrix, teens double-counted as tens plus
# ones, the descending suffix never reversed, case-insensitive run
# grouping, greedy leftmost segmentation with no backtracking, empty path
# segments kept literally, no 32-bit clamp, and no cycle detection so a
# repeating fraction never gets parenthesized.
_BAD_BODIES: dict[str, str] = {
    "wildcard_match": (
        "def wildcard_match(pattern,text):\n"
        " m,n=len(pattern),len(text)\n"
        " dp=[[False]*(n+1) for _ in range(m+1)]\n"
        " dp[0][0]=True\n"
        " for i in range(1,m+1):\n"
        "  for j in range(1,n+1):\n"
        "   pc=pattern[i-1]\n"
        "   if pc=='*':dp[i][j]=dp[i-1][j] or dp[i][j-1]\n"
        "   elif pc=='?' or pc==text[j-1]:dp[i][j]=dp[i-1][j-1]\n"
        "   else:dp[i][j]=False\n"
        " return dp[m][n]"
    ),
    "edit_distance": (
        "def edit_distance(a,b):\n"
        " m,n=len(a),len(b)\n"
        " dp=[[0]*(n+1) for _ in range(m+1)]\n"
        " for i in range(m+1):dp[i][0]=i\n"
        " for j in range(n+1):dp[0][j]=j\n"
        " for i in range(1,m+1):\n"
        "  for j in range(1,n+1):\n"
        "   if a[i-1]==b[j-1]:dp[i][j]=dp[i-1][j-1]\n"
        "   else:dp[i][j]=1+min(dp[i-1][j],dp[i][j-1])\n"
        " return dp[m][n]"
    ),
    "longest_palindromic_substring": (
        "def longest_palindromic_substring(s):\n"
        " if not s:return ''\n"
        " start,best_len=0,1\n"
        " def expand(l,r):\n"
        "  while l>=0 and r<len(s) and s[l]==s[r]:l-=1;r+=1\n"
        "  return l+1,r-l-1\n"
        " for i in range(len(s)):\n"
        "  st,length=expand(i,i)\n"
        "  if length>best_len:start,best_len=st,length\n"
        " return s[start:start+best_len]"
    ),
    "is_valid_number": (
        "def is_valid_number(s):\n"
        " s=s.strip()\n"
        " try:\n"
        "  float(s)\n"
        "  return True\n"
        " except ValueError:\n"
        "  return False"
    ),
    "basic_calculator": (
        "def basic_calculator(s):\n"
        " import re\n"
        " toks=re.findall(r'\\d+|[+\\-*/]', s.replace(' ','').replace('(','').replace(')',''))\n"
        " total=int(toks[0])\n"
        " i=1\n"
        " while i<len(toks):\n"
        "  op=toks[i];val=int(toks[i+1])\n"
        "  if op=='+':total+=val\n"
        "  elif op=='-':total-=val\n"
        "  elif op=='*':total*=val\n"
        "  else:total=int(total/val)\n"
        "  i+=2\n"
        " return total"
    ),
    "decode_ways": (
        "def decode_ways(s):\n"
        " if not s:return 0\n"
        " n=len(s)\n"
        " dp=[0]*(n+1)\n"
        " dp[0]=1\n"
        " dp[1]=1\n"
        " for i in range(2,n+1):\n"
        "  one=s[i-1];two=s[i-2:i]\n"
        "  cnt=0\n"
        "  if one!='0':cnt+=dp[i-1]\n"
        "  if 10<=int(two)<=26:cnt+=dp[i-2]\n"
        "  dp[i]=cnt\n"
        " return dp[n]"
    ),
    "spiral_order": (
        "def spiral_order(matrix):\n"
        " if not matrix:return []\n"
        " result=[]\n"
        " n=len(matrix)\n"
        " top,bottom,left,right=0,n-1,0,n-1\n"
        " while top<=bottom and left<=right:\n"
        "  for c in range(left,right+1):result.append(matrix[top][c])\n"
        "  top+=1\n"
        "  for r in range(top,bottom+1):result.append(matrix[r][right])\n"
        "  right-=1\n"
        "  if top<=bottom:\n"
        "   for c in range(right,left-1,-1):result.append(matrix[bottom][c])\n"
        "   bottom-=1\n"
        "  if left<=right:\n"
        "   for r in range(bottom,top-1,-1):result.append(matrix[r][left])\n"
        "   left+=1\n"
        " return result"
    ),
    "integer_to_words": (
        "_B=['Zero','One','Two','Three','Four','Five','Six','Seven','Eight','Nine']\n"
        "_T=['','','Twenty','Thirty','Forty','Fifty','Sixty','Seventy','Eighty','Ninety']\n"
        "def _three(n):\n"
        " parts=[]\n"
        " if n>=100:parts.append(_B[n//100]);parts.append('Hundred');n%=100\n"
        " if n>=10:\n"
        "  parts.append(_T[n//10])\n"
        "  if n%10:parts.append(_B[n%10])\n"
        " elif n>0:parts.append(_B[n])\n"
        " return ' '.join(parts)\n"
        "def integer_to_words(n):\n"
        " if n==0:return 'Zero'\n"
        " groups=[(1000000000,'Billion'),(1000000,'Million'),(1000,'Thousand'),(1,'')]\n"
        " parts=[]\n"
        " for size,label in groups:\n"
        "  if n>=size:\n"
        "   count=n//size;n%=size\n"
        "   words=_three(count)\n"
        "   parts.append(f'{words} {label}'.strip())\n"
        " return ' '.join(parts)"
    ),
    "next_permutation": (
        "def next_permutation(nums):\n"
        " out=list(nums)\n"
        " n=len(out)\n"
        " i=n-2\n"
        " while i>=0 and out[i]>=out[i+1]:i-=1\n"
        " if i>=0:\n"
        "  j=n-1\n"
        "  while out[j]<=out[i]:j-=1\n"
        "  out[i],out[j]=out[j],out[i]\n"
        " return out"
    ),
    "rle_compress": (
        "def rle_compress(s):\n"
        " if not s:return ''\n"
        " out,ch,n=[],s[0].lower(),1\n"
        " for c in s[1:]:\n"
        "  if c.lower()==ch:n+=1\n"
        "  else:out.append(f'{ch}{n}');ch,n=c.lower(),1\n"
        " out.append(f'{ch}{n}');return ''.join(out)"
    ),
    "word_break": (
        "def word_break(s,words):\n"
        " word_set=set(words)\n"
        " i=0\n"
        " while i<len(s):\n"
        "  for j in range(len(s),i,-1):\n"
        "   if s[i:j] in word_set:\n"
        "    i=j;break\n"
        "  else:\n"
        "   return False\n"
        " return True"
    ),
    "simplify_unix_path": (
        "def simplify_unix_path(path):\n"
        " parts=path.split('/')\n"
        " stack=[]\n"
        " for part in parts:\n"
        "  if part=='.':continue\n"
        "  if part=='..':\n"
        "   if stack:stack.pop()\n"
        "  elif part!='':stack.append(part)\n"
        "  else:stack.append(part)\n"
        " return '/'+'/'.join(p for p in stack if p!='')"
    ),
    "atoi": (
        "def atoi(s):\n"
        " i,n=0,len(s)\n"
        " while i<n and s[i]==' ':i+=1\n"
        " sign=1\n"
        " if i<n and s[i] in '+-':\n"
        "  if s[i]=='-':sign=-1\n"
        "  i+=1\n"
        " start=i\n"
        " while i<n and s[i].isdigit():i+=1\n"
        " digits=s[start:i]\n"
        " value=int(digits) if digits else 0\n"
        " return value*sign"
    ),
    "fraction_to_decimal": (
        "def fraction_to_decimal(numerator,denominator):\n"
        " if numerator==0:return '0'\n"
        " sign='-' if (numerator<0)!=(denominator<0) else ''\n"
        " num,den=abs(numerator),abs(denominator)\n"
        " whole=num//den\n"
        " remainder=num%den\n"
        " if remainder==0:return f'{sign}{whole}'\n"
        " digits=[]\n"
        " for _ in range(20):\n"
        "  remainder*=10\n"
        "  digits.append(str(remainder//den))\n"
        "  remainder%=den\n"
        "  if remainder==0:break\n"
        " return f'{sign}{whole}.{\"\".join(digits)}'"
    ),
}


def _good_solution(task: CodeTask) -> str:
    """A scripted, always-correct implementation for `task`, mock-only."""
    return f"```python\n{_GOOD_BODIES[task.task_id]}\n```"


def _bad_solution(task: CodeTask) -> str:
    """A scripted, plausible-but-wrong implementation for `task`, mock-only."""
    return f"```python\n{_BAD_BODIES[task.task_id]}\n```"


if __name__ == "__main__":
    result = run_mock()
    print(f"bench_reflection [mock] model={result.model} n={result.n}")
    for variant, rate in result.variants.items():
        print(f"  {variant:<12} pass_rate={rate:.2%}")
    print(f"  mean_refine_rounds={result.detail['mean_refine_rounds']:.2f}")
    print(f"  cost_usd={result.usage.get('cost_usd', 0.0)}")
    print(result.headline)
