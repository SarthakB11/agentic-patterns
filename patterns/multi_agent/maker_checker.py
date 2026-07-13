"""Maker-checker / generator-critic loop, as a two-agent pair with explicit pass/fail.

One agent (the maker) produces work; a second agent (the checker) evaluates
it against acceptance criteria and returns pass or fail plus feedback. The
maker revises on feedback until the checker approves or an attempt cap is
hit. This differs from the single-agent reflection pattern elsewhere in the
repo in that the checker is a distinct agent with its own scoped role
(reviewer, not author), matching the brief's requirement for "explicit pass
or fail criteria and a fallback when the cap is reached."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, get_provider

MAKER_SYSTEM = "You produce work for a task and revise it based on checker feedback."
CHECKER_SYSTEM = (
    "You review work against the task's acceptance criteria. Reply with a 'RESULT: PASS' or "
    "'RESULT: FAIL' line, then a 'FEEDBACK:' line explaining why."
)


@dataclass
class CheckResult:
    """A checker's verdict on one maker attempt.

    Attributes:
        passed: True if the checker's RESULT line was PASS.
        feedback: The checker's FEEDBACK line, fed back to the maker on the
            next attempt.
    """

    passed: bool
    feedback: str


@dataclass
class MakerCheckerResult:
    """The outcome of a maker-checker run.

    Attributes:
        attempts: Every output the maker produced, in order.
        checks: The checker's verdict on each attempt, same order.
        approved: True if the checker passed the final attempt.
        final_output: The approved attempt, or the fallback value if the
            cap was reached without approval.
        stop_reason: "approved" or "cap_reached".
    """

    attempts: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    approved: bool = False
    final_output: str = ""
    stop_reason: str = "cap_reached"


def _run_check(checker: Provider, task: str, output: str) -> CheckResult:
    """Ask the checker to evaluate one maker attempt and parse its verdict."""
    reply = checker.complete(
        [Message.user(f"Task: {task}\n\nSubmitted work:\n{output}")], system=CHECKER_SYSTEM
    ).content
    passed = any(line.strip().upper() == "RESULT: PASS" for line in reply.splitlines())
    feedback_line = next((line for line in reply.splitlines() if line.upper().startswith("FEEDBACK:")), "")
    feedback = feedback_line.split(":", 1)[1].strip() if feedback_line else reply.strip()
    return CheckResult(passed=passed, feedback=feedback)


def run_maker_checker(
    maker: Provider,
    checker: Provider,
    task: str,
    *,
    max_attempts: int = 3,
    fallback: str | None = None,
) -> MakerCheckerResult:
    """Run the maker-checker loop until approval or `max_attempts` is reached.

    Args:
        maker: Provider for the maker agent, scripted with one attempt per
            round it is expected to produce.
        checker: Provider for the checker agent, scripted with one verdict
            per attempt it reviews.
        task: The task description and acceptance criteria, given to both
            agents.
        max_attempts: Hard cap on maker attempts.
        fallback: Value returned as `final_output` when the cap is reached
            without approval. Defaults to a fixed escalation message rather
            than silently returning the last (rejected) attempt, so a
            caller cannot mistake an unapproved draft for a passing one.
    """
    attempts: list[str] = []
    checks: list[CheckResult] = []
    feedback = ""

    for attempt_index in range(1, max_attempts + 1):
        prompt = task if attempt_index == 1 else f"{task}\n\nChecker feedback on your previous attempt:\n{feedback}\nRevise your work."
        output = maker.complete([Message.user(prompt)], system=MAKER_SYSTEM).content
        attempts.append(output)

        check = _run_check(checker, task, output)
        checks.append(check)
        if check.passed:
            return MakerCheckerResult(attempts, checks, True, output, "approved")
        feedback = check.feedback

    final = fallback if fallback is not None else "Escalated for manual review: automatic revision did not converge within budget."
    return MakerCheckerResult(attempts, checks, False, final, "cap_reached")


# --- demos -------------------------------------------------------------


def run_maker_checker_demo() -> MakerCheckerResult:
    """A checker fails a SQL query twice, then approves the third revision.

    The first attempt omits the date filter, the second uses non-portable
    date syntax without stating the assumption, and the third fixes both
    and is approved. This should run exactly three maker turns.
    """
    task = (
        "Write a SQL query that returns each customer's total spend over the last 30 days, "
        "using the orders table (customer_id, amount, created_at)."
    )
    maker = get_provider(
        script=[
            "SELECT customer_id, SUM(amount) AS total_spend FROM orders GROUP BY customer_id;",
            "SELECT customer_id, SUM(amount) AS total_spend FROM orders "
            "WHERE created_at >= NOW() - INTERVAL 30 DAY GROUP BY customer_id;",
            "SELECT customer_id, SUM(amount) AS total_spend FROM orders "
            "WHERE created_at >= CURRENT_DATE - 30 GROUP BY customer_id; "
            "-- assumes Postgres date arithmetic",
        ]
    )
    checker = get_provider(
        script=[
            "RESULT: FAIL\nFEEDBACK: Missing a WHERE clause filtering to the last 30 days; this sums all-time spend.",
            "RESULT: FAIL\nFEEDBACK: NOW() - INTERVAL 30 DAY is MySQL-specific syntax; use a portable expression "
            "or state the database assumption explicitly.",
            "RESULT: PASS\nFEEDBACK: Filters to the last 30 days, aggregates per customer, and states its date "
            "assumption. Approved.",
        ]
    )
    return run_maker_checker(maker, checker, task, max_attempts=3)


def run_cap_demo() -> MakerCheckerResult:
    """A checker never passes within a two-attempt budget; the loop returns the fallback.

    Demonstrates the defined fallback behavior: the cap is reached with the
    checker still failing the work, so the result carries the fallback
    escalation message rather than the last, still-rejected attempt.
    """
    task = "Write a one-sentence privacy summary for a new location-sharing feature that legal can sign off on."
    maker = get_provider(
        script=[
            "We use your location to show nearby stores.",
            "We use your location to show nearby stores and share it with our analytics vendor.",
        ]
    )
    checker = get_provider(
        script=[
            "RESULT: FAIL\nFEEDBACK: Doesn't disclose retention or third-party sharing; legal cannot sign off on this.",
            "RESULT: FAIL\nFEEDBACK: Still missing a retention period and an opt-out mention.",
        ]
    )
    return run_maker_checker(
        maker,
        checker,
        task,
        max_attempts=2,
        fallback="Escalated to legal for manual review: automatic drafting did not reach a compliant summary within budget.",
    )
