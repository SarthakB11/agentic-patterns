"""Regression gate: compare a candidate run against a stored baseline.

The operational wrapper around everything else in this pattern: run the
exact evaluators over the eval set, aggregate to one metric, and compare it
to a baseline from a prior run. A tolerance band absorbs ordinary noise so
the gate does not flap on a change that did not actually regress anything;
only a drop past the tolerance band fails the gate. This models the CI
check a real pipeline would run on every change, including the non-zero
exit code a shell pipeline checks.

This gate is deliberately scoped to the deterministic exact-evaluator suite
(`regex` and `json_schema`) with one global threshold. A real pipeline
would also gate judge metrics (mean score, pass rate, an accepted judge's
verdicts) and gate per slice, using `EvalCase.tags`, so a gain on easy
cases cannot hide a regression on a hard slice.
"""

from __future__ import annotations

from dataclasses import dataclass

from patterns.evaluation.aggregate import pass_rate
from patterns.evaluation.eval_set import get_case
from patterns.evaluation.exact import json_schema_evaluator, regex_match_evaluator

DEFAULT_TOLERANCE = 0.02

# The stored baseline: the exact-evaluator pass rate recorded on a prior,
# known-good run. A real pipeline would load this from a committed file or
# a metrics store; it is a literal here for a self-contained demo.
BASELINE_METRICS: dict[str, float] = {"exact_pass_rate": 1.0}


@dataclass
class RegressionResult:
    """The outcome of comparing a candidate metric to its baseline.

    Attributes:
        metric_name: Name of the metric being gated, e.g. "exact_pass_rate".
        candidate_metric: The metric's value on the run under test.
        baseline_metric: The metric's stored value from a prior run.
        tolerance: How far below the baseline the candidate may fall and
            still pass.
        passed: True if `candidate_metric >= baseline_metric - tolerance`.
        delta: `candidate_metric - baseline_metric`, negative on a drop.
    """

    metric_name: str
    candidate_metric: float
    baseline_metric: float
    tolerance: float
    passed: bool
    delta: float


def evaluate_regression(
    candidate_metric: float,
    baseline_metric: float,
    *,
    metric_name: str = "metric",
    tolerance: float = DEFAULT_TOLERANCE,
) -> RegressionResult:
    """Compare one candidate metric to its baseline within a tolerance band.

    Args:
        candidate_metric: The metric's value on the run under test.
        baseline_metric: The metric's stored value from a prior run.
        metric_name: Label for the metric, carried into the result.
        tolerance: How far below the baseline the candidate may fall and
            still pass.
    """
    delta = candidate_metric - baseline_metric
    passed = candidate_metric >= baseline_metric - tolerance
    return RegressionResult(
        metric_name=metric_name,
        candidate_metric=candidate_metric,
        baseline_metric=baseline_metric,
        tolerance=tolerance,
        passed=passed,
        delta=delta,
    )


def exit_code(result: RegressionResult) -> int:
    """Return the shell exit code a CI job would use for `result`: 0 or 1."""
    return 0 if result.passed else 1


def _run_exact_suite(candidate_outputs: dict[str, str]) -> float:
    """Run the exact evaluators over the eval set and return their pass rate."""
    lookup_case = get_case("order_status_lookup")
    extraction_case = get_case("order_extraction")
    results = [
        regex_match_evaluator(lookup_case, candidate_outputs["order_status_lookup"]).passed,
        json_schema_evaluator(extraction_case, candidate_outputs["order_extraction"]).passed,
    ]
    return pass_rate(results)


def run_regression_demo(candidate_outputs: dict[str, str] | None = None) -> RegressionResult:
    """Run the exact-evaluator suite and gate it against the stored baseline.

    Args:
        candidate_outputs: Mapping from case id to candidate output, for the
            two exact-evaluator cases. Defaults to outputs that pass both
            checks, matching the baseline exactly.
    """
    if candidate_outputs is None:
        candidate_outputs = {
            "order_status_lookup": "Order 48213 is currently out for delivery.",
            "order_extraction": '{"order_id": "48213", "status": "shipped"}',
        }
    candidate_metric = _run_exact_suite(candidate_outputs)
    return evaluate_regression(
        candidate_metric, BASELINE_METRICS["exact_pass_rate"], metric_name="exact_pass_rate"
    )


def run_regression_failing_demo() -> RegressionResult:
    """Run the gate against candidate outputs that break the extraction case.

    The extraction output drops the required `status` key, so the exact
    pass rate falls to 0.5, well past the default 0.02 tolerance below the
    1.0 baseline, and the gate fails.
    """
    candidate_outputs = {
        "order_status_lookup": "Order 48213 is currently out for delivery.",
        "order_extraction": '{"order_id": "48213"}',
    }
    return run_regression_demo(candidate_outputs)
