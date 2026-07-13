"""Evaluation pattern: eval set, scorers, and a regression gate.

An evaluation loop turns "did this change make the system better or worse?"
into a repeatable, automatable answer. It has three parts: a versioned eval
set of input cases, one or more scorers that grade a candidate output per
case (from exact programmatic checks to an LLM acting as a judge), and a
regression gate that aggregates a run's scores and compares them to a
baseline, so CI can block a change that degrades quality.

This demo runs every variant end to end, entirely offline against
`MockProvider` with scripted, coherent conversations, no network call and no
API key:

1. The eval set: five cases spanning exact-checkable and open-ended tasks.
2. Exact evaluators: regex match and JSON-schema validity.
3. Semantic similarity: embedding-distance comparison to a reference.
4. Pointwise LLM judge: reference-based vs reference-free, an
   instruction-specific checklist judge, and a position-order check.
5. Pairwise judge: both orderings run and aggregated to cancel position
   bias, contrasting a fair comparison with a purely position-biased one.
6. Ensemble jury: three independent judges, majority vote.
7. Agent-as-judge trajectory evaluation, contrasted with final-answer-only
   judging on the same shortcut trajectory.
8. Metrics aggregation: mean score, pass rate, pairwise win rate, and an
   Elo-style ranking rolled up from pairwise verdicts.
9. A regression gate: a passing run and a failing run against a stored
   baseline, with the CI exit code each would produce.
10. Meta-evaluation: judge-vs-human agreement (Cohen's kappa) and a
    test-retest same-verdict rate.

Run it from the repository root:

    python -m patterns.evaluation.main

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key in the environment) to run the same code against
a real model instead of the mock. No source change is required; every demo
function builds its provider through `agentic_patterns.get_provider`.
"""

from __future__ import annotations

from patterns.evaluation import aggregate, ensemble, exact, meta, pairwise, pointwise, regression, semantic, trajectory
from patterns.evaluation.eval_set import EVAL_SET, EVAL_SET_VERSION, get_case


def main() -> None:
    """Run every evaluation sub-variant demo and print a readable transcript."""
    print("EVALUATION PATTERN: eval set, scorers, regression gate\n")

    _section_eval_set()
    _section_exact()
    _section_semantic()
    _section_pointwise()
    _section_pairwise()
    _section_ensemble()
    _section_trajectory()
    _section_aggregate()
    _section_regression()
    _section_meta()

    print("All ten sections completed without exhausting their scripts.")


def _section_eval_set() -> None:
    print(f"=== 1. Eval set (version {EVAL_SET_VERSION}) ===")
    for case in EVAL_SET:
        ref = "yes" if case.reference else "no"
        prop = case.expected_property or "-"
        print(f"  [{case.id}] tags={case.tags} reference={ref} expected_property={prop}")
    print()


def _section_exact() -> None:
    print("=== 2. Exact / programmatic evaluators ===")
    lookup_score = exact.regex_match_evaluator(
        get_case("order_status_lookup"), "Order 48213 is currently out for delivery."
    )
    print(f"regex: passed={lookup_score.passed} ({lookup_score.detail})")

    good_extraction = exact.json_schema_evaluator(
        get_case("order_extraction"), '{"order_id": "48213", "status": "shipped"}'
    )
    print(f"json_schema (valid): passed={good_extraction.passed} ({good_extraction.detail})")

    bad_extraction = exact.json_schema_evaluator(get_case("order_extraction"), '{"order_id": "48213"}')
    print(f"json_schema (missing key): passed={bad_extraction.passed} ({bad_extraction.detail})")
    assert good_extraction.passed and not bad_extraction.passed
    print()


def _section_semantic() -> None:
    print("=== 3. Semantic similarity evaluator ===")
    case = get_case("refund_policy")
    paraphrase = "You can get a refund within 30 days if you still have the receipt or the order number."
    off_topic = "Our support hours are 9 to 5 Monday through Friday."
    good = semantic.semantic_similarity_evaluator(case, paraphrase)
    bad = semantic.semantic_similarity_evaluator(case, off_topic)
    print(f"paraphrase: passed={good.passed} ({good.detail})")
    print(f"off-topic:  passed={bad.passed} ({bad.detail})")
    assert good.passed and not bad.passed
    print()


def _section_pointwise() -> None:
    print("=== 4. Pointwise LLM judge (rubric, chain-of-thought) ===")
    ref_based, ref_free = pointwise.run_pointwise_demo()
    print(f"reference-based: score={ref_based.score} passed={ref_based.passed}")
    print(f"reference-free:  score={ref_free.score} passed={ref_free.passed}")

    checklist, checklist_verdict = pointwise.run_checklist_demo()
    first_item = checklist.splitlines()[0]
    print(f"checklist judge: derived {first_item!r} ... score={checklist_verdict.score:.2f}")

    forward, reversed_, bias = pointwise.run_pointwise_order_check_demo()
    print(
        f"order check: forward score={forward.score} reversed score={reversed_.score} "
        f"position_bias_detected={bias}"
    )
    print()


def _section_pairwise() -> None:
    print("=== 5. Pairwise judge (both orderings, position-bias cancellation) ===")
    fair = pairwise.run_pairwise_fair_demo()
    print(f"fair comparison: winner={fair.winner} position_bias_detected={fair.position_bias_detected}")

    biased = pairwise.run_pairwise_biased_demo()
    print(f"biased comparison: winner={biased.winner} position_bias_detected={biased.position_bias_detected}")
    assert biased.position_bias_detected and biased.winner == "tie"
    print()


def _section_ensemble() -> None:
    print("=== 6. Ensemble / jury of judges ===")
    jury = ensemble.run_jury_demo()
    votes = [v.passed for v in jury.verdicts]
    print(f"juror verdicts: {votes} -> {jury.pass_votes}/{len(votes)} pass, majority_passed={jury.majority_passed}")
    print()


def _section_trajectory() -> None:
    print("=== 7. Agent-as-judge: trajectory evaluation ===")
    grounded = trajectory.run_trajectory_grounded_demo()
    shortcut = trajectory.run_trajectory_shortcut_demo()
    final_answer_only = trajectory.run_final_answer_only_comparison()
    print(f"grounded trajectory:   passed={grounded.passed} (score={grounded.score})")
    print(f"shortcut trajectory:   passed={shortcut.passed} (score={shortcut.score})")
    print(f"same final answer, judged alone: passed={final_answer_only.passed} (score={final_answer_only.score})")
    print("  -> trajectory judging catches the skipped verification step that final-answer-only judging misses")
    assert grounded.passed and not shortcut.passed and final_answer_only.passed
    print()


def _section_aggregate() -> None:
    print("=== 8. Metrics aggregation ===")
    ref_based, ref_free = pointwise.run_pointwise_demo()
    scores = [v.score for v in (ref_based, ref_free) if v.score is not None]
    passes = [bool(v.passed) for v in (ref_based, ref_free)]
    print(f"mean_score={aggregate.mean_score(scores):.2f} pass_rate={aggregate.pass_rate(passes):.2f}")

    fair = pairwise.run_pairwise_fair_demo()
    winners = [fair.winner]
    print(f"pairwise_win_rate(candidate_b)={aggregate.pairwise_win_rate(winners, 'candidate_b'):.2f}")

    matches = [("draft_v1", "draft_v2", "draft_v2"), ("draft_v2", "draft_v3", "tie")]
    rankings = aggregate.compute_rankings(matches)
    ranked = ", ".join(f"{label}={rating:.0f}" for label, rating in sorted(rankings.items()))
    print(f"elo rankings: {ranked}")
    print()


def _section_regression() -> None:
    print("=== 9. Regression gate ===")
    passing = regression.run_regression_demo()
    print(
        f"candidate run: metric={passing.candidate_metric:.2f} baseline={passing.baseline_metric:.2f} "
        f"passed={passing.passed} exit_code={regression.exit_code(passing)}"
    )

    failing = regression.run_regression_failing_demo()
    print(
        f"regressed run: metric={failing.candidate_metric:.2f} baseline={failing.baseline_metric:.2f} "
        f"passed={failing.passed} exit_code={regression.exit_code(failing)}"
    )
    assert passing.passed and not failing.passed
    print()


def _section_meta() -> None:
    print("=== 10. Meta-evaluation (judging the judge) ===")
    kappa = meta.run_meta_evaluation_demo()
    print(f"Cohen's kappa (judge vs human, 5 labeled cases): {kappa:.3f}")
    retest_rate = meta.run_test_retest_demo()
    print(f"test-retest same-verdict rate (3 cases, run twice): {retest_rate:.2f}")
    print()


if __name__ == "__main__":
    main()
