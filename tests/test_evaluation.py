"""Tests for the evaluation pattern.

Deterministic and offline: every test drives `MockProvider` scripts through
the pattern's own demo modules or through small scripts built inline, with
no network call and no API key.
"""

from __future__ import annotations

import pytest

from agentic_patterns import MockProvider

from patterns.evaluation import aggregate, ensemble, exact, meta, pairwise, pointwise, regression, semantic, trajectory
from patterns.evaluation.eval_set import get_case
from patterns.evaluation.pointwise import build_pointwise_judge, run_checklist_judgment
from patterns.evaluation.verdict import parse_pairwise_verdict, parse_pointwise_verdict

# --- eval_set ----------------------------------------------------------


def test_get_case_returns_expected_case() -> None:
    case = get_case("refund_policy")
    assert case.id == "refund_policy"
    assert case.reference is not None


def test_get_case_unknown_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        get_case("does_not_exist")


# --- verdict parsing -----------------------------------------------------


def test_parse_pointwise_verdict_extracts_score_and_verdict() -> None:
    v = parse_pointwise_verdict("some reasoning first\nSCORE: 7\nVERDICT: pass")
    assert v.score == 7.0
    assert v.passed is True
    assert v.malformed is False


def test_parse_pointwise_verdict_malformed_falls_back_to_fail() -> None:
    v = parse_pointwise_verdict("this is just unstructured prose with no fields")
    assert v.malformed is True
    assert v.passed is False
    assert v.score is None


def test_parse_pairwise_verdict_extracts_winner() -> None:
    v = parse_pairwise_verdict("candidate b is more specific\nWINNER: b")
    assert v.winner == "b"
    assert v.malformed is False


def test_parse_pairwise_verdict_malformed_falls_back_to_tie() -> None:
    v = parse_pairwise_verdict("no structured field anywhere in this text")
    assert v.winner == "tie"
    assert v.malformed is True


# --- exact evaluators ------------------------------------------------------


def test_regex_match_evaluator_pass_and_fail() -> None:
    case = get_case("order_status_lookup")
    passing = exact.regex_match_evaluator(case, "order 48213 shipped yesterday")
    failing = exact.regex_match_evaluator(case, "order 99999 shipped yesterday")
    assert passing.passed is True
    assert failing.passed is False


def test_regex_match_evaluator_requires_regex_property() -> None:
    case = get_case("refund_policy")
    with pytest.raises(ValueError):
        exact.regex_match_evaluator(case, "text")


def test_json_schema_evaluator_valid() -> None:
    case = get_case("order_extraction")
    score = exact.json_schema_evaluator(case, '{"order_id": "48213", "status": "shipped"}')
    assert score.passed is True


def test_json_schema_evaluator_missing_key() -> None:
    case = get_case("order_extraction")
    score = exact.json_schema_evaluator(case, '{"order_id": "48213"}')
    assert score.passed is False
    assert "missing required keys" in score.detail


def test_json_schema_evaluator_invalid_json() -> None:
    case = get_case("order_extraction")
    score = exact.json_schema_evaluator(case, "not json at all")
    assert score.passed is False
    assert "invalid JSON" in score.detail


def test_json_schema_evaluator_bad_status_value() -> None:
    case = get_case("order_extraction")
    score = exact.json_schema_evaluator(case, '{"order_id": "48213", "status": "lost"}')
    assert score.passed is False


# --- semantic similarity ---------------------------------------------------


def test_semantic_similarity_evaluator_paraphrase_passes() -> None:
    case = get_case("refund_policy")
    reply = "You can get a refund within 30 days if you still have the receipt or the order number."
    score = semantic.semantic_similarity_evaluator(case, reply)
    assert score.passed is True


def test_semantic_similarity_evaluator_off_topic_fails() -> None:
    case = get_case("refund_policy")
    reply = "Our support hours are 9 to 5 Monday through Friday."
    score = semantic.semantic_similarity_evaluator(case, reply)
    assert score.passed is False


def test_semantic_similarity_evaluator_requires_reference() -> None:
    case = get_case("order_status_lookup")
    with pytest.raises(ValueError):
        semantic.semantic_similarity_evaluator(case, "text")


# --- pointwise judge ---------------------------------------------------


def test_pointwise_judge_reference_based_includes_reference_in_prompt() -> None:
    provider = MockProvider(["SCORE: 8\nVERDICT: pass"])
    case = get_case("refund_policy")
    judge = build_pointwise_judge(provider, reference_mode=True)
    judge(case, "some reply")
    prompt = provider.calls[0]["messages"][0].content
    assert "Reference answer" in prompt
    assert case.reference in prompt


def test_pointwise_judge_reference_free_omits_reference() -> None:
    provider = MockProvider(["SCORE: 7\nVERDICT: pass"])
    case = get_case("refund_policy")
    judge = build_pointwise_judge(provider, reference_mode=False)
    judge(case, "some reply")
    prompt = provider.calls[0]["messages"][0].content
    assert "Reference answer" not in prompt


def test_checklist_judgment_makes_two_calls_and_computes_fraction() -> None:
    provider = MockProvider(
        [
            "CHECKLIST:\n1. names the window\n2. names proof needed\n3. professional tone",
            "1. names the window: PASS\n2. names proof needed: FAIL\n"
            "3. professional tone: PASS\nCHECKLIST_SCORE: 2/3",
        ]
    )
    case = get_case("cancel_subscription")
    checklist, verdict = run_checklist_judgment(provider, case, "reply text")
    assert "CHECKLIST" in checklist
    assert len(provider.calls) == 2
    assert verdict.score == pytest.approx(2 / 3)
    assert verdict.passed is True


def test_pointwise_order_check_demo_detects_bias() -> None:
    forward, reversed_, bias_detected = pointwise.run_pointwise_order_check_demo()
    assert forward.score == 9.0
    assert reversed_.score == 7.0
    assert bias_detected is True


# --- pairwise judge ---------------------------------------------------


def test_pairwise_fair_demo_agrees_across_orders_no_bias() -> None:
    result = pairwise.run_pairwise_fair_demo()
    assert result.winner == "candidate_b"
    assert result.position_bias_detected is False


def test_pairwise_biased_demo_disagrees_across_orders_and_ties() -> None:
    result = pairwise.run_pairwise_biased_demo()
    assert result.winner == "tie"
    assert result.position_bias_detected is True


def test_pairwise_judgment_malformed_responses_fall_back_to_tie() -> None:
    provider = MockProvider(["no structured field here", "still nothing structured"])
    case = get_case("cancel_subscription")
    result = pairwise.run_pairwise_judgment(provider, case, "candidate one", "candidate two")
    assert result.order_ab.malformed is True
    assert result.order_ba.malformed is True
    assert result.winner == "tie"


def test_pairwise_judgment_calls_provider_exactly_twice() -> None:
    provider = MockProvider(["WINNER: a", "WINNER: b"])
    case = get_case("cancel_subscription")
    pairwise.run_pairwise_judgment(provider, case, "candidate one", "candidate two")
    assert len(provider.calls) == 2


# --- ensemble jury ---------------------------------------------------


def test_jury_demo_majority_vote_resolves_to_pass() -> None:
    jury = ensemble.run_jury_demo()
    assert [v.passed for v in jury.verdicts] == [True, True, False]
    assert jury.pass_votes == 2
    assert jury.majority_passed is True


def test_jury_uses_one_independent_provider_per_juror() -> None:
    provider_a = MockProvider(["SCORE: 9\nVERDICT: pass"])
    provider_b = MockProvider(["SCORE: 8\nVERDICT: pass"])
    provider_c = MockProvider(["SCORE: 4\nVERDICT: fail"])
    case = get_case("refund_policy")
    result = ensemble.run_jury([provider_a, provider_b, provider_c], case, "some reply")
    assert len(provider_a.calls) == 1
    assert len(provider_b.calls) == 1
    assert len(provider_c.calls) == 1
    assert result.majority_passed is True


def test_jury_tied_vote_fails_closed() -> None:
    tied_providers = [MockProvider(["SCORE: 9\nVERDICT: pass"]), MockProvider(["SCORE: 3\nVERDICT: fail"])]
    case = get_case("refund_policy")
    result = ensemble.run_jury(tied_providers, case, "some reply")
    assert result.pass_votes == 1
    assert result.majority_passed is False


# --- trajectory judge ---------------------------------------------------


def test_trajectory_grounded_demo_passes() -> None:
    verdict = trajectory.run_trajectory_grounded_demo()
    assert verdict.passed is True


def test_trajectory_shortcut_fails_while_final_answer_alone_passes() -> None:
    shortcut = trajectory.run_trajectory_shortcut_demo()
    final_answer_only = trajectory.run_final_answer_only_comparison()
    assert shortcut.passed is False
    assert final_answer_only.passed is True


# --- aggregate metrics ---------------------------------------------------


def test_mean_score_and_pass_rate() -> None:
    assert aggregate.mean_score([2.0, 4.0, 6.0]) == 4.0
    assert aggregate.mean_score([]) == 0.0
    assert aggregate.pass_rate([True, True, False]) == pytest.approx(2 / 3)
    assert aggregate.pass_rate([]) == 0.0


def test_pairwise_win_rate_counts_ties_as_half() -> None:
    winners = ["a", "a", "tie", "b"]
    assert aggregate.pairwise_win_rate(winners, "a") == pytest.approx(2.5 / 4)


def test_compute_rankings_winner_gains_rating_loser_loses() -> None:
    ratings = aggregate.compute_rankings([("x", "y", "x")])
    assert ratings["x"] > 1000.0
    assert ratings["y"] < 1000.0


def test_compute_rankings_tie_keeps_ratings_equal() -> None:
    ratings = aggregate.compute_rankings([("x", "y", "tie")])
    assert ratings["x"] == pytest.approx(ratings["y"])


# --- regression gate ---------------------------------------------------


def test_regression_gate_passes_within_tolerance() -> None:
    result = regression.evaluate_regression(0.99, 1.0, tolerance=0.02)
    assert result.passed is True


def test_regression_gate_fails_below_tolerance() -> None:
    result = regression.evaluate_regression(0.9, 1.0, tolerance=0.02)
    assert result.passed is False
    assert result.delta == pytest.approx(-0.1)


def test_regression_exit_code_matches_passed() -> None:
    assert regression.exit_code(regression.evaluate_regression(1.0, 1.0)) == 0
    assert regression.exit_code(regression.evaluate_regression(0.5, 1.0)) == 1


def test_regression_demo_passes_and_failing_demo_fails() -> None:
    assert regression.run_regression_demo().passed is True
    assert regression.run_regression_failing_demo().passed is False


# --- meta-evaluation ---------------------------------------------------


def test_cohens_kappa_matches_hand_calculation() -> None:
    judge_labels = [True, True, False, True, False]
    human_labels = [True, False, False, True, False]
    kappa = meta.cohens_kappa(judge_labels, human_labels)
    assert kappa == pytest.approx(0.6153846153846154)


def test_cohens_kappa_requires_equal_length_lists() -> None:
    with pytest.raises(ValueError):
        meta.cohens_kappa([True], [True, False])


def test_cohens_kappa_perfect_agreement_returns_one() -> None:
    assert meta.cohens_kappa([True, True, False], [True, True, False]) == 1.0


def test_test_retest_rate_matches_hand_calculation() -> None:
    rate = meta.test_retest_rate([True, True, False], [True, False, False])
    assert rate == pytest.approx(2 / 3)


def test_meta_demo_functions_match_hand_calculated_values() -> None:
    assert meta.run_meta_evaluation_demo() == pytest.approx(0.6153846153846154)
    assert meta.run_test_retest_demo() == pytest.approx(2 / 3)


# --- determinism ---------------------------------------------------


def test_full_demo_is_deterministic_across_two_runs() -> None:
    first_a, first_b = pointwise.run_pointwise_demo()
    second_a, second_b = pointwise.run_pointwise_demo()
    assert (first_a.score, first_a.passed) == (second_a.score, second_a.passed)
    assert (first_b.score, first_b.passed) == (second_b.score, second_b.passed)

    first_fair = pairwise.run_pairwise_fair_demo()
    second_fair = pairwise.run_pairwise_fair_demo()
    assert first_fair.winner == second_fair.winner
    assert first_fair.position_bias_detected == second_fair.position_bias_detected

    assert meta.run_meta_evaluation_demo() == meta.run_meta_evaluation_demo()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
