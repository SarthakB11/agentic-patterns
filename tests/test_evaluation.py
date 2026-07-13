"""Tests for the evaluation pattern.

Deterministic and offline: every test drives `MockProvider` scripts through
the pattern's own demo modules or through small scripts built inline, with
no network call and no API key.
"""

from __future__ import annotations

import pytest

from agentic_patterns import MockProvider

from patterns.evaluation import (
    aggregate,
    ensemble,
    exact,
    leakage,
    meta,
    pairwise,
    pointwise,
    process_reward,
    regression,
    selective,
    semantic,
    trajectory,
    validation_protocol,
)
from patterns.evaluation.eval_set import get_case
from patterns.evaluation.pairwise import PairwiseResult
from patterns.evaluation.pointwise import build_pointwise_judge, run_checklist_judgment
from patterns.evaluation.trajectory import TrajectoryStep
from patterns.evaluation.verdict import PairwiseVerdict, parse_pairwise_verdict, parse_pointwise_verdict

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


def test_pairwise_fair_demo_scripted_reasoning_matches_winner_slot() -> None:
    """Each scripted judge response must describe its own WINNER slot as the
    strong one, not the vague one. Regression test for a swapped-order
    response whose prose called slot A vague while declaring WINNER: a."""
    result = pairwise.run_pairwise_fair_demo()
    for verdict in (result.order_ab, result.order_ba):
        winner_label = {"a": "Candidate A", "b": "Candidate B"}.get(verdict.winner)
        assert winner_label is not None
        winner_idx = verdict.raw.index(winner_label)
        sentence_end = verdict.raw.index(";", winner_idx)
        winner_sentence = verdict.raw[winner_idx:sentence_end]
        assert "vague" not in winner_sentence


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


# --- validation protocol ---------------------------------------------------


def _fake_pairwise_result(*, bias: bool) -> PairwiseResult:
    verdict = PairwiseVerdict(winner="a", reasoning="", raw="")
    winner = "tie" if bias else "candidate_a"
    return PairwiseResult(order_ab=verdict, order_ba=verdict, winner=winner, position_bias_detected=bias)


def test_position_bias_rate_two_flip_one_agree() -> None:
    results = [_fake_pairwise_result(bias=True), _fake_pairwise_result(bias=True), _fake_pairwise_result(bias=False)]
    assert validation_protocol.position_bias_rate(results) == pytest.approx(2 / 3)


def test_validate_judge_accepts_when_all_axes_pass() -> None:
    result = validation_protocol.validate_judge(kappa=0.6, test_retest=0.97, bias_rate=0.05)
    assert result.accepted is True
    assert result.paradox_flag is False


def test_validate_judge_paradox_flag_high_retest_high_bias() -> None:
    result = validation_protocol.validate_judge(kappa=0.6, test_retest=0.97, bias_rate=0.30)
    assert result.accepted is False
    assert result.paradox_flag is True


def test_validate_judge_rejects_on_each_single_failing_axis() -> None:
    fails_agreement = validation_protocol.validate_judge(kappa=0.1, test_retest=0.97, bias_rate=0.05)
    fails_consistency = validation_protocol.validate_judge(kappa=0.6, test_retest=0.50, bias_rate=0.05)
    fails_bias = validation_protocol.validate_judge(kappa=0.6, test_retest=0.97, bias_rate=0.50)
    assert fails_agreement.accepted is False and fails_agreement.paradox_flag is False
    assert fails_consistency.accepted is False and fails_consistency.paradox_flag is False
    assert fails_bias.accepted is False and fails_bias.paradox_flag is True


def test_validation_protocol_demo_deterministic_across_two_runs() -> None:
    healthy_1, paradox_1 = validation_protocol.run_validation_protocol_demo()
    healthy_2, paradox_2 = validation_protocol.run_validation_protocol_demo()
    assert healthy_1 == healthy_2
    assert paradox_1 == paradox_2


# --- selective judging ---------------------------------------------------


def test_selective_judgment_high_confidence_answers() -> None:
    provider = MockProvider(["reasoning\nSCORE: 8\nVERDICT: pass"] * 3)
    case = get_case("refund_policy")
    result = selective.run_selective_judgment(provider, case, "some reply", resamples=3, tau=0.67)
    assert result.confidence == 1.0
    assert result.answered is True
    assert result.verdict is not None
    assert result.verdict.passed is True


def test_selective_judgment_low_confidence_abstains() -> None:
    provider = MockProvider(
        [
            "reasoning\nSCORE: 8\nVERDICT: pass",
            "reasoning\nSCORE: 4\nVERDICT: fail",
            "reasoning\nSCORE: 3\nVERDICT: fail",
        ]
    )
    case = get_case("refund_policy")
    result = selective.run_selective_judgment(provider, case, "some reply", resamples=3, tau=0.67)
    assert result.confidence == pytest.approx(2 / 3)
    assert result.answered is False
    assert result.verdict is None


def test_selective_demo_coverage_is_two_of_three() -> None:
    result = selective.run_selective_demo()
    assert result.coverage == pytest.approx(2 / 3)


def test_selective_coverage_reliability_monotonicity() -> None:
    low_tau, high_tau = selective.run_coverage_tau_curve_demo()
    assert high_tau.coverage <= low_tau.coverage
    assert high_tau.selective_agreement >= low_tau.selective_agreement


def test_selective_escalation_count_matches_abstained_count() -> None:
    result = selective.run_selective_demo()
    assert result.abstained_count == 1
    assert result.escalated_count == result.abstained_count


# --- preference leakage ---------------------------------------------------


def test_leakage_same_model_judge_detected() -> None:
    same_model, _, _ = leakage.run_leakage_demo()
    assert same_model.leakage_score > 0
    assert same_model.leakage_detected is True


def test_leakage_unrelated_judge_near_zero_not_detected() -> None:
    _, _, unrelated = leakage.run_leakage_demo()
    assert unrelated.leakage_score == pytest.approx(0.0)
    assert unrelated.leakage_detected is False


def test_leakage_tier_ordering_same_model_ge_inheritance_ge_unrelated() -> None:
    same_model, inheritance, unrelated = leakage.run_leakage_demo()
    assert same_model.leakage_score >= inheritance.leakage_score >= unrelated.leakage_score


def test_leakage_mitigation_unrelated_judge_drops_detection_on_same_outputs() -> None:
    same_model, _, unrelated = leakage.run_leakage_demo()
    assert same_model.leakage_detected is True
    assert unrelated.leakage_detected is False


def test_leakage_demo_deterministic_across_two_runs() -> None:
    first = leakage.run_leakage_demo()
    second = leakage.run_leakage_demo()
    assert [r.leakage_score for r in first] == [r.leakage_score for r in second]


# --- process reward ---------------------------------------------------


def test_score_steps_parses_exact_score_vector() -> None:
    provider = MockProvider(["reasoning\nSCORE: 9", "reasoning\nSCORE: 3", "reasoning\nSCORE: 8"])
    steps = [TrajectoryStep("a1", "o1"), TrajectoryStep("a2", "o2"), TrajectoryStep("a3", "o3")]
    scores = process_reward.score_steps(provider, "goal", steps)
    assert scores == [9.0, 3.0, 8.0]


def test_aggregate_step_scores_rules_on_fixed_vector() -> None:
    scores = [9.0, 3.0, 8.0]
    min_score = process_reward.aggregate_step_scores(scores, "min")
    mean_score = process_reward.aggregate_step_scores(scores, "mean")
    last_score = process_reward.aggregate_step_scores(scores, "last")
    product_score = process_reward.aggregate_step_scores(scores, "product")
    assert min_score == 3.0
    assert mean_score == pytest.approx(20 / 3)
    assert last_score == 8.0
    assert product_score < min(min_score, mean_score, last_score)


def test_process_reward_min_gates_while_mean_passes() -> None:
    mean_result, min_result = process_reward.run_process_reward_demo()
    assert mean_result.passed is True
    assert min_result.passed is False


def test_weakest_step_index_first_minimum_tiebreak() -> None:
    assert process_reward.weakest_step_index([9.0, 3.0, 8.0]) == 1
    assert process_reward.weakest_step_index([3.0, 3.0, 8.0]) == 0


def test_process_reward_demo_deterministic_across_two_runs() -> None:
    first_mean, first_min = process_reward.run_process_reward_demo()
    second_mean, second_min = process_reward.run_process_reward_demo()
    assert first_mean.step_scores == second_mean.step_scores
    assert first_mean.aggregate_score == second_mean.aggregate_score
    assert first_min.aggregate_score == second_min.aggregate_score


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
