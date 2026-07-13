"""Tests for the routing pattern.

Deterministic and offline: every test drives `MockProvider` scripts (or, for
the rule-based and semantic routers, no provider at all) through the
pattern's own modules, with no network call and no API key.
"""

from __future__ import annotations

import pytest

from agentic_patterns import Message, MockProvider, get_provider

from patterns.routing import (
    cascade,
    escalation,
    fallback,
    handoff,
    llm_classifier,
    main,
    reasoning_mode,
    robustness,
    router_eval,
    rule_based,
    semantic,
    threshold_sweep,
    verified_cascade,
)
from patterns.routing.registry import Route, RouteDecision, RouteRegistry

# --- registry mechanics -----------------------------------------------------


def test_registry_register_and_get() -> None:
    registry = RouteRegistry([Route(name="billing", description="d")])
    assert registry.get("billing") is not None
    assert registry.get("missing") is None


def test_registry_names_and_contains() -> None:
    registry = RouteRegistry([Route(name="billing", description="d"), Route(name="technical", description="d")])
    assert registry.names() == ["billing", "technical"]
    assert "billing" in registry
    assert "shipping" not in registry


def test_registry_validate_returns_known_candidate() -> None:
    registry = RouteRegistry([Route(name="billing", description="d")])
    assert registry.validate("billing", default="general") == "billing"


def test_registry_validate_falls_back_on_unknown_candidate() -> None:
    registry = RouteRegistry([Route(name="billing", description="d")])
    assert registry.validate("shipping", default="general") == "general"


def test_registry_dispatch_runs_handler() -> None:
    registry = RouteRegistry([Route(name="billing", description="d", handler=lambda text: f"handled: {text}")])
    decision = RouteDecision(route="billing", score=1.0, method="rule")
    assert registry.dispatch(decision, "help") == "handled: help"


def test_registry_dispatch_unknown_route_raises_key_error() -> None:
    registry = RouteRegistry()
    decision = RouteDecision(route="billing", score=1.0, method="rule")
    with pytest.raises(KeyError):
        registry.dispatch(decision, "help")


def test_registry_dispatch_missing_handler_raises_value_error() -> None:
    registry = RouteRegistry([Route(name="billing", description="d")])
    decision = RouteDecision(route="billing", score=1.0, method="rule")
    with pytest.raises(ValueError):
        registry.dispatch(decision, "help")


# --- rule-based router -------------------------------------------------------


def test_rule_based_matches_keyword() -> None:
    decision = rule_based.classify("I was charged twice for my subscription")
    assert decision.route == "billing"
    assert decision.metadata["matched_keyword"] == "charge"


def test_rule_based_unmatched_returns_default() -> None:
    decision = rule_based.classify("do you offer support in French")
    assert decision.route == rule_based.DEFAULT_ROUTE
    assert decision.score == 0.0


# --- semantic router ----------------------------------------------------------


def test_semantic_near_query_routes_to_matching_route() -> None:
    decision = semantic.classify("the mobile app keeps crashing on launch")
    assert decision.route == "technical"


def test_semantic_far_query_routes_to_no_match() -> None:
    decision = semantic.classify("what is the weather like today in Paris")
    assert decision.route == semantic.NO_MATCH_ROUTE


def test_semantic_threshold_boundary_flips_route() -> None:
    above = semantic.classify_scores({"billing": 0.21}, threshold=0.2)
    below = semantic.classify_scores({"billing": 0.19}, threshold=0.2)
    assert above.route == "billing"
    assert below.route == semantic.NO_MATCH_ROUTE


def test_semantic_embedder_is_deterministic() -> None:
    scores_a = semantic.route_scores("I was charged twice for my subscription")
    scores_b = semantic.route_scores("I was charged twice for my subscription")
    assert scores_a == scores_b


def test_route_scores_embeds_route_utterances_once_across_calls() -> None:
    from agentic_patterns import HashEmbedder

    class CountingEmbedder:
        """Wraps `HashEmbedder`, counting every `embed()` call it receives."""

        def __init__(self) -> None:
            self._inner = HashEmbedder()
            self.calls = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            return self._inner.embed(texts)

    registry = RouteRegistry(
        [
            Route(name="billing", description="d", utterances=["I was charged twice"]),
            Route(name="technical", description="d", utterances=["the app crashes"]),
        ]
    )
    embedder = CountingEmbedder()

    semantic.route_scores("why was I billed", registry, embedder)
    semantic.route_scores("why was I billed again", registry, embedder)

    # One embed() call per route's utterances (2 routes) plus one per query
    # text (2 calls), not one per route on every call (which would be 4).
    assert embedder.calls == 4


# --- LLM-classifier router ----------------------------------------------------


def test_llm_classifier_valid_label_dispatches() -> None:
    provider = get_provider(script=["ROUTE: technical"])
    decision = llm_classifier.classify("the app won't stop crashing", provider)
    assert decision.route == "technical"
    assert decision.metadata["valid"] is True


def test_llm_classifier_sends_route_descriptions_in_system_prompt() -> None:
    provider = get_provider(script=["ROUTE: billing"])
    llm_classifier.classify("why was I charged", provider)
    assert isinstance(provider, MockProvider)
    sent_system = provider.calls[0]["system"]
    assert "billing" in sent_system
    assert "technical" in sent_system


def test_llm_classifier_unknown_label_falls_back() -> None:
    provider = get_provider(script=["ROUTE: shipping"])
    decision = llm_classifier.classify("where is my package", provider)
    assert decision.route == llm_classifier.DEFAULT_ROUTE
    assert decision.metadata["valid"] is False


def test_llm_classifier_malformed_reply_falls_back() -> None:
    provider = get_provider(script=["I'm not sure how to categorize this."])
    decision = llm_classifier.classify("hmm", provider)
    assert decision.route == llm_classifier.DEFAULT_ROUTE


# --- cascade and capability selection -----------------------------------------


def test_cascade_passing_quality_check_never_calls_strong_tier() -> None:
    provider = get_provider(
        script=[
            "The invoice total is $482.10, due on the 15th, covering the March and April subscription periods.",
            "unused strong-tier answer",
        ]
    )
    decision = cascade.run_cascade("what does my invoice total?", provider)
    assert decision.route == "cheap"
    assert len(provider.calls) == 1


def test_cascade_failing_quality_check_escalates_to_strong_tier() -> None:
    provider = get_provider(
        script=["I'm not sure, I don't have enough information.", "Break-even price is $21.00 per unit."]
    )
    decision = cascade.run_cascade("derive the break-even price", provider)
    assert decision.route == "strong"
    assert decision.attempts == 2
    assert len(provider.calls) == 2


def test_select_tier_easy_question_is_cheap() -> None:
    assert cascade.select_tier("What is the capital of France?").route == "cheap"


def test_select_tier_hard_question_is_strong() -> None:
    assert cascade.select_tier("Derive the break-even price for this product.").route == "strong"


def test_baseline_comparison_heuristic_beats_always_strong_on_calls() -> None:
    baselines = cascade.run_baseline_comparison_demo()
    assert baselines["heuristic_strong_calls"] < baselines["always_strong_calls"]
    assert baselines["heuristic_accuracy"] >= baselines["random_accuracy"]


# --- fallback chain -------------------------------------------------------------


def test_fallback_chain_recovers_on_second_handler() -> None:
    def fails() -> str:
        raise fallback.HandlerFailure("timeout")

    def succeeds() -> str:
        return "recovered answer"

    handlers = [fallback.FallbackHandler("primary", fails), fallback.FallbackHandler("secondary", succeeds)]
    decision = fallback.run_fallback_chain(handlers)
    assert decision.route == "secondary"
    assert decision.attempts == 2
    assert "primary" in decision.metadata["errors"][0]


def test_fallback_chain_all_fail_returns_terminal_human_route() -> None:
    def fails_one() -> str:
        raise fallback.HandlerFailure("error: broken")

    def fails_two() -> str:
        raise fallback.HandlerFailure("refusal: cannot help")

    handlers = [fallback.FallbackHandler("a", fails_one), fallback.FallbackHandler("b", fails_two)]
    decision = fallback.run_fallback_chain(handlers)
    assert decision.route == "human"
    assert decision.attempts == 2
    assert len(decision.metadata["errors"]) == 2


def test_make_provider_handler_converts_refusal_to_failure() -> None:
    provider = get_provider(script=["I can't help with that."])
    handler = fallback.make_provider_handler("policy_bot", provider, "question")
    with pytest.raises(fallback.HandlerFailure):
        handler.call()


def test_make_provider_handler_returns_answer_on_success() -> None:
    provider = get_provider(script=["Here is your answer."])
    handler = fallback.make_provider_handler("bot", provider, "question")
    assert handler.call() == "Here is your answer."


def test_make_provider_handler_converts_provider_exception_to_failure() -> None:
    # An empty script makes the very first call() raise MockScriptExhausted,
    # a real provider-side exception rather than a scripted refusal.
    provider = get_provider(script=[])
    handler = fallback.make_provider_handler("bot", provider, "question")
    with pytest.raises(fallback.HandlerFailure):
        handler.call()


def test_run_fallback_chain_recovers_when_a_handler_provider_raises() -> None:
    broken_provider = get_provider(script=[])
    healthy_provider = get_provider(script=["Here is your answer."])
    handlers = [
        fallback.make_provider_handler("broken_bot", broken_provider, "question"),
        fallback.make_provider_handler("healthy_bot", healthy_provider, "question"),
    ]
    decision = fallback.run_fallback_chain(handlers)
    assert decision.route == "healthy_bot"
    assert decision.attempts == 2
    assert "broken_bot" in decision.metadata["errors"][0]


# --- human escalation -----------------------------------------------------------


def test_escalation_below_threshold_routes_to_human() -> None:
    decision = RouteDecision(route="technical", score=0.1, method="semantic")
    result = escalation.apply_escalation(decision, "not sure what this is")
    assert result.route == escalation.HUMAN_ROUTE
    assert result.metadata["escalation_reason"] == "below_threshold"


def test_escalation_sensitive_topic_overrides_confident_decision() -> None:
    decision = RouteDecision(route="billing", score=0.95, method="semantic")
    result = escalation.apply_escalation(decision, "I'm considering legal action over this charge")
    assert result.route == escalation.HUMAN_ROUTE
    assert result.metadata["escalation_reason"] == "sensitive_topic"


def test_escalation_confident_and_not_sensitive_passes_through() -> None:
    decision = RouteDecision(route="account", score=0.9, method="semantic")
    result = escalation.apply_escalation(decision, "I forgot my password")
    assert result is decision


# --- reasoning-mode router -------------------------------------------------------


def test_reasoning_mode_simple_question_is_direct() -> None:
    assert reasoning_mode.classify_reasoning_mode("What is the capital of Japan?").route == "direct"


def test_reasoning_mode_calculation_question_is_reason() -> None:
    assert reasoning_mode.classify_reasoning_mode("Calculate the prorated total cost for this month.").route == "reason"


def test_reasoning_mode_long_prompt_routes_to_reason() -> None:
    long_prompt = " ".join(["word"] * 30)
    assert reasoning_mode.classify_reasoning_mode(long_prompt).route == "reason"


def test_reasoning_mode_uses_different_system_prompt_per_route() -> None:
    provider = get_provider(script=["Tokyo."])
    reasoning_mode.answer_with_reasoning_mode("What is the capital of Japan?", provider)
    assert provider.calls[0]["system"] == reasoning_mode._DIRECT_SYSTEM


# --- handoff routing -------------------------------------------------------------


def test_handoff_transfers_to_named_sub_agent() -> None:
    from agentic_patterns import scripted_tool_call

    triage = get_provider(script=[scripted_tool_call("transfer_to_billing", {})])
    billing_agent = get_provider(script=["Your invoice was paid in full."])
    decision = handoff.run_handoff("did my invoice go through?", triage, {"billing": billing_agent})
    assert decision.route == "billing"
    assert decision.metadata["transferred"] is True
    assert decision.metadata["answer"] == "Your invoice was paid in full."


def test_handoff_no_transfer_keeps_triage_as_route() -> None:
    triage = get_provider(script=["We're open 9 to 6 Eastern."])
    decision = handoff.run_handoff("what are your hours?", triage, {})
    assert decision.route == "triage"
    assert decision.metadata["transferred"] is False


def test_handoff_unknown_transfer_target_raises() -> None:
    from agentic_patterns import scripted_tool_call

    triage = get_provider(script=[scripted_tool_call("transfer_to_shipping", {})])
    with pytest.raises(ValueError):
        handoff.run_handoff("where is my order?", triage, {"billing": get_provider(script=[])})


# --- reasoning-mode / escalation safety corrections --------------------------------


def test_reasoning_mode_enforce_safety_overrides_direct_for_sensitive_input() -> None:
    text = "there was a data breach affecting my account"
    decision = reasoning_mode.classify_reasoning_mode(text)
    assert decision.route == "direct"
    enforced = reasoning_mode.enforce_reasoning_safety(decision, text)
    assert enforced.route == "reason"
    assert enforced.metadata["safety_override"] is True


def test_reasoning_mode_enforce_safety_leaves_non_sensitive_direct_alone() -> None:
    text = "What is the capital of Japan?"
    decision = reasoning_mode.classify_reasoning_mode(text)
    enforced = reasoning_mode.enforce_reasoning_safety(decision, text)
    assert enforced is decision


def test_escalation_is_sensitive_survives_whitespace_perturbation() -> None:
    # Regression test: a multi-word keyword like "legal action" must still
    # match after whitespace is mangled, not just on exact spacing.
    policy = escalation.EscalationPolicy()
    assert escalation.is_sensitive("I'm  considering  legal   action  over  this", policy)


# --- router_eval: benchmark against baselines and an oracle ------------------------


def test_router_eval_oracle_ceiling() -> None:
    dataset = cascade._DIFFICULTY_DATASET
    oracle = router_eval.oracle_score(dataset, "tier")
    assert oracle.accuracy == 1.0
    assert oracle.total_cost == sum(router_eval.route_cost(label) for _, label in dataset)


def test_router_eval_weak_llm_classifier_caught_by_benchmark() -> None:
    # Correct on only 2 of 8 rows (25%), the same accuracy the fixed-seed
    # random baseline gets on this dataset: a router this weak must not be
    # reported as beating random.
    dataset = router_eval._CATEGORY_DATASET
    weak_script = [
        "ROUTE: technical",  # wrong (billing)
        "ROUTE: account",  # wrong (technical)
        "ROUTE: general",  # wrong (account)
        "ROUTE: billing",  # wrong (general)
        "ROUTE: billing",  # correct
        "ROUTE: technical",  # correct
        "ROUTE: general",  # wrong (account)
        "ROUTE: account",  # wrong (general)
    ]
    weak_provider = get_provider(script=weak_script)
    weak_score = router_eval.score_llm_classifier(weak_provider, dataset)
    category_labels = ["billing", "technical", "account", "general"]
    random_baseline = router_eval.random_score(dataset, "category", category_labels, seed=0)
    oracle = router_eval.oracle_score(dataset, "category")
    router_eval._finalize(weak_score, oracle, random_baseline, always_strong=None)
    assert weak_score.accuracy == pytest.approx(0.25)
    assert weak_score.beats_random is False


def test_router_eval_cascade_cost_includes_burned_cheap_attempt() -> None:
    dataset = cascade._DIFFICULTY_DATASET
    score = router_eval.score_cascade(dataset)
    strong_rows = sum(1 for _, tier in dataset if tier == "strong")
    cheap_rows = len(dataset) - strong_rows
    honest_cost = cheap_rows * router_eval._TIER_COSTS["cheap"] + strong_rows * (
        router_eval._TIER_COSTS["cheap"] + router_eval._TIER_COSTS["strong"]
    )
    naive_strong_only_cost = cheap_rows * router_eval._TIER_COSTS["cheap"] + strong_rows * router_eval._TIER_COSTS["strong"]
    assert score.total_cost == honest_cost
    assert score.total_cost > naive_strong_only_cost


def test_router_eval_llm_classifier_overhead_exceeds_rule_based_on_identical_routes() -> None:
    dataset = router_eval._CATEGORY_DATASET
    rule_score = router_eval.score_rule_based(dataset)
    good_provider = get_provider(script=[f"ROUTE: {label}" for _, label in dataset])
    llm_score = router_eval.score_llm_classifier(good_provider, dataset)
    # Both get every row right (identical routes to the labels), so the
    # entire cost gap is the LLM classifier's own per-call overhead.
    assert rule_score.accuracy == llm_score.accuracy == 1.0
    assert llm_score.total_cost > rule_score.total_cost


def test_router_eval_benchmark_is_deterministic() -> None:
    scores_a = router_eval.run_benchmark()
    scores_b = router_eval.run_benchmark()
    assert scores_a == scores_b


# --- threshold_sweep: continuous score vs. swept threshold -------------------------


def test_threshold_sweep_cost_is_non_increasing_as_threshold_rises() -> None:
    frontier = threshold_sweep.sweep(cascade._DIFFICULTY_DATASET)
    costs = [p.cost for p in frontier]
    assert costs == sorted(costs, reverse=True)


def test_threshold_sweep_pick_operating_point_meets_budget() -> None:
    frontier = threshold_sweep.sweep(cascade._DIFFICULTY_DATASET)
    point = threshold_sweep.pick_operating_point(frontier, cost_budget=50.0)
    assert point.cost <= 50.0
    assert all(p.accuracy <= point.accuracy for p in frontier if p.cost <= 50.0)


def test_threshold_sweep_boundary_flip_at_exact_score() -> None:
    score = 0.5
    assert threshold_sweep.route_at_threshold(score, 0.5) == "strong"
    assert threshold_sweep.route_at_threshold(score, 0.5 + 1e-9) == "cheap"


def test_threshold_sweep_degenerate_thresholds_match_baselines() -> None:
    dataset = cascade._DIFFICULTY_DATASET
    frontier = threshold_sweep.sweep(dataset, thresholds=(0.0, 1.1))
    always_strong_point, always_cheap_point = frontier
    always_strong = router_eval.always_score("always_strong", dataset, "tier", "strong")
    always_cheap = router_eval.always_score("always_cheap", dataset, "tier", "cheap")
    assert always_strong_point.accuracy == always_strong.accuracy
    assert always_strong_point.cost == always_strong.total_cost
    assert always_cheap_point.accuracy == always_cheap.accuracy
    assert always_cheap_point.cost == always_cheap.total_cost


def test_threshold_sweep_is_deterministic() -> None:
    a = threshold_sweep.sweep(cascade._DIFFICULTY_DATASET)
    b = threshold_sweep.sweep(cascade._DIFFICULTY_DATASET)
    assert a == b


# --- verified_cascade: model-judge cascade with abstention -------------------------


def test_verified_cascade_accepts_on_cheap_tier() -> None:
    provider = get_provider(script=["A complete cheap-tier answer.", "ACCEPT"])
    decision = verified_cascade.run_verified_cascade("a question", provider)
    assert decision.route == "cheap"
    assert decision.attempts == 1
    assert len(provider.calls) == 2


def test_verified_cascade_escalates_after_cheap_deferred() -> None:
    provider = get_provider(script=["weak cheap answer", "DEFER", "solid strong answer", "ACCEPT"])
    decision = verified_cascade.run_verified_cascade("a question", provider)
    assert decision.route == "strong"
    assert decision.attempts == 2
    assert decision.metadata["escalated"] is True


def test_verified_cascade_abstains_when_both_tiers_deferred() -> None:
    provider = get_provider(script=["weak cheap", "DEFER", "weak strong", "DEFER"])
    decision = verified_cascade.run_verified_cascade("a question", provider)
    assert decision.route == "human"
    assert decision.attempts == 3
    assert decision.metadata["abstained"] is True


def test_verified_cascade_cost_counts_judge_calls() -> None:
    provider = get_provider(script=["A complete cheap-tier answer.", "ACCEPT"])
    decision = verified_cascade.run_verified_cascade("a question", provider)
    # Answer call plus judge call, not just the answer.
    assert decision.metadata["provider_calls"] == 2


def test_verified_cascade_rejects_short_but_wrong_answer_the_heuristic_would_pass() -> None:
    plausible_but_wrong = "The refund will be issued within thirty days to your original payment method."
    assert cascade.quality_check(plausible_but_wrong) is True  # the heuristic cascade would accept this
    provider = get_provider(script=[plausible_but_wrong, "DEFER", "The actual correct strong-tier answer.", "ACCEPT"])
    decision = verified_cascade.run_verified_cascade("a question", provider)
    assert decision.route == "strong"  # the judge escalates where the heuristic would not


# --- robustness: route stability under perturbation ---------------------------------


def test_robustness_rule_based_flips_when_keyword_paraphrased_out() -> None:
    query = "the app crashes every time I open settings"
    rate = robustness.flip_rate(robustness._rule_route, query, [robustness.perturb_synonym_swap])
    assert rate > 0


def test_robustness_semantic_steadier_than_rule_on_same_perturbation() -> None:
    query = "the app crashes every time I open settings"
    rule_rate = robustness.flip_rate(robustness._rule_route, query, [robustness.perturb_synonym_swap])
    semantic_rate = robustness.flip_rate(robustness._semantic_route, query, [robustness.perturb_synonym_swap])
    assert semantic_rate < rule_rate


def test_robustness_boundary_probe_flips_near_threshold_not_far() -> None:
    boundary_rate, far_rate = robustness.boundary_probe()
    assert boundary_rate > 0
    assert far_rate == 0


def test_robustness_escalation_safety_invariant_holds() -> None:
    result = robustness.check_escalation_safety(escalation.apply_escalation)
    assert result.passed
    assert result.failures == []


def test_robustness_reasoning_safety_invariant_holds() -> None:
    result = robustness.check_reasoning_safety()
    assert result.passed


def test_robustness_safety_invariant_catches_broken_escalate_fn() -> None:
    def ignores_sensitivity(decision: RouteDecision, text: str) -> RouteDecision:
        return decision  # a regression: never checks whether text is sensitive

    result = robustness.check_escalation_safety(ignores_sensitivity)
    assert result.passed is False
    assert len(result.failures) > 0


# --- end-to-end pipeline (main.py) -----------------------------------------------


def test_end_to_end_dispatches_confident_non_sensitive_input() -> None:
    decision = main.run_end_to_end_demo("I forgot my account password and cannot log in")
    assert decision.route == "account"
    assert "answer" in decision.metadata


def test_end_to_end_escalates_sensitive_input_to_human() -> None:
    decision = main.run_end_to_end_demo("I'm considering legal action over this billing dispute")
    assert decision.route == escalation.HUMAN_ROUTE


def test_message_helpers_used_consistently_with_core_types() -> None:
    # Sanity check that this pattern's handlers build well-formed core Message
    # objects, the same shape every other pattern in the repo uses.
    msg = Message.user("hello")
    assert msg.role == "user"
    assert msg.content == "hello"
