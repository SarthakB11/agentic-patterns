"""Tests for the routing pattern.

Deterministic and offline: every test drives `MockProvider` scripts (or, for
the rule-based and semantic routers, no provider at all) through the
pattern's own modules, with no network call and no API key.
"""

from __future__ import annotations

import pytest

from agentic_patterns import Message, MockProvider, get_provider

from patterns.routing import cascade, escalation, fallback, handoff, llm_classifier, main, reasoning_mode, rule_based, semantic
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
