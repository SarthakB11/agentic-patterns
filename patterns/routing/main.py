"""Routing pattern: classify an input, then dispatch it to a specialized handler.

Routing separates "who should answer this" from the answering itself: a
classifier looks at an input and picks a route, then a handler built for
that route (a different prompt, tool, sub-agent, or model) produces the
answer. Anthropic frames this as a core agent workflow; it works at two
altitudes, task routing (billing questions to the billing handler) and
model routing (easy questions to a cheap model, hard ones to a strong one).

This demo runs eight sub-variants end to end, entirely offline against
`MockProvider` (or no provider at all, for the variants that need none)
with scripted, coherent conversations, no network call and no API key:

1. Rule-based routing: keyword dispatch, standard library only.
2. Semantic routing: embedding-similarity match against per-route example
   utterances, with a below-threshold input falling to "no match".
3. LLM-classifier routing: a structured label, validated against the route
   set, with fallback on an invalid label.
4. Cost/quality cascade and capability model selection: try the cheap tier
   and escalate on a failed quality check, versus deciding the tier
   up front from a difficulty heuristic; both compared against random-
   choice and always-strong baselines.
5. Fallback chain: an ordered list of handlers tried until one succeeds,
   covering a simulated timeout, a refusal, and a raised error.
6. Human escalation: a below-threshold score and a sensitive-topic flag
   both override a decision to the human route, including a case where a
   fully confident decision is overridden because the input is sensitive.
7. Reasoning-mode routing: a binary "reason or not" decision, separate
   from which model or category answers.
8. Handoff routing: the triage model transfers the conversation to a
   sub-agent by calling a tool, and the sub-agent answers directly with no
   return trip through triage.
9. Router benchmark (`router_eval.py`): every router above, scored against
   random-choice, always-cheapest, always-strongest, and an oracle.
10. Threshold sweep (`threshold_sweep.py`): a continuous score against a
    swept cost threshold, tracing the cost-quality frontier.
11. Verified cascade (`verified_cascade.py`): a three-tier cascade gated by
    a scripted model judge, escalating on a low verdict and abstaining to
    a human when even the strong tier fails review.
12. Robustness (`robustness.py`): route flip rate under paraphrase for the
    rule, semantic, and reasoning-mode routers, plus the safety invariant
    that a sensitive input never flips toward a weaker handler.

It closes with one end-to-end pipeline wiring an input through semantic
classification, a threshold check, dispatch, and a fallback to human if
dispatch itself fails, returning routing metadata (route, score, attempts)
so the whole decision is observable, not just the final answer.

Run it from the repository root:

    python -m patterns.routing.main

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key in the environment) to run the same code against
a real model instead of the mock. No source change is required; every demo
that calls a model builds its provider through `agentic_patterns.get_provider`.
The rule-based and semantic routers make no model call at all, by design,
and are unaffected by that setting.
"""

from __future__ import annotations

from agentic_patterns import Message, get_provider

from patterns.routing import (
    cascade,
    escalation,
    fallback,
    handoff,
    llm_classifier,
    reasoning_mode,
    robustness,
    router_eval,
    rule_based,
    semantic,
    threshold_sweep,
    verified_cascade,
)
from patterns.routing.registry import Route, RouteDecision, RouteRegistry
from patterns.routing.transcript import format_decision

_PIPELINE_HANDLERS = {
    "billing": "Your last invoice was $482.10, billed on the 1st for the March subscription period.",
    "technical": "That crash is a known issue in v3.2; updating to v3.3 resolves it.",
    "account": "I've sent a password reset link to your account email.",
}


def _build_pipeline_registry() -> RouteRegistry:
    """Build the route registry the end-to-end demo dispatches through.

    Each route's handler is backed by its own `MockProvider`, scripted with
    one canned answer, so the dispatch step in the demo below makes a real
    (mock) model call rather than returning a hardcoded string directly.
    """
    registry = RouteRegistry()
    for name, canned_answer in _PIPELINE_HANDLERS.items():
        provider = get_provider(script=[canned_answer])

        def handler(text: str, p=provider) -> str:
            return p.complete([Message.user(text)]).content

        registry.register(Route(name=name, description=f"Handles {name} questions.", handler=handler))
    return registry


def run_end_to_end_demo(text: str) -> RouteDecision:
    """Wire one input through classification, threshold check, dispatch, and fallback.

    Args:
        text: The input to route and answer.

    Returns:
        The final `RouteDecision`, with `metadata["answer"]` set if a
        handler produced one, or escalated to the human route if
        classification was not confident enough, the topic was sensitive,
        or dispatch itself failed.
    """
    registry = _build_pipeline_registry()
    decision = semantic.classify(text)
    decision = escalation.apply_escalation(decision, text)
    if decision.route == escalation.HUMAN_ROUTE:
        return decision

    try:
        answer = registry.dispatch(decision, text)
    except (KeyError, ValueError) as exc:
        return RouteDecision(
            route=escalation.HUMAN_ROUTE,
            score=decision.score,
            method="escalation",
            attempts=decision.attempts + 1,
            metadata={**decision.metadata, "escalation_reason": "dispatch_failed", "error": str(exc)},
        )
    decision.metadata["answer"] = answer
    return decision


def main() -> None:
    """Run every routing sub-variant demo and print a readable transcript."""
    print("ROUTING PATTERN: classify, then dispatch\n")

    decisions = rule_based.run_rule_based_demo()
    for d in decisions:
        print(format_decision(d, title="1. Rule-based routing"))
    print()

    decisions = semantic.run_semantic_demo()
    for d in decisions:
        print(format_decision(d, title="2. Semantic (embedding-similarity) routing"))
    print()

    valid, fallback_label = llm_classifier.run_llm_classifier_demo()
    print(format_decision(valid, title="3. LLM-classifier routing (valid label)"))
    print(format_decision(fallback_label, title="3b. LLM-classifier routing (invalid label, fallback)"))
    print()

    passed, escalated = cascade.run_cascade_demo()
    print(format_decision(passed, title="4a. Cascade (cheap tier passes quality check)"))
    print(format_decision(escalated, title="4b. Cascade (cheap tier fails, escalates to strong)"))
    tier_decisions = cascade.run_capability_selection_demo()
    print("4c. Capability model selection (decided up front):")
    for (question, _), d in zip(cascade._DIFFICULTY_DATASET, tier_decisions):
        print(f"    {d.route:6} <- {question}")
    baselines = cascade.run_baseline_comparison_demo()
    print("4d. Baseline comparison on the difficulty dataset:")
    for key, value in baselines.items():
        print(f"    {key}: {value}")
    print()

    recovered, exhausted = fallback.run_fallback_demo()
    print(format_decision(recovered, title="5a. Fallback chain (secondary handler recovers)"))
    print(format_decision(exhausted, title="5b. Fallback chain (all handlers fail, terminal human route)"))
    print()

    low_score, sensitive_override, pass_through = escalation.run_escalation_demo()
    print(format_decision(low_score, title="6a. Human escalation (below threshold)"))
    print(format_decision(sensitive_override, title="6b. Human escalation (sensitive topic overrides confidence)"))
    print(format_decision(pass_through, title="6c. Human escalation (confident, non-sensitive: passes through)"))
    print()

    simple_decision, simple_answer, complex_decision, complex_answer = reasoning_mode.run_reasoning_mode_demo()
    print(format_decision(simple_decision, title="7a. Reasoning-mode routing (direct)"))
    print(f"    answer: {simple_answer}")
    print(format_decision(complex_decision, title="7b. Reasoning-mode routing (reason)"))
    print(f"    answer: {complex_answer}")
    print()

    handed_off, answered_directly = handoff.run_handoff_demo()
    print(format_decision(handed_off, title="8a. Handoff routing (transferred to billing agent)"))
    print(format_decision(answered_directly, title="8b. Handoff routing (triage answers directly, no transfer)"))
    print()

    print("9. Router benchmark: every router vs. random, always-cheap/strong, and an oracle")
    print(router_eval.render_table(router_eval.run_benchmark()))
    print()

    frontier, tight_point, loose_point = threshold_sweep.run_threshold_sweep_demo()
    print("10. Threshold sweep: continuous score vs. a swept cost threshold")
    for p in frontier:
        print(f"    t={p.t:<4} accuracy={p.accuracy:.3f} cost={p.cost:5.1f} strong_fraction={p.strong_fraction:.2f} flips={p.flips_from_previous}")
    print(f"    tight budget (20.0) picks t={tight_point.t} (cost={tight_point.cost}); loose budget (50.0) picks t={loose_point.t} (cost={loose_point.cost})")
    print()

    accepted, escalated, abstained = verified_cascade.run_verified_cascade_demo()
    print("11. Verified cascade: model-judge escalation with abstention")
    for label, d in (("accept-on-cheap", accepted), ("defer-then-accept-on-strong", escalated), ("defer-defer-abstain", abstained)):
        print(f"    {label:28} route={d.route:6} attempts={d.attempts}  escalated={d.metadata['escalated']}  abstained={d.metadata['abstained']}  provider_calls={d.metadata['provider_calls']}")
    print()

    table, boundary, escalation_safety, reasoning_safety = robustness.run_robustness_demo()
    print("12. Robustness: route flip rate under paraphrase, and the safety invariant")
    for router_name, per_query in table.items():
        avg = sum(per_query.values()) / len(per_query)
        print(f"    {router_name:15} avg_flip_rate={avg:.2f}")
    print(f"    semantic boundary query flip_rate={boundary[0]:.2f}  far-from-boundary query flip_rate={boundary[1]:.2f}")
    print(f"    escalation safety invariant: {'passed' if escalation_safety.passed else 'FAILED'}")
    print(f"    reasoning-mode safety invariant: {'passed' if reasoning_safety.passed else 'FAILED'}")
    print()

    end_to_end = run_end_to_end_demo("I forgot my account password and cannot log in")
    print(format_decision(end_to_end, title="13. End-to-end: classify, threshold, dispatch, fallback"))
    print()

    print("All twelve sub-variants and the end-to-end pipeline completed without exhausting their scripts.")


if __name__ == "__main__":
    main()
