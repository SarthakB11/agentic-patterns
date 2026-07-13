"""Guardrails pattern: checkpoints around a model, plus architectural guards.

A guardrail is a checkpoint that inspects data crossing a trust boundary
around a language model and decides whether to allow it, change it, or
block it. Input guards run before the model sees a request; output guards
run after generation; a pre-tool guard sits between the model and any tool
call it wants to make. The guiding principle is defense in depth: no single
guard is enough, so guards are layered and every decision is logged. The
2025-2026 shift this folder also covers: checkpoint inspection alone is not
a reliable defense against indirect prompt injection, so several modules
here constrain the architecture instead, removing an injection's path to a
side effect rather than trying to detect it.

This demo runs fifteen scenarios end to end, entirely offline against
`MockProvider` with scripted, coherent conversations, no network call and
no API key:

1. Input guards: a prompt-injection tripwire, an off-topic refusal, and a
   length-fixed request that still reaches the model.
2. PII masking: a raw email never reaches the model; the reply is unmasked
   for the human afterward.
3. PII redaction: a reply that itself surfaces personal data is redacted,
   irreversibly, before it reaches the user.
4. Retrieval guard: one clean chunk, one chunk with PII redacted, and one
   poisoned chunk dropped before any of it enters the prompt.
5. Output schema guard: a malformed triage response triggers exactly one
   reask, then validates.
6. Output schema guard: two malformed responses in a row exhaust the retry
   budget and the pipeline returns a safe fallback, never raw JSON.
7. Moderation guard: a blocklisted draft reply is refrained, not sent.
8. Groundedness guard: one answer is fully supported by context, a second
   contains a fabricated claim the context does not support.
9. Pre-tool (execution) guard: a disallowed tool, an out-of-range argument,
   an over-threshold call that a human approves, one a human denies, and a
   clean call.
10. Architectural guard (Plan-Then-Execute): a tool's return value carries
    an embedded, injected instruction; the plan was already committed
    before that text existed, so the injection has no path to a new or
    different action.
11. Dual LLM (CaMeL-lite): a quarantined extraction strips an embedded
    instruction from a tool result, and a capability policy blocks an
    unauthorized recipient and raw, unquarantined tool text at the sink.
12. Policy engine (Progent-lite): a declarative policy narrows without
    approval and blocks an expansion that was denied, and a hard deny rule
    still wins over an LLM-authored policy that tries to allow everything.
13. Reasoning auditor (AlignmentCheck-style): a hijacked reasoning trace
    tripwires on a keyword match, and a subtler one escalates to a
    scripted auditor model.
14. Injection suite (AgentDojo-lite): utility versus attack-success-rate
    across three defenses, showing a regex guard stopped by an adaptively
    phrased case that the capability layer still blocks.
15. Design patterns: Action-Selector never lets a tool result reach the
    model at all; Context-Minimization drops the raw request once its
    intent is extracted.

Run it from the repository root:

    python -m patterns.guardrails.main

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key in the environment) to run the same code against
a real model instead of the mock. No source change is required; every demo
function builds its provider through `agentic_patterns.get_provider`.
"""

from __future__ import annotations

from patterns.guardrails import (
    architecture,
    design_patterns,
    dual_llm,
    groundedness,
    injection_suite,
    policy_engine,
    pretool_guard,
    reasoning_auditor,
    retrieval_guard,
    scenarios,
)


def main() -> None:
    """Run every guardrails sub-variant demo and print a readable transcript."""
    print("GUARDRAILS PATTERN: checkpoints around a model, plus architectural guards\n")

    scenarios.run_input_guard_demo()
    print()

    pii_result = scenarios.run_pii_masked_demo()
    assert "jane.doe@example.com" not in str(pii_result.value)
    print()

    redact_result = scenarios.run_pii_redact_demo()
    assert "jane.doe@example.com" not in str(redact_result.value)
    print()

    kept = retrieval_guard.run_retrieval_guard_demo()
    assert "doc-3" not in [c.id for c in kept]
    print()

    reask_result = scenarios.run_schema_reask_demo()
    assert reask_result.passed and reask_result.retries == 1
    print()

    exhausted_result = scenarios.run_retries_exhausted_demo()
    assert not exhausted_result.passed and exhausted_result.stop_reason == "retries_exhausted"
    print()

    moderation_result = scenarios.run_moderation_demo()
    assert not moderation_result.passed
    print()

    grounded, ungrounded = groundedness.run_groundedness_demo()
    assert grounded.passed and not ungrounded.passed
    print()

    pretool_guard.run_pretool_guard_demo()
    print()

    plan_result = architecture.run_plan_then_execute_demo()
    executed_as_planned = [(s.tool_name, s.arguments) for s in plan_result.executed] == plan_result.planned_calls
    assert executed_as_planned
    print()

    legit, blocked_dest, blocked_quarantine = dual_llm.run_dual_llm_demo()
    assert not any(s.blocked for s in legit.executed)
    assert any(s.blocked for s in blocked_dest.executed) and any(s.blocked for s in blocked_quarantine.executed)
    print()

    _, policy_results = policy_engine.run_policy_engine_demo()
    assert not policy_results[-1].passed  # the LLM-authored policy cannot override the hard deny rule
    print()

    aligned, hijacked, subtle = reasoning_auditor.run_reasoning_auditor_demo()
    assert aligned.passed and not hijacked.passed and not subtle.passed
    print()

    suite_rows = injection_suite.run_injection_suite_demo()
    by_name = {row.config_name: row for row in suite_rows}
    assert by_name["undefended"].attack_success_rate == 1.0
    assert 0.0 < by_name["regex_input_guard"].attack_success_rate < 1.0
    assert by_name["capability_layer"].attack_success_rate == 0.0 and by_name["capability_layer"].utility == 1.0
    print()

    action_clean, action_injected = design_patterns.run_action_selector_demo()
    assert action_clean.model_calls == 1 and not action_clean.saw_tool_observation
    print()

    minimization_result = design_patterns.run_context_minimization_demo()
    assert not minimization_result.raw_request_leaked
    print()

    print("All fifteen scenarios completed without exhausting their scripts.")


if __name__ == "__main__":
    main()
