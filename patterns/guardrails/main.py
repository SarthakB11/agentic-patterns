"""Guardrails pattern: checkpoints around a model, plus one architectural guard.

A guardrail is a checkpoint that inspects data crossing a trust boundary
around a language model and decides whether to allow it, change it, or
block it. Input guards run before the model sees a request; output guards
run after generation; a pre-tool guard sits between the model and any tool
call it wants to make. The guiding principle is defense in depth: no single
guard is enough, so guards are layered and every decision is logged.

This demo runs ten scenarios end to end, entirely offline against
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

Run it from the repository root:

    python -m patterns.guardrails.main

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key in the environment) to run the same code against
a real model instead of the mock. No source change is required; every demo
function builds its provider through `agentic_patterns.get_provider`.
"""

from __future__ import annotations

from patterns.guardrails import architecture, groundedness, pretool_guard, retrieval_guard, scenarios


def main() -> None:
    """Run every guardrails sub-variant demo and print a readable transcript."""
    print("GUARDRAILS PATTERN: checkpoints around a model, plus one architectural guard\n")

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

    print("All ten scenarios completed without exhausting their scripts.")


if __name__ == "__main__":
    main()
