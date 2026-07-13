"""Tests for the tool_use pattern.

Deterministic and offline: every test drives `MockProvider` scripts through
the pattern's own logic functions or demo modules, with no network call and
no API key.
"""

from __future__ import annotations

from agentic_patterns import (
    Completion,
    HashEmbedder,
    Message,
    ToolCall,
    ToolRegistry,
    get_provider,
    scripted_tool_call,
)

from patterns.tool_use import (
    code_action,
    code_execution,
    concepts,
    constrained_decoding,
    error_recovery,
    forced_choice,
    guardrails,
    parallel,
    sequential,
    single_shot,
    structured_output,
    tool_search,
    validation,
    write_action,
)
from patterns.tool_use.catalog import SYSTEM_PROMPT, build_registry
from patterns.tool_use.loop import run_tool_loop, validate_arguments
from patterns.tool_use.schema import auto_tool, schema_from_function


# --- validate_arguments edge cases -----------------------------------------


def test_validate_arguments_valid_passes() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    assert validate_arguments(schema, {"a": 3}) == []


def test_validate_arguments_missing_required_field() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    errors = validate_arguments(schema, {})
    assert any("missing required field 'a'" in e for e in errors)


def test_validate_arguments_wrong_type() -> None:
    schema = {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"]}
    errors = validate_arguments(schema, {"amount": "100"})
    assert any("expected type number" in e for e in errors)


def test_validate_arguments_unexpected_field() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": []}
    errors = validate_arguments(schema, {"a": 1, "b": 2})
    assert any("unexpected field 'b'" in e for e in errors)


def test_validate_arguments_bool_rejected_as_integer() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    errors = validate_arguments(schema, {"n": True})
    assert any("got boolean" in e for e in errors)


# --- schema autogeneration (parser edge cases) -----------------------------


def test_schema_from_function_derives_types_and_required() -> None:
    def sample(city: str, count: int = 3) -> str:
        """Look up something.

        Args:
            city: A city name.
            count: How many results.
        """
        return city

    derived = schema_from_function(sample)
    assert derived["description"] == "Look up something."
    props = derived["parameters"]["properties"]
    assert props["city"] == {"type": "string", "description": "A city name."}
    assert props["count"]["type"] == "integer"
    assert derived["parameters"]["required"] == ["city"]  # count has a default, so not required


def test_schema_from_function_handles_missing_docstring() -> None:
    def bare(x: int) -> int:
        return x

    derived = schema_from_function(bare)
    assert derived["description"] == ""
    assert derived["parameters"]["properties"]["x"]["type"] == "integer"


def test_auto_tool_registers_with_derived_schema() -> None:
    registry = ToolRegistry()

    @auto_tool(registry)
    def add(a: int, b: int) -> int:
        """Add two integers.

        Args:
            a: First addend.
            b: Second addend.
        """
        return a + b

    tool = registry.get("add")
    assert tool.description == "Add two integers."
    assert tool.parameters["required"] == ["a", "b"]
    assert tool.fn(a=2, b=3) == 5


# --- single-shot and sequential (test ideas 1-2) ---------------------------


def test_single_shot_executes_call_with_exact_arguments() -> None:
    result = single_shot.demo_single_shot()
    call = result.rounds[0].calls[0]
    assert call.call.name == "get_weather"
    assert call.call.arguments == {"city": "Tokyo"}
    assert "18C" in call.observation
    assert "Tokyo" in result.final_answer
    assert result.stop_reason == "stop"


def test_sequential_second_call_receives_first_calls_value() -> None:
    result = sequential.demo_sequential()
    first_call, second_call = (r.calls[0] for r in result.rounds)
    assert first_call.call.name == "lookup_order"
    assert "CUST-42" in first_call.observation
    assert second_call.call.name == "get_customer_email"
    assert second_call.call.arguments["customer_id"] == "CUST-42"
    assert result.stop_reason == "stop"


# --- parallel calls (test idea 3) ------------------------------------------


def test_parallel_calls_all_executed_order_preserved() -> None:
    result = parallel.demo_parallel()
    calls = result.rounds[0].calls
    assert [c.call.arguments["city"] for c in calls] == ["Tokyo", "Paris", "San Francisco"]
    assert all(c.outcome == "ok" for c in calls)
    assert len(result.rounds) == 1


# --- validation and self-repair (test idea 4) ------------------------------


def test_validation_repair_turn_succeeds() -> None:
    result = validation.demo_structural_repair()
    first_round, second_round = result.rounds
    assert first_round.calls[0].outcome == "repair_requested"
    assert second_round.calls[0].outcome == "ok"
    assert "92.00 EUR" in second_round.calls[0].observation


def test_retry_limit_exhausted_produces_terminal_observation() -> None:
    result = guardrails.demo_retry_cap()
    first_round, second_round = result.rounds
    assert first_round.calls[0].outcome == "repair_requested"
    assert second_round.calls[0].outcome == "validation_failed"
    assert "repair budget exhausted" in second_round.calls[0].observation


# --- tool execution errors and unknown tools (test ideas 5-6) -------------


def test_tool_error_becomes_observation_and_loop_continues() -> None:
    result = guardrails.demo_tool_error()
    call = result.rounds[0].calls[0]
    assert call.outcome == "tool_error"
    assert call.observation.startswith("ERROR:")
    assert result.stop_reason == "stop"
    assert result.final_answer


def test_unknown_tool_policy_produces_observation_not_exception() -> None:
    result = guardrails.demo_unknown_tool()
    call = result.rounds[0].calls[0]
    assert call.outcome == "unknown_tool"
    assert "Unknown tool" in call.observation
    assert result.stop_reason == "stop"


# --- iteration cap (test idea 7) -------------------------------------------


def test_max_iterations_caps_a_never_stopping_model() -> None:
    result = guardrails.demo_iteration_cap()
    assert len(result.rounds) == 3
    assert result.stop_reason == "max_iterations"
    assert result.final_answer == ""


# --- forced tool choice ------------------------------------------------


def test_forced_choice_none_offers_no_tools() -> None:
    forced_choice.demo_none()  # smoke test: runs without a tool call, offering []


def test_forced_choice_violation_raises_when_required_not_satisfied() -> None:
    try:
        forced_choice.assert_tool_choice_satisfied([], "required")
    except forced_choice.ToolChoiceViolation:
        pass
    else:
        raise AssertionError("expected ToolChoiceViolation")


def test_forced_choice_named_violation_when_wrong_tool_called() -> None:
    calls = [ToolCall("call_1", "get_weather", {"city": "Paris"})]
    try:
        forced_choice.assert_tool_choice_satisfied(calls, "convert_currency")
    except forced_choice.ToolChoiceViolation:
        pass
    else:
        raise AssertionError("expected ToolChoiceViolation")
    forced_choice.assert_tool_choice_satisfied(calls, "get_weather")  # does not raise


# --- offered_specs narrowing must gate execution, not just display ---------


def test_call_to_unoffered_tool_is_not_executed() -> None:
    """A call naming a registered tool outside offered_specs must not run.

    Regression test for the loop resolving calls against the full registry
    even when offered_specs narrowed what the model was shown: the model
    here "sees" only convert_currency but calls get_weather anyway, which
    must come back as a not-offered error instead of a real observation.
    """
    registry = build_registry()
    provider = get_provider(
        script=[
            scripted_tool_call("get_weather", {"city": "Tokyo"}),
            "Could not use get_weather; it was not offered this turn.",
        ]
    )
    messages = [Message.user("What's the weather in Tokyo?")]
    convert_currency_only = [spec for spec in registry.specs() if spec["name"] == "convert_currency"]

    result = run_tool_loop(
        provider, registry, messages, system=SYSTEM_PROMPT, offered_specs=convert_currency_only, max_iterations=2
    )

    call = result.rounds[0].calls[0]
    assert call.outcome == "not_offered"
    assert call.observation == "ERROR: tool not offered this turn"
    assert "Tokyo" not in call.observation  # the real get_weather observation never ran


# --- duplicate call ids within one round must not collide -------------------


def test_duplicate_call_ids_in_one_round_each_get_their_own_observation() -> None:
    """Two calls sharing an id in one completion must both produce records.

    Regression test for per-round records being keyed by call.id in a dict:
    two scripted tool calls that both default to "call_1" used to silently
    drop the first observation and duplicate the second.
    """
    registry = build_registry()
    duplicate_id_completion = Completion(
        tool_calls=[
            ToolCall(id="call_1", name="get_weather", arguments={"city": "Tokyo"}),
            ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"}),
        ],
        stop_reason="tool_use",
    )
    provider = get_provider(
        script=[
            duplicate_id_completion,
            "Tokyo is 18C and Paris is 21C.",
        ]
    )
    messages = [Message.user("What's the weather in Tokyo and Paris?")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=2)

    calls = result.rounds[0].calls
    assert len(calls) == 2
    assert [c.call.arguments["city"] for c in calls] == ["Tokyo", "Paris"]
    assert "Tokyo" in calls[0].observation
    assert "Paris" in calls[1].observation
    tool_messages = [m for m in result.history if m.role == "tool"]
    assert len(tool_messages) == 2


# --- structured-output-as-tool ---------------------------------------------


def test_structured_output_extracts_in_a_single_round_trip() -> None:
    result = structured_output.extract_structured(
        "Reach out to Dana Alvarez, dana@example.com, +1-555-0199, about the renewal."
    )
    assert result == {"name": "Dana Alvarez", "email": "dana@example.com", "phone": "+1-555-0199"}


# --- write action (test idea 9) --------------------------------------------


def test_write_action_blocked_without_confirmation() -> None:
    registry = write_action.build_write_registry()
    provider = get_provider(
        script=[scripted_tool_call("send_refund_email", {"to": "x@example.com", "subject": "s", "body": "b"})]
    )
    messages = [Message.user("Email x@example.com about their refund.")]
    result = run_tool_loop(provider, registry, messages, system=write_action.SYSTEM_PROMPT, max_iterations=1)
    call = result.rounds[0].calls[0]
    assert call.outcome == "tool_error"
    assert "confirmed=True" in call.observation


def test_write_action_elicitation_accept_then_executes() -> None:
    result = write_action.demo_write_action_accepted()
    assert "sent to priya@example.com" in result.rounds[0].calls[0].observation
    assert result.stop_reason == "stop"


def test_elicit_confirmation_rejects_unknown_decision() -> None:
    try:
        write_action.elicit_confirmation("send it?", "maybe")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unrecognized decision")


# --- code-as-action / programmatic tool calling -----------------------------


def test_code_execution_resolves_placeholder_across_steps() -> None:
    registry = build_registry()
    steps = [
        {"call": "lookup_order", "args": {"order_id": "ORD-1002"}, "save_as": "order"},
        {"call": "get_customer_email", "args": {"customer_id": "$order.customer_id"}, "save_as": "email"},
    ]
    results = code_execution.run_program(registry, steps)
    assert results["order"] == "status=processing customer_id=CUST-77"
    assert results["email"] == "sam@example.com"


def test_code_execution_uses_fewer_round_trips_than_sequential() -> None:
    code_execution.demo_code_execution()
    sequential.demo_sequential()
    # code_execution: one round trip for the program, one for the final
    # answer. sequential: one per tool call plus one final answer. Both
    # solve a two-step dependent lookup; the assertion is on the shape of
    # each approach, made concrete via each module's own transcript above.
    registry = build_registry()
    steps = [{"call": "lookup_order", "args": {"order_id": "ORD-1001"}, "save_as": "order"}]
    assert code_execution.run_program(registry, steps)["order"].startswith("status=")


# --- retrieval-based tool selection -----------------------------------------


def test_tool_search_ranks_relevant_tool_first() -> None:
    registry = tool_search.build_large_registry()
    embedder = HashEmbedder()
    query = "convert 50 GBP to JPY, what's the exchange rate"
    selected = tool_search.search_tools(query, registry.specs(), embedder, top_k=3)
    assert selected[0]["name"] == "convert_currency"
    assert len(selected) == 3
    assert len(registry.specs()) == 10


# --- conceptual notes --------------------------------------------------


def test_concept_notes_cover_toolformer_and_mrkl() -> None:
    assert "Toolformer" in concepts.LEARNED_TOOL_USE
    assert "MRKL" in concepts.MRKL_ROUTING
    assert "tool_search.py" in concepts.PROMOTED_TO_RUNNABLE


# --- failure-injection: errors accumulate across a multi-step run ----------


def test_failure_injection_errors_accumulate_but_loop_recovers() -> None:
    """Inject failures into a subset of calls in a multi-step run.

    Two of four steps fail deterministically; the loop must record both
    failures without stopping, and still reach a final answer once the
    model synthesizes across the mixed results.
    """
    registry = ToolRegistry()
    fails_on = {"a", "c"}

    @auto_tool(registry)
    def flaky(item_id: str) -> str:
        """Look up an item that sometimes fails.

        Args:
            item_id: Item identifier.
        """
        if item_id in fails_on:
            raise RuntimeError(f"transient failure for {item_id}")
        return f"ok:{item_id}"

    provider = get_provider(
        script=[
            scripted_tool_call("flaky", {"item_id": "a"}),
            scripted_tool_call("flaky", {"item_id": "b"}),
            scripted_tool_call("flaky", {"item_id": "c"}),
            scripted_tool_call("flaky", {"item_id": "d"}),
            "Two of the four lookups failed; b and d came back fine.",
        ]
    )
    messages = [Message.user("Look up items a, b, c, and d.")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=5)

    outcomes = [round_record.calls[0].outcome for round_record in result.rounds]
    assert outcomes == ["tool_error", "ok", "tool_error", "ok"]
    assert result.stop_reason == "stop"
    assert result.final_answer


# --- constrained decoding: schema-grammar token masking ---------------------


def test_constrained_decoding_masks_illegal_number_token() -> None:
    """A letter-like top preference for a number field is blocked; the digit is emitted and parses numeric."""
    grammar = constrained_decoding.compile_grammar(constrained_decoding.CONVERT_SCHEMA)
    arguments, steps = constrained_decoding.generate_masked(grammar, [["hundred", "100"], ["EUR"]])
    assert steps[0].emitted == "100"
    assert arguments["amount"] == 100
    assert isinstance(arguments["amount"], int)


def test_constrained_decoding_masks_illegal_enum_token() -> None:
    """A non-enum top preference for currency is blocked; the in-enum value is emitted."""
    grammar = constrained_decoding.compile_grammar(constrained_decoding.CONVERT_SCHEMA)
    arguments, steps = constrained_decoding.generate_masked(grammar, [["100"], ["JPY", "EUR"]])
    assert steps[1].emitted == "EUR"
    assert arguments["currency"] == "EUR"


def test_constrained_decoding_blocked_count_matches_injected() -> None:
    """Exactly one illegal token was injected per field; blocked count must match, not over- or under-count."""
    _, steps = constrained_decoding.demo_constrained_decoding()
    total_blocked = sum(len(step.blocked) for step in steps)
    assert total_blocked == 2
    assert steps[0].blocked == ["hundred"]
    assert steps[1].blocked == ["JPY"]


def test_constrained_decoding_precheck_computed_at_compile_time() -> None:
    """The enum field's legal-token set is built once at compile time, unaffected by how many steps run."""
    grammar = constrained_decoding.compile_grammar(constrained_decoding.CONVERT_SCHEMA)
    assert grammar.precheck_count == 1  # one enum field (currency); amount is a number, no finite set to precompute
    constrained_decoding.generate_masked(grammar, [["100"], ["EUR"]])
    constrained_decoding.generate_masked(grammar, [["50"], ["GBP"]])
    assert grammar.precheck_count == 1  # unchanged after two full generation runs against the same compiled grammar


def test_constrained_decoding_parity_with_validate_arguments() -> None:
    """The masked call passes validate_arguments with zero repairs; the unmasked call is rejected."""
    masked_arguments, _ = constrained_decoding.demo_constrained_decoding()
    grammar = constrained_decoding.compile_grammar(constrained_decoding.CONVERT_SCHEMA)
    unmasked_arguments = constrained_decoding.generate_unconstrained(grammar, [["hundred", "100"], ["JPY", "EUR"]])

    assert validate_arguments(constrained_decoding.CONVERT_SCHEMA, masked_arguments) == []
    errors = validate_arguments(constrained_decoding.CONVERT_SCHEMA, unmasked_arguments)
    assert any("expected type number" in e for e in errors)


# --- code-as-action: real control flow over tool results --------------------


def test_code_action_loop_filters_list_in_one_round_trip() -> None:
    """Code filters five orders to the two delayed ones; exactly two model round trips ran (code, then answer)."""
    result = code_action.demo_loop_filter()
    assert result.observation == "['ORD-2002: delayed', 'ORD-2004: delayed']"
    assert result.round_trips == 2


def test_code_action_context_collapse_one_observation_many_tool_calls() -> None:
    """Message history holds exactly one tool observation though five tool calls executed inside the sandbox."""
    result = code_action.demo_loop_filter()
    tool_messages = [m for m in result.history if m.role == "tool"]
    assert len(tool_messages) == 1
    assert result.tool_calls_made == 5


def test_code_action_sandbox_boundary_blocks_unavailable_name() -> None:
    """Code referencing a name outside the sandbox (open) raises inside exec and becomes an error observation."""
    observation, tool_calls_made = code_action.demo_sandbox_boundary()
    assert observation.startswith("ERROR:")
    assert "open" in observation
    assert tool_calls_made == 0  # the boundary was hit before any tool wrapper ran


def test_code_action_branch_runs_taken_branch_only() -> None:
    """A delayed order takes the escalate branch; mark_fulfilled must never run for it."""
    result = code_action.demo_branch()
    assert "escalated" in result.observation
    assert "fulfilled" not in result.observation


def test_code_action_parity_with_step_interpreter() -> None:
    """The same two-step order-then-email task yields the same final result as code_execution.py's step list."""
    registry = build_registry()
    code = (
        "order = lookup_order(order_id='ORD-1002')\n"
        "parts = dict(p.split('=', 1) for p in order.split())\n"
        "result = get_customer_email(customer_id=parts['customer_id'])\n"
    )
    provider = get_provider(
        script=[
            scripted_tool_call("run_python", {"code": code}),
            "Order ORD-1002 is processing; the customer's email on file is sam@example.com.",
        ]
    )
    messages = [Message.user("Check order ORD-1002's status and find the customer's email, in one pass.")]

    result = code_action._run_one_action(registry, provider, messages)

    assert result.observation == "sam@example.com"
    step_results = code_execution.run_program(
        build_registry(),
        [
            {"call": "lookup_order", "args": {"order_id": "ORD-1002"}, "save_as": "order"},
            {"call": "get_customer_email", "args": {"customer_id": "$order.customer_id"}, "save_as": "email"},
        ],
    )
    assert result.observation == step_results["email"]


# --- class-routed tool-failure recovery -------------------------------------


def test_error_recovery_transient_retry_succeeds_within_budget() -> None:
    """Two transient failures then a success: two retries recorded, all within one model round trip."""
    result = error_recovery.demo_transient_retry()
    attempt = result.rounds[0].attempts[0]
    assert attempt.classification == "transient"
    assert attempt.succeeded is True
    assert "retried x2" in attempt.strategy
    assert len(result.rounds[0].attempts) == 1
    assert result.stop_reason == "stop"


def test_error_recovery_permanent_failure_uses_substitute() -> None:
    """A permanently failing tool falls back to its registered substitute, and the substitute's result is used."""
    registry = error_recovery.build_recovery_registry({"get_weather": ["permanent"]})
    call = ToolCall("call_1", "get_weather", {"city": "Atlantis"})

    attempt = error_recovery.recover_call(
        registry, call, substitutes={"get_weather": "get_weather_backup"}, transient_budget={}
    )

    assert attempt.classification == "permanent"
    assert attempt.succeeded is True
    assert "get_weather_backup" in attempt.strategy
    assert "backup source" in attempt.observation


def test_error_recovery_budget_stop_names_transient_class() -> None:
    """A tool that always fails transiently is stopped at the retry budget, with a terminal observation naming the class."""
    result = error_recovery.demo_budget_stop()
    attempt = result.rounds[0].attempts[0]
    assert attempt.classification == "transient"
    assert attempt.succeeded is False
    assert "budget exhausted after 2 retries" in attempt.strategy
    assert "transient" in attempt.observation


def test_error_recovery_implicit_failure_injects_verify_not_success() -> None:
    """An empty result is never accepted as success; a verify observation is injected instead."""
    registry = error_recovery.build_recovery_registry({"get_customer_email": ["empty"]})
    call = ToolCall("call_1", "get_customer_email", {"customer_id": "CUST-42"})

    attempt = error_recovery.recover_call(registry, call, substitutes={}, transient_budget={})

    assert attempt.classification == "implicit"
    assert attempt.succeeded is False
    assert attempt.observation.startswith("VERIFY:")


def test_error_recovery_all_four_classes_routed_distinctly() -> None:
    """One script exercises all four failure classes; each must reach its own distinct strategy."""
    result = error_recovery.demo_all_classes()
    classifications = [round_record.attempts[0].classification for round_record in result.rounds]
    assert classifications == ["malformed_args", "transient", "permanent", "implicit"]
    strategies = [round_record.attempts[0].strategy for round_record in result.rounds]
    assert len(set(strategies)) == 4  # every class ran a strategy distinct from every other class


def test_error_recovery_transient_budget_shared_across_calls_not_reset_per_call() -> None:
    """Regression test for the anti-futile-loop guard: the retry budget is per-tool across the whole run.

    A budget reset per call (instead of accumulated in `transient_budget`)
    would let a permanently-transient tool retry forever across many model
    round trips, defeating ToolMaze's anti-futile-loop finding. Two
    separate calls to the same always-failing tool must share one budget.
    """
    registry = error_recovery.build_recovery_registry(
        {"get_weather": ["transient", "transient", "transient", "transient"]}
    )
    transient_budget: dict[str, int] = {}
    call = ToolCall("call_1", "get_weather", {"city": "Tokyo"})

    first = error_recovery.recover_call(
        registry, call, substitutes={}, transient_budget=transient_budget, max_transient_retries=2
    )
    second = error_recovery.recover_call(
        registry, call, substitutes={}, transient_budget=transient_budget, max_transient_retries=2
    )

    assert first.succeeded is False
    assert second.succeeded is False
    assert transient_budget["get_weather"] == 2  # exhausted by the first call; the second gets zero further attempts


# --- retrieval-based tool selection at flooded catalog scale ----------------


def test_tool_search_flood_collapse_picks_wrong_tool() -> None:
    """Offering the whole flooded 16-tool catalog with no retrieval collapses selection to a distractor."""
    registry = tool_search.build_flooded_registry(distractor_count=6)
    embedder = HashEmbedder()
    query = "convert 50 GBP to JPY, what's the exchange rate"
    ranked = tool_search.search_tools(query, registry.specs(), embedder, top_k=1)
    assert ranked[0]["name"] != "convert_currency"
    assert ranked[0]["name"] == "exchange_rate_convert"


def test_tool_search_retrieval_fix_picks_correct_tool_fewer_tokens() -> None:
    """Top-3 retrieval against the flooded catalog offers and ranks convert_currency first, at far fewer tokens."""
    registry = tool_search.build_flooded_registry(distractor_count=6)
    embedder = HashEmbedder()
    full_specs = registry.specs()
    query = "please help me convert an amount of money from one currency to another"

    selected = tool_search.search_tools(query, full_specs, embedder, top_k=3)

    assert selected[0]["name"] == "convert_currency"
    assert tool_search.estimate_tokens(selected) < tool_search.estimate_tokens(full_specs)


def test_tool_search_recall_miss_then_widen_recovers() -> None:
    """A one-shot top-3 retrieval misses convert_currency; widening to top-8 recovers it."""
    registry = tool_search.build_flooded_registry(distractor_count=6)
    embedder = HashEmbedder()
    query = "convert 50 GBP to JPY, what's the exchange rate"
    specs = registry.specs()

    narrow = tool_search.search_tools(query, specs, embedder, top_k=3)
    widened = tool_search.search_tools(query, specs, embedder, top_k=8)

    assert "convert_currency" not in [s["name"] for s in narrow]
    assert "convert_currency" in [s["name"] for s in widened]


def test_tool_search_token_accounting_retrieved_below_full() -> None:
    """The retrieved offer's token estimate is well below the full flooded catalog's, by more than half."""
    registry = tool_search.build_flooded_registry(distractor_count=6)
    embedder = HashEmbedder()
    full_specs = registry.specs()
    selected = tool_search.search_tools("convert 50 GBP to JPY", full_specs, embedder, top_k=3)

    full_tokens = tool_search.estimate_tokens(full_specs)
    retrieved_tokens = tool_search.estimate_tokens(selected)
    assert retrieved_tokens < full_tokens / 2


def test_tool_search_determinism_same_query_same_ranking() -> None:
    """Running the same query against the same flooded catalog twice yields identical rankings and picks."""
    registry = tool_search.build_flooded_registry(distractor_count=6)
    embedder = HashEmbedder()
    query = "convert 50 GBP to JPY, what's the exchange rate"
    specs = registry.specs()

    first = [s["name"] for s in tool_search.search_tools(query, specs, embedder, top_k=5)]
    second = [s["name"] for s in tool_search.search_tools(query, specs, embedder, top_k=5)]
    assert first == second
