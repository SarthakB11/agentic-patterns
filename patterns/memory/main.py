"""Memory pattern: short-term buffers plus long-term vector retrieval.

Memory is the set of mechanisms that let an agent carry information across
turns and sessions, beyond a single prompt. Short-term memory is the
working context the model can currently see, bounded by the token window.
Long-term memory is an external store, written to selectively and pulled
back into the window by similarity search when relevant. This pattern is
the plumbing deciding what to keep, evict, persist, and retrieve.

This demo runs every sub-variant end to end, entirely offline against
`MockProvider` and `HashEmbedder`, with scripted, coherent conversations,
no network call and no API key:

1. Vector store fundamentals: upsert, cosine top-k, threshold exclusion.
2. Short-term memory: full buffer, sliding window, summarization/
   compaction, token-budget eviction, and context editing.
3. Retrieval scoring: plain top-k, recency-weighted, and diversity re-rank.
4. Semantic memory: facts keyed by subject, conflict resolved by overwrite.
5. Episodic memory: a failed attempt's lesson recalled by a later attempt.
6. Procedural memory: standing rules pinned to the system prompt.
7. Write policy: hot-path vs background writing, and sleep-time
   consolidation of a contradiction.
8. The pattern's headline scenario: a fact told to the agent in session
   one is recalled in session two, from an empty short-term window.
9. Context assembler: every source merged under a tight token budget,
   procedural rules kept pinned regardless of what gets evicted.
10. MemGPT-style paged memory: recursive summarization on overflow, plus
    model-driven paging through function calls.
11. Memory-as-tools: the model decides when to store or retrieve, instead
    of a fixed pre-retrieval step.
12. File-directory memory: a peer of the vector store, no embedding index.
13. Mem0-style update decision: ADD/UPDATE/DELETE/NOOP resolves a
    same-claim, different-key conflict that plain overwrite misses.
14. Forgetting: decay, reinforcement, TTL, capacity bound, and
    intent-aware deletion, so the store no longer only ever grows.
15. Offline recall benchmark: a LongMemEval-style ability taxonomy with
    abstention scoring, comparing the overwrite and Mem0-style backends.
16. Sleep-time compute: one offline pre-derivation amortized across
    several later queries, with the crossover made visible in call counts.

Run it from the repository root:

    python -m patterns.memory.main

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key set) and/or `AGENTIC_PATTERNS_EMBEDDER=openai`
(with `OPENAI_API_KEY` set) to run the same code against real APIs instead
of the mock and hash embedder. No source change is required: every demo
function builds its provider and embedder through
`agentic_patterns.get_provider` / `agentic_patterns.get_embedder`.
"""

from __future__ import annotations

from patterns.memory import (
    assembler,
    episodic,
    file_memory,
    forgetting,
    mem0_update,
    memgpt,
    memory_bench,
    memory_tools,
    procedural,
    retrieval,
    semantic,
    short_term,
    sleep_time,
    vector_store,
    write_policy,
)


def main() -> None:
    """Run every memory sub-variant demo and print a readable transcript."""
    print("MEMORY PATTERN: short-term buffers + long-term vector retrieval\n")

    print("=== 1. Vector store fundamentals ===")
    hits = vector_store.run_vector_store_demo()
    for h in hits:
        print(f"  {h.record.id}: similarity={h.similarity:.3f}  {h.record.text}")
    print("  (a sub-threshold item was excluded, not padded into the results)")
    print()

    print("=== 2. Short-term memory: buffer, window, summary, budget, context editing ===")
    st = short_term.run_short_term_demo()
    print(f"  full buffer holds all {st['full_turn_count']} turns")
    print(f"  sliding window (N=3) holds {st['window_turn_count']}: {st['window_contents']}")
    print(f"  summarization compacted on turn {st['summary_compacted_on_turn']}")
    print(f"    running summary: {st['running_summary']}")
    print(f"    kept verbatim after compaction: {st['summary_kept_verbatim']}")
    print(f"  token budget {st['budget_limit']}: trimmed to {st['trimmed_token_total']} tokens "
          f"across {st['trimmed_message_count']} messages")
    print(f"  context editing: {st['tool_messages_before_edit']} tool results -> "
          f"{st['tool_messages_after_edit']} kept (stale ones dropped, not summarized)")
    print()

    print("=== 3. Retrieval scoring strategies ===")
    rt = retrieval.run_retrieval_demo()
    print(f"  plain top-k order: {rt['plain_top_k']}")
    print(f"  recency-weighted order: {rt['recency_weighted']}")
    print(f"  diversity re-ranked (top 2): {rt['diversity_reranked']}")
    print()

    print("=== 4. Semantic memory (facts / profile) ===")
    sem = semantic.run_semantic_demo()
    print(f"  records stored: {sem['record_count']} (2 subjects, not 3 writes: no accumulation)")
    print(f"  plan write was a conflict overwrite: {sem['plan_write_was_overwrite']}")
    print(f"  recalled: {sem['recalled_fact']}")
    print(f"  answer: {sem['answer']}")
    print()

    print("=== 5. Episodic memory (past outcomes, Reflexion-style) ===")
    epi = episodic.run_episodic_demo()
    print(f"  lesson recorded: {epi['lesson_recorded']}")
    print(f"  lessons recalled for a similar task: {epi['lessons_recalled']}")
    print(f"  retry result: {epi['retry_result']}")
    print()

    print("=== 6. Procedural memory (rules injected into the prompt) ===")
    proc_answer = procedural.run_procedural_demo()
    print(f"  answer, following standing rules: {proc_answer}")
    print()

    print("=== 7. Write policy: hot path vs background, plus consolidation ===")
    wp = write_policy.run_write_policy_demo()
    print(f"  hot-path store state: {wp['hot_path_state']}")
    print(f"  background store state: {wp['background_state']}")
    print(f"  states equal: {wp['states_equal']}")
    print(f"  consolidated 'plan' -> {wp['resolved_plan']}")
    print()

    print("=== 8. End-to-end round trip across two sessions ===")
    two_session = write_policy.run_two_session_demo()
    print(f"  session 1 wrote: {two_session['facts_written_in_session_1']}")
    print(f"  session 2 window size at query time: {two_session['session_2_window_size_at_query']} "
          "(fresh, nothing carried over)")
    print(f"  session 2 retrieved: {two_session['retrieved_in_session_2']}")
    print(f"  session 2 answer: {two_session['answer']}")
    print()

    print("=== 9. Context assembler: merge under budget, rules stay pinned ===")
    ctx = assembler.run_assembler_demo()
    print(f"  system prompt (pinned, never trimmed): {ctx.system!r}")
    print(f"  assembled {len(ctx.messages)} message(s), {ctx.total_tokens} tokens")
    print(f"  dropped to fit budget: {ctx.dropped}")
    assert "metric units" in ctx.system, "procedural rule must stay pinned even under a tight budget"
    print()

    print("=== 10. MemGPT-style paged memory ===")
    mem = memgpt.run_memgpt_demo()
    print(f"  main context now: {mem.main}")
    print(f"  external context: {list(mem.external)}")
    print(f"  page-call sequence: {mem.page_events}")
    print()

    print("=== 11. Memory-as-tools (model decides when to store/retrieve) ===")
    tools_answer = memory_tools.run_memory_tools_demo()
    print(f"  final answer via tool-driven retrieval: {tools_answer}")
    print()

    print("=== 12. File-directory memory backend ===")
    fm = file_memory.run_file_memory_demo()
    print(f"  {fm['create_result']}")
    print(f"  read before update: {fm['read_before_update']!r}")
    print(f"  {fm['update_result']}")
    print(f"  read after update: {fm['read_after_update']!r}")
    print(f"  files: {fm['files']}")
    print()

    print("=== 13. Mem0-style update decision (ADD/UPDATE/DELETE/NOOP) ===")
    mem0 = mem0_update.run_mem0_update_demo()
    print(f"  operations: {mem0['operations']}")
    print(f"  record count after each turn: {mem0['record_count_after_each_turn']}")
    print("  (a same-key overwrite would have missed the plan/subscription rename)")
    print()

    print("=== 14. Forgetting: decay, TTL, capacity bound, intent-aware delete ===")
    forget = forgetting.run_forgetting_demo()
    print(f"  decay swept: {forget['decay_deleted']} (reinforced sibling survived: "
          f"{forget['active_note_survived_decay']})")
    print(f"  TTL swept: {forget['ttl_deleted']}")
    print(f"  capacity bound evicted: {forget['capacity_deleted']} -> "
          f"{forget['capacity_final_size']} records left (reinforced note still survives: "
          f"{forget['active_note_survived_capacity']})")
    print(f"  intent-aware delete removed: {forget['intent_deleted']} "
          f"(unrelated note survived: {forget['current_job_note_survived_intent_delete']})")
    print()

    print("=== 15. Offline recall benchmark (LongMemEval-style abilities) ===")
    bench = memory_bench.run_memory_bench_demo()
    print(f"  overwrite backend accuracy: {bench['overwrite_accuracy']:.2f} "
          f"by ability: {bench['overwrite_accuracy_by_ability']}")
    print(f"  overwrite knowledge-update answer: {bench['overwrite_knowledge_update_answer']!r} "
          f"(correct: {bench['overwrite_knowledge_update_correct']})")
    print(f"  mem0 knowledge-update answer: {bench['mem0_knowledge_update_answer']!r} "
          f"(correct: {bench['mem0_knowledge_update_correct']})")
    print()

    print("=== 16. Sleep-time compute: offline pre-derivation, amortized ===")
    sleep = sleep_time.run_sleep_time_demo()
    print(f"  learned context: {sleep['learned_context']}")
    print(f"  path A (test-time only) online calls: {sleep['path_a_online_calls']}")
    print(f"  path B (sleep-time) online calls: {sleep['path_b_online_calls']} "
          f"(+1 offline sleep pass = {sleep['path_b_total_calls']} total)")
    print(f"  fallback queries (not anticipated by the sleep pass): {sleep['fallback_queries']}")
    print(f"  at n=1, no online-call advantage: {sleep['single_query_no_advantage']}")
    print()

    print("All sixteen sections completed without exhausting their scripts.")


if __name__ == "__main__":
    main()
