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
    memgpt,
    memory_tools,
    procedural,
    retrieval,
    semantic,
    short_term,
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

    print("All twelve sections completed without exhausting their scripts.")


if __name__ == "__main__":
    main()
