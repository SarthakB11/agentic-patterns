"""RAG pattern: naive, hybrid, and reranked retrieval-augmented generation.

Retrieval-augmented generation grounds a model's answer in text fetched at
query time from a corpus, instead of relying only on what the model
memorized during training. This demo builds a small internal knowledge base
for a fictional company, "Aurora Cloud," and runs the same corpus and a
handful of related questions through ten variants of the pattern:

1. Ingestion: fixed-size, overlapping, word-boundary chunking.
2. Naive dense RAG: one embed-and-cosine lookup, stuffed into the prompt.
3. Term-based retrieval: a pure-Python BM25 ranker, strong on the exact
   error code a dense lookup only partially weighs.
4. Hybrid retrieval: dense and BM25 fused with Reciprocal Rank Fusion.
5. Late-interaction retrieval: a ColBERT-style per-token MaxSim ranker,
   the middle tier between dense recall and cross-encoder precision.
6. Reranking: an LLM reads a shortlist and corrects a bag-of-words
   retriever's confused ordering.
7. Query transformation: multi-query expansion recovers both halves of a
   two-part question; HyDE recovers a vague question dense search misses.
8. Contextual retrieval: a model-written blurb makes a pronoun-orphaned
   chunk findable again.
9. Grading: a sufficiency gate catches a narrow fetch that misses half the
   answer (corrective RAG), and a relevance threshold drives an abstain
   path when nothing in the corpus matches the question at all.
10. Agentic RAG: retrieval as a tool the model calls in a loop, broadening
    its own search when the first result is incomplete.
11. Deep research: decompose into sub-questions, read into an evidence
    notebook, spawn a gap-driven follow-up round, and synthesize a cited report.
12. Graph RAG: an entity co-occurrence graph reaches a two-hop fact flat
    retrieval misses, answers a corpus-level question by map-reduce over
    community summaries, and is shown adding nothing on a single-hop question.
13. Reasoning reranking: a pointwise judge grades each candidate 0-3 with a
    reasoning trace, promoting a buried chunk and dropping zero-graded noise.
14. Order-preserving assembly: chunks kept in document order instead of
    score order, plus a k-sweep surfacing the inverted-U in chunk count.

Every step runs entirely offline against `MockProvider` with scripted,
coherent conversations: no network call, no API key, and the corpus and
queries stay the same across variants so a reader can compare how each one
handles the same material.

Run it from the repository root:

    python -m patterns.rag.main

Set `AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(with the matching API key in the environment) to run the same code against
a real model instead of the mock. No source change is required; every demo
function builds its provider through `agentic_patterns.get_provider` and its
embedder through `agentic_patterns.get_embedder`.
"""

from __future__ import annotations

from agentic_patterns import Provider, get_embedder, get_provider
from patterns.rag import (
    agentic,
    contextual,
    deep_research,
    grading,
    graph_rag,
    order_preserve,
    pipeline,
    query_transform,
    reasoning_rerank,
    rerank,
)
from patterns.rag.bm25 import bm25_retrieve, build_bm25_index
from patterns.rag.chunking import ScoredChunk
from patterns.rag.corpus import DOCUMENTS, default_chunks
from patterns.rag.dense import build_dense_index, dense_retrieve
from patterns.rag.hybrid import reciprocal_rank_fusion
from patterns.rag.late_interaction import build_late_interaction_index, late_interaction_retrieve

_RULE = "-" * 72


def _print_ranking(label: str, results: list[ScoredChunk]) -> None:
    for scored in results:
        print(f"  {label:>6}  {scored.chunk.id:<22} score={scored.score:.4f}")


def _print_answer(result: pipeline.RagResult) -> None:
    print(f"  query: {result.query}")
    print(f"  context: {[c.id for c in result.context_chunks]}")
    print(f"  answer: {result.answer.answer}")
    print(f"  citations: {result.answer.citations}  abstained: {result.answer.abstained}")


def main() -> None:
    """Run all fourteen RAG variant demos and print a readable transcript."""
    print("RAG PATTERN: naive, hybrid, graph, and reasoning-reranked retrieval-augmented generation\n")

    # 1. Ingestion --------------------------------------------------------
    chunks = default_chunks()
    print("=== 1. Ingestion: chunk the Aurora Cloud knowledge base ===")
    print(f"  {len(DOCUMENTS)} documents chunked into {len(chunks)} overlapping chunks")
    for chunk in chunks[:3]:
        print(f"  {chunk.id:<22} [{chunk.start}:{chunk.end}] {chunk.text[:60]!r}...")
    print(f"  ... and {len(chunks) - 3} more\n")

    embedder = get_embedder()
    dense_index = build_dense_index(chunks, embedder)
    bm25_index = build_bm25_index(chunks)

    # 2. Naive dense RAG ----------------------------------------------------
    print("=== 2. Naive dense RAG (one embed-and-cosine lookup) ===")
    naive_result = pipeline.run_naive_rag_demo(dense_index=dense_index, bm25_index=bm25_index, embedder=embedder)
    _print_answer(naive_result)
    print()

    # 3. Term-based retrieval (BM25) --------------------------------------
    print("=== 3. Term-based retrieval: BM25 favors the exact error code ===")
    bm25_query = "What does the error code ERR_RATE_LIMIT_1004 mean and how is it returned?"
    bm25_result = pipeline.answer_question(
        bm25_query,
        dense_index=dense_index,
        bm25_index=bm25_index,
        embedder=embedder,
        provider=_provider_for_error_code_answer(),
        retrieval="bm25",
        fetch_k=1,
        top_k=1,
    )
    _print_answer(bm25_result)
    print()

    # 4. Hybrid retrieval + RRF --------------------------------------------
    print("=== 4. Hybrid retrieval: dense + BM25 fused with RRF ===")
    hybrid_query = "How does the aurora-primary on-call rotation escalate if the primary does not respond?"
    dense_ranking = dense_retrieve(hybrid_query, dense_index, embedder, top_k=5)
    bm25_ranking = bm25_retrieve(hybrid_query, bm25_index, top_k=5)
    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=60)
    print(f"  query: {hybrid_query}")
    _print_ranking("dense", dense_ranking)
    _print_ranking("bm25", bm25_ranking)
    _print_ranking("fused", fused[:5])
    print()

    # 5. Late-interaction retrieval ----------------------------------------
    print("=== 5. Late-interaction retrieval: per-token MaxSim (ColBERT-style) ===")
    late_index = build_late_interaction_index(chunks, embedder)
    late_query = "What is the first mitigation step for a SEV1 incident caused by a recent deploy?"
    late_results = late_interaction_retrieve(late_query, late_index, embedder, top_k=3)
    print(f"  query: {late_query}")
    _print_ranking("late", late_results)
    print()

    # 6. Reranking -----------------------------------------------------------
    print("=== 6. Reranking: an LLM corrects a bag-of-words retriever's ordering ===")
    rerank_query, before, after = rerank.run_rerank_demo(dense_index=dense_index, embedder=embedder)
    print(f"  query: {rerank_query}")
    print("  before rerank (dense order):")
    _print_ranking("dense", before)
    print("  after rerank (LLM listwise order):")
    _print_ranking("rerank", after)
    print()

    # 7a. Query transformation: multi-query --------------------------------
    print("=== 7a. Query transformation: multi-query expansion ===")
    mq_query, sub_queries, mq_context, mq_answer = query_transform.run_multi_query_demo(
        dense_index=dense_index, embedder=embedder
    )
    print(f"  query: {mq_query}")
    print(f"  sub-queries: {sub_queries}")
    print(f"  fused context: {[c.id for c in mq_context]}")
    print(f"  answer: {mq_answer.answer}")
    print(f"  citations: {mq_answer.citations}")
    print()

    # 7b. Query transformation: HyDE ----------------------------------------
    print("=== 7b. Query transformation: HyDE (hypothetical document embedding) ===")
    hyde_query, hypothetical, hyde_results = query_transform.run_hyde_demo(dense_index=dense_index, embedder=embedder)
    print(f"  query: {hyde_query}")
    print(f"  hypothetical document: {hypothetical}")
    _print_ranking("hyde", hyde_results)
    print()

    # 8. Contextual retrieval ------------------------------------------------
    print("=== 8. Contextual retrieval: a blurb rescues a pronoun-orphaned chunk ===")
    ctx_query, blurb, ctx_before, ctx_after = contextual.run_contextual_demo(chunks=chunks, embedder=embedder)
    print(f"  query: {ctx_query}")
    print(f"  generated blurb: {blurb!r}")
    print("  before (orphan chunk buried):")
    _print_ranking("dense", ctx_before)
    print("  after (orphan chunk on top):")
    _print_ranking("dense", ctx_after)
    print()

    # 9a. Grading: sufficient-context gate (corrective RAG) -----------------
    print("=== 9a. Grading: sufficient-context gate widens a narrow fetch ===")
    suff_query, narrow, narrow_verdict, wide, wide_verdict = grading.run_sufficiency_demo(
        dense_index=dense_index, embedder=embedder
    )
    print(f"  query: {suff_query}")
    narrow_ids = [c.id for c in narrow]
    wide_ids = [c.id for c in wide]
    print(f"  narrow fetch {narrow_ids}: sufficient={narrow_verdict.sufficient} ({narrow_verdict.reasoning})")
    print(f"  widened fetch {wide_ids}: sufficient={wide_verdict.sufficient} ({wide_verdict.reasoning})")
    print()

    # 9b. Grading: relevance threshold and the abstain path -----------------
    print("=== 9b. Grading: abstain when nothing clears the relevance threshold ===")
    abstain_result = pipeline.run_abstain_demo(dense_index=dense_index, bm25_index=bm25_index, embedder=embedder)
    _print_answer(abstain_result)
    assert abstain_result.answer.abstained
    print()

    # 10. Agentic RAG -----------------------------------------------------
    print("=== 10. Agentic RAG: retrieval as a tool, called in a loop ===")
    agentic_result = agentic.run_agentic_rag_demo(dense_index=dense_index, embedder=embedder)
    for line in agentic_result.transcript:
        print(f"  {line}")
    print(f"  citations: {agentic_result.answer.citations}")
    print()

    # 11. Deep research -----------------------------------------------------
    print("=== 11. Deep research: decompose, read, spawn a gap-driven round, synthesize ===")
    dr_result = deep_research.run_deep_research_demo(dense_index=dense_index, embedder=embedder)
    print(f"  query: {dr_result.query}")
    print(f"  sub-questions: {dr_result.sub_question_tree}")
    print(f"  rounds used: {dr_result.rounds_used}, notebook entries: {len(dr_result.notebook)}")
    print(f"  report: {dr_result.answer.answer}")
    print(f"  citations: {dr_result.answer.citations}")
    print()

    # 12. Graph RAG -----------------------------------------------------
    print("=== 12. Graph RAG: local two-hop win, global map-reduce, skeptic no-benefit ===")
    graph_result = graph_rag.run_graph_rag_demo(dense_index=dense_index, embedder=embedder)
    local_win = graph_rag.graph_adds_value(graph_result.local_result.chunk_ids, graph_result.local_flat_baseline)
    skeptic_win = graph_rag.graph_adds_value(
        graph_result.skeptic_result.chunk_ids, graph_result.skeptic_flat_baseline
    )
    print(
        f"  local search chunks: {graph_result.local_result.chunk_ids} "
        f"(flat top-1: {graph_result.local_flat_baseline})"
    )
    print(f"  local search adds value over flat retrieval: {local_win}")
    print(f"  global answer: {graph_result.global_result.answer.answer}")
    print(
        f"  skeptic search chunks: {graph_result.skeptic_result.chunk_ids} "
        f"(flat: {graph_result.skeptic_flat_baseline})"
    )
    print(f"  skeptic search adds value over flat retrieval: {skeptic_win}")
    print()

    # 13. Reasoning reranking -----------------------------------------------
    print("=== 13. Reasoning reranking: a pointwise judge grades, then reasons before it ranks ===")
    rr_query, rr_before, rr_after, rr_judgments = reasoning_rerank.run_reasoning_rerank_demo(
        dense_index=dense_index, embedder=embedder
    )
    print(f"  query: {rr_query}")
    print(f"  before rerank (dense order): {[sc.chunk.id for sc in rr_before]}")
    print(f"  after rerank (graded, reasoned order): {[sc.chunk.id for sc in rr_after]}")
    top_judgment = rr_judgments[0]
    print(f"  top judgment: [{top_judgment.chunk_id}] grade={top_judgment.grade} because: {top_judgment.rationale}")
    print()

    # 14. Order-preserving assembly and the k sweep -------------------------
    print("=== 14. Order-preserving assembly: document order, plus the inverted-U on k ===")
    op_query, op_score_ordered, op_order_preserved, op_sweep = order_preserve.run_order_preserve_demo(
        dense_index=dense_index, embedder=embedder
    )
    print(f"  query: {op_query}")
    print(f"  score order:      {[c.id for c in op_score_ordered]}")
    print(f"  document order:   {[c.id for c in op_order_preserved]}")
    print(f"  k sweep proxy: {[(p.k, p.proxy_score) for p in op_sweep.points]}, sweet spot k={op_sweep.sweet_spot_k}")
    print()

    print(_RULE)
    print("All fourteen RAG variant demos completed without exhausting their scripts.")


def _provider_for_error_code_answer() -> Provider:
    """Scripted provider for the BM25 demo's grounded-generation call."""
    return get_provider(
        script=[
            "ERR_RATE_LIMIT_1004 is the error code returned when a request exceeds Aurora's API "
            "rate limit; the client receives an HTTP 429 response with a Retry-After header "
            "telling it when to retry [api-rate-limits#1]."
        ]
    )


if __name__ == "__main__":
    main()
