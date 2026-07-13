"""Tests for the RAG pattern.

Deterministic and offline: every test drives `MockProvider` scripts and
`HashEmbedder` through the pattern's own modules, with no network call and
no API key.
"""

from __future__ import annotations

import pytest

from agentic_patterns import HashEmbedder, MockProvider

from patterns.rag.agentic import run_agentic_rag, run_agentic_rag_demo
from patterns.rag.assembly import assemble_context, deduplicate, edge_order, fit_to_budget
from patterns.rag.bm25 import build_bm25_index, bm25_retrieve, tokenize
from patterns.rag.chunking import Chunk, Document, ScoredChunk, chunk_document
from patterns.rag.contextual import run_contextual_demo
from patterns.rag.corpus import default_chunks
from patterns.rag.dense import build_dense_index, dense_retrieve
from patterns.rag.generation import ABSTAIN_ANSWER, extract_citations, generate_grounded_answer
from patterns.rag.grading import grade_relevance, parse_sufficiency, run_sufficiency_demo
from patterns.rag.hybrid import hybrid_retrieve, reciprocal_rank_fusion
from patterns.rag.late_interaction import build_late_interaction_index, late_interaction_retrieve
from patterns.rag.pipeline import answer_question, run_abstain_demo, run_hybrid_rerank_demo, run_naive_rag_demo
from patterns.rag.query_transform import parse_multi_queries, run_hyde_demo, run_multi_query_demo
from patterns.rag.rerank import parse_rerank_order, rerank_chunks, run_rerank_demo


# --- chunking ---------------------------------------------------------------


def test_chunk_document_is_deterministic() -> None:
    doc = Document(id="doc", text="alpha beta gamma delta epsilon zeta eta theta iota kappa")
    first = chunk_document(doc, size=20, overlap=5)
    second = chunk_document(doc, size=20, overlap=5)
    assert [c.id for c in first] == [c.id for c in second]
    assert [(c.start, c.end) for c in first] == [(c.start, c.end) for c in second]
    assert len(first) > 1


def test_chunk_document_breaks_on_word_boundaries() -> None:
    doc = Document(id="doc", text="one two three four five six seven eight nine ten")
    chunks = chunk_document(doc, size=15, overlap=3)
    for chunk in chunks:
        assert not chunk.text.startswith(" ")
        # a chunk boundary never lands inside a word: every chunk is a clean
        # run of whole words once surrounding whitespace is stripped.
        assert chunk.text == chunk.text.strip()


def test_chunk_document_rejects_overlap_not_smaller_than_size() -> None:
    doc = Document(id="doc", text="short text")
    with pytest.raises(ValueError):
        chunk_document(doc, size=10, overlap=10)


# --- dense retrieval ----------------------------------------------------


def test_dense_retrieve_ranks_by_similarity() -> None:
    embedder = HashEmbedder()
    match = Chunk(id="match", source_id="d", text="rollback deploy release rollback", start=0, end=1)
    other = Chunk(id="other", source_id="d", text="refund invoice billing cycle", start=0, end=1)
    index = build_dense_index([other, match], embedder)
    results = dense_retrieve("rollback deploy release", index, embedder, top_k=2)
    assert results[0].chunk.id == "match"
    assert results[0].score > results[1].score


# --- BM25 -----------------------------------------------------------------


def test_tokenize_lowercases_and_splits_alphanumerics() -> None:
    assert tokenize("Error Code ERR_1004!") == ["error", "code", "err", "1004"]


def test_bm25_favors_rare_exact_term_over_frequent_paraphrase() -> None:
    """A rare exact term (BM25's high-IDF signal) outranks a wordier paraphrase
    that a dense bag-of-words embedder ranks first, matching the research
    brief's BM25 test idea.
    """
    exact = Chunk(id="doc-a#0", source_id="doc-a", text="System log: error code zeta9000 detected.", start=0, end=42)
    paraphrase = Chunk(
        id="doc-b#0",
        source_id="doc-b",
        text="There was a connection error during a windy tulip morning, a tulip afternoon, and a windy evening.",
        start=0,
        end=100,
    )
    embedder = HashEmbedder()
    dense_index = build_dense_index([exact, paraphrase], embedder)
    bm25_index = build_bm25_index([exact, paraphrase])
    query = "connection error code"

    dense_results = dense_retrieve(query, dense_index, embedder, top_k=2)
    bm25_results = bm25_retrieve(query, bm25_index, top_k=2)

    assert dense_results[0].chunk.id == "doc-b#0"  # dense prefers the wordier paraphrase
    assert bm25_results[0].chunk.id == "doc-a#0"  # BM25 prefers the exact rare term


def test_bm25_zero_overlap_scores_zero() -> None:
    chunk = Chunk(id="c", source_id="d", text="alpha beta gamma", start=0, end=1)
    index = build_bm25_index([chunk])
    results = bm25_retrieve("delta epsilon", index, top_k=1)
    assert results[0].score == 0.0


# --- hybrid / RRF -----------------------------------------------------------


def test_reciprocal_rank_fusion_matches_hand_computed_scores() -> None:
    c1 = Chunk(id="c1", source_id="d", text="x", start=0, end=1)
    c2 = Chunk(id="c2", source_id="d", text="y", start=0, end=1)
    c3 = Chunk(id="c3", source_id="d", text="z", start=0, end=1)
    list1 = [ScoredChunk(c1, 0.9), ScoredChunk(c2, 0.5), ScoredChunk(c3, 0.1)]
    list2 = [ScoredChunk(c3, 0.8), ScoredChunk(c1, 0.4)]

    fused = reciprocal_rank_fusion([list1, list2], k=60)
    scores = {sc.chunk.id: sc.score for sc in fused}

    assert scores["c1"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c3"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["c2"] == pytest.approx(1 / 62)
    assert [sc.chunk.id for sc in fused] == ["c1", "c3", "c2"]


def test_hybrid_retrieve_returns_fused_top_k() -> None:
    chunks = default_chunks()
    embedder = HashEmbedder()
    dense_index = build_dense_index(chunks, embedder)
    bm25_index = build_bm25_index(chunks)
    results = hybrid_retrieve(
        "How does the aurora-primary on-call rotation escalate?", dense_index, bm25_index, embedder, top_k=3, fetch_k=5
    )
    assert len(results) == 3
    assert results[0].chunk.id == "oncall-rotation#0"


# --- late interaction --------------------------------------------------


def test_late_interaction_retrieve_ranks_matching_chunk_first() -> None:
    embedder = HashEmbedder()
    match = Chunk(id="match", source_id="d", text="rollback release stable minutes", start=0, end=1)
    other = Chunk(id="other", source_id="d", text="invoice refund billing team", start=0, end=1)
    index = build_late_interaction_index([other, match], embedder)
    results = late_interaction_retrieve("rollback release minutes", index, embedder, top_k=2)
    assert results[0].chunk.id == "match"


# --- reranking --------------------------------------------------------------


def test_parse_rerank_order_reads_rank_line() -> None:
    assert parse_rerank_order("RANK: c3, c1, c2") == ["c3", "c1", "c2"]


def test_parse_rerank_order_falls_back_without_rank_prefix() -> None:
    assert parse_rerank_order("c2, c1") == ["c2", "c1"]


def test_rerank_chunks_moves_planted_chunk_to_top() -> None:
    c1 = Chunk(id="c1", source_id="d", text="irrelevant text", start=0, end=1)
    c2 = Chunk(id="c2", source_id="d", text="the actually relevant passage", start=0, end=1)
    candidates = [ScoredChunk(c1, 0.9), ScoredChunk(c2, 0.1)]
    provider = MockProvider(["RANK: c2, c1"])

    reranked = rerank_chunks("a question", candidates, provider, top_k=2)

    assert [sc.chunk.id for sc in reranked] == ["c2", "c1"]
    assert reranked[0].score > reranked[1].score
    sent_prompt = provider.calls[0]["messages"][0].content
    assert "c1" in sent_prompt and "c2" in sent_prompt


def test_run_rerank_demo_promotes_dense_last_place_chunk() -> None:
    query, before, after = run_rerank_demo()
    assert before[0].chunk.id != "billing-faq#0"  # dense ranks it last of the shortlist
    assert after[0].chunk.id == "billing-faq#0"  # reranking promotes it to first


# --- query transformation ------------------------------------------------


def test_parse_multi_queries_strips_list_markers() -> None:
    text = "1. first sub-query\n- second sub-query\n\n* third sub-query"
    assert parse_multi_queries(text) == ["first sub-query", "second sub-query", "third sub-query"]


def test_run_multi_query_demo_fuses_both_sub_query_results() -> None:
    query, sub_queries, context_chunks, answer = run_multi_query_demo()
    assert len(sub_queries) == 2
    context_ids = {c.id for c in context_chunks}
    assert "incident-runbook#1" in context_ids
    assert "incident-runbook#2" in context_ids
    assert set(answer.citations) <= context_ids


def test_run_hyde_demo_hypothetical_outranks_raw_query() -> None:
    query, hypothetical, results = run_hyde_demo()
    assert hypothetical  # a hypothetical passage was generated
    assert results[0].chunk.id == "incident-runbook#1"


# --- contextual retrieval -------------------------------------------------


def test_contextual_demo_moves_orphan_chunk_to_top() -> None:
    query, blurb, before, after = run_contextual_demo()
    before_ids = [sc.chunk.id for sc in before]
    after_ids = [sc.chunk.id for sc in after]
    assert before_ids.index("billing-faq#orphan") > 0  # buried before contextualizing
    assert after_ids[0] == "billing-faq#orphan"  # on top after contextualizing


# --- context assembly ------------------------------------------------------


def test_deduplicate_drops_near_identical_chunks() -> None:
    c1 = Chunk(id="c1", source_id="d", text="the rollback command reverts the release", start=0, end=1)
    c2 = Chunk(id="c2", source_id="d", text="the rollback command reverts the release now", start=0, end=1)
    kept = deduplicate([ScoredChunk(c1, 0.9), ScoredChunk(c2, 0.8)], threshold=0.7)
    assert [sc.chunk.id for sc in kept] == ["c1"]


def test_fit_to_budget_always_keeps_first_even_if_over_budget() -> None:
    long_chunk = Chunk(id="long", source_id="d", text=" ".join(["word"] * 50), start=0, end=1)
    kept = fit_to_budget([ScoredChunk(long_chunk, 1.0)], token_budget=5)
    assert len(kept) == 1


def test_fit_to_budget_stops_before_exceeding() -> None:
    a = Chunk(id="a", source_id="d", text=" ".join(["word"] * 5), start=0, end=1)
    b = Chunk(id="b", source_id="d", text=" ".join(["word"] * 5), start=0, end=1)
    kept = fit_to_budget([ScoredChunk(a, 1.0), ScoredChunk(b, 0.9)], token_budget=8)
    assert [sc.chunk.id for sc in kept] == ["a"]


def test_edge_order_places_top_score_at_an_edge_not_middle() -> None:
    chunks = [Chunk(id=f"c{i}", source_id="d", text=f"word{i}", start=0, end=1) for i in range(4)]
    scored = [ScoredChunk(chunks[0], 0.9), ScoredChunk(chunks[1], 0.7), ScoredChunk(chunks[2], 0.5), ScoredChunk(chunks[3], 0.3)]
    ordered = edge_order(scored)
    assert ordered[0].id == "c0"  # best score at the front
    middle_ids = {c.id for c in ordered[1:-1]}
    assert "c0" not in middle_ids and ordered[-1].id != ordered[len(ordered) // 2].id


def test_assemble_context_full_pipeline_orders_and_dedups() -> None:
    base = "the rollback command reverts the previous stable release quickly today"
    c1 = Chunk(id="c1", source_id="d", text=base, start=0, end=1)
    c2 = Chunk(id="c2", source_id="d", text=base + " now", start=0, end=1)  # near-dup of c1
    c3 = Chunk(id="c3", source_id="d", text="delta epsilon zeta", start=0, end=1)
    result = assemble_context([ScoredChunk(c1, 0.9), ScoredChunk(c2, 0.8), ScoredChunk(c3, 0.5)], token_budget=100)
    assert [c.id for c in result] == ["c1", "c3"]  # c2 dropped as a near-duplicate of c1


# --- grading ----------------------------------------------------------------


def test_grade_relevance_splits_kept_and_dropped() -> None:
    a = Chunk(id="a", source_id="d", text="x", start=0, end=1)
    b = Chunk(id="b", source_id="d", text="y", start=0, end=1)
    kept, dropped = grade_relevance([ScoredChunk(a, 0.8), ScoredChunk(b, 0.1)], threshold=0.5)
    assert [sc.chunk.id for sc in kept] == ["a"]
    assert [sc.chunk.id for sc in dropped] == ["b"]


def test_parse_sufficiency_reads_yes_and_no() -> None:
    assert parse_sufficiency("SUFFICIENT: yes\nCovers both parts.").sufficient is True
    assert parse_sufficiency("SUFFICIENT: no\nMissing the refund window.").sufficient is False


def test_sufficiency_demo_widens_from_insufficient_to_sufficient() -> None:
    query, narrow, narrow_verdict, wide, wide_verdict = run_sufficiency_demo()
    assert narrow_verdict.sufficient is False
    assert wide_verdict.sufficient is True
    assert len(wide) > len(narrow)


# --- grounded generation -----------------------------------------------


def test_extract_citations_drops_ids_not_in_valid_set() -> None:
    text = "The rollback is described here [chunk-1] and also supposedly here [chunk-99]."
    citations = extract_citations(text, valid_ids={"chunk-1"})
    assert citations == ["chunk-1"]


def test_generate_grounded_answer_abstains_without_calling_provider() -> None:
    provider = MockProvider([])
    answer = generate_grounded_answer("any question", [], provider)
    assert answer.abstained is True
    assert answer.answer == ABSTAIN_ANSWER
    assert provider.calls == []  # abstain path never reaches the model


def test_generate_grounded_answer_cites_only_supplied_chunk_ids() -> None:
    chunk = Chunk(id="incident-runbook#1", source_id="incident-runbook", text="rollback is the first step", start=0, end=1)
    provider = MockProvider(["The first step is a rollback [incident-runbook#1], also see [made-up-id]."])
    answer = generate_grounded_answer("what is the first step?", [chunk], provider)
    assert answer.citations == ["incident-runbook#1"]  # the fabricated id never resolves


# --- pipeline ---------------------------------------------------------------


def test_naive_rag_demo_produces_grounded_two_citation_answer() -> None:
    result = run_naive_rag_demo()
    assert result.answer.abstained is False
    assert set(result.answer.citations) <= {c.id for c in result.context_chunks}


def test_hybrid_rerank_demo_covers_both_halves_of_the_question() -> None:
    result = run_hybrid_rerank_demo()
    context_ids = {c.id for c in result.context_chunks}
    assert context_ids == {"billing-faq#0", "billing-faq#1"}


def test_abstain_demo_makes_no_generation_call() -> None:
    provider = MockProvider([])
    from patterns.rag.pipeline import _build_indexes

    dense_index, bm25_index = _build_indexes()
    result = answer_question(
        "xylophone quokka marmalade skateboard umbrella",
        dense_index=dense_index,
        bm25_index=bm25_index,
        embedder=HashEmbedder(),
        provider=provider,
        retrieval="dense",
        fetch_k=5,
        top_k=3,
        relevance_threshold=0.5,
    )
    assert result.answer.abstained is True
    assert provider.calls == []


def test_answer_question_rejects_unknown_retrieval_mode() -> None:
    from patterns.rag.pipeline import _build_indexes

    dense_index, bm25_index = _build_indexes()
    with pytest.raises(ValueError):
        answer_question(
            "a question",
            dense_index=dense_index,
            bm25_index=bm25_index,
            embedder=HashEmbedder(),
            provider=MockProvider([]),
            retrieval="not-a-real-mode",
        )


def test_full_pipeline_is_reproducible() -> None:
    first = run_naive_rag_demo()
    second = run_naive_rag_demo()
    assert first.answer.answer == second.answer.answer
    assert [c.id for c in first.context_chunks] == [c.id for c in second.context_chunks]
    assert first.answer.citations == second.answer.citations


# --- agentic RAG --------------------------------------------------------


def test_agentic_rag_stops_on_first_final_answer_with_no_tool_call() -> None:
    embedder = HashEmbedder()
    chunk = Chunk(id="policy#0", source_id="policy", text="Refunds are honored within fourteen days.", start=0, end=1)
    index = build_dense_index([chunk], embedder)
    provider = MockProvider(
        [
            {"tool": "search_knowledge_base", "args": {"query": "refund window"}},
            "Refunds are honored within fourteen days [policy#0].",
        ]
    )
    result = run_agentic_rag("What is the refund window?", provider, index, embedder)
    assert result.answer.citations == ["policy#0"]
    assert provider.calls[0]["tools"] is not None  # the tool spec was actually offered to the model


def test_agentic_rag_abstains_after_exhausting_max_rounds() -> None:
    embedder = HashEmbedder()
    chunk = Chunk(id="policy#0", source_id="policy", text="Refunds are honored within fourteen days.", start=0, end=1)
    index = build_dense_index([chunk], embedder)
    provider = MockProvider(
        [
            {"tool": "search_knowledge_base", "args": {"query": "q1"}},
            {"tool": "search_knowledge_base", "args": {"query": "q2"}},
        ]
    )
    result = run_agentic_rag("a question", provider, index, embedder, max_rounds=2)
    assert result.answer.abstained is True
    assert result.rounds_used == 2


def test_agentic_rag_demo_narrows_its_second_search() -> None:
    result = run_agentic_rag_demo()
    tool_calls = [line for line in result.transcript if "tool call" in line]
    assert len(tool_calls) == 2
    assert tool_calls[0] != tool_calls[1]
    assert set(result.answer.citations) == {"incident-runbook#1", "deploy-policy#1"}
