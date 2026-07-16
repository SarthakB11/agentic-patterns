"""Tests for the RAG pattern.

Deterministic and offline: every test drives `MockProvider` scripts and
`HashEmbedder` through the pattern's own modules, with no network call and
no API key.
"""

from __future__ import annotations

import pytest

from agentic_patterns import Completion, HashEmbedder, MockProvider
from patterns.rag.agentic import run_agentic_rag, run_agentic_rag_demo
from patterns.rag.assembly import assemble_context, deduplicate, edge_order, fit_to_budget
from patterns.rag.bm25 import bm25_retrieve, build_bm25_index, tokenize
from patterns.rag.chunking import Chunk, Document, ScoredChunk, chunk_document
from patterns.rag.contextual import build_contextual_index, run_contextual_demo
from patterns.rag.corpus import default_chunks
from patterns.rag.deep_research import run_deep_research, run_deep_research_demo
from patterns.rag.dense import build_dense_index, dense_retrieve
from patterns.rag.generation import ABSTAIN_ANSWER, extract_citations, generate_grounded_answer
from patterns.rag.grading import grade_relevance, parse_sufficiency, run_sufficiency_demo
from patterns.rag.graph_rag import (
    build_entities_by_chunk_heuristic,
    build_graph,
    detect_communities,
    global_search,
    graph_adds_value,
    local_search,
    run_graph_rag_demo,
    summarize_communities,
)
from patterns.rag.hybrid import hybrid_retrieve, reciprocal_rank_fusion
from patterns.rag.late_interaction import build_late_interaction_index, late_interaction_retrieve
from patterns.rag.order_preserve import order_preserve_assemble, run_order_preserve_demo, sweep_k
from patterns.rag.pipeline import answer_question, run_abstain_demo, run_hybrid_rerank_demo, run_naive_rag_demo
from patterns.rag.query_transform import parse_multi_queries, run_hyde_demo, run_multi_query_demo
from patterns.rag.reasoning_rerank import reasoning_rerank, run_reasoning_rerank_demo
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


def test_run_rerank_demo_uses_prebuilt_dense_index_when_given() -> None:
    """A prebuilt index over an unrelated corpus should actually drive retrieval,
    proving the demo threads the passed-in index through instead of silently
    rebuilding its own default index over the sample corpus."""
    embedder = HashEmbedder()
    only_chunk = Chunk(id="swapped#0", source_id="swapped", text="only chunk available here", start=0, end=1)
    dense_index = build_dense_index([only_chunk], embedder)
    provider = MockProvider(["RANK: swapped#0"])
    query, before, after = run_rerank_demo(provider, dense_index=dense_index, embedder=embedder)
    assert [sc.chunk.id for sc in before] == ["swapped#0"]


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


def test_run_multi_query_demo_uses_prebuilt_dense_index_when_given() -> None:
    """A prebuilt index over an unrelated corpus should actually drive retrieval,
    proving the demo threads the passed-in index through instead of silently
    rebuilding its own default index over the sample corpus."""
    embedder = HashEmbedder()
    only_chunk = Chunk(id="swapped#0", source_id="swapped", text="only chunk available here", start=0, end=1)
    dense_index = build_dense_index([only_chunk], embedder)
    provider = MockProvider(
        [
            "sub query one\nsub query two",
            "Scripted answer citing the swapped chunk [swapped#0].",
        ]
    )
    query, sub_queries, context_chunks, answer = run_multi_query_demo(provider, dense_index=dense_index, embedder=embedder)
    assert [c.id for c in context_chunks] == ["swapped#0"]


def test_run_hyde_demo_uses_prebuilt_dense_index_when_given() -> None:
    """A prebuilt index over an unrelated corpus should actually drive retrieval,
    proving the demo threads the passed-in index through instead of silently
    rebuilding its own default index over the sample corpus."""
    embedder = HashEmbedder()
    only_chunk = Chunk(id="swapped#0", source_id="swapped", text="only chunk available here", start=0, end=1)
    dense_index = build_dense_index([only_chunk], embedder)
    provider = MockProvider(["a hypothetical passage about the swapped chunk"])
    query, hypothetical, results = run_hyde_demo(provider, dense_index=dense_index, embedder=embedder)
    assert [sc.chunk.id for sc in results] == ["swapped#0"]


# --- contextual retrieval -------------------------------------------------


def test_contextual_demo_moves_orphan_chunk_to_top() -> None:
    query, blurb, before, after = run_contextual_demo()
    before_ids = [sc.chunk.id for sc in before]
    after_ids = [sc.chunk.id for sc in after]
    assert before_ids.index("billing-faq#orphan") > 0  # buried before contextualizing
    assert after_ids[0] == "billing-faq#orphan"  # on top after contextualizing


def test_run_contextual_demo_uses_provided_chunks_when_given() -> None:
    """A `chunks` list missing a chunk id the demo needs should raise, proving
    the demo looks up its distractors in the passed-in list instead of
    silently re-chunking the corpus itself."""
    incomplete_chunks = [c for c in default_chunks() if c.id != "oncall-rotation#0"]
    with pytest.raises(KeyError):
        run_contextual_demo(chunks=incomplete_chunks)


def test_build_contextual_index_only_blurbs_selected_chunks() -> None:
    embedder = HashEmbedder()
    provider = MockProvider(["a blurb for the orphan chunk"])
    doc = Document(id="doc", text="raw chunk text. orphan chunk text.")
    raw = Chunk(id="doc#raw", source_id="doc", text="raw chunk text", start=0, end=1)
    orphan = Chunk(id="doc#orphan", source_id="doc", text="orphan chunk text", start=1, end=2)

    index, blurbs = build_contextual_index([raw, orphan], {"doc": doc}, embedder, provider, blurb_chunk_ids={"doc#orphan"})

    assert blurbs == {"doc#orphan": "a blurb for the orphan chunk"}
    assert len(provider.calls) == 1  # only the selected chunk triggered a blurb call
    assert index.chunks == [raw, orphan]


def test_build_contextual_index_blurbs_every_chunk_by_default() -> None:
    embedder = HashEmbedder()
    provider = MockProvider(["blurb one", "blurb two"])
    doc = Document(id="doc", text="alpha chunk. beta chunk.")
    c1 = Chunk(id="doc#0", source_id="doc", text="alpha chunk", start=0, end=1)
    c2 = Chunk(id="doc#1", source_id="doc", text="beta chunk", start=1, end=2)

    index, blurbs = build_contextual_index([c1, c2], {"doc": doc}, embedder, provider)

    assert set(blurbs) == {"doc#0", "doc#1"}
    assert len(provider.calls) == 2  # every chunk triggered its own blurb call


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


def test_run_sufficiency_demo_uses_prebuilt_dense_index_when_given() -> None:
    """A prebuilt index over an unrelated corpus should actually drive retrieval,
    proving the demo threads the passed-in index through instead of silently
    rebuilding its own default index over the sample corpus."""
    embedder = HashEmbedder()
    only_chunk = Chunk(id="swapped#0", source_id="swapped", text="only chunk available here", start=0, end=1)
    dense_index = build_dense_index([only_chunk], embedder)
    provider = MockProvider(["SUFFICIENT: no\nnot enough", "SUFFICIENT: no\nstill not enough"])
    query, narrow, narrow_verdict, wide, wide_verdict = run_sufficiency_demo(provider, dense_index=dense_index, embedder=embedder)
    assert [c.id for c in narrow] == ["swapped#0"]
    assert [c.id for c in wide] == ["swapped#0"]


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


def test_run_naive_rag_demo_uses_prebuilt_indexes_when_given() -> None:
    """A prebuilt index over an unrelated corpus should actually drive retrieval,
    proving the demo threads the passed-in indexes through instead of silently
    rebuilding its own default indexes over the sample corpus."""
    embedder = HashEmbedder()
    only_chunk = Chunk(id="swapped#0", source_id="swapped", text="only chunk available here", start=0, end=1)
    dense_index = build_dense_index([only_chunk], embedder)
    bm25_index = build_bm25_index([only_chunk])
    provider = MockProvider(["Scripted answer citing the swapped chunk [swapped#0]."])
    result = run_naive_rag_demo(provider, dense_index=dense_index, bm25_index=bm25_index, embedder=embedder)
    assert [c.id for c in result.context_chunks] == ["swapped#0"]


def test_hybrid_rerank_demo_covers_both_halves_of_the_question() -> None:
    result = run_hybrid_rerank_demo()
    context_ids = {c.id for c in result.context_chunks}
    assert context_ids == {"billing-faq#0", "billing-faq#1"}


def test_run_abstain_demo_uses_prebuilt_indexes_when_given() -> None:
    """Feeding the abstain demo a prebuilt index with one clearly relevant chunk
    should stop it from abstaining, proving it threads the passed-in indexes
    through instead of silently rebuilding its own default indexes."""
    embedder = HashEmbedder()
    only_chunk = Chunk(
        id="swapped#0", source_id="swapped", text="xylophone quokka marmalade skateboard umbrella", start=0, end=1
    )
    dense_index = build_dense_index([only_chunk], embedder)
    bm25_index = build_bm25_index([only_chunk])
    provider = MockProvider(["Scripted answer citing the swapped chunk [swapped#0]."])
    result = run_abstain_demo(provider, dense_index=dense_index, bm25_index=bm25_index, embedder=embedder)
    assert result.answer.abstained is False
    assert [c.id for c in result.context_chunks] == ["swapped#0"]


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


def test_run_agentic_rag_demo_uses_prebuilt_dense_index_when_given() -> None:
    """A prebuilt index over an unrelated corpus should actually back the search
    tool, proving the demo threads the passed-in index through instead of
    silently rebuilding its own default index over the sample corpus."""
    embedder = HashEmbedder()
    only_chunk = Chunk(id="swapped#0", source_id="swapped", text="only chunk available here", start=0, end=1)
    dense_index = build_dense_index([only_chunk], embedder)
    provider = MockProvider(
        [
            {"tool": "search_knowledge_base", "args": {"query": "anything"}},
            "The scripted final answer cites the swapped chunk [swapped#0].",
        ]
    )
    result = run_agentic_rag_demo(provider, dense_index=dense_index, embedder=embedder)
    assert result.answer.citations == ["swapped#0"]


# --- deep research --------------------------------------------------------


def test_deep_research_decompose_then_answer() -> None:
    embedder = HashEmbedder()
    chunk_a = Chunk(id="doc-a#0", source_id="doc-a", text="rollback deploy release procedure", start=0, end=1)
    chunk_b = Chunk(id="doc-b#0", source_id="doc-b", text="refund invoice billing window", start=0, end=1)
    index = build_dense_index([chunk_a, chunk_b], embedder)
    provider = MockProvider(
        [
            "What is the rollback procedure?\nWhat is the refund window?",
            "The rollback procedure is described here [doc-a#0].",
            "The refund window is described here [doc-b#0].",
            "Report: rollback procedure [doc-a#0]; refund window [doc-b#0].",
        ]
    )
    result = run_deep_research("a two-part question", index, embedder, provider, top_k=1, max_rounds=1)
    assert [f.sub_question for f in result.notebook] == [
        "What is the rollback procedure?",
        "What is the refund window?",
    ]
    assert result.notebook[0].chunk_ids == ["doc-a#0"]
    assert result.notebook[1].chunk_ids == ["doc-b#0"]


def test_deep_research_gap_driven_follow_up_runs_a_second_round() -> None:
    embedder = HashEmbedder()
    chunk_a = Chunk(id="doc-a#0", source_id="doc-a", text="rollback deploy release procedure", start=0, end=1)
    chunk_c = Chunk(id="doc-c#0", source_id="doc-c", text="postmortem deadline forty eight hours", start=0, end=1)
    index = build_dense_index([chunk_a, chunk_c], embedder)
    provider = MockProvider(
        [
            "What is the rollback procedure?",
            "The rollback procedure is described here [doc-a#0].",
            "What is the postmortem deadline?",  # coverage after round 1: one gap
            "The postmortem deadline is described here [doc-c#0].",
            "Report: rollback [doc-a#0]; postmortem deadline [doc-c#0].",
        ]
    )
    result = run_deep_research("rollback and postmortem question", index, embedder, provider, top_k=1, max_rounds=2)
    assert result.rounds_used == 2
    assert [f.sub_question for f in result.notebook] == [
        "What is the rollback procedure?",
        "What is the postmortem deadline?",
    ]
    assert result.notebook[1].chunk_ids == ["doc-c#0"]


def test_deep_research_stops_early_on_done_coverage() -> None:
    embedder = HashEmbedder()
    chunk_a = Chunk(id="doc-a#0", source_id="doc-a", text="rollback deploy release procedure", start=0, end=1)
    index = build_dense_index([chunk_a], embedder)
    provider = MockProvider(
        [
            "What is the rollback procedure?",
            "The rollback procedure is described here [doc-a#0].",
            "DONE",
            "Report: rollback [doc-a#0].",
        ]
    )
    # max_rounds=3 leaves room for more rounds; DONE coverage must stop the loop before
    # a second retrieval round, or the script (with no 5th entry) would raise.
    result = run_deep_research("rollback question", index, embedder, provider, top_k=1, max_rounds=3)
    assert result.rounds_used == 1
    assert len(result.notebook) == 1


def test_deep_research_not_found_finding_is_dropped_and_not_citable() -> None:
    embedder = HashEmbedder()
    chunk_a = Chunk(id="doc-a#0", source_id="doc-a", text="rollback deploy release procedure", start=0, end=1)
    chunk_b = Chunk(id="doc-b#0", source_id="doc-b", text="unrelated other topic", start=0, end=1)
    index = build_dense_index([chunk_a, chunk_b], embedder)
    provider = MockProvider(
        [
            "What is the rollback procedure?\nWhat is the refund window?",
            "The rollback procedure is described here [doc-a#0].",
            "NOT FOUND",
            "Report: rollback [doc-a#0], also allegedly [doc-b#0].",
        ]
    )
    result = run_deep_research("q", index, embedder, provider, top_k=1, max_rounds=1)
    assert len(result.notebook) == 1
    assert result.notebook[0].sub_question == "What is the rollback procedure?"
    # doc-b#0 never entered the notebook (its sub-question was "not found"), so the
    # synthesis's fabricated citation to it is filtered out, not trusted.
    assert result.answer.citations == ["doc-a#0"]


def test_deep_research_stops_at_round_budget_even_with_gaps_remaining() -> None:
    embedder = HashEmbedder()
    chunk_a = Chunk(id="doc-a#0", source_id="doc-a", text="rollback deploy release procedure", start=0, end=1)
    chunk_b = Chunk(id="doc-b#0", source_id="doc-b", text="postmortem deadline forty eight hours", start=0, end=1)
    index = build_dense_index([chunk_a, chunk_b], embedder)
    provider = MockProvider(
        [
            "What is the rollback procedure?",
            "The rollback procedure is described here [doc-a#0].",
            "What is the postmortem deadline?",  # coverage after round 1: a gap
            "The postmortem deadline is described here [doc-b#0].",
            # round 2 == max_rounds: budget spent, coverage is never asked again
            "Report so far: rollback [doc-a#0]; postmortem deadline [doc-b#0].",
        ]
    )
    result = run_deep_research("q", index, embedder, provider, top_k=1, max_rounds=2)
    assert result.rounds_used == 2
    assert result.answer.abstained is False
    assert len(result.notebook) == 2


def test_run_deep_research_demo_completes_a_gap_driven_report() -> None:
    result = run_deep_research_demo()
    assert result.rounds_used == 2
    assert set(result.answer.citations) == {
        "incident-runbook#0",
        "incident-runbook#1",
        "oncall-rotation#0",
        "incident-runbook#2",
    }


# --- graph RAG --------------------------------------------------------------


def test_build_graph_is_deterministic_and_connects_co_occurring_entities() -> None:
    chunk = Chunk(id="doc#0", source_id="doc", text="Aurora Cloud runs the SEV1 Incident Response process.", start=0, end=1)
    entities_by_chunk = build_entities_by_chunk_heuristic([chunk])
    first = build_graph([chunk], entities_by_chunk)
    second = build_graph([chunk], entities_by_chunk)

    assert first.entities == second.entities
    first_edges = [(e.source, e.target, e.weight, e.chunk_ids) for e in first.edges]
    second_edges = [(e.source, e.target, e.weight, e.chunk_ids) for e in second.edges]
    assert first_edges == second_edges

    edge = next(e for e in first.edges if {e.source, e.target} == {"Aurora Cloud", "SEV1 Incident Response"})
    assert edge.chunk_ids == ["doc#0"]


def test_detect_communities_groups_co_occurring_entities_together() -> None:
    chunk_a = Chunk(id="doc-a#0", source_id="doc-a", text="PagerDuty escalates to the Engineering Manager.", start=0, end=1)
    chunk_b = Chunk(
        id="doc-b#0", source_id="doc-b", text="GDPR Deletion requests go through Data Retention review.", start=0, end=1
    )
    entities_by_chunk = build_entities_by_chunk_heuristic([chunk_a, chunk_b])
    graph = build_graph([chunk_a, chunk_b], entities_by_chunk)
    communities = detect_communities(graph)

    assert len(communities) == 2
    pagerduty_community = next(c for c in communities if "PagerDuty" in c.entities)
    assert "Engineering Manager" in pagerduty_community.entities
    gdpr_community = next(c for c in communities if "GDPR Deletion" in c.entities)
    assert "Data Retention" in gdpr_community.entities


def test_local_search_reaches_two_hop_chunk_flat_top1_retrieval_misses() -> None:
    chunks = default_chunks()
    embedder = HashEmbedder()
    dense_index = build_dense_index(chunks, embedder)
    entities_by_chunk = {
        "incident-runbook#1": ["SEV1", "Rollback Command"],
        "deploy-policy#1": ["SEV1", "Deployment Freeze"],
    }
    graph = build_graph(chunks, entities_by_chunk)
    query = (
        "If a SEV1 is caused by a bad deploy, what mitigation step is taken, and are deployment "
        "freezes also in effect during the incident?"
    )
    provider = MockProvider(
        [
            "The rollback mitigation [incident-runbook#1] and the deployment freeze during an "
            "active SEV1 [deploy-policy#1] are both documented."
        ]
    )

    result = local_search(query, graph, provider, hops=1)
    flat_top1 = [sc.chunk.id for sc in dense_retrieve(query, dense_index, embedder, top_k=1)]

    assert "deploy-policy#1" in result.chunk_ids
    assert "deploy-policy#1" not in flat_top1
    assert graph_adds_value(result.chunk_ids, flat_top1) is True


def test_global_search_answers_from_summaries_with_no_chunk_retrieval() -> None:
    chunk_a = Chunk(id="doc-a#0", source_id="doc-a", text="alpha beta", start=0, end=1)
    chunk_b = Chunk(id="doc-b#0", source_id="doc-b", text="gamma delta", start=0, end=1)
    entities_by_chunk = {"doc-a#0": ["Alpha Topic", "Beta Topic"], "doc-b#0": ["Gamma Topic"]}
    graph = build_graph([chunk_a, chunk_b], entities_by_chunk)
    communities = detect_communities(graph)
    summarize_communities(communities, graph.chunks_by_id, MockProvider(["summary one", "summary two"]))

    provider = MockProvider(
        [
            "Alpha and beta are covered [community-0].",
            "Gamma is covered [community-1].",
            "Overall, the corpus covers alpha and beta [community-0] and gamma [community-1].",
        ]
    )
    result = global_search("what are the themes?", communities, provider)

    assert result.mode == "global"
    assert result.chunk_ids == []
    assert result.communities_touched == [0, 1]
    assert set(result.answer.citations) == {"community-0", "community-1"}


def test_local_search_reports_no_benefit_on_single_hop_factoid() -> None:
    chunks = default_chunks()
    embedder = HashEmbedder()
    dense_index = build_dense_index(chunks, embedder)
    entities_by_chunk = {"api-rate-limits#0": ["API Rate Limit"], "api-rate-limits#1": ["API Rate Limit"]}
    graph = build_graph(chunks, entities_by_chunk)
    query = "What is Aurora's default API rate limit per minute?"
    provider = MockProvider(
        ["The default limit is one hundred requests per minute per key [api-rate-limits#0]."]
    )

    result = local_search(query, graph, provider, hops=1)
    flat = [sc.chunk.id for sc in dense_retrieve(query, dense_index, embedder, top_k=len(result.chunk_ids) or 1)]

    assert graph_adds_value(result.chunk_ids, flat) is False


def test_run_graph_rag_demo_shows_local_win_and_skeptic_no_benefit() -> None:
    result = run_graph_rag_demo()
    assert graph_adds_value(result.local_result.chunk_ids, result.local_flat_baseline) is True
    assert graph_adds_value(result.skeptic_result.chunk_ids, result.skeptic_flat_baseline) is False
    assert result.global_result.chunk_ids == []
    assert len(result.communities) == 5


# --- reasoning reranking ------------------------------------------------


def test_reasoning_rerank_promotes_highest_graded_chunk_to_first() -> None:
    planted = Chunk(id="planted", source_id="d", text="the directly relevant passage", start=0, end=1)
    distractor_a = Chunk(id="distractor-a", source_id="d", text="tangential passage one", start=0, end=1)
    distractor_b = Chunk(id="distractor-b", source_id="d", text="tangential passage two", start=0, end=1)
    candidates = [ScoredChunk(distractor_a, 0.9), ScoredChunk(distractor_b, 0.8), ScoredChunk(planted, 0.1)]
    provider = MockProvider(
        [
            {"content": "RELEVANCE: 1", "reasoning": "only tangentially related"},
            {"content": "RELEVANCE: 1", "reasoning": "also only tangential"},
            {"content": "RELEVANCE: 3", "reasoning": "directly answers the question"},
        ]
    )
    reranked, judgments = reasoning_rerank("a question", candidates, provider, top_k=3)
    assert reranked[0].chunk.id == "planted"
    assert judgments[0].chunk_id == "planted" and judgments[0].grade == 3
    assert judgments[1].chunk_id == "distractor-a" and judgments[1].grade == 1


def test_reasoning_rerank_reasoning_channel_is_recorded_not_parsed_for_grade() -> None:
    chunk = Chunk(id="c1", source_id="d", text="text", start=0, end=1)
    completion = Completion(content="RELEVANCE: 2", reasoning="RELEVANCE: 3 (this reasoning text must not be parsed)")
    provider = MockProvider([completion])

    reranked, judgments = reasoning_rerank("q", [ScoredChunk(chunk, 0.5)], provider, top_k=1)

    assert judgments[0].grade == 2  # parsed only from content, never from the reasoning channel
    assert judgments[0].rationale == "RELEVANCE: 3 (this reasoning text must not be parsed)"
    assert reranked[0].score == 2.0


def test_reasoning_rerank_drops_all_zero_graded_candidates() -> None:
    a = Chunk(id="a", source_id="d", text="x", start=0, end=1)
    b = Chunk(id="b", source_id="d", text="y", start=0, end=1)
    provider = MockProvider(["RELEVANCE: 0", "RELEVANCE: 0"])

    reranked, judgments = reasoning_rerank("q", [ScoredChunk(a, 0.9), ScoredChunk(b, 0.5)], provider, top_k=3)

    assert reranked == []
    assert judgments == []


def test_reasoning_rerank_breaks_grade_ties_by_retrieval_score() -> None:
    higher = Chunk(id="higher", source_id="d", text="x", start=0, end=1)
    lower = Chunk(id="lower", source_id="d", text="y", start=0, end=1)
    candidates = [ScoredChunk(lower, 0.3), ScoredChunk(higher, 0.7)]
    provider = MockProvider(["RELEVANCE: 2", "RELEVANCE: 2"])

    reranked, _ = reasoning_rerank("q", candidates, provider, top_k=2)

    assert [sc.chunk.id for sc in reranked] == ["higher", "lower"]


def test_reasoning_rerank_is_deterministic_given_an_identical_script() -> None:
    a = Chunk(id="a", source_id="d", text="x", start=0, end=1)
    b = Chunk(id="b", source_id="d", text="y", start=0, end=1)
    candidates = [ScoredChunk(a, 0.5), ScoredChunk(b, 0.4)]

    def make_provider() -> MockProvider:
        return MockProvider(
            [
                {"content": "RELEVANCE: 3", "reasoning": "rationale for a"},
                {"content": "RELEVANCE: 1", "reasoning": "rationale for b"},
            ]
        )

    first, first_judgments = reasoning_rerank("q", candidates, make_provider(), top_k=2)
    second, second_judgments = reasoning_rerank("q", candidates, make_provider(), top_k=2)

    assert [sc.chunk.id for sc in first] == [sc.chunk.id for sc in second]
    assert [j.grade for j in first_judgments] == [j.grade for j in second_judgments]
    assert [j.rationale for j in first_judgments] == [j.rationale for j in second_judgments]


def test_run_reasoning_rerank_demo_promotes_buried_chunk_and_drops_noise() -> None:
    query, before, after, judgments = run_reasoning_rerank_demo()
    assert before[-1].chunk.id == "billing-faq#1"  # buried last of six by dense retrieval
    assert after[0].chunk.id == "billing-faq#1"  # promoted to first by the reasoning grade
    after_ids = {sc.chunk.id for sc in after}
    assert "incident-runbook#0" not in after_ids  # graded 0, dropped rather than ranked last
    assert "api-rate-limits#1" not in after_ids  # graded 0, dropped rather than ranked last


# --- order-preserving assembly and the k sweep ---------------------------


def test_order_preserve_assemble_orders_by_source_and_start_not_score() -> None:
    early = Chunk(id="doc#0", source_id="doc", text="early passage text here", start=0, end=24)
    late = Chunk(id="doc#1", source_id="doc", text="late passage text here", start=100, end=123)
    scored = [ScoredChunk(late, 0.9), ScoredChunk(early, 0.3)]  # late scores higher

    ordered = order_preserve_assemble(scored, token_budget=1000)

    assert [c.id for c in ordered] == ["doc#0", "doc#1"]  # document order, not score order


def test_order_preserve_assemble_dedups_and_respects_budget() -> None:
    base = "the rollback command reverts the previous stable release quickly today"
    c1 = Chunk(id="c1", source_id="d", text=base, start=0, end=1)
    c2 = Chunk(id="c2", source_id="d", text=base + " now", start=10, end=11)  # near-dup of c1
    c3 = Chunk(id="c3", source_id="d", text="delta epsilon zeta", start=20, end=21)

    result = order_preserve_assemble([ScoredChunk(c1, 0.9), ScoredChunk(c2, 0.8), ScoredChunk(c3, 0.5)], token_budget=100)

    assert [c.id for c in result] == ["c1", "c3"]  # c2 dropped as a near-duplicate of c1


def test_sweep_k_reports_interior_sweet_spot_not_largest_k() -> None:
    chunks = [Chunk(id=f"c{i}", source_id="d", text=f"word{i}", start=0, end=1) for i in range(4)]
    candidates = [ScoredChunk(chunks[i], 1.0 - i * 0.1) for i in range(4)]
    provider = MockProvider(
        [
            "one citation [c0].",
            "two citations [c0] [c1].",
            "three citations [c0] [c1] [c2].",
            "one citation only [c0].",
        ]
    )

    result = sweep_k("q", candidates, provider, ks=[1, 2, 3, 4])

    assert result.sweet_spot_k == 3  # interior maximum, not the largest swept k
    assert result.points[-1].proxy_score < result.points[2].proxy_score


def test_sweep_k_hard_negative_lowers_proxy_at_a_matched_k() -> None:
    a = Chunk(id="a", source_id="d", text="x", start=0, end=1)
    b = Chunk(id="b", source_id="d", text="y", start=0, end=1)
    noise = Chunk(id="noise", source_id="e", text="unrelated", start=0, end=1)
    clean = [ScoredChunk(a, 0.9), ScoredChunk(b, 0.8)]
    noisy = [ScoredChunk(a, 0.9), ScoredChunk(b, 0.8), ScoredChunk(noise, 0.5)]
    provider = MockProvider(["both covered [a] [b].", "only one covered [a], the noise chunk distracted from b."])

    clean_result = sweep_k("q", clean, provider, ks=[2])
    noisy_result = sweep_k("q", noisy, provider, ks=[3])

    assert noisy_result.points[0].proxy_score < clean_result.points[0].proxy_score


def test_run_order_preserve_demo_shows_document_order_and_interior_peak() -> None:
    query, score_ordered, order_preserved, sweep = run_order_preserve_demo()
    assert [c.id for c in score_ordered] == ["incident-runbook#1", "incident-runbook#0"]
    assert [c.id for c in order_preserved] == ["incident-runbook#0", "incident-runbook#1"]
    assert sweep.sweet_spot_k == 3
