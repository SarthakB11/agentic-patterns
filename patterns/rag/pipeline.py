"""The canonical RAG control flow, wired end to end.

Ingest and index happen once, ahead of any query (see `corpus.py`,
`dense.py`, `bm25.py`). For each query this module runs: retrieve, grade
relevance, optionally rerank, assemble context, and generate a grounded
answer or abstain. It is one function so a reader can see the whole flow in
one place; each stage's own logic lives in its own module and is tested on
its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Embedder, Provider, get_embedder, get_provider

from patterns.rag.assembly import assemble_context
from patterns.rag.bm25 import BM25Index, bm25_retrieve, build_bm25_index
from patterns.rag.chunking import Chunk, ScoredChunk
from patterns.rag.corpus import default_chunks
from patterns.rag.dense import DenseIndex, build_dense_index, dense_retrieve
from patterns.rag.generation import GroundedAnswer, generate_grounded_answer
from patterns.rag.grading import grade_relevance
from patterns.rag.hybrid import hybrid_retrieve
from patterns.rag.rerank import rerank_chunks

_RETRIEVAL_MODES = ("dense", "bm25", "hybrid")


@dataclass
class RagResult:
    """The full record of one query answered through the pipeline.

    Attributes:
        query: The question that was asked.
        retrieval_mode: Which first-stage retriever(s) were used.
        candidates: All chunks the retrieval stage returned, before grading.
        kept: Candidates that survived relevance grading (and reranking, if used).
        dropped: Candidates relevance grading dropped.
        context_chunks: The chunks actually assembled into the generation prompt.
        answer: The grounded answer, or an abstain result.
    """

    query: str
    retrieval_mode: str
    candidates: list[ScoredChunk]
    kept: list[ScoredChunk]
    dropped: list[ScoredChunk]
    context_chunks: list[Chunk]
    answer: GroundedAnswer


def answer_question(
    query: str,
    *,
    dense_index: DenseIndex,
    bm25_index: BM25Index,
    embedder: Embedder,
    provider: Provider,
    retrieval: str = "dense",
    fetch_k: int = 6,
    top_k: int = 3,
    relevance_threshold: float | None = None,
    token_budget: int = 220,
    rerank: bool = False,
) -> RagResult:
    """Run the full ingest-to-answer pipeline for one query.

    Args:
        query: The user's question.
        dense_index: A `DenseIndex` over the corpus.
        bm25_index: A `BM25Index` over the same corpus.
        embedder: Embedder for the dense side of retrieval.
        provider: `Provider` used for reranking (if enabled) and generation.
        retrieval: One of "dense", "bm25", "hybrid".
        fetch_k: Candidates fetched before grading/reranking.
        top_k: Chunks kept for the final context, after grading/reranking.
        relevance_threshold: Minimum retrieval score to keep a candidate.
            `None` skips grading and keeps the top-scored candidates as is,
            since dense, BM25, and fused scores are on different scales and
            a single default threshold would not suit all three.
        token_budget: Word-count budget passed to `assemble_context`.
        rerank: Whether to rerank the graded candidates before assembly.

    Returns:
        A `RagResult` carrying every intermediate stage plus the final answer.

    Raises:
        ValueError: If `retrieval` is not a recognized mode.
    """
    if retrieval not in _RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval mode {retrieval!r}. Valid modes: {', '.join(_RETRIEVAL_MODES)}")

    if retrieval == "dense":
        candidates = dense_retrieve(query, dense_index, embedder, top_k=fetch_k)
    elif retrieval == "bm25":
        candidates = bm25_retrieve(query, bm25_index, top_k=fetch_k)
    else:
        candidates = hybrid_retrieve(query, dense_index, bm25_index, embedder, top_k=fetch_k, fetch_k=fetch_k)

    if relevance_threshold is None:
        kept, dropped = candidates, []
    else:
        kept, dropped = grade_relevance(candidates, threshold=relevance_threshold)

    if rerank and kept:
        kept = rerank_chunks(query, kept, provider, top_k=top_k)
    else:
        kept = kept[:top_k]

    if not kept:
        return RagResult(
            query=query,
            retrieval_mode=retrieval,
            candidates=candidates,
            kept=[],
            dropped=dropped,
            context_chunks=[],
            answer=generate_grounded_answer(query, [], provider),
        )

    context_chunks = assemble_context(kept, token_budget=token_budget)
    answer = generate_grounded_answer(query, context_chunks, provider)
    return RagResult(
        query=query,
        retrieval_mode=retrieval,
        candidates=candidates,
        kept=kept,
        dropped=dropped,
        context_chunks=context_chunks,
        answer=answer,
    )


def _build_indexes() -> tuple[DenseIndex, BM25Index]:
    """Build the dense and BM25 indexes over the sample corpus once."""
    chunks = default_chunks()
    embedder = get_embedder()
    return build_dense_index(chunks, embedder), build_bm25_index(chunks)


_NAIVE_DEMO_QUERY = "What is the first mitigation step for a SEV1 incident caused by a recent deploy?"
_ADVANCED_DEMO_QUERY = "What does Aurora do automatically for a customer who upgrades mid cycle, and who handles invoice disputes?"
_ABSTAIN_DEMO_QUERY = "xylophone quokka marmalade skateboard umbrella"


def run_naive_rag_demo(
    provider: Provider | None = None,
    *,
    dense_index: DenseIndex | None = None,
    bm25_index: BM25Index | None = None,
    embedder: Embedder | None = None,
) -> RagResult:
    """Run naive RAG end to end: one dense lookup, no fusion, no reranking.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with a grounded, two-citation answer.
        dense_index: A prebuilt `DenseIndex` over the sample corpus. Built
            fresh, together with `bm25_index`, when either is omitted, so
            the demo still runs standalone with no arguments.
        bm25_index: A prebuilt `BM25Index` over the same corpus.
        embedder: Embedder for query encoding. Defaults to
            `agentic_patterns.get_embedder`.

    Returns:
        The full `RagResult` for the demo query.
    """
    if dense_index is None or bm25_index is None:
        dense_index, bm25_index = _build_indexes()
    if embedder is None:
        embedder = get_embedder()
    if provider is None:
        provider = get_provider(
            script=[
                "The first mitigation step for a SEV1 caused by a recent deploy is an immediate "
                "rollback rather than a forward fix [incident-runbook#1]. The on-call engineer "
                "must already have declared the severity level within five minutes of the first "
                "alert before that step begins [incident-runbook#0]."
            ]
        )
    return answer_question(
        _NAIVE_DEMO_QUERY,
        dense_index=dense_index,
        bm25_index=bm25_index,
        embedder=embedder,
        provider=provider,
        retrieval="dense",
        fetch_k=2,
        top_k=2,
    )


def run_hybrid_rerank_demo(
    provider: Provider | None = None,
    *,
    dense_index: DenseIndex | None = None,
    bm25_index: BM25Index | None = None,
    embedder: Embedder | None = None,
) -> RagResult:
    """Run the advanced pipeline: hybrid retrieval, then an LLM rerank pass.

    The demo question has two parts answered by two different chunks from
    the same document (`billing-faq#0` and `billing-faq#1`). Hybrid fusion
    fetches both, but ranks the second part's chunk last of six candidates;
    the scripted rerank call promotes it, so the final top two chunks cover
    both parts of the question.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with the rerank order and a
            two-citation grounded answer.
        dense_index: A prebuilt `DenseIndex` over the sample corpus. Built
            fresh, together with `bm25_index`, when either is omitted, so
            the demo still runs standalone with no arguments.
        bm25_index: A prebuilt `BM25Index` over the same corpus.
        embedder: Embedder for query encoding. Defaults to
            `agentic_patterns.get_embedder`.

    Returns:
        The full `RagResult` for the demo query.
    """
    if dense_index is None or bm25_index is None:
        dense_index, bm25_index = _build_indexes()
    if embedder is None:
        embedder = get_embedder()
    if provider is None:
        provider = get_provider(
            script=[
                "RANK: billing-faq#0, billing-faq#1, data-retention#0, oncall-rotation#0, "
                "data-retention#1, deploy-policy#0",
                "Aurora automatically credits the unused portion of a plan when a customer "
                "upgrades mid cycle [billing-faq#0]. Invoice disputes go to the billing team "
                "through the support portal [billing-faq#1].",
            ]
        )
    return answer_question(
        _ADVANCED_DEMO_QUERY,
        dense_index=dense_index,
        bm25_index=bm25_index,
        embedder=embedder,
        provider=provider,
        retrieval="hybrid",
        fetch_k=6,
        top_k=2,
        rerank=True,
    )


def run_abstain_demo(
    provider: Provider | None = None,
    *,
    dense_index: DenseIndex | None = None,
    bm25_index: BM25Index | None = None,
    embedder: Embedder | None = None,
) -> RagResult:
    """Run the pipeline on a query with no relevant chunks in the corpus.

    A relevance threshold well above the best score any chunk can reach for
    an off-topic query makes every candidate get dropped, so the pipeline
    never calls the model for generation: it returns the fixed abstain
    answer instead of guessing from irrelevant context.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` with an empty script, since a correctly abstaining
            pipeline makes no generation call at all.
        dense_index: A prebuilt `DenseIndex` over the sample corpus. Built
            fresh, together with `bm25_index`, when either is omitted, so
            the demo still runs standalone with no arguments.
        bm25_index: A prebuilt `BM25Index` over the same corpus.
        embedder: Embedder for query encoding. Defaults to
            `agentic_patterns.get_embedder`.

    Returns:
        The full `RagResult` for the demo query, with `answer.abstained` True.
    """
    if dense_index is None or bm25_index is None:
        dense_index, bm25_index = _build_indexes()
    if embedder is None:
        embedder = get_embedder()
    if provider is None:
        provider = get_provider(script=[])
    return answer_question(
        _ABSTAIN_DEMO_QUERY,
        dense_index=dense_index,
        bm25_index=bm25_index,
        embedder=embedder,
        provider=provider,
        retrieval="dense",
        fetch_k=5,
        top_k=3,
        relevance_threshold=0.5,
    )
