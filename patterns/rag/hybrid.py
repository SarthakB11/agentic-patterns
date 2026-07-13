"""Hybrid retrieval: run a dense retriever and a term-based retriever, then
merge their ranked lists with Reciprocal Rank Fusion (RRF).

RRF sidesteps the problem of two retrievers scoring on different, hard to
compare scales (a cosine similarity and a BM25 score are not the same unit).
It uses only each list's rank order: a chunk's fused score is the sum, over
every list it appears in, of `1 / (k + rank)`, with `rank` starting at 1 and
`k` a constant (60 is the default from Cormack et al., SIGIR 2009) that
flattens the influence of any single very-high rank. A chunk that ranks
respectably in both lists usually outranks a chunk that ranks first in one
list and is absent from the other.
"""

from __future__ import annotations

from agentic_patterns import Embedder

from patterns.rag.bm25 import BM25Index, bm25_retrieve
from patterns.rag.chunking import Chunk, ScoredChunk
from patterns.rag.dense import DenseIndex, dense_retrieve


def reciprocal_rank_fusion(rankings: list[list[ScoredChunk]], *, k: int = 60) -> list[ScoredChunk]:
    """Fuse multiple ranked lists into one list, ranked by RRF score.

    Args:
        rankings: One or more ranked lists of `ScoredChunk`, best first. The
            same chunk id may appear in more than one list.
        k: RRF's rank-damping constant.

    Returns:
        Every distinct chunk that appeared in any input list, with a fused
        RRF score, sorted highest first. Original per-list scores are
        discarded; only rank position is used.
    """
    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, scored in enumerate(ranking, start=1):
            fused_scores[scored.chunk.id] = fused_scores.get(scored.chunk.id, 0.0) + 1.0 / (k + rank)
            chunk_by_id[scored.chunk.id] = scored.chunk
    fused = [ScoredChunk(chunk=chunk_by_id[cid], score=score) for cid, score in fused_scores.items()]
    fused.sort(key=lambda sc: sc.score, reverse=True)
    return fused


def hybrid_retrieve(
    query: str,
    dense_index: DenseIndex,
    bm25_index: BM25Index,
    embedder: Embedder,
    *,
    top_k: int = 5,
    fetch_k: int = 10,
    rrf_k: int = 60,
) -> list[ScoredChunk]:
    """Retrieve with dense and BM25 in parallel, then fuse with RRF.

    Args:
        query: The user's question.
        dense_index: A `DenseIndex` over the corpus.
        bm25_index: A `BM25Index` over the same corpus.
        embedder: Embedder used to embed the query for the dense side.
        top_k: Number of fused chunks to return.
        fetch_k: Number of candidates each retriever contributes before fusion.
        rrf_k: RRF's rank-damping constant.

    Returns:
        Up to `top_k` `ScoredChunk`s ranked by fused RRF score, highest first.
    """
    dense_results = dense_retrieve(query, dense_index, embedder, top_k=fetch_k)
    bm25_results = bm25_retrieve(query, bm25_index, top_k=fetch_k)
    fused = reciprocal_rank_fusion([dense_results, bm25_results], k=rrf_k)
    return fused[:top_k]
