"""Late-interaction retrieval: a ColBERT-style middle tier between the recall
of a single-vector (bi-encoder) dense retriever and the precision of a
cross-encoder reranker.

Instead of collapsing a chunk into one vector, a late-interaction retriever
keeps one vector per token and scores a query against a chunk with MaxSim:
for each query token, take its highest cosine similarity against any token
in the chunk, then sum those maxima. This lets a query match a chunk on its
strongest overlapping terms without needing the whole chunk's meaning to
compress into a single vector, and it stays cheap because token vectors are
precomputed per chunk, unlike a cross-encoder that must read the query and
chunk together at query time. ColBERT introduced this scoring; ColPali
(arXiv:2407.01449) extends the same idea to image patches for OCR-free PDF
retrieval, which this module does not attempt since it works over text only.

This is a teaching-scale stub: it reuses `HashEmbedder` (or whatever
embedder `get_embedder()` resolves to) per token rather than a trained
late-interaction model, and caps how many tokens it embeds per chunk and
query to keep the demo fast.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Embedder, cosine_similarity

from patterns.rag.bm25 import tokenize
from patterns.rag.chunking import Chunk, ScoredChunk


@dataclass
class LateInteractionIndex:
    """A corpus indexed with one embedding vector per token, per chunk.

    Attributes:
        chunks: The indexed chunks, in a fixed order.
        token_vectors: For each chunk, one vector per (capped) token.
    """

    chunks: list[Chunk]
    token_vectors: list[list[list[float]]]


def build_late_interaction_index(
    chunks: list[Chunk], embedder: Embedder, *, max_tokens: int = 24
) -> LateInteractionIndex:
    """Tokenize every chunk and embed each token separately.

    Args:
        chunks: Chunks to index.
        embedder: Embedder used per token.
        max_tokens: Maximum tokens embedded per chunk, keeping the demo fast
            on longer chunks at the cost of ignoring tokens past the cap.
    """
    token_vectors: list[list[list[float]]] = []
    for chunk in chunks:
        tokens = tokenize(chunk.text)[:max_tokens] or [chunk.text]
        token_vectors.append(embedder.embed(tokens))
    return LateInteractionIndex(chunks=list(chunks), token_vectors=token_vectors)


def _max_sim(query_vectors: list[list[float]], doc_vectors: list[list[float]]) -> float:
    """Sum, over each query token vector, its best cosine match in a chunk."""
    return sum(max(cosine_similarity(qv, dv) for dv in doc_vectors) for qv in query_vectors)


def late_interaction_retrieve(
    query: str, index: LateInteractionIndex, embedder: Embedder, *, top_k: int = 5, max_tokens: int = 24
) -> list[ScoredChunk]:
    """Score every chunk against a query with MaxSim and return the top-k.

    Args:
        query: The user's question.
        index: A `LateInteractionIndex` over the corpus.
        embedder: Embedder used per query token; should match the embedder
            used to build `index`.
        top_k: Number of chunks to return, best first.
        max_tokens: Maximum query tokens scored, matching the index's cap.

    Returns:
        Up to `top_k` `ScoredChunk`s sorted by MaxSim score, highest first.
    """
    query_tokens = tokenize(query)[:max_tokens] or [query]
    query_vectors = embedder.embed(query_tokens)
    scored = [
        ScoredChunk(chunk=chunk, score=_max_sim(query_vectors, doc_vectors))
        for chunk, doc_vectors in zip(index.chunks, index.token_vectors)
    ]
    scored.sort(key=lambda sc: sc.score, reverse=True)
    return scored[:top_k]
