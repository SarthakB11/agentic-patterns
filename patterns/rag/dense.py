"""Naive dense retrieval: embed the corpus once, embed the query, rank by
cosine similarity, and take the top-k. This is the baseline the research
brief names "naive RAG": one dense-vector lookup, no fusion, no reranking.

The embedder is whatever `agentic_patterns.get_embedder()` resolves to. The
default, `HashEmbedder`, is a deterministic, network-free stand-in: it
cannot capture meaning, but texts that share vocabulary land closer together
in vector space than texts that do not, which is the property retrieval
demos need.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Embedder, cosine_similarity

from patterns.rag.chunking import Chunk, ScoredChunk


@dataclass
class DenseIndex:
    """A corpus of chunks with one embedding vector per chunk.

    Attributes:
        chunks: The indexed chunks, in a fixed order.
        vectors: One embedding vector per chunk, same order as `chunks`.
    """

    chunks: list[Chunk]
    vectors: list[list[float]]


def build_dense_index(chunks: list[Chunk], embedder: Embedder) -> DenseIndex:
    """Embed every chunk once and hold the vectors alongside the chunks.

    Args:
        chunks: Chunks to index.
        embedder: Embedder used for every chunk, so query embeddings must
            come from the same embedder for cosine scores to be meaningful.
    """
    vectors = embedder.embed([chunk.text for chunk in chunks])
    return DenseIndex(chunks=list(chunks), vectors=vectors)


def dense_retrieve(query: str, index: DenseIndex, embedder: Embedder, *, top_k: int = 5) -> list[ScoredChunk]:
    """Embed a query and return its top-k nearest chunks by cosine similarity.

    Args:
        query: The user's question.
        index: A `DenseIndex` built from the same embedder.
        embedder: Embedder used to embed the query.
        top_k: Number of chunks to return, best first.

    Returns:
        Up to `top_k` `ScoredChunk`s sorted by cosine similarity, highest first.
    """
    query_vector = embedder.embed([query])[0]
    scored = [
        ScoredChunk(chunk=chunk, score=cosine_similarity(query_vector, vector))
        for chunk, vector in zip(index.chunks, index.vectors)
    ]
    scored.sort(key=lambda sc: sc.score, reverse=True)
    return scored[:top_k]
