"""Term-based retrieval: a pure-Python BM25 ranker.

BM25 scores a chunk by how often query terms appear in it, weighted by how
rare each term is across the whole corpus (inverse document frequency) and
normalized for chunk length. It has no notion of synonyms or paraphrase, but
it rewards an exact rare term, such as an error code or a proper noun, far
more than a bag-of-words embedding does, since embeddings spread a rare
token's signal across many overlapping dimensions instead of weighting it by
rarity.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from patterns.rag.chunking import Chunk, ScoredChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """A corpus indexed for BM25 scoring.

    Attributes:
        chunks: The indexed chunks, in a fixed order.
        doc_tokens: Tokenized text for each chunk, same order as `chunks`.
        doc_freq: Number of chunks each term appears in at least once.
        avg_doc_len: Average token count across all chunks.
        k1: Term-frequency saturation parameter.
        b: Length-normalization parameter, 0 (off) to 1 (full).
    """

    chunks: list[Chunk]
    doc_tokens: list[list[str]]
    doc_freq: dict[str, int]
    avg_doc_len: float
    k1: float = 1.5
    b: float = 0.75


def build_bm25_index(chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> BM25Index:
    """Tokenize every chunk and compute the document-frequency table.

    Args:
        chunks: Chunks to index.
        k1: BM25 term-frequency saturation parameter.
        b: BM25 length-normalization parameter.
    """
    doc_tokens = [tokenize(chunk.text) for chunk in chunks]
    doc_freq: dict[str, int] = {}
    for tokens in doc_tokens:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    avg_doc_len = sum(len(tokens) for tokens in doc_tokens) / len(doc_tokens) if doc_tokens else 0.0
    return BM25Index(chunks=list(chunks), doc_tokens=doc_tokens, doc_freq=doc_freq, avg_doc_len=avg_doc_len, k1=k1, b=b)


def _idf(index: BM25Index, term: str) -> float:
    """Inverse document frequency for one term, using the BM25+ smoothing form."""
    n = len(index.chunks)
    df = index.doc_freq.get(term, 0)
    return math.log((n - df + 0.5) / (df + 0.5) + 1)


def bm25_retrieve(query: str, index: BM25Index, *, top_k: int = 5) -> list[ScoredChunk]:
    """Rank every chunk against a query with BM25 and return the top-k.

    Args:
        query: The user's question.
        index: A `BM25Index` built over the corpus.
        top_k: Number of chunks to return, best first.

    Returns:
        Up to `top_k` `ScoredChunk`s sorted by BM25 score, highest first.
        A chunk that shares no query terms scores 0.0.
    """
    query_terms = tokenize(query)
    scored: list[ScoredChunk] = []
    for chunk, tokens in zip(index.chunks, index.doc_tokens):
        term_counts = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            freq = term_counts.get(term, 0)
            if freq == 0:
                continue
            idf = _idf(index, term)
            denom = freq + index.k1 * (1 - index.b + index.b * doc_len / (index.avg_doc_len or 1))
            score += idf * (freq * (index.k1 + 1)) / denom
        scored.append(ScoredChunk(chunk=chunk, score=score))
    scored.sort(key=lambda sc: sc.score, reverse=True)
    return scored[:top_k]
