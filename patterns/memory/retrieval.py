"""Retrieval scoring strategies beyond plain top-k cosine similarity.

`retrieve` is the canonical control-flow step: embed the query, run
similarity search, keep items above a threshold, then optionally re-rank by
a blend of recency and importance (the Generative Agents memory-stream
formula: relevance + recency + importance). `hybrid_retrieve` adds a
keyword channel alongside the vector one. `diversity_rerank` is a small
MMR-style pass that avoids returning several near-duplicate memories in one
result set.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Embedder, cosine_similarity, get_embedder

from patterns.memory.vector_store import ScoredRecord, VectorStore


@dataclass
class RetrievalConfig:
    """Retrieval knobs.

    Attributes:
        top_k: Maximum number of results to return.
        min_similarity: Records scoring below this on raw relevance are
            excluded before any re-ranking happens.
        recency_weight: Weight given to a recency score in [0, 1] when
            re-ranking. 0 disables recency re-ranking.
        importance_weight: Weight given to each record's stored importance
            when re-ranking. 0 disables importance re-ranking.
        half_life: Number of logical clock ticks after which recency decays
            to half its original score.
    """

    top_k: int = 3
    min_similarity: float = 0.1
    recency_weight: float = 0.0
    importance_weight: float = 0.0
    half_life: int = 5


def _recency_score(written_at: int, now: int, half_life: int) -> float:
    """Exponential recency decay, 1.0 for a record written this tick."""
    if half_life <= 0:
        return 1.0
    age = max(now - written_at, 0)
    return 0.5 ** (age / half_life)


def retrieve(
    store: VectorStore,
    embedder: Embedder,
    namespace: str,
    query: str,
    config: RetrievalConfig,
) -> list[ScoredRecord]:
    """Query long-term memory following the canonical control flow.

    Embeds `query`, runs cosine similarity search, keeps items at or above
    `config.min_similarity`, then, if `recency_weight` or
    `importance_weight` is set, re-ranks the surviving candidates by a
    blended score before taking the final top-k. Blending is applied on a
    wider candidate pool than `top_k` so re-ranking can actually change
    which items make the final cut, not just their order.
    """
    query_vec = embedder.embed([query])[0]
    now = store.clock
    pool_size = max(config.top_k * 3, config.top_k)
    candidates = store.search(namespace, query_vec, top_k=pool_size, min_similarity=config.min_similarity)

    if config.recency_weight == 0.0 and config.importance_weight == 0.0:
        return candidates[: config.top_k]

    relevance_weight = max(1.0 - config.recency_weight - config.importance_weight, 0.0)
    blended: list[ScoredRecord] = []
    for c in candidates:
        recency = _recency_score(c.record.written_at, now, config.half_life)
        score = (
            relevance_weight * c.similarity
            + config.recency_weight * recency
            + config.importance_weight * c.record.importance
        )
        blended.append(ScoredRecord(c.record, score))
    blended.sort(key=lambda s: s.similarity, reverse=True)
    return blended[: config.top_k]


def keyword_overlap(query: str, text: str) -> float:
    """Jaccard token overlap between `query` and `text`, the keyword half of
    hybrid keyword-plus-vector search.
    """
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


def hybrid_retrieve(
    store: VectorStore,
    embedder: Embedder,
    namespace: str,
    query: str,
    config: RetrievalConfig,
    keyword_weight: float = 0.3,
) -> list[ScoredRecord]:
    """Hybrid keyword-plus-vector search: blend cosine similarity with token
    overlap before ranking and filtering by `config.min_similarity`.
    """
    query_vec = embedder.embed([query])[0]
    scored: list[ScoredRecord] = []
    for record in store.all(namespace):
        vec_score = cosine_similarity(query_vec, record.embedding)
        kw_score = keyword_overlap(query, record.text)
        blended = (1 - keyword_weight) * vec_score + keyword_weight * kw_score
        if blended >= config.min_similarity:
            scored.append(ScoredRecord(record, blended))
    scored.sort(key=lambda s: s.similarity, reverse=True)
    return scored[: config.top_k]


def diversity_rerank(candidates: list[ScoredRecord], lambda_: float = 0.5, k: int | None = None) -> list[ScoredRecord]:
    """Diversity re-ranking (MMR-lite).

    Greedily selects the next candidate that balances its own relevance
    against how similar it is to items already selected, so a top-k result
    set is not dominated by several near-duplicate memories.

    Args:
        candidates: Scored records, typically the output of `retrieve`.
        lambda_: Trade-off between relevance (1.0) and diversity (0.0).
        k: Number of items to select. Defaults to all of `candidates`.
    """
    remaining = list(candidates)
    selected: list[ScoredRecord] = []
    limit = k if k is not None else len(candidates)

    while remaining and len(selected) < limit:
        def mmr_score(c: ScoredRecord) -> float:
            if not selected:
                return c.similarity
            max_sim = max(cosine_similarity(c.record.embedding, s.record.embedding) for s in selected)
            return lambda_ * c.similarity - (1 - lambda_) * max_sim

        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)

    return selected


def run_retrieval_demo() -> dict[str, list[str]]:
    """Compare plain top-k, recency-weighted, and hybrid retrieval over the
    same small store of near-duplicate and unrelated memories.
    """
    embedder = get_embedder()
    store = VectorStore()
    items = {
        "note-1": "The user likes hiking trails with mountain views.",
        "note-2": "The user enjoys hiking on scenic mountain trails.",
        "note-3": "The user's favorite food is ramen.",
    }
    for key, text in items.items():
        store.upsert(key, "demo", text, embedder.embed([text])[0])
    # note-3 written most recently, so recency weighting should promote it
    store.upsert("note-3", "demo", items["note-3"], embedder.embed([items["note-3"]])[0], importance=0.9)

    query = "What outdoor activities does the user enjoy?"
    plain = retrieve(store, embedder, "demo", query, RetrievalConfig(top_k=3, min_similarity=0.0))
    recency = retrieve(
        store, embedder, "demo", query, RetrievalConfig(top_k=3, min_similarity=0.0, recency_weight=0.6)
    )
    diverse = diversity_rerank(plain, lambda_=0.3, k=2)

    return {
        "plain_top_k": [s.record.id for s in plain],
        "recency_weighted": [s.record.id for s in recency],
        "diversity_reranked": [s.record.id for s in diverse],
    }
