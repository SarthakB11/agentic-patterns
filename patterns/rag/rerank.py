"""Reranking: over-fetch candidates with a cheap retriever, then have a model
read the query against each candidate and reorder the shortlist.

A bi-encoder (dense retrieval) embeds the query and each chunk separately,
so it never lets the two interact. A reranker reads query and chunk
together, which is more accurate but too slow to run over a whole corpus, so
it only ever reorders a small shortlist a first-stage retriever already
narrowed down. A production reranker is usually a trained cross-encoder
(open-weight options like Qwen3-Reranker, arXiv:2506.05176, now compete with
closed APIs); this module uses the LLM listwise variant instead, asking the
provider to rank the shortlist directly, which trades some latency for
needing no extra model. Rerankers cannot recover a chunk the first-stage
retriever never fetched, so they help most when recall is already good and
only the ordering is weak.
"""

from __future__ import annotations

from agentic_patterns import Embedder, Message, Provider, get_embedder, get_provider

from patterns.rag.chunking import ScoredChunk
from patterns.rag.corpus import default_chunks
from patterns.rag.dense import DenseIndex, build_dense_index, dense_retrieve

_DEMO_QUERY = (
    "If a customer moves to a higher tier before their billing period ends, "
    "what automatic adjustment do they receive?"
)

_RERANK_SYSTEM = (
    "You are a careful relevance judge. Read the question and each "
    "candidate passage, then rank the candidates from most to least "
    "relevant to answering the question."
)


def build_rerank_prompt(query: str, candidates: list[ScoredChunk]) -> str:
    """Build the listwise reranking prompt for a shortlist of candidates."""
    lines = [f"Question: {query}", "", "Candidates:"]
    for scored in candidates:
        lines.append(f"[{scored.chunk.id}] {scored.chunk.text}")
    lines.append("")
    lines.append(
        "Reply with exactly one line: RANK: id, id, id (best match first, "
        "using every candidate id exactly once). No other text."
    )
    return "\n".join(lines)


def parse_rerank_order(text: str) -> list[str]:
    """Parse a `RANK: id, id, id` line into an ordered list of chunk ids.

    Falls back to treating the whole reply as the id list if no `RANK:`
    line is found, so a slightly off-format reply still degrades gracefully
    instead of raising.
    """
    rank_line = next((line for line in text.splitlines() if line.strip().upper().startswith("RANK:")), text)
    ids_part = rank_line.split(":", 1)[1] if ":" in rank_line else rank_line
    return [part.strip() for part in ids_part.split(",") if part.strip()]


def rerank_chunks(query: str, candidates: list[ScoredChunk], provider: Provider, *, top_k: int = 3) -> list[ScoredChunk]:
    """Rerank a shortlist with a model call and return the top-k.

    Candidates the model's reply omits keep their original relative order
    and are appended after the ones the model did rank, so a partial or
    malformed reply still returns a complete, sane list instead of dropping
    candidates.

    Args:
        query: The user's question.
        candidates: The shortlist to reorder, any order, any prior score.
        provider: `Provider` used for the rerank call. Its returned score is
            discarded; only the rank order the model gives matters, since a
            listwise call does not produce comparable per-item scores.
        top_k: Number of chunks to keep after reranking.

    Returns:
        Up to `top_k` `ScoredChunk`s in the model's ranked order. The score
        field is replaced with a synthetic descending rank score so the
        result composes with `assembly.assemble_context`, which sorts by score.
    """
    if not candidates:
        return []

    prompt = build_rerank_prompt(query, candidates)
    completion = provider.complete([Message.user(prompt)], system=_RERANK_SYSTEM)
    order = parse_rerank_order(completion.content)

    by_id = {scored.chunk.id: scored for scored in candidates}
    ranked = [by_id[cid] for cid in order if cid in by_id]
    seen = {scored.chunk.id for scored in ranked}
    ranked.extend(scored for scored in candidates if scored.chunk.id not in seen)

    limited = ranked[:top_k]
    total = len(limited)
    return [ScoredChunk(chunk=sc.chunk, score=float(total - i)) for i, sc in enumerate(limited)]


def run_rerank_demo(
    provider: Provider | None = None,
    *,
    dense_index: DenseIndex | None = None,
    embedder: Embedder | None = None,
) -> tuple[str, list[ScoredChunk], list[ScoredChunk]]:
    """Demonstrate a reranker fixing a first-stage retriever's weak ordering.

    `HashEmbedder`'s bag-of-hashed-tokens similarity has no real semantics,
    so it can rank a lexically-overlapping but off-topic chunk above the
    chunk that actually answers the question. Here the correct chunk
    (`billing-faq#0`, describing Aurora's automatic proration credit) is
    fetched but ranked last of six candidates by dense retrieval; the
    scripted reranker reads all six against the question and promotes it
    to first.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with a single `RANK:` reply that reads
            like a model that actually read each candidate.
        dense_index: A prebuilt `DenseIndex` over the sample corpus. Built
            fresh with `embedder` when omitted, so the demo still runs
            standalone with no arguments.
        embedder: Embedder for query encoding, and for building
            `dense_index` when it is not supplied. Defaults to
            `agentic_patterns.get_embedder`.

    Returns:
        A tuple of the query, the dense-ranked candidates before reranking,
        and the reranked top-3.
    """
    if embedder is None:
        embedder = get_embedder()
    if dense_index is None:
        dense_index = build_dense_index(default_chunks(), embedder)
    candidates = dense_retrieve(_DEMO_QUERY, dense_index, embedder, top_k=6)

    if provider is None:
        provider = get_provider(
            script=[
                "RANK: billing-faq#0, api-rate-limits#1, incident-runbook#1, "
                "deploy-policy#0, data-retention#1, oncall-rotation#1"
            ]
        )
    reranked = rerank_chunks(_DEMO_QUERY, candidates, provider, top_k=3)
    return _DEMO_QUERY, candidates, reranked
