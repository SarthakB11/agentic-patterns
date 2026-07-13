"""Order-preserving assembly and the inverted-U on retrieved-chunk count.

`assembly.py` edge-orders the assembled context by score to fight
lost-in-the-middle (Liu et al., TACL 2024, arXiv:2307.03172). Two pieces of
2024-2025 long-context evidence complicate that as the only answer. OP-RAG
(Yu et al., arXiv:2409.01666) finds that for a long-context reader, sorting
kept chunks back into their original document order (by source id, then
start offset) is often competitive with or better than similarity order,
keeping a document's internal logic intact instead of interleaving unrelated
passages. It also finds something larger: answer quality is an inverted-U in
the number of retrieved chunks `k`, because past a sweet spot the extra
chunks are hard negatives that pull the answer down. "Long-Context LLMs Meet
RAG" (Jin et al., arXiv:2410.05983) names hard negatives as the mechanism
and finds retrieval reordering a training-free mitigation. Together these
say `k`, not order, is usually the bigger lever, and "retrieve more" is not free.

`order_preserve_assemble` is a sibling to `assembly.assemble_context`,
identical except for its final ordering step. `sweep_k` illustrates the
inverted-U itself: offline, with no real accuracy metric available, the
"answer quality" proxy at each `k` is the grounded answer's citation count
from a scripted `MockProvider` reply, chosen to trace the U-shape rather than
measured from a real model. This demonstrates the mechanism and the shape,
not an empirical result.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Embedder, Provider, get_embedder, get_provider

from patterns.rag.assembly import deduplicate, fit_to_budget
from patterns.rag.chunking import Chunk, ScoredChunk
from patterns.rag.corpus import default_chunks
from patterns.rag.dense import DenseIndex, build_dense_index, dense_retrieve
from patterns.rag.generation import GroundedAnswer, generate_grounded_answer

_DEMO_QUERY = (
    "What is the first mitigation step for a SEV1 incident caused by a recent deploy, and how "
    "does the on-call rotation escalate?"
)


def order_preserve_assemble(scored_chunks: list[ScoredChunk], *, token_budget: int = 200, dedup: bool = True) -> list[Chunk]:
    """Assemble context ordered by source document position, not by score.

    Runs the same dedup and budget steps as `assembly.assemble_context`;
    only the final ordering step differs. Chunks from the same document keep
    their original reading order, and different documents are ordered by
    source id, so a document's internal logic stays intact in the prompt
    instead of interleaving chunks by how well each one scored.

    Args:
        scored_chunks: Retrieved (and possibly reranked) candidates, any order.
        token_budget: Maximum combined word count of the assembled context.
        dedup: Whether to drop near-duplicate chunks before budgeting.

    Returns:
        The chunks to place in the generation prompt, ordered by
        `(source_id, start)`.
    """
    ranked = sorted(scored_chunks, key=lambda sc: sc.score, reverse=True)
    candidates = deduplicate(ranked) if dedup else ranked
    budgeted = fit_to_budget(candidates, token_budget=token_budget)
    ordered = sorted(budgeted, key=lambda sc: (sc.chunk.source_id, sc.chunk.start))
    return [scored.chunk for scored in ordered]


@dataclass
class KSweepPoint:
    """One point on a `k` sweep.

    Attributes:
        k: Number of top-scored candidates kept at this point.
        chunk_ids: Ids of the chunks assembled at this `k`.
        proxy_score: The offline answer-quality proxy at this `k`, the
            grounded answer's citation count from a scripted reply.
    """

    k: int
    chunk_ids: list[str]
    proxy_score: float


@dataclass
class KSweepResult:
    """The outcome of sweeping `k` and tracking an answer-quality proxy.

    Attributes:
        points: One `KSweepPoint` per swept `k`, in the order swept.
        sweet_spot_k: The `k` with the highest proxy score. When it is an
            interior point (neither the smallest nor the largest swept `k`),
            that is the inverted-U: more chunks past this point hurt rather
            than help.
    """

    points: list[KSweepPoint]
    sweet_spot_k: int


def sweep_k(query: str, candidates: list[ScoredChunk], provider: Provider, *, ks: list[int]) -> KSweepResult:
    """Assemble and generate at each `k` in `ks`, tracking a citation-count proxy.

    Args:
        query: The user's question.
        candidates: Retrieved candidates, ranked best first. `candidates[:k]`
            is assembled and generated from at each swept `k`.
        provider: `Provider` used for one generation call per swept `k`.
        ks: The chunk counts to sweep, in the order to call the provider.

    Returns:
        A `KSweepResult` with one point per `k` and the `k` where the proxy peaked.
    """
    points: list[KSweepPoint] = []
    for k in ks:
        chunks = order_preserve_assemble(candidates[:k], token_budget=10_000, dedup=False)
        answer: GroundedAnswer = generate_grounded_answer(query, chunks, provider)
        points.append(KSweepPoint(k=k, chunk_ids=[c.id for c in chunks], proxy_score=float(len(answer.citations))))
    best = max(points, key=lambda point: point.proxy_score)
    return KSweepResult(points=points, sweet_spot_k=best.k)


def run_order_preserve_demo(
    provider: Provider | None = None,
    *,
    dense_index: DenseIndex | None = None,
    embedder: Embedder | None = None,
) -> tuple[str, list[Chunk], list[Chunk], KSweepResult]:
    """Demonstrate order-preserving assembly, then the inverted-U on `k`.

    Dense retrieval scores `incident-runbook#1` (a later passage) above
    `incident-runbook#0` (the earlier one it overlaps with), so score-order
    assembly would place the later passage first. Order-preserving assembly
    instead keeps `incident-runbook#0` before `incident-runbook#1`, their
    original document order.

    The `k` sweep then retrieves 1 through 5 candidates for the same query
    and generates from each. The scripted proxy rises through `k=3`, where
    both parts of the question are covered, then falls as unrelated
    candidates from other documents are added at `k=4` and `k=5`: the
    scripted stand-in for OP-RAG's hard-negative decline.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with one generation reply per swept `k`,
            citation counts shaped to trace the inverted-U.
        dense_index: A prebuilt `DenseIndex` over the sample corpus. Built
            fresh with `embedder` when omitted, so the demo still runs
            standalone with no arguments.
        embedder: Embedder for query encoding, and for building
            `dense_index` when it is not supplied. Defaults to
            `agentic_patterns.get_embedder`.

    Returns:
        A tuple of the query, the score-ordered assembly, the
        order-preserving assembly, and the `k`-sweep result.
    """
    if embedder is None:
        embedder = get_embedder()
    if dense_index is None:
        dense_index = build_dense_index(default_chunks(), embedder)
    candidates = dense_retrieve(_DEMO_QUERY, dense_index, embedder, top_k=5)

    score_ordered = [sc.chunk for sc in sorted(candidates[:2], key=lambda sc: sc.score, reverse=True)]
    order_preserved = order_preserve_assemble(candidates[:2], token_budget=10_000, dedup=False)

    if provider is None:
        provider = get_provider(
            script=[
                "The first mitigation step for a SEV1 caused by a recent deploy is an immediate "
                "rollback [incident-runbook#1].",
                "The on-call engineer declares severity within five minutes [incident-runbook#0], "
                "and the first mitigation step is an immediate rollback [incident-runbook#1].",
                "The rollback mitigation step is well covered [incident-runbook#1], the severity "
                "declaration is covered [incident-runbook#0], and rate-limit responses are unrelated "
                "but present in context [api-rate-limits#1].",
                "The rollback step is covered [incident-runbook#1] and severity declaration is "
                "covered [incident-runbook#0], but the added rate-limit and on-call passages are "
                "distractions from this question's incident focus.",
                "Only the rollback mitigation step is clearly supported here [incident-runbook#1]; "
                "the other four candidates are off-topic noise that crowd the context.",
            ]
        )
    sweep = sweep_k(_DEMO_QUERY, candidates, provider, ks=[1, 2, 3, 4, 5])
    return _DEMO_QUERY, score_ordered, order_preserved, sweep
