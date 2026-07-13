"""Context assembly: turn a ranked list of candidates into the chunks that
actually go in the prompt.

Three concerns, handled in order: drop near-duplicate chunks so the same
sentence is not paid for twice in the token budget; fit what remains inside
a token budget, since more chunks raise recall but crowd the prompt; and
order the surviving chunks so the strongest evidence sits at the edges of
the context rather than the middle. That last step counters the
lost-in-the-middle effect (Liu et al., TACL 2024, arXiv:2307.03172), where
models attend least to text in the middle of a long context.
"""

from __future__ import annotations

from patterns.rag.chunking import Chunk, ScoredChunk


def _word_count(text: str) -> int:
    """Approximate a token count with a word count, needing no tokenizer."""
    return len(text.split())


def _jaccard_overlap(a: str, b: str) -> float:
    """Word-set Jaccard overlap between two texts, 0.0 if either is empty."""
    tokens_a, tokens_b = set(a.lower().split()), set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def deduplicate(scored_chunks: list[ScoredChunk], *, threshold: float = 0.8) -> list[ScoredChunk]:
    """Drop chunks that are near-duplicates of a chunk already kept.

    Args:
        scored_chunks: Candidates in the order they should be considered;
            earlier chunks win over later near-duplicates.
        threshold: Word-set Jaccard overlap at or above which two chunks
            count as near-duplicates.

    Returns:
        `scored_chunks` with later near-duplicates removed, order preserved.
    """
    kept: list[ScoredChunk] = []
    for candidate in scored_chunks:
        if any(_jaccard_overlap(candidate.chunk.text, k.chunk.text) >= threshold for k in kept):
            continue
        kept.append(candidate)
    return kept


def fit_to_budget(scored_chunks: list[ScoredChunk], *, token_budget: int) -> list[ScoredChunk]:
    """Keep candidates, best first, until the next one would blow the budget.

    Args:
        scored_chunks: Candidates ordered best first.
        token_budget: Maximum combined word count to keep. The single
            best-ranked candidate is always kept even if it alone exceeds
            the budget, so a query never comes back with zero context solely
            because the top match is long.

    Returns:
        A prefix of `scored_chunks` whose combined word count fits the budget.
    """
    kept: list[ScoredChunk] = []
    used = 0
    for candidate in scored_chunks:
        cost = _word_count(candidate.chunk.text)
        if kept and used + cost > token_budget:
            break
        kept.append(candidate)
        used += cost
    return kept


def edge_order(scored_chunks: list[ScoredChunk]) -> list[Chunk]:
    """Reorder chunks, best first, so the top matches sit at both edges.

    Alternates candidates between the front and the back of the output,
    strongest first: rank 1 goes to the front, rank 2 to the back, rank 3
    to the front (after rank 1), and so on. The back half is then reversed,
    so the second-best chunk lands at the very end and the weakest surviving
    chunks land in the middle, the position models attend to least.

    Args:
        scored_chunks: Candidates ordered best first.

    Returns:
        Chunks reordered for assembly into a prompt.
    """
    front: list[Chunk] = []
    back: list[Chunk] = []
    for index, scored in enumerate(scored_chunks):
        (front if index % 2 == 0 else back).append(scored.chunk)
    return front + list(reversed(back))


def assemble_context(scored_chunks: list[ScoredChunk], *, token_budget: int = 200, dedup: bool = True) -> list[Chunk]:
    """Run the full assembly step: dedup, fit to budget, then edge-order.

    Args:
        scored_chunks: Retrieved (and possibly reranked) candidates, any order.
        token_budget: Maximum combined word count of the assembled context.
        dedup: Whether to drop near-duplicate chunks before budgeting.

    Returns:
        The chunks to place in the generation prompt, in prompt order.
    """
    ranked = sorted(scored_chunks, key=lambda sc: sc.score, reverse=True)
    candidates = deduplicate(ranked) if dedup else ranked
    budgeted = fit_to_budget(candidates, token_budget=token_budget)
    return edge_order(budgeted)
