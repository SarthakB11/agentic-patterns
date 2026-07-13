"""Grading: decide whether retrieved evidence is worth generating from.

Two independent gates, corresponding to the corrective and self-reflective
RAG variants (CRAG, Self-RAG) and to the newer sufficient-context framing:

- `grade_relevance` grades each chunk on its own, against a numeric
  similarity threshold, and is the mechanism behind the abstain path: if no
  chunk clears the bar, there is nothing to generate from.
- `grade_sufficient_context` grades the retrieved set as a whole, asking the
  model whether the chunks together can actually support an answer. The
  sufficient-context study (arXiv:2411.06037, ICLR 2025) found that strong
  models tend to hallucinate rather than abstain when context is
  insufficient, so this gate is meant to run before generation, not left for
  the generator to self-police.

The two gates are deliberately separate: a chunk can individually clear the
relevance threshold while the set as a whole still fails to answer a
multi-part question, which per-chunk scoring alone cannot catch.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Embedder, Message, Provider, get_embedder, get_provider

from patterns.rag.chunking import Chunk, ScoredChunk
from patterns.rag.corpus import default_chunks
from patterns.rag.dense import DenseIndex, build_dense_index, dense_retrieve

_SUFFICIENCY_DEMO_QUERY = "What is the refund window and what is the SEV1 escalation window, together?"

_SUFFICIENCY_SYSTEM = (
    "Judge whether the context chunks below contain enough information to "
    "answer the question, even partially through combining them. Reply "
    "with SUFFICIENT: yes or SUFFICIENT: no on the first line, then one "
    "sentence of reasoning."
)


def grade_relevance(scored_chunks: list[ScoredChunk], *, threshold: float) -> tuple[list[ScoredChunk], list[ScoredChunk]]:
    """Split scored chunks into ones that clear a relevance threshold and ones that don't.

    Args:
        scored_chunks: Candidates from any retriever.
        threshold: Minimum score to keep. Scale is retriever-specific: a
            cosine similarity threshold and a BM25 threshold are not
            interchangeable.

    Returns:
        A tuple `(kept, dropped)`, each preserving `scored_chunks`' order.
    """
    kept = [sc for sc in scored_chunks if sc.score >= threshold]
    dropped = [sc for sc in scored_chunks if sc.score < threshold]
    return kept, dropped


@dataclass
class ContextSufficiency:
    """The result of grading a retrieved set for sufficiency.

    Attributes:
        sufficient: True if the model judged the context adequate to answer.
        reasoning: The model's stated reasoning, kept for the transcript.
    """

    sufficient: bool
    reasoning: str


def build_sufficiency_prompt(query: str, chunks: list[Chunk]) -> str:
    """Build the prompt asking the model to grade a retrieved set as a whole."""
    context = "\n\n".join(f"[{chunk.id}] {chunk.text}" for chunk in chunks)
    return f"Question: {query}\n\nContext chunks:\n{context}"


def parse_sufficiency(text: str) -> ContextSufficiency:
    """Parse a `SUFFICIENT: yes|no` line plus reasoning into `ContextSufficiency`."""
    stripped = text.strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    sufficient = "yes" in first_line.lower()
    reasoning = "\n".join(stripped.splitlines()[1:]).strip() or stripped
    return ContextSufficiency(sufficient=sufficient, reasoning=reasoning)


def grade_sufficient_context(query: str, chunks: list[Chunk], provider: Provider) -> ContextSufficiency:
    """Ask the model whether a retrieved set can support an answer at all.

    Args:
        query: The user's question.
        chunks: The retrieved (and possibly assembled) context chunks. An
            empty list is graded as insufficient without a model call.
        provider: `Provider` used for the grading call.

    Returns:
        A `ContextSufficiency` verdict and the model's stated reasoning.
    """
    if not chunks:
        return ContextSufficiency(sufficient=False, reasoning="No chunks were retrieved.")
    completion = provider.complete([Message.user(build_sufficiency_prompt(query, chunks))], system=_SUFFICIENCY_SYSTEM)
    return parse_sufficiency(completion.content)


def run_sufficiency_demo(
    provider: Provider | None = None,
    *,
    dense_index: DenseIndex | None = None,
    embedder: Embedder | None = None,
) -> tuple[str, list[Chunk], ContextSufficiency, list[Chunk], ContextSufficiency]:
    """Demonstrate the corrective-RAG pattern: grade, widen, regrade.

    A two-part question retrieves cleanly for its first half but a narrow
    top-2 fetch misses the chunk answering the second half entirely. The
    sufficiency gate catches this even though both fetched chunks
    individually look relevant to the topic, then a wider fetch brings in
    the missing chunk and the gate passes.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with an insufficient verdict, then a
            sufficient one after widening.
        dense_index: A prebuilt `DenseIndex` over the sample corpus. Built
            fresh with `embedder` when omitted, so the demo still runs
            standalone with no arguments.
        embedder: Embedder for query encoding, and for building
            `dense_index` when it is not supplied. Defaults to
            `agentic_patterns.get_embedder`.

    Returns:
        A tuple of the query, the narrow chunk set, its verdict, the widened
        chunk set, and its verdict.
    """
    if embedder is None:
        embedder = get_embedder()
    if dense_index is None:
        dense_index = build_dense_index(default_chunks(), embedder)

    narrow = [sc.chunk for sc in dense_retrieve(_SUFFICIENCY_DEMO_QUERY, dense_index, embedder, top_k=2)]
    wide = [sc.chunk for sc in dense_retrieve(_SUFFICIENCY_DEMO_QUERY, dense_index, embedder, top_k=4)]

    if provider is None:
        provider = get_provider(
            script=[
                "SUFFICIENT: no\n"
                "The chunks describe SEV1 declaration and mitigation but say nothing about a "
                "refund window.",
                "SUFFICIENT: yes\n"
                "The wider set now covers both the SEV1 escalation timing and the fourteen day "
                "refund window.",
            ]
        )
    narrow_verdict = grade_sufficient_context(_SUFFICIENCY_DEMO_QUERY, narrow, provider)
    wide_verdict = grade_sufficient_context(_SUFFICIENCY_DEMO_QUERY, wide, provider)
    return _SUFFICIENCY_DEMO_QUERY, narrow, narrow_verdict, wide, wide_verdict
