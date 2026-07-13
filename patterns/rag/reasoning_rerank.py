"""Reasoning reranking: a pointwise reranker that reasons briefly before
grading each candidate, instead of a silent cross-encoder score or a
listwise call that only emits an order.

`rerank.py` asks the model to read a whole shortlist and reply with one
ranked list: no per-item score, no rationale, and no signal comparable
across candidates beyond relative order. Rank1 (Weller et al.,
arXiv:2502.18418, "Test-Time Compute for Reranking") distills R1-style
reasoning traces onto a reranker, so the frontier moved from a trained
cross-encoder's silent score (Qwen3-Reranker, arXiv:2506.05176) to a model
that spends test-time compute reasoning about each candidate before it
grades it. This module is that shape: one call per candidate, each reasoning
briefly and then emitting a graded relevance label from 0 to 3, with the
reasoning kept as a rationale for the transcript.

The reasoning trace is read from `Completion.reasoning`, the opaque
provider-neutral channel core already carries for reasoning models, and is
stored verbatim; it is never parsed for content and never rewritten. Only
the parsed `RELEVANCE:` label drives sorting and the drop decision. Grading
each candidate 0-3 instead of only ranking it also composes better with the
rest of this folder than a synthetic listwise rank score would: a candidate
graded 0 is an explicit not-relevant judgment, so dropping it (rather than
merely ranking it last) feeds the same empty-context abstain path
`generation.generate_grounded_answer` already uses when nothing is left to
answer from.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Embedder, Message, Provider, get_embedder, get_provider

from patterns.rag.chunking import ScoredChunk
from patterns.rag.corpus import default_chunks
from patterns.rag.dense import DenseIndex, build_dense_index, dense_retrieve

_POINTWISE_SYSTEM = (
    "Judge how relevant the candidate passage is to answering the question. Reason briefly, then "
    "on its own final line emit exactly: RELEVANCE: <0-3>, using 0 for not relevant, 1 for "
    "tangential, 2 for partially relevant, and 3 for directly answers the question."
)

_DEMO_QUERY = "What happens when a customer disputes an invoice?"


@dataclass
class RerankJudgment:
    """One candidate's pointwise reasoning-reranker judgment.

    Attributes:
        chunk_id: The judged chunk's id.
        grade: Parsed relevance label, 0 (not relevant) to 3 (directly answers).
        rationale: The reasoning trace, read verbatim from `Completion.reasoning`.
    """

    chunk_id: str
    grade: int
    rationale: str


def build_pointwise_prompt(query: str, candidate: ScoredChunk) -> str:
    """Build the prompt judging a single candidate against the query."""
    return f"Question: {query}\n\nCandidate [{candidate.chunk.id}]: {candidate.chunk.text}"


def parse_relevance_grade(text: str) -> int:
    """Parse a `RELEVANCE: n` line into an integer grade, clamped to 0-3.

    Scans from the last line backward so any reasoning prose that happens to
    contain a stray digit earlier in the reply cannot be mistaken for the
    label; a reply with no parseable label grades 0.
    """
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.upper().startswith("RELEVANCE:"):
            digits = "".join(ch for ch in stripped.split(":", 1)[1] if ch.isdigit())
            if digits:
                return max(0, min(3, int(digits[0])))
    return 0


def reasoning_rerank(
    query: str, candidates: list[ScoredChunk], provider: Provider, *, top_k: int = 3
) -> tuple[list[ScoredChunk], list[RerankJudgment]]:
    """Grade every candidate pointwise with a reasoning call, then rerank.

    Each candidate gets its own provider call, so the reasoning channel and
    the grade are per-item, unlike a listwise call's single shared reply.
    Candidates graded 0 are dropped outright rather than ranked last: a 0 is
    an explicit not-relevant judgment, so it should not silently occupy a
    context slot.

    Args:
        query: The user's question.
        candidates: The shortlist to grade, any order, any prior score.
        provider: `Provider` used for one call per candidate.
        top_k: Number of graded, non-zero candidates to keep.

    Returns:
        A tuple of the kept `ScoredChunk`s, score replaced by their integer
        grade so the result composes with `assembly.assemble_context`, and
        the matching `RerankJudgment`s in the same order. Ties in grade are
        broken by the candidate's original retrieval score, highest first,
        so the reranker never has to invent an order among equal grades.
    """
    judgments: list[RerankJudgment] = []
    for candidate in candidates:
        completion = provider.complete([Message.user(build_pointwise_prompt(query, candidate))], system=_POINTWISE_SYSTEM)
        grade = parse_relevance_grade(completion.content)
        judgments.append(RerankJudgment(chunk_id=candidate.chunk.id, grade=grade, rationale=completion.reasoning))

    paired = list(zip(candidates, judgments))
    kept = [(candidate, judgment) for candidate, judgment in paired if judgment.grade > 0]
    kept.sort(key=lambda pair: (pair[1].grade, pair[0].score), reverse=True)
    limited = kept[:top_k]

    reranked = [ScoredChunk(chunk=candidate.chunk, score=float(judgment.grade)) for candidate, judgment in limited]
    kept_judgments = [judgment for _, judgment in limited]
    return reranked, kept_judgments


def run_reasoning_rerank_demo(
    provider: Provider | None = None,
    *,
    dense_index: DenseIndex | None = None,
    embedder: Embedder | None = None,
) -> tuple[str, list[ScoredChunk], list[ScoredChunk], list[RerankJudgment]]:
    """Demonstrate a reasoning reranker promoting a buried chunk and dropping noise.

    Dense retrieval buries `billing-faq#1` (invoice disputes route to the
    billing team) in last place of a six-candidate shortlist, since its
    vocabulary barely overlaps the question. The reasoning reranker grades
    it 3 with a rationale explaining why it directly answers, grades two
    off-topic candidates 0 (an incident-severity chunk and a rate-limit
    error chunk), and drops them outright rather than ranking them last.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with one pointwise judgment per
            candidate, each carrying a distinct `reasoning` trace.
        dense_index: A prebuilt `DenseIndex` over the sample corpus. Built
            fresh with `embedder` when omitted, so the demo still runs
            standalone with no arguments.
        embedder: Embedder for query encoding, and for building
            `dense_index` when it is not supplied. Defaults to
            `agentic_patterns.get_embedder`.

    Returns:
        A tuple of the query, the dense-ranked candidates before reranking,
        the reranked top-3, and their judgments (rationale included).
    """
    if embedder is None:
        embedder = get_embedder()
    if dense_index is None:
        dense_index = build_dense_index(default_chunks(), embedder)
    candidates = dense_retrieve(_DEMO_QUERY, dense_index, embedder, top_k=6)

    if provider is None:
        provider = get_provider(
            script=[
                {
                    "content": "The passage covers SEV1 mitigation and postmortem timing, nothing about billing "
                    "or invoices.\nRELEVANCE: 1",
                    "reasoning": "Checked for billing vocabulary: none present; this is incident-response content.",
                },
                {
                    "content": "The passage is about GDPR deletion requests, not invoice disputes.\nRELEVANCE: 1",
                    "reasoning": "Data-retention topic; shares only generic customer-request phrasing with the question.",
                },
                {
                    "content": "This describes proration credits on upgrade, which is billing-adjacent but not "
                    "about disputing an invoice.\nRELEVANCE: 2",
                    "reasoning": "Same document and team as invoice handling, but answers a different sub-question.",
                },
                {
                    "content": "This defines what counts as a SEV1 incident; unrelated to invoices.\nRELEVANCE: 0",
                    "reasoning": "No billing, invoice, or dispute vocabulary anywhere in this passage.",
                },
                {
                    "content": "This is about API rate-limit error codes, unrelated to invoices.\nRELEVANCE: 0",
                    "reasoning": "Rate-limiting and invoicing are different subsystems with no overlap here.",
                },
                {
                    "content": "This states plainly that invoice disputes go to the billing team through the "
                    "support portal, which is exactly what was asked.\nRELEVANCE: 3",
                    "reasoning": "Direct match: the passage names the invoice-dispute process the question asks about.",
                },
            ]
        )
    reranked, judgments = reasoning_rerank(_DEMO_QUERY, candidates, provider, top_k=3)
    return _DEMO_QUERY, candidates, reranked, judgments
