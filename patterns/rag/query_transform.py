"""Query transformation: rewrite the query before retrieval.

Two sub-variants live here. Multi-query expansion asks the model to split a
vague or multi-part question into several narrower sub-queries, retrieves
for each one separately, and fuses the ranked lists with the same
`reciprocal_rank_fusion` used for hybrid search, since RRF's rank-only
merge is not specific to combining a dense list with a term-based one. HyDE
(Gao et al., arXiv:2212.10496) goes the other direction: instead of
embedding the question, it asks the model to write a hypothetical answer and
embeds that, on the idea that a hypothetical answer's vocabulary is closer
to a real passage's vocabulary than the question's vocabulary is.
"""

from __future__ import annotations

from agentic_patterns import Embedder, Message, Provider, get_embedder, get_provider

from patterns.rag.assembly import assemble_context
from patterns.rag.chunking import Chunk, ScoredChunk
from patterns.rag.corpus import default_chunks
from patterns.rag.dense import DenseIndex, build_dense_index, dense_retrieve
from patterns.rag.generation import GroundedAnswer, generate_grounded_answer
from patterns.rag.hybrid import reciprocal_rank_fusion

_MULTI_QUERY_DEMO_QUERY = "What should I do about the incident, and how long until the report on it is due?"
_HYDE_DEMO_QUERY = "What happens right after a customer's login stops working because of a bad release?"

_MULTI_QUERY_SYSTEM = (
    "Split the user's question into 2-3 focused sub-queries that together "
    "cover what it is asking. Reply with one sub-query per line, no "
    "numbering, no other text."
)

_HYDE_SYSTEM = (
    "Write a short, plausible passage (2-3 sentences) that would answer the "
    "user's question, as if it were pulled straight from internal "
    "documentation. Do not hedge or say you are unsure; write it as fact."
)


def build_multi_query_prompt(query: str) -> str:
    """Build the prompt asking the model to split a query into sub-queries."""
    return f"Question: {query}"


def parse_multi_queries(text: str) -> list[str]:
    """Parse one sub-query per non-blank line, stripping list markers."""
    queries: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789.() ").strip()
        if cleaned:
            queries.append(cleaned)
    return queries


def multi_query_expand(query: str, provider: Provider) -> list[str]:
    """Ask the model to rewrite a query into narrower sub-queries.

    Args:
        query: The original, possibly vague or multi-part, question.
        provider: `Provider` used for the rewrite call.

    Returns:
        The sub-queries the model proposed, in the order given. Never
        includes the original query; callers that want it retrieved too
        should add it to the list themselves.
    """
    completion = provider.complete([Message.user(build_multi_query_prompt(query))], system=_MULTI_QUERY_SYSTEM)
    return parse_multi_queries(completion.content)


def multi_query_retrieve(
    query: str,
    dense_index: DenseIndex,
    embedder: Embedder,
    provider: Provider,
    *,
    top_k: int = 5,
    fetch_k: int = 5,
    rrf_k: int = 60,
) -> tuple[list[str], list[ScoredChunk]]:
    """Expand a query, retrieve for each sub-query, and fuse the results.

    Args:
        query: The original question.
        dense_index: A `DenseIndex` over the corpus.
        embedder: Embedder used for each sub-query.
        provider: `Provider` used to generate the sub-queries.
        top_k: Number of fused chunks to return.
        fetch_k: Number of candidates each sub-query contributes before fusion.
        rrf_k: RRF's rank-damping constant.

    Returns:
        A tuple of the sub-queries used and the fused, top-k `ScoredChunk`s.
    """
    sub_queries = multi_query_expand(query, provider)
    rankings = [dense_retrieve(sub_query, dense_index, embedder, top_k=fetch_k) for sub_query in sub_queries]
    fused = reciprocal_rank_fusion(rankings, k=rrf_k)
    return sub_queries, fused[:top_k]


def build_hyde_prompt(query: str) -> str:
    """Build the prompt asking the model to write a hypothetical answer passage."""
    return f"Question: {query}"


def hyde_generate(query: str, provider: Provider) -> str:
    """Generate a hypothetical answer passage to embed instead of the query.

    Args:
        query: The user's question.
        provider: `Provider` used for the generation call.

    Returns:
        A short passage written as if it answered the question, meant to be
        embedded and used in place of the raw query for dense retrieval.
    """
    completion = provider.complete([Message.user(build_hyde_prompt(query))], system=_HYDE_SYSTEM)
    return completion.content


def hyde_retrieve(
    query: str, dense_index: DenseIndex, embedder: Embedder, provider: Provider, *, top_k: int = 5
) -> tuple[str, list[ScoredChunk]]:
    """Retrieve using a HyDE hypothetical document instead of the raw query.

    Args:
        query: The user's question.
        dense_index: A `DenseIndex` over the corpus.
        embedder: Embedder used to embed the hypothetical document.
        provider: `Provider` used to generate the hypothetical document.
        top_k: Number of chunks to return.

    Returns:
        A tuple of the generated hypothetical document and the retrieved,
        top-k `ScoredChunk`s.
    """
    hypothetical = hyde_generate(query, provider)
    results = dense_retrieve(hypothetical, dense_index, embedder, top_k=top_k)
    return hypothetical, results


def run_multi_query_demo(
    provider: Provider | None = None,
) -> tuple[str, list[str], list[Chunk], GroundedAnswer]:
    """Demonstrate multi-query expansion recovering both halves of a vague question.

    The demo query bundles two unrelated asks ("what do I do about the
    incident" and "when is the report due") into one sentence that shares
    little exact vocabulary with either answer. Dense retrieval on the raw
    query alone ranks the correct chunks poorly; splitting into two focused
    sub-queries and fusing their results with RRF surfaces both.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with the sub-query split and the final
            grounded answer.

    Returns:
        A tuple of the original query, the sub-queries the model proposed,
        the assembled context chunks, and the grounded answer.
    """
    if provider is None:
        provider = get_provider(
            script=[
                "What is the first mitigation step for a SEV1 incident caused by a recent deploy?\n"
                "How long after resolution is a postmortem report due?",
                "The first mitigation step for a SEV1 caused by a recent deploy is an immediate "
                "rollback, not a forward fix [incident-runbook#1]. Once the incident is resolved, "
                "the postmortem report is due within forty eight hours [incident-runbook#2].",
            ]
        )
    embedder = get_embedder()
    dense_index = build_dense_index(default_chunks(), embedder)
    sub_queries, fused = multi_query_retrieve(_MULTI_QUERY_DEMO_QUERY, dense_index, embedder, provider, top_k=2, fetch_k=2)
    context_chunks = assemble_context(fused, token_budget=200)
    answer = generate_grounded_answer(_MULTI_QUERY_DEMO_QUERY, context_chunks, provider)
    return _MULTI_QUERY_DEMO_QUERY, sub_queries, context_chunks, answer


def run_hyde_demo(provider: Provider | None = None) -> tuple[str, str, list[ScoredChunk]]:
    """Demonstrate HyDE recovering a match a vague raw query misses.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with a plausible hypothetical passage.

    Returns:
        A tuple of the original query, the generated hypothetical document,
        and the chunks retrieved using it.
    """
    if provider is None:
        provider = get_provider(
            script=[
                "When a SEV1 outage blocks customer logins after a recent deploy, the on-call "
                "engineer's first mitigation step is an immediate rollback of the release rather "
                "than attempting a forward fix."
            ]
        )
    embedder = get_embedder()
    dense_index = build_dense_index(default_chunks(), embedder)
    hypothetical, results = hyde_retrieve(_HYDE_DEMO_QUERY, dense_index, embedder, provider, top_k=3)
    return _HYDE_DEMO_QUERY, hypothetical, results
