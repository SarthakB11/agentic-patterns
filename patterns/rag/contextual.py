"""Contextual retrieval: prepend a short, model-written description of a
chunk's place in its document before embedding it, so a chunk that reads
like "It also credits the unused portion of a plan..." stays findable on its
own instead of only making sense next to the sentence before it.

Fixed-size chunking can cut a chunk off from the pronoun or topic it needs to
be self-contained. Anthropic's contextual retrieval (2024) fixes this by
asking a model to write one or two sentences situating each chunk in its
source document, and embedding that blurb together with the chunk. A newer,
cheaper alternative is late chunking (Gunther et al., arXiv:2409.04701),
which embeds the whole document first with a long-context model and pools
token vectors into chunk vectors, needing no per-chunk model call; this
module implements the LLM-blurb approach since it composes directly with
this repo's mock-provider pattern, and notes late chunking as the production
alternative rather than re-implementing it.
"""

from __future__ import annotations

from agentic_patterns import Embedder, Message, Provider, get_embedder, get_provider

from patterns.rag.chunking import Chunk, Document, ScoredChunk
from patterns.rag.corpus import DOCUMENTS_BY_ID, default_chunks
from patterns.rag.dense import DenseIndex, build_dense_index, dense_retrieve

_CONTEXT_SYSTEM = (
    "Write one short sentence that situates the given chunk within its "
    "source document, naming the document and the chunk's topic, so the "
    "chunk can be understood on its own. Reply with only that sentence."
)

_ORPHAN_TEXT = "It also credits the unused portion of a plan automatically when a customer upgrades mid cycle."
_CONTEXTUAL_DEMO_QUERY = "How does Aurora's proration engine credit customers who upgrade their plan mid cycle?"


def build_context_prompt(document: Document, chunk: Chunk) -> str:
    """Build the prompt asking the model to describe one chunk's context."""
    return (
        f"Document ({document.id}):\n{document.text}\n\n"
        f"Chunk to situate:\n{chunk.text}"
    )


def contextualize_chunk(document: Document, chunk: Chunk, provider: Provider) -> str:
    """Ask the model for a one-sentence blurb situating a chunk in its document.

    Args:
        document: The chunk's source document, for the model to read in full.
        chunk: The chunk to describe.
        provider: `Provider` used for the description call.

    Returns:
        A short blurb, meant to be prepended to the chunk before embedding.
    """
    completion = provider.complete([Message.user(build_context_prompt(document, chunk))], system=_CONTEXT_SYSTEM)
    return completion.content.strip()


def build_contextual_index(
    chunks: list[Chunk], documents_by_id: dict[str, Document], embedder: Embedder, provider: Provider
) -> tuple[DenseIndex, dict[str, str]]:
    """Build a dense index where each chunk's embedding includes a context blurb.

    Args:
        chunks: Chunks to index.
        documents_by_id: Source documents keyed by id, for prompt context.
        embedder: Embedder used on `blurb + chunk text`, not the chunk alone.
        provider: `Provider` used to generate each chunk's blurb.

    Returns:
        A tuple of the resulting `DenseIndex` (chunk text left unmodified for
        display; only the embedded text changes) and a mapping from chunk id
        to the blurb generated for it.
    """
    blurbs: dict[str, str] = {}
    embedded_texts: list[str] = []
    for chunk in chunks:
        document = documents_by_id[chunk.source_id]
        blurb = contextualize_chunk(document, chunk, provider)
        blurbs[chunk.id] = blurb
        embedded_texts.append(f"{blurb} {chunk.text}")
    vectors = embedder.embed(embedded_texts)
    return DenseIndex(chunks=list(chunks), vectors=vectors), blurbs


def run_contextual_demo(
    provider: Provider | None = None,
) -> tuple[str, str, list[ScoredChunk], list[ScoredChunk]]:
    """Demonstrate a pronoun-orphaned chunk becoming findable once contextualized.

    A small chunk from the billing FAQ ("It also credits the unused portion
    of a plan...") shares no vocabulary with a query about Aurora's
    proration engine beyond generic words, so it ranks below several
    unrelated chunks. Prepending a one-sentence, model-written blurb naming
    the document and topic adds exactly the vocabulary the query needs, and
    the same chunk moves to the top of the ranking.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with one contextual blurb.

    Returns:
        A tuple of the query, the generated blurb, the ranking before
        contextualization, and the ranking after.
    """
    all_chunks = default_chunks()
    by_id = {chunk.id: chunk for chunk in all_chunks}
    orphan = Chunk(id="billing-faq#orphan", source_id="billing-faq", text=_ORPHAN_TEXT, start=213, end=309)
    distractors = [by_id["oncall-rotation#0"], by_id["billing-faq#0"], by_id["deploy-policy#0"], by_id["data-retention#0"]]
    demo_chunks = [*distractors, orphan]

    embedder = get_embedder()
    baseline_index = build_dense_index(demo_chunks, embedder)
    before = dense_retrieve(_CONTEXTUAL_DEMO_QUERY, baseline_index, embedder, top_k=len(demo_chunks))

    if provider is None:
        provider = get_provider(
            script=[
                "This chunk is from the Aurora Cloud billing FAQ and describes the proration "
                "engine that automatically credits unused plan time."
            ]
        )
    blurb = contextualize_chunk(DOCUMENTS_BY_ID["billing-faq"], orphan, provider)
    contextual_texts = [chunk.text for chunk in distractors] + [f"{blurb} {orphan.text}"]
    vectors = embedder.embed(contextual_texts)
    contextual_index = DenseIndex(chunks=demo_chunks, vectors=vectors)
    after = dense_retrieve(_CONTEXTUAL_DEMO_QUERY, contextual_index, embedder, top_k=len(demo_chunks))

    return _CONTEXTUAL_DEMO_QUERY, blurb, before, after
