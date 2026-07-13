"""Grounded generation: answer only from the retrieved chunks, cite the
chunks a claim came from, and abstain when there is nothing to ground an
answer in.

The prompt gives the model each chunk labeled with its id and instructs it
to cite ids in square brackets rather than restate them. Citations in the
model's reply are checked against the ids actually supplied. Retrieval can
fail silently: an empty or irrelevant result set can still produce a fluent,
confident answer, so a caller with zero context chunks gets a fixed abstain
message instead of a generation call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentic_patterns import Message, Provider

from patterns.rag.chunking import Chunk

ABSTAIN_ANSWER = "I don't have enough information in the retrieved documents to answer that."

_GENERATION_SYSTEM = (
    "Answer the question using only the numbered context chunks below. "
    "Cite the chunk id of every claim in square brackets, like [chunk-id]. "
    "If the chunks do not contain the answer, say so instead of guessing."
)

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_\-#]+)\]")


@dataclass
class GroundedAnswer:
    """The result of a grounded-generation call.

    Attributes:
        answer: The model's answer text, or `ABSTAIN_ANSWER` when abstaining.
        citations: Chunk ids the answer cited, in first-mention order,
            limited to ids that were actually supplied in the prompt.
        abstained: True if no context was available and generation was
            skipped entirely.
    """

    answer: str
    citations: list[str] = field(default_factory=list)
    abstained: bool = False


def format_context_block(chunks: list[Chunk]) -> str:
    """Render chunks as a labeled context block for a generation prompt."""
    return "\n\n".join(f"[{chunk.id}] {chunk.text}" for chunk in chunks)


def build_grounded_prompt(query: str, chunks: list[Chunk]) -> str:
    """Build the user-turn prompt for grounded generation.

    Args:
        query: The user's question.
        chunks: Context chunks to answer from, already assembled and ordered.
    """
    return f"Context:\n{format_context_block(chunks)}\n\nQuestion: {query}"


def extract_citations(text: str, valid_ids: set[str]) -> list[str]:
    """Pull `[chunk-id]` citation markers out of generated text.

    Args:
        text: The model's answer text.
        valid_ids: Chunk ids that were actually supplied to the model; any
            bracketed token outside this set is not a real citation and is
            dropped rather than trusted.

    Returns:
        Distinct valid chunk ids, in the order they first appear.
    """
    seen: list[str] = []
    for candidate in _CITATION_RE.findall(text):
        if candidate in valid_ids and candidate not in seen:
            seen.append(candidate)
    return seen


def generate_grounded_answer(query: str, chunks: list[Chunk], provider: Provider) -> GroundedAnswer:
    """Generate an answer grounded in the given chunks, or abstain.

    Args:
        query: The user's question.
        chunks: Assembled context chunks. An empty list skips the model
            call entirely and returns the abstain answer.
        provider: `Provider` used for the generation call.

    Returns:
        A `GroundedAnswer` with the model's text and the citations that
        resolved to a supplied chunk id.
    """
    if not chunks:
        return GroundedAnswer(answer=ABSTAIN_ANSWER, citations=[], abstained=True)

    prompt = build_grounded_prompt(query, chunks)
    completion = provider.complete([Message.user(prompt)], system=_GENERATION_SYSTEM)
    valid_ids = {chunk.id for chunk in chunks}
    citations = extract_citations(completion.content, valid_ids)
    return GroundedAnswer(answer=completion.content, citations=citations, abstained=False)
