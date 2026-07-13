"""Ingestion: split documents into overlapping chunks.

Every retrieval variant in this package operates on `Chunk` objects, not raw
documents, so chunking is the first stage of the canonical control flow
(see `pipeline.py`). Chunks keep their source document id and character
offsets so a citation can point back to somewhere real in the original text.
This module uses fixed-size chunking with overlap, the baseline strategy the
research brief names; `contextual.py` shows a refinement that changes what
gets embedded, not the chunk boundaries themselves.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """A single source document before chunking.

    Attributes:
        id: Stable identifier for the document, used as the chunk id prefix.
        text: Full document text.
    """

    id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document, the unit retrieval operates on.

    Attributes:
        id: Stable chunk identifier, `f"{source_id}#{index}"`. Citations in
            generated answers refer back to this id.
        source_id: Id of the `Document` this chunk was cut from.
        text: The chunk's text.
        start: Character offset of `text` within the source document.
        end: Character offset one past the end of `text`.
    """

    id: str
    source_id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk paired with a retrieval or reranking score.

    Attributes:
        chunk: The chunk itself.
        score: A similarity, BM25, fusion, or rerank score. Scale differs by
            retriever, so scores are only comparable within one ranked list.
    """

    chunk: Chunk
    score: float


def chunk_document(document: Document, *, size: int = 220, overlap: int = 40) -> list[Chunk]:
    """Split one document into fixed-size, overlapping chunks.

    Chunk boundaries are deterministic: given the same document, size, and
    overlap, the chunk count and offsets are always the same. A chunk end
    snaps back to the nearest preceding space so chunks break on word
    boundaries rather than mid-word, when a space is available within the
    chunk. The final chunk may be shorter than `size` if the document does
    not divide evenly, and leading whitespace at a chunk's start (left over
    from snapping the previous chunk's end) is skipped.

    Args:
        document: The document to split.
        size: Maximum characters per chunk.
        overlap: Characters shared between consecutive chunks, so a sentence
            that falls near a boundary is still whole in at least one chunk.

    Returns:
        Chunks in document order, each carrying its source id and offsets.

    Raises:
        ValueError: If `overlap` is not smaller than `size`, which would
            make the chunker fail to advance.
    """
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")
    if size <= 0:
        raise ValueError("size must be positive")

    text = document.text
    total = len(text)
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < total:
        if start > 0 and not text[start - 1].isspace():
            next_space = text.find(" ", start)
            start = next_space + 1 if next_space != -1 else total
        while start < total and text[start].isspace():
            start += 1
        if start >= total:
            break
        raw_end = min(start + size, total)
        end = raw_end
        if end < total and not text[end].isspace():
            snapped = text.rfind(" ", start, end)
            if snapped > start:
                end = snapped
        chunks.append(
            Chunk(id=f"{document.id}#{index}", source_id=document.id, text=text[start:end], start=start, end=end)
        )
        if end >= total:
            break
        index += 1
        start = max(end - overlap, start + 1)
    return chunks


def chunk_corpus(documents: list[Document], *, size: int = 220, overlap: int = 40) -> list[Chunk]:
    """Chunk every document in a corpus, in order.

    Args:
        documents: Documents to chunk.
        size: Maximum characters per chunk, passed to `chunk_document`.
        overlap: Overlap in characters, passed to `chunk_document`.

    Returns:
        All chunks from all documents, grouped by document and in document
        order within each group.
    """
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, size=size, overlap=overlap))
    return chunks
