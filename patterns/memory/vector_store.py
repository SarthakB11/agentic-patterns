"""In-memory vector store: the shared machinery under semantic and episodic
memory.

Each record pairs raw text with its embedding and a small metadata bag,
scoped to a namespace so one user's or thread's memories never surface in
another's search results. Records are keyed by an explicit id, so writing
the same id twice is an upsert (create or overwrite) rather than an append.
That single property is what dedup and conflict resolution build on
elsewhere in this pattern: callers who want "one record per subject" just
use the subject as the id.

The store keeps a logical clock (a counter incremented on every write)
instead of wall-clock time, so recency scoring in `retrieval.py` stays
deterministic across runs and machines.

`delete` exists on this class, but nothing in `retrieval.py` ever calls it:
recency there is a ranking weight, not an eviction trigger, so a namespace
only ever grows unless a caller deletes explicitly. `forgetting.py` is the
module that actually removes records, by decay, TTL, capacity bound, or
model-judged intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_patterns import cosine_similarity, get_embedder


@dataclass
class VectorRecord:
    """One stored memory item.

    Attributes:
        id: Caller-assigned identifier, unique within a namespace. Upserting
            the same id replaces the record in place.
        namespace: Isolation boundary, typically a user or thread id.
        text: The raw text this record represents.
        embedding: The embedding vector for `text`.
        metadata: Arbitrary structured data attached to the record.
        written_at: Logical clock tick at the time of the last write.
        importance: A 0-1 salience score, used by importance-weighted
            retrieval. Defaults to a neutral middle value.
    """

    id: str
    namespace: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    written_at: int = 0
    importance: float = 0.5


@dataclass
class ScoredRecord:
    """A `VectorRecord` paired with a similarity or blended score from a query."""

    record: VectorRecord
    similarity: float


class VectorStore:
    """A namespaced, in-memory store of (text, embedding, metadata) records."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], VectorRecord] = {}
        self._clock = 0

    def upsert(
        self,
        id: str,
        namespace: str,
        text: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
    ) -> VectorRecord:
        """Create or overwrite the record at `(namespace, id)`.

        Args:
            id: Record id, unique within `namespace`.
            namespace: Isolation boundary for this record.
            text: Raw text the record represents.
            embedding: Embedding vector for `text`.
            metadata: Structured data to attach to the record.
            importance: Salience score in [0, 1].
        """
        self._clock += 1
        record = VectorRecord(
            id=id,
            namespace=namespace,
            text=text,
            embedding=embedding,
            metadata=dict(metadata or {}),
            written_at=self._clock,
            importance=importance,
        )
        self._records[(namespace, id)] = record
        return record

    def get(self, namespace: str, id: str) -> VectorRecord | None:
        """Look up a record by namespace and id, or return None."""
        return self._records.get((namespace, id))

    def delete(self, namespace: str, id: str) -> bool:
        """Remove a record. Returns True if a record was removed."""
        return self._records.pop((namespace, id), None) is not None

    def all(self, namespace: str) -> list[VectorRecord]:
        """Return every record in `namespace`, in no particular order."""
        return [r for (ns, _rid), r in self._records.items() if ns == namespace]

    def search(
        self,
        namespace: str,
        query_embedding: list[float],
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> list[ScoredRecord]:
        """Cosine-similarity top-k search within one namespace.

        Args:
            namespace: Only records in this namespace are searched.
            query_embedding: Embedding of the query text.
            top_k: Maximum number of results to return.
            min_similarity: Records scoring below this are excluded before
                the top-k cut, so a small top-k never pads results with
                irrelevant items.
        """
        scored = [
            ScoredRecord(r, cosine_similarity(query_embedding, r.embedding)) for r in self.all(namespace)
        ]
        scored = [s for s in scored if s.similarity >= min_similarity]
        scored.sort(key=lambda s: s.similarity, reverse=True)
        return scored[:top_k]

    @property
    def clock(self) -> int:
        """The current logical clock value (number of writes so far)."""
        return self._clock


def run_vector_store_demo() -> list[ScoredRecord]:
    """Insert a few records and run a top-k search, showing that a
    sub-threshold item is excluded from the results rather than padding
    the tail of the list.
    """
    embedder = get_embedder()
    store = VectorStore()
    docs = {
        "rec-coffee": "The user drinks dark roast coffee every morning.",
        "rec-allergy": "The user is allergic to peanuts.",
        "rec-standup": "Team standup happens every day at 9:30am.",
    }
    for key, text in docs.items():
        store.upsert(key, "demo", text, embedder.embed([text])[0])
    query = "What does the user drink in the morning?"
    return store.search("demo", embedder.embed([query])[0], top_k=3, min_similarity=0.05)
