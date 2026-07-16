"""Semantic memory: a persistent store of facts and user preferences,
decoupled from when they were learned.

Held here as a profile schema, one record per subject key, rather than a
free-form collection: writing a fact under a subject that already exists
overwrites the record in place instead of accumulating a second, possibly
contradictory, one. That overwrite rule is the write policy for semantic
memory; `write_policy.py` covers the two other write concerns, when to
write (hot path vs background) and what to write (extraction).

This conflict resolution is narrower than it looks: `write_fact` only
catches a conflict when the new fact reuses the exact subject key. A fact
stored as `plan: pro tier` and a later `subscription: free tier` are the
same underlying claim under different keys, so both persist and both
surface in retrieval, exactly the accumulation-and-contradiction failure
mode this pattern's own design brief warns against. `mem0_update.py`'s
similarity-gated ADD/UPDATE/DELETE/NOOP decision is the fix for that case:
it compares a new candidate fact against its nearest existing memories by
embedding, not by exact key.
"""

from __future__ import annotations

from typing import Any

from agentic_patterns import Embedder, Message, Provider, get_embedder, get_provider
from patterns.memory.retrieval import RetrievalConfig, retrieve
from patterns.memory.vector_store import ScoredRecord, VectorStore


class SemanticMemory:
    """Facts and preferences for one namespace, one record per subject."""

    def __init__(self, store: VectorStore, embedder: Embedder, namespace: str) -> None:
        self.store = store
        self.embedder = embedder
        self.namespace = namespace

    def write_fact(self, subject: str, value: str, metadata: dict[str, Any] | None = None) -> bool:
        """Upsert a fact keyed by `subject`.

        Args:
            subject: The fact's key, e.g. "timezone" or "plan".
            value: The fact's current value.
            metadata: Extra structured data to attach.

        Returns:
            True if this write replaced an existing value for `subject`
            (a conflict was resolved by overwrite), False if `subject` was
            new.

        Note:
            Conflict detection is exact-key only: a fact under a different
            subject that says the same thing in different words is not
            caught here and will accumulate as a second record. See
            `mem0_update.apply_candidate_fact` for similarity-gated
            resolution across differently-keyed facts.
        """
        existing = self.store.get(self.namespace, subject)
        text = f"{subject}: {value}"
        embedding = self.embedder.embed([text])[0]
        meta = dict(metadata or {})
        if existing is not None:
            meta["previous_value"] = existing.text
        self.store.upsert(subject, self.namespace, text, embedding, metadata=meta, importance=0.8)
        return existing is not None

    def recall(self, query: str, top_k: int = 3, min_similarity: float = 0.1) -> list[ScoredRecord]:
        """Retrieve facts relevant to `query` by similarity search."""
        config = RetrievalConfig(top_k=top_k, min_similarity=min_similarity)
        return retrieve(self.store, self.embedder, self.namespace, query, config)


def run_semantic_demo(provider: Provider | None = None) -> dict[str, Any]:
    """Write two facts, then overwrite one when the user's situation
    changes, and confirm the store holds exactly two records, not three:
    the conflicting write replaced the old value instead of accumulating.
    """
    if provider is None:
        provider = get_provider(
            script=["The user is on the pro tier, with a 1M request/month limit."]
        )
    embedder = get_embedder()
    store = VectorStore()
    memory = SemanticMemory(store, embedder, namespace="user:alex")

    memory.write_fact("plan", "free tier, 10k requests/month")
    memory.write_fact("timezone", "America/Chicago")
    was_overwrite = memory.write_fact("plan", "pro tier, 1M requests/month")

    hits = memory.recall("What plan is the user on?", top_k=1)
    memory_text = hits[0].record.text if hits else "none"
    answer = provider.complete([Message.user(f"What plan is the user on? Memory: {memory_text}")])

    return {
        "record_count": len(store.all("user:alex")),
        "plan_write_was_overwrite": was_overwrite,
        "recalled_fact": memory_text,
        "answer": answer.content,
    }
