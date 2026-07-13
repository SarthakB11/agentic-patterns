"""Episodic memory: a log of specific past events and task outcomes.

Written after a task completes, so a later attempt at a similar task can
retrieve what happened and learn from it. This is Reflexion-style verbal
self-reflection stored as memory: a failed attempt's lesson is recorded
once and read back before the next try, rather than re-derived from
scratch every time.
"""

from __future__ import annotations

from agentic_patterns import Embedder, Message, Provider, get_embedder, get_provider

from patterns.memory.retrieval import RetrievalConfig, retrieve
from patterns.memory.vector_store import ScoredRecord, VectorStore


class EpisodicMemory:
    """One record per attempt at a task: what was tried, what happened, and
    the lesson drawn from it.
    """

    def __init__(self, store: VectorStore, embedder: Embedder, namespace: str) -> None:
        self.store = store
        self.embedder = embedder
        self.namespace = namespace
        self._count = 0

    def record_episode(self, task: str, outcome: str, lesson: str) -> str:
        """Write one episode. Failed outcomes are weighted more important,
        since a failure's lesson is usually more worth retrieving again than
        a routine success.

        Returns:
            The generated episode id.
        """
        self._count += 1
        episode_id = f"episode-{self._count}"
        text = f"task: {task}\noutcome: {outcome}\nlesson: {lesson}"
        embedding = self.embedder.embed([text])[0]
        importance = 0.9 if outcome.lower().startswith("fail") else 0.5
        self.store.upsert(
            episode_id,
            self.namespace,
            text,
            embedding,
            metadata={"task": task, "outcome": outcome, "lesson": lesson},
            importance=importance,
        )
        return episode_id

    def recall_lessons(self, query: str, top_k: int = 2) -> list[ScoredRecord]:
        """Retrieve past episodes relevant to `query`, biased toward
        important (typically failed) ones."""
        config = RetrievalConfig(top_k=top_k, min_similarity=0.1, importance_weight=0.3)
        return retrieve(self.store, self.embedder, self.namespace, query, config)


def run_episodic_demo(provider: Provider | None = None) -> dict[str, str]:
    """A failed Terraform apply writes a verbal lesson to episodic memory;
    a later, similar task retrieves that lesson before trying again.
    """
    if provider is None:
        provider = get_provider(
            script=[
                "The apply failed because the S3 backend bucket was not versioned; "
                "enable bucket versioning before the next apply.",
                "Enabled S3 bucket versioning first, per the earlier lesson, then ran "
                "terraform apply successfully against the new environment.",
            ]
        )
    embedder = get_embedder()
    store = VectorStore()
    memory = EpisodicMemory(store, embedder, namespace="user:alex")

    lesson_completion = provider.complete(
        [Message.user("terraform apply failed with: Error: S3 backend bucket must be versioned.")],
        system="State in one sentence what went wrong and the concrete fix for next time.",
    )
    memory.record_episode(
        task="terraform apply to us-west-2",
        outcome="failed: backend bucket not versioned",
        lesson=lesson_completion.content.strip(),
    )

    hits = memory.recall_lessons("terraform apply to a new environment")
    lessons_text = "; ".join(h.record.metadata["lesson"] for h in hits)
    retry_completion = provider.complete(
        [Message.user(f"Apply Terraform to a new environment. Known lessons: {lessons_text}")],
        system="Use the known lessons to avoid repeating past failures.",
    )

    return {
        "lesson_recorded": lesson_completion.content.strip(),
        "lessons_recalled": lessons_text,
        "retry_result": retry_completion.content,
    }
