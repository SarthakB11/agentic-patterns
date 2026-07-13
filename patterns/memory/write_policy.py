"""Write policy: what to persist, when to persist it, and how to resolve
conflicts with what is already stored.

Three concerns, kept distinct:

- **What**: `extract_facts` pulls durable, subject/value facts out of a raw
  user turn instead of storing the turn verbatim, so the store holds
  compact facts rather than accumulating unbounded raw text.
- **When**: `hot_path_write` persists synchronously, inside the turn,
  immediately available to the next retrieval at the cost of the
  extraction call's latency. `BackgroundWriteQueue` defers the same
  extraction to a later `drain` call, keeping the turn itself fast at the
  cost of staleness until drained.
- **Conflicts**: `SemanticMemory.write_fact` already resolves same-subject
  conflicts by clean overwrite (see `semantic.py`, and its docstring for
  the narrower limit that overwrite does not cover). `consolidate` covers
  the harder case, offline contradiction reconciliation, where a model
  must reconcile several candidate values already recorded under one
  known subject in prose rather than replace one key. This is a one-off
  reconcile, not sleep-time compute: it derives nothing that a later,
  unseen query reuses. `sleep_time.py` is where that amortization claim
  actually gets exercised, and `mem0_update.py` is where similarity-gated
  conflict resolution across different subject keys lives.

This module also holds the pattern's headline end-to-end demo: a fact told
to the agent in one session is recalled in a second, unrelated session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_patterns import Message, Provider, get_embedder, get_provider

from patterns.memory.semantic import SemanticMemory
from patterns.memory.short_term import ShortTermMemory
from patterns.memory.vector_store import VectorStore


@dataclass
class ExtractedFact:
    """One `subject: value` fact pulled out of a raw turn."""

    subject: str
    value: str


def extract_facts(provider: Provider, user_text: str) -> list[ExtractedFact]:
    """Ask the model to extract durable facts from `user_text` as lines of
    the form "subject: value". A scripted mock returns a fixed extraction,
    so this runs deterministically offline.

    Args:
        provider: The model to extract with.
        user_text: The raw user turn to extract facts from.

    Returns:
        One `ExtractedFact` per recognized line. Empty if the model
        replies "NONE" or returns no parseable lines.
    """
    completion = provider.complete(
        [Message.user(user_text)],
        system=(
            "Extract durable facts about the user from their message as "
            "lines of the form 'subject: value'. Only include facts worth "
            "remembering across sessions. If there are none, reply NONE."
        ),
    )
    text = completion.content.strip()
    if not text or text.upper() == "NONE":
        return []
    facts: list[ExtractedFact] = []
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        subject, _, value = line.partition(":")
        facts.append(ExtractedFact(subject.strip(), value.strip()))
    return facts


def hot_path_write(provider: Provider, memory: SemanticMemory, user_text: str) -> list[ExtractedFact]:
    """Extract facts from `user_text` and write each one synchronously,
    inside the current turn.
    """
    facts = extract_facts(provider, user_text)
    for fact in facts:
        memory.write_fact(fact.subject, fact.value)
    return facts


@dataclass
class BackgroundWriteQueue:
    """Queues raw turns during the interactive turn; extraction and
    persistence happen later, offline, when `drain` runs.
    """

    pending: list[str] = field(default_factory=list)

    def enqueue(self, user_text: str) -> None:
        """Queue one raw turn for later extraction."""
        self.pending.append(user_text)

    def drain(self, provider: Provider, memory: SemanticMemory) -> list[ExtractedFact]:
        """Extract and write every queued turn, in order, then clear the
        queue. Returns every fact written, across all queued turns.
        """
        written: list[ExtractedFact] = []
        while self.pending:
            user_text = self.pending.pop(0)
            written.extend(hot_path_write(provider, memory, user_text))
        return written


def consolidate(provider: Provider, memory: SemanticMemory, subject: str, candidates: list[str]) -> str:
    """Offline contradiction reconciliation: reconcile several candidate
    values already known to share one subject into a single current value,
    and overwrite the record.

    This is the offline analogue of `write_fact`'s in-line overwrite, used
    when a conflict needs a model to weigh candidates in prose (e.g. one
    stale, one current) rather than a clean same-subject replacement. It is
    a single inline reconcile, not sleep-time compute: it does not
    pre-derive anything a later, unseen query reuses, and it still requires
    the caller to already know the candidates share one subject, which
    `mem0_update.py`'s similarity-gated decision does not.
    """
    completion = provider.complete(
        [Message.user(f"Candidates for '{subject}': {'; '.join(candidates)}. Which is current?")],
        system="Resolve conflicting facts to the single most current, most specific value. Reply with only the value.",
    )
    resolved = completion.content.strip()
    memory.write_fact(subject, resolved)
    return resolved


def run_write_policy_demo(provider: Provider | None = None) -> dict[str, Any]:
    """Hot-path and background writing must converge to the same store
    state for the same input, and a consolidation pass resolves a
    contradiction recorded under one subject.
    """
    if provider is None:
        provider = get_provider(
            script=[
                "favorite_language: Python",
                "favorite_language: Python",
                "pro tier, 1M requests/month (seen today, supersedes the free tier)",
            ]
        )
    embedder = get_embedder()

    hot_store = VectorStore()
    hot_memory = SemanticMemory(hot_store, embedder, namespace="user:alex")
    hot_path_write(provider, hot_memory, "My favorite language is Python.")

    bg_store = VectorStore()
    bg_memory = SemanticMemory(bg_store, embedder, namespace="user:alex")
    queue = BackgroundWriteQueue()
    queue.enqueue("My favorite language is Python.")
    queue.drain(provider, bg_memory)

    hot_state = sorted((r.id, r.text) for r in hot_store.all("user:alex"))
    bg_state = sorted((r.id, r.text) for r in bg_store.all("user:alex"))

    resolved_plan = consolidate(
        provider, hot_memory, "plan", ["free tier (seen last month)", "pro tier (seen today)"]
    )

    return {
        "hot_path_state": hot_state,
        "background_state": bg_state,
        "states_equal": hot_state == bg_state,
        "resolved_plan": resolved_plan,
    }


def run_two_session_demo(provider: Provider | None = None) -> dict[str, Any]:
    """The pattern's headline scenario: a fact told to the agent in session
    one is recalled in session two, from a brand-new, empty short-term
    window, purely through long-term retrieval.
    """
    if provider is None:
        provider = get_provider(
            script=[
                "deployment_region: us-west-2\niac_tool: Terraform",
                "Your deployment is in us-west-2, provisioned with Terraform.",
            ]
        )
    embedder = get_embedder()
    store = VectorStore()
    memory = SemanticMemory(store, embedder, namespace="user:alex")

    # --- Session 1 ---
    session_1 = ShortTermMemory(mode="full")
    user_turn = "My deployment target is us-west-2 and I use Terraform for provisioning."
    session_1.append(Message.user(user_turn))
    written = hot_path_write(provider, memory, user_turn)
    session_1.append(Message.assistant("Noted, I'll remember that."))
    # Session 1's buffer is discarded here; nothing carries over in-window.

    # --- Session 2: a fresh, empty short-term window ---
    session_2 = ShortTermMemory(mode="full")
    query = "Tell me my deployment region and IaC tool."
    session_2.append(Message.user(query))
    hits = memory.recall(query, top_k=2)
    retrieved_text = "; ".join(h.record.text for h in hits)
    answer = provider.complete(
        [Message.user(f"{query}\n\nRelevant memory: {retrieved_text}")],
        system="Answer using only the relevant memory provided, in one sentence.",
    )

    return {
        "facts_written_in_session_1": [f"{f.subject}={f.value}" for f in written],
        "session_2_window_size_at_query": len(session_2.turns),
        "retrieved_in_session_2": retrieved_text,
        "answer": answer.content,
    }
