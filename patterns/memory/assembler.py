"""Context assembler: merges every memory source into one prompt under a
token budget.

Implements the "assemble context" and "enforce the short-term budget" steps
of the canonical control flow in one place: system prompt, procedural
rules, the running summary and recent turns from short-term memory, and
retrieved long-term items, trimmed to fit a `TokenBudget` by priority when
the sum overflows.

Priority, highest to lowest:

1. The system prompt and procedural rules are pinned: always included,
   never evicted, regardless of how much history has accumulated. Stable
   rules suffer positional decay when buried mid-transcript (context rot),
   so they are kept at the top rather than fit into the evictable budget.
2. Retrieved long-term items, lowest-relevance dropped first.
3. Short-term turns, oldest dropped first.

Namespace isolation is enforced upstream, by `VectorStore.search` and every
memory wrapper that calls it with an explicit namespace; the assembler only
ever sees whatever the caller already retrieved for one namespace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Message, get_embedder

from patterns.memory.procedural import ProceduralMemory
from patterns.memory.semantic import SemanticMemory
from patterns.memory.short_term import ShortTermMemory, TokenBudget, count_tokens
from patterns.memory.vector_store import ScoredRecord, VectorStore


@dataclass
class AssembledContext:
    """The final assembled prompt.

    Attributes:
        system: The system prompt, base instructions plus rendered
            procedural rules. Always included in full, never trimmed.
        messages: Retrieved long-term items followed by short-term turns,
            trimmed to fit the budget.
        total_tokens: Approximate token total of `system` plus `messages`.
        dropped: Content of every message evicted to make room, in the
            order it was dropped, for inspection and testing.
    """

    system: str
    messages: list[Message]
    total_tokens: int
    dropped: list[str] = field(default_factory=list)


def assemble_context(
    *,
    base_system: str,
    procedural: ProceduralMemory | None,
    short_term: ShortTermMemory,
    retrieved: list[ScoredRecord],
    budget: TokenBudget,
) -> AssembledContext:
    """Assemble a full prompt from every memory source, trimmed to `budget`.

    Args:
        base_system: The task-specific system prompt text.
        procedural: Standing rules to pin alongside `base_system`, if any.
        short_term: The current conversation buffer.
        retrieved: Long-term items already retrieved for this query,
            typically the output of `retrieval.retrieve`.
        budget: Token ceiling for `messages` plus the pinned system prompt.
    """
    system_parts = [base_system]
    if procedural is not None and procedural.render():
        system_parts.append(procedural.render())
    system = "\n\n".join(part for part in system_parts if part)
    pinned_tokens = count_tokens(system)

    retrieved_sorted = sorted(retrieved, key=lambda s: s.similarity, reverse=True)
    retrieved_messages = [Message.user(f"[retrieved memory] {s.record.text}") for s in retrieved_sorted]
    short_term_messages = short_term.render()

    def total_tokens(messages: list[Message]) -> int:
        return pinned_tokens + sum(count_tokens(m.content) for m in messages)

    dropped: list[str] = []
    combined = retrieved_messages + short_term_messages
    while total_tokens(combined) > budget.limit and combined:
        if retrieved_messages:
            dropped.append(retrieved_messages.pop().content)
        elif short_term_messages:
            dropped.append(short_term_messages.pop(0).content)
        combined = retrieved_messages + short_term_messages

    final_messages = retrieved_messages + short_term_messages
    return AssembledContext(
        system=system, messages=final_messages, total_tokens=total_tokens(final_messages), dropped=dropped
    )


def run_assembler_demo() -> AssembledContext:
    """Assemble a prompt under a tight budget: a procedural rule stays
    pinned to the system prompt even as retrieved items and old turns are
    trimmed to fit.
    """
    procedural = ProceduralMemory(namespace="user:alex")
    procedural.add_rule("Always answer in metric units.")

    short_term = ShortTermMemory(mode="full")
    for m in [
        Message.user("How far is the airport?"),
        Message.assistant("About 18 kilometers from downtown."),
        Message.user("And the drive time?"),
        Message.assistant("Roughly 25 minutes without traffic."),
        Message.user("What about the train?"),
    ]:
        short_term.append(m)

    embedder = get_embedder()
    store = VectorStore()
    memory = SemanticMemory(store, embedder, namespace="user:alex")
    memory.write_fact("home_airport", "ORD, 18km from downtown")
    retrieved = memory.recall("airport distance", top_k=1)

    budget = TokenBudget(limit=25)
    return assemble_context(
        base_system="You are a travel assistant.",
        procedural=procedural,
        short_term=short_term,
        retrieved=retrieved,
        budget=budget,
    )
