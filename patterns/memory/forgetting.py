"""Forgetting: decay, reinforcement, TTL, capacity bound, and intent-aware
deletion.

Nothing else in this pattern ever removes a record. `retrieval._recency_score`
down-weights old items in ranking, but the store only ever grows, so a stale
or superseded fact stays in the namespace forever and keeps competing for
the top-k. This module makes forgetting a first-class store operation and
keeps the two mechanisms MemoryBank (Zhong, Guo, Gao, Ye, Wang, 2023,
arXiv:2305.10250) and Control-Plane Placement Shapes Forgetting (Yang,
arXiv:2606.15903) argue belong in different places, deliberately separate:

- **Deterministic primitives** (`sweep_decay`, `sweep_ttl`,
  `enforce_capacity`): pure arithmetic over the store's logical clock and
  each record's access history, no model call. These handle lexical and
  time-based forgetting cheaply, mirroring MemoryBank's Ebbinghaus-curve
  decay and reinforcement.
- **Mutation-time model judgment** (`intent_aware_delete`): a natural
  language "forget X" request needs a model to decide which stored records
  it covers, since a deterministic keyword rule misses canonicalization
  (the request names something the record does not spell out verbatim)
  and intent (which records the request actually means). Yang's 2026 study
  finds this split, not a single mechanism, is what reaches high deletion
  accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agentic_patterns import Embedder, Message, Provider, get_embedder, get_provider
from patterns.memory.vector_store import VectorRecord, VectorStore

_ACCESS_COUNT_KEY = "access_count"
_LAST_ACCESS_KEY = "last_access_tick"
_TTL_KEY = "ttl_tick"


@dataclass
class DeletionLogEntry:
    """One record removed by a forgetting pass.

    Attributes:
        record_id: The id of the removed record.
        mechanism: Which pass removed it: "decay", "ttl", "capacity", or
            "intent".
        reason: A short human-readable explanation, for test assertions
            and demo output.
    """

    record_id: str
    mechanism: str
    reason: str


def touch(record: VectorRecord, now: int) -> None:
    """Record a retrieval hit against `record`: increment its access count
    and stamp the current tick as its last access, the reinforcement half
    of the Ebbinghaus curve. Call this from a retrieval path when a record
    is actually surfaced to a query, not merely considered.
    """
    record.metadata[_ACCESS_COUNT_KEY] = record.metadata.get(_ACCESS_COUNT_KEY, 0) + 1
    record.metadata[_LAST_ACCESS_KEY] = now


def strength(record: VectorRecord, now: int, decay_rate: float = 0.25) -> float:
    """Compute a record's current memory strength in (0, 1].

    An exponential decay over ticks since last access, its rate slowed by
    how often the record has been reinforced: `exp(-decay_rate * age /
    (1 + access_count))`. A record touched often decays slowly; a
    never-accessed record decays at the bare `decay_rate`.
    """
    last_access = record.metadata.get(_LAST_ACCESS_KEY, record.written_at)
    access_count = record.metadata.get(_ACCESS_COUNT_KEY, 0)
    age = max(now - last_access, 0)
    stability = 1.0 + access_count
    return math.exp(-decay_rate * age / stability)


def set_ttl(record: VectorRecord, expires_at_tick: int) -> None:
    """Mark `record` for TTL-based deletion once `store.clock` passes
    `expires_at_tick`, independent of its decay strength.
    """
    record.metadata[_TTL_KEY] = expires_at_tick


def sweep_decay(
    store: VectorStore, namespace: str, now: int, floor: float = 0.1, decay_rate: float = 0.25
) -> list[DeletionLogEntry]:
    """Delete every record in `namespace` whose current strength falls
    below `floor`. A deterministic primitive: no model call, arithmetic
    only.
    """
    log: list[DeletionLogEntry] = []
    for record in store.all(namespace):
        s = strength(record, now, decay_rate)
        if s < floor:
            store.delete(namespace, record.id)
            log.append(DeletionLogEntry(record.id, "decay", f"strength {s:.3f} below floor {floor}"))
    return log


def sweep_ttl(store: VectorStore, namespace: str, now: int) -> list[DeletionLogEntry]:
    """Delete every record in `namespace` whose TTL (set via `set_ttl`) is
    at or past `now`, regardless of decay strength: retention imposed by
    policy, not earned by recall.
    """
    log: list[DeletionLogEntry] = []
    for record in store.all(namespace):
        ttl = record.metadata.get(_TTL_KEY)
        if ttl is not None and now >= ttl:
            store.delete(namespace, record.id)
            log.append(DeletionLogEntry(record.id, "ttl", f"now={now} past ttl={ttl}"))
    return log


def enforce_capacity(
    store: VectorStore, namespace: str, now: int, max_size: int, decay_rate: float = 0.25
) -> list[DeletionLogEntry]:
    """If `namespace` holds more than `max_size` records, evict the
    lowest-strength records until it fits: an LFU-and-recency blend since
    strength already folds in both access count and recency.
    """
    records = store.all(namespace)
    overflow = len(records) - max_size
    if overflow <= 0:
        return []
    weakest_first = sorted(records, key=lambda r: strength(r, now, decay_rate))
    log: list[DeletionLogEntry] = []
    for record in weakest_first[:overflow]:
        store.delete(namespace, record.id)
        log.append(DeletionLogEntry(record.id, "capacity", f"namespace over max_size={max_size}"))
    return log


def intent_aware_delete(
    provider: Provider, embedder: Embedder, store: VectorStore, namespace: str, request: str, top_k: int = 5
) -> list[DeletionLogEntry]:
    """Delete every record a natural-language forget request covers.

    Retrieves the `top_k` records most similar to `request`, then asks the
    model which of those ids the request actually means to remove. This is
    the mutation-time model-judged path deterministic rules cannot express:
    a request like "forget everything about my old employer" names an
    intent, not a literal string any stored record contains.

    Args:
        provider: Model to run the deletion-intent call with.
        embedder: Embedder used to find candidate records.
        store: Vector store to delete from.
        namespace: Isolation boundary to search and delete within.
        request: The natural-language forget request.
        top_k: Number of candidate records shown to the model.
    """
    query_vec = embedder.embed([request])[0]
    candidates = store.search(namespace, query_vec, top_k=top_k, min_similarity=0.0)
    if not candidates:
        return []
    listing = "\n".join(f"{c.record.id}: {c.record.text}" for c in candidates)
    completion = provider.complete(
        [Message.user(f"Forget request: {request}\n\nCandidate memories:\n{listing}")],
        system=(
            "Identify which candidate memory ids the forget request covers. "
            "Reply with a comma-separated list of ids, or NONE if none apply."
        ),
    )
    text = completion.content.strip()
    if not text or text.upper() == "NONE":
        return []
    target_ids = {part.strip() for part in text.split(",") if part.strip()}
    log: list[DeletionLogEntry] = []
    for record_id in target_ids:
        if store.delete(namespace, record_id):
            log.append(DeletionLogEntry(record_id, "intent", f"matched forget request: {request!r}"))
    return log


def run_forgetting_demo(provider: Provider | None = None) -> dict[str, object]:
    """Walk one namespace through every forgetting mechanism: an unaccessed
    record decays and is swept while a reinforced sibling survives the same
    span, a TTL-marked record is removed on schedule regardless of
    strength, a capacity bound evicts the weakest record once the
    namespace overflows, and an intent-aware request removes two records a
    lexical rule would miss.
    """
    if provider is None:
        provider = get_provider(script=["old-job-note, ex-employer-note"])
    embedder = get_embedder()
    store = VectorStore()
    namespace = "user:alex"

    # --- decay + reinforcement ---
    store.upsert("stale-note", namespace, "User once mentioned liking jazz.", embedder.embed(["jazz"])[0])
    store.upsert("active-note", namespace, "User's home airport is ORD.", embedder.embed(["ORD airport"])[0])
    now = store.clock
    active_note = store.get(namespace, "active-note")
    assert active_note is not None, "just upserted above"
    touch(active_note, now)  # reinforced; decays slower
    later = now + 20
    decay_log = sweep_decay(store, namespace, later, floor=0.05, decay_rate=0.25)
    active_note_survived_decay = store.get(namespace, "active-note") is not None

    # --- TTL ---
    store.upsert("promo-code", namespace, "20% off code SAVE20, valid this week.", embedder.embed(["promo"])[0])
    promo_code = store.get(namespace, "promo-code")
    assert promo_code is not None, "just upserted above"
    set_ttl(promo_code, store.clock + 2)
    store.upsert("filler-1", namespace, "filler", embedder.embed(["filler one"])[0])
    store.upsert("filler-2", namespace, "filler", embedder.embed(["filler two"])[0])
    ttl_log = sweep_ttl(store, namespace, store.clock)

    # --- capacity bound ---
    active_note = store.get(namespace, "active-note")
    assert active_note is not None, "still present; decay sweep left it alone"
    touch(active_note, store.clock)  # reinforced again; still important
    for i in range(3):
        store.upsert(f"pad-{i}", namespace, f"padding item {i}", embedder.embed([f"padding {i}"])[0])
    capacity_log = enforce_capacity(store, namespace, store.clock, max_size=3)
    capacity_final_size = len(store.all(namespace))
    active_note_survived_capacity = store.get(namespace, "active-note") is not None

    # --- intent-aware deletion ---
    store.upsert(
        "old-job-note", namespace, "User worked at Acme Corp until 2023.", embedder.embed(["Acme Corp job"])[0]
    )
    store.upsert(
        "ex-employer-note",
        namespace,
        "Acme Corp's timezone was America/New_York.",
        embedder.embed(["Acme Corp timezone"])[0],
    )
    store.upsert(
        "current-job-note", namespace, "User now works at a startup called Nimbus.", embedder.embed(["Nimbus job"])[0]
    )
    forget_request = "Forget everything about my old employer, Acme Corp."
    intent_log = intent_aware_delete(provider, embedder, store, namespace, forget_request)
    current_job_note_survived_intent_delete = store.get(namespace, "current-job-note") is not None

    return {
        "decay_deleted": [e.record_id for e in decay_log],
        "active_note_survived_decay": active_note_survived_decay,
        "ttl_deleted": [e.record_id for e in ttl_log],
        "capacity_deleted": [e.record_id for e in capacity_log],
        "capacity_final_size": capacity_final_size,
        "active_note_survived_capacity": active_note_survived_capacity,
        "intent_deleted": [e.record_id for e in intent_log],
        "current_job_note_survived_intent_delete": current_job_note_survived_intent_delete,
    }
