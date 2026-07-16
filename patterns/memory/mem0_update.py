"""Mem0-style extract-then-update: ADD / UPDATE / DELETE / NOOP.

`semantic.write_fact` resolves a conflict only when the new fact reuses the
exact subject key: a fact stored as `plan: pro tier` and a later
`subscription: free tier` are semantically the same claim under different
keys, so today both persist and both surface in retrieval, which is the
accumulation-and-contradiction failure mode this repo's own memory brief
warns against. Mem0 (Chhikara, Khant, Aryan, Singh, Yadav, ECAI 2025,
arXiv:2504.19413) closes that gap with a similarity-gated update decision:
retrieve the top-`s` existing memories similar to a new candidate fact and
ask the model to choose one of ADD, UPDATE, DELETE, or NOOP, instead of
keying by exact subject string.

Two calls per turn drive the pipeline: one extraction call (reusing
`write_policy.extract_facts`'s shape) turns the raw turn into candidate
facts, then one decision call per candidate fact turns that candidate plus
its `top_s` nearest existing memories into a single operation. This module
is deliberately not a replacement for `semantic.write_fact`: it is the
similarity-gated alternative for callers who need dedup across rephrasing,
at the cost of one extra model call per candidate fact.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentic_patterns import Embedder, Message, Provider, get_embedder, get_provider
from patterns.memory.vector_store import VectorStore
from patterns.memory.write_policy import ExtractedFact, extract_facts

DECISION_SYSTEM_PROMPT = (
    "You reconcile a new candidate memory against similar memories already "
    "stored. Reply with exactly one of:\n"
    "ADD\n"
    "UPDATE <id>: <merged text>\n"
    "DELETE <id>\n"
    "NOOP\n"
    "Choose ADD if the candidate is genuinely new. Choose UPDATE if it "
    "revises or refines an existing memory (merge the two into <merged "
    "text>, keeping the existing id). Choose DELETE if the candidate "
    "contradicts and supersedes an existing memory with no merge possible. "
    "Choose NOOP if the candidate is already fully captured by an existing "
    "memory."
)


@dataclass
class UpdateOp:
    """One resolved operation from the extract-then-update decision step.

    Attributes:
        fact_text: The candidate fact text this operation was decided for.
        operation: One of "ADD", "UPDATE", "DELETE", "NOOP".
        record_id: The store id this operation acted on. The fresh id for
            ADD, the target id for UPDATE/DELETE, None for NOOP.
        applied_text: The text written to the store for ADD/UPDATE. None
            for DELETE/NOOP.
    """

    fact_text: str
    operation: str
    record_id: str | None
    applied_text: str | None


def _next_id(store: VectorStore, namespace: str, prefix: str) -> str:
    """Compute a fresh `prefix-N` id, one past the highest `N` already used
    in `namespace`, so ids stay stable and collision-free without external
    counter state.
    """
    highest = 0
    for record in store.all(namespace):
        if record.id.startswith(f"{prefix}-"):
            suffix = record.id[len(prefix) + 1 :]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    return f"{prefix}-{highest + 1}"


def _parse_verdict(text: str) -> tuple[str, str | None, str]:
    """Parse a decision completion into (operation, target_id, payload).

    `payload` is the merged text for UPDATE, empty for the rest.
    """
    stripped = text.strip()
    head, _, rest = stripped.partition(" ")
    operation = head.upper().rstrip(":")
    if operation not in {"ADD", "UPDATE", "DELETE", "NOOP"}:
        return "NOOP", None, ""
    if operation in {"UPDATE", "DELETE"}:
        target_id, _, payload = rest.partition(":")
        return operation, target_id.strip(), payload.strip()
    return operation, None, ""


def apply_candidate_fact(
    provider: Provider,
    store: VectorStore,
    embedder: Embedder,
    namespace: str,
    fact_text: str,
    *,
    top_s: int = 3,
    id_prefix: str = "mem",
) -> UpdateOp:
    """Decide and apply one operation for a single candidate fact already in
    text form (steps 2-7 of the mechanism, with extraction already done).

    Exposed separately from `mem0_update` so callers that already have a
    fact string (for example `memory_bench.py`, replaying a benchmark
    dataset that is already fact-shaped) can skip the extraction call.

    Args:
        provider: Model to run the decision call with.
        store: Vector store the operation reads from and writes to.
        embedder: Embedder used to find similar existing memories.
        namespace: Isolation boundary to search and write within.
        fact_text: The candidate fact, e.g. "plan: pro tier".
        top_s: Number of nearest existing memories to show the model.
        id_prefix: Prefix for freshly assigned ids on ADD.
    """
    candidate_vec = embedder.embed([fact_text])[0]
    neighbors = store.search(namespace, candidate_vec, top_k=top_s, min_similarity=0.0)

    if neighbors:
        listing = "\n".join(f"{n.record.id}: {n.record.text}" for n in neighbors)
    else:
        listing = "(no existing memories in this namespace)"
    completion = provider.complete(
        [Message.user(f"Candidate memory: {fact_text}\n\nExisting memories:\n{listing}")],
        system=DECISION_SYSTEM_PROMPT,
    )
    operation, target_id, payload = _parse_verdict(completion.content)

    if operation == "ADD":
        new_id = _next_id(store, namespace, id_prefix)
        store.upsert(new_id, namespace, fact_text, candidate_vec, importance=0.7)
        return UpdateOp(fact_text, "ADD", new_id, fact_text)

    if operation == "UPDATE" and target_id and store.get(namespace, target_id) is not None:
        merged_text = payload or fact_text
        merged_vec = embedder.embed([merged_text])[0]
        store.upsert(target_id, namespace, merged_text, merged_vec, importance=0.7)
        return UpdateOp(fact_text, "UPDATE", target_id, merged_text)

    if operation == "DELETE" and target_id and store.get(namespace, target_id) is not None:
        store.delete(namespace, target_id)
        return UpdateOp(fact_text, "DELETE", target_id, None)

    return UpdateOp(fact_text, "NOOP", None, None)


def mem0_update(
    provider: Provider,
    store: VectorStore,
    embedder: Embedder,
    namespace: str,
    user_text: str,
    *,
    top_s: int = 3,
    id_prefix: str = "mem",
) -> list[UpdateOp]:
    """Run the full extract-then-update pipeline for one raw user turn.

    Args:
        provider: Model to run extraction and each decision call with.
        store: Vector store the operations read from and write to.
        embedder: Embedder used for the candidate/neighbor similarity search.
        namespace: Isolation boundary to search and write within.
        user_text: The raw user turn to extract candidate facts from.
        top_s: Number of nearest existing memories shown per decision.
        id_prefix: Prefix for freshly assigned ids on ADD.

    Returns:
        One `UpdateOp` per extracted candidate fact, in extraction order,
        so a test can assert the exact ordered op log.
    """
    candidates: list[ExtractedFact] = extract_facts(provider, user_text)
    ops: list[UpdateOp] = []
    for candidate in candidates:
        fact_text = f"{candidate.subject}: {candidate.value}"
        op = apply_candidate_fact(provider, store, embedder, namespace, fact_text, top_s=top_s, id_prefix=id_prefix)
        ops.append(op)
    return ops


def run_mem0_update_demo(provider: Provider | None = None) -> dict[str, object]:
    """Walk one namespace through all four operations in a coherent story:
    a new plan fact is ADDed, a rephrased downgrade UPDATEs it in place
    instead of accumulating a second record, a restatement is a NOOP, and a
    cancellation DELETEs the now-contradicted record.
    """
    if provider is None:
        provider = get_provider(
            script=[
                # Turn 1: brand-new fact into an empty namespace.
                "plan: pro tier, 1M requests/month",
                "ADD",
                # Turn 2: same underlying claim, different subject key and
                # phrasing; a same-key overwrite would miss this entirely.
                "subscription: free tier, 10k requests/month",
                "UPDATE mem-1: free tier, 10k requests/month",
                # Turn 3: restating the now-current value changes nothing.
                "subscription: free tier, 10k requests/month",
                "NOOP",
                # Turn 4: an explicit cancellation contradicts and removes it.
                "plan: canceled",
                "DELETE mem-1",
            ]
        )
    embedder = get_embedder()
    store = VectorStore()
    namespace = "user:alex"

    turns = [
        "I'm on the pro tier, with a 1M request/month limit.",
        "Actually, I downgraded to the free plan, still 10k requests a month.",
        "Just confirming, I'm still on the free tier.",
        "I canceled my account entirely.",
    ]
    op_log: list[UpdateOp] = []
    state_after_each_turn: list[int] = []
    for turn in turns:
        ops = mem0_update(provider, store, embedder, namespace, turn)
        op_log.extend(ops)
        state_after_each_turn.append(len(store.all(namespace)))

    return {
        "operations": [f"{op.operation}({op.record_id or '-'})" for op in op_log],
        "record_count_after_each_turn": state_after_each_turn,
        "final_record_count": len(store.all(namespace)),
    }
