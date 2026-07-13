"""Tests for the memory pattern.

Deterministic and offline: every test drives `MockProvider` scripts and
`HashEmbedder` through the pattern's own modules, with no network call and
no API key.
"""

from __future__ import annotations

from agentic_patterns import HashEmbedder, Message, MockProvider, Tool, ToolRegistry

from patterns.memory.assembler import assemble_context
from patterns.memory.episodic import EpisodicMemory
from patterns.memory.file_memory import FileMemoryStore
from patterns.memory.memgpt import MemGPTMemory
from patterns.memory.memory_tools import build_memory_toolset
from patterns.memory.procedural import ProceduralMemory
from patterns.memory.retrieval import RetrievalConfig, retrieve
from patterns.memory.semantic import SemanticMemory
from patterns.memory.short_term import (
    ShortTermMemory,
    TokenBudget,
    drop_stale_tool_results,
    evict_to_budget,
)
from patterns.memory.vector_store import VectorStore
from patterns.memory.write_policy import (
    BackgroundWriteQueue,
    consolidate,
    extract_facts,
    hot_path_write,
)


# --- short-term memory: modes ----------------------------------------------


def test_sliding_window_holds_exactly_last_n_in_order() -> None:
    memory = ShortTermMemory(mode="window", window_turns=3)
    for i in range(6):
        memory.append(Message.user(f"turn {i}"))
    assert [m.content for m in memory.turns] == ["turn 3", "turn 4", "turn 5"]


def test_full_buffer_never_evicts() -> None:
    memory = ShortTermMemory(mode="full")
    for i in range(10):
        memory.append(Message.user(f"turn {i}"))
    assert len(memory.turns) == 10


def test_summarization_fires_at_threshold_and_keeps_recent_verbatim() -> None:
    provider = MockProvider(script=["condensed summary of early turns"])
    memory = ShortTermMemory(mode="summary", summary_threshold=4, window_turns=2)

    def summarize(old_turns, running_summary):
        return provider.complete([Message.user("summarize")]).content

    fired_on = []
    for i in range(1, 7):
        memory.append(Message.user(f"turn {i}"))
        if memory.maybe_compact(summarize):
            fired_on.append(i)

    assert fired_on == [5]  # 5th append pushes past threshold=4
    assert memory.running_summary == "condensed summary of early turns"
    # turn 6 was appended after the only compaction, so it stays alongside
    # the two turns compaction kept verbatim
    assert [m.content for m in memory.turns] == ["turn 4", "turn 5", "turn 6"]
    assert len(provider.calls) == 1  # compaction only fired once


def test_context_editing_drops_stale_tool_results_keeps_other_roles() -> None:
    messages = [
        Message.user("q1"),
        Message.tool("call_1", "obs1"),
        Message.assistant("a1"),
        Message.user("q2"),
        Message.tool("call_2", "obs2"),
        Message.assistant("a2"),
    ]
    edited = drop_stale_tool_results(messages, keep_last=1)
    tool_msgs = [m for m in edited if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "obs2"
    assert len(edited) == 5  # only the stale tool message was removed


def test_evict_to_budget_protects_leading_messages() -> None:
    messages = [Message.system("sys"), Message.user("aaaa"), Message.user("bbbb"), Message.user("cccc")]
    budget = TokenBudget(limit=2)  # each Message content is a single "word" here
    trimmed = evict_to_budget(messages, budget, protected=1)
    assert trimmed[0].content == "sys"  # protected message is never evicted
    assert len(trimmed) == 2  # trimmed down to protected + one more, oldest-first
    assert trimmed[-1].content == "cccc"  # most recent evictable message survives


# --- vector store: cosine ranking and namespacing ---------------------------


def test_vector_store_top_k_order_and_threshold_exclusion() -> None:
    store = VectorStore()
    store.upsert("a", "ns", "a", [1.0, 0.0])
    store.upsert("b", "ns", "b", [0.9, 0.1])
    store.upsert("c", "ns", "c", [0.0, 1.0])
    results = store.search("ns", [1.0, 0.0], top_k=5, min_similarity=0.5)
    assert [r.record.id for r in results] == ["a", "b"]  # "c" excluded, orthogonal
    assert results[0].similarity >= results[1].similarity


def test_vector_store_namespacing_isolates_queries() -> None:
    store = VectorStore()
    store.upsert("shared-key", "user:a", "user a's secret note", [1.0, 0.0])
    store.upsert("shared-key", "user:b", "user b's unrelated note", [1.0, 0.0])
    results = store.search("user:a", [1.0, 0.0], top_k=5, min_similarity=0.0)
    assert len(results) == 1
    assert results[0].record.text == "user a's secret note"


def test_vector_store_upsert_same_id_overwrites_not_appends() -> None:
    store = VectorStore()
    store.upsert("k", "ns", "first value", [1.0, 0.0])
    store.upsert("k", "ns", "second value", [1.0, 0.0])
    assert len(store.all("ns")) == 1
    assert store.get("ns", "k").text == "second value"


# --- retrieval scoring -------------------------------------------------------


def test_retrieval_recency_weighting_can_change_order() -> None:
    embedder = HashEmbedder()
    store = VectorStore()
    text = "the user likes hiking"
    # write the same-ish content twice; later write has higher recency
    store.upsert("old", "ns", text, embedder.embed([text])[0])
    store.upsert("new", "ns", text, embedder.embed([text])[0])

    plain = retrieve(store, embedder, "ns", "hiking", RetrievalConfig(top_k=2, min_similarity=0.0))
    by_recency = retrieve(
        store, embedder, "ns", "hiking", RetrievalConfig(top_k=2, min_similarity=0.0, recency_weight=0.9)
    )
    assert {r.record.id for r in plain} == {"old", "new"}
    assert by_recency[0].record.id == "new"  # most recently written wins under heavy recency weight


def test_retrieval_min_similarity_filters_before_top_k() -> None:
    embedder = HashEmbedder()
    store = VectorStore()
    store.upsert("close", "ns", "dark roast coffee in the morning", embedder.embed(["dark roast coffee in the morning"])[0])
    store.upsert("far", "ns", "team standup at 9:30am", embedder.embed(["team standup at 9:30am"])[0])
    results = retrieve(store, embedder, "ns", "coffee in the morning", RetrievalConfig(top_k=5, min_similarity=0.3))
    assert all(r.record.id != "far" for r in results)


# --- semantic memory: conflict resolution -----------------------------------


def test_semantic_memory_overwrite_not_accumulate() -> None:
    embedder = HashEmbedder()
    store = VectorStore()
    memory = SemanticMemory(store, embedder, namespace="user:x")
    first_write_was_overwrite = memory.write_fact("plan", "free tier")
    second_write_was_overwrite = memory.write_fact("plan", "pro tier")
    assert first_write_was_overwrite is False
    assert second_write_was_overwrite is True
    assert len(store.all("user:x")) == 1
    assert store.get("user:x", "plan").text == "plan: pro tier"


# --- write policy: extraction, hot path, background, consolidation ---------


def test_extract_facts_parses_subject_value_lines() -> None:
    provider = MockProvider(script=["timezone: America/Chicago\nplan: pro"])
    facts = extract_facts(provider, "I'm in Chicago and on the pro plan.")
    assert [(f.subject, f.value) for f in facts] == [("timezone", "America/Chicago"), ("plan", "pro")]


def test_extract_facts_none_reply_yields_no_facts() -> None:
    provider = MockProvider(script=["NONE"])
    assert extract_facts(provider, "What's the weather like?") == []


def test_extract_facts_skips_lines_without_colon() -> None:
    provider = MockProvider(script=["just a stray sentence\ntimezone: America/Chicago"])
    facts = extract_facts(provider, "some text")
    assert [(f.subject, f.value) for f in facts] == [("timezone", "America/Chicago")]


def test_hot_path_write_persists_immediately() -> None:
    provider = MockProvider(script=["favorite_color: blue"])
    embedder = HashEmbedder()
    memory = SemanticMemory(VectorStore(), embedder, namespace="user:x")
    hot_path_write(provider, memory, "My favorite color is blue.")
    assert memory.store.get("user:x", "favorite_color").text == "favorite_color: blue"


def test_background_queue_matches_hot_path_final_state() -> None:
    embedder = HashEmbedder()

    hot_provider = MockProvider(script=["favorite_color: blue"])
    hot_memory = SemanticMemory(VectorStore(), embedder, namespace="user:x")
    hot_path_write(hot_provider, hot_memory, "My favorite color is blue.")

    bg_provider = MockProvider(script=["favorite_color: blue"])
    bg_memory = SemanticMemory(VectorStore(), embedder, namespace="user:x")
    queue = BackgroundWriteQueue()
    queue.enqueue("My favorite color is blue.")
    queue.drain(bg_provider, bg_memory)

    hot_state = sorted((r.id, r.text) for r in hot_memory.store.all("user:x"))
    bg_state = sorted((r.id, r.text) for r in bg_memory.store.all("user:x"))
    assert hot_state == bg_state
    assert queue.pending == []  # queue drained


def test_consolidate_resolves_contradiction_offline() -> None:
    provider = MockProvider(script=["pro tier (current)"])
    embedder = HashEmbedder()
    memory = SemanticMemory(VectorStore(), embedder, namespace="user:x")
    memory.write_fact("plan", "free tier (stale)")
    resolved = consolidate(provider, memory, "plan", ["free tier (stale)", "pro tier (current)"])
    assert resolved == "pro tier (current)"
    assert memory.store.get("user:x", "plan").text == "plan: pro tier (current)"
    assert len(memory.store.all("user:x")) == 1  # resolution overwrote, did not add a second record


# --- round-trip recall across sessions --------------------------------------


def test_round_trip_recall_across_fresh_sessions_via_assembler() -> None:
    embedder = HashEmbedder()
    store = VectorStore()
    memory = SemanticMemory(store, embedder, namespace="user:x")

    # session 1
    memory.write_fact("timezone", "America/Chicago")

    # session 2: brand-new, empty short-term buffer
    session_2 = ShortTermMemory(mode="full")
    session_2.append(Message.user("What timezone am I in?"))
    hits = memory.recall("What timezone am I in?", top_k=1)
    assembled = assemble_context(
        base_system="You are an assistant.",
        procedural=None,
        short_term=session_2,
        retrieved=hits,
        budget=TokenBudget(limit=1000),
    )
    assert len(hits) == 1
    assert any("America/Chicago" in m.content for m in assembled.messages)


# --- assembler: token budget and pinned constraints -------------------------


def test_assembler_never_exceeds_budget_and_drops_retrieved_before_short_term() -> None:
    embedder = HashEmbedder()
    store = VectorStore()
    memory = SemanticMemory(store, embedder, namespace="user:x")
    memory.write_fact("note", "one two three four")
    hits = memory.recall("note", top_k=1, min_similarity=0.0)
    assert len(hits) == 1  # sanity: there is a retrieved item to drop

    short_term = ShortTermMemory(mode="full")
    short_term.append(Message.user("aaaa bbbb"))
    short_term.append(Message.assistant("cccc dddd"))

    budget = TokenBudget(limit=3)  # "sys" (1 token pinned) leaves room for 3 message tokens
    result = assemble_context(
        base_system="sys",
        procedural=None,
        short_term=short_term,
        retrieved=hits,
        budget=budget,
    )
    assert result.total_tokens <= budget.limit
    # the retrieved memory item is the first thing dropped, before any short-term turn
    assert result.dropped[0].startswith("[retrieved memory]")
    assert all(m.role != "user" or "[retrieved memory]" not in m.content for m in result.messages)


def test_assembler_pins_procedural_rule_even_under_heavy_eviction() -> None:
    procedural = ProceduralMemory(namespace="user:x")
    procedural.add_rule("Always answer in metric units.")
    short_term = ShortTermMemory(mode="full")
    for i in range(20):
        short_term.append(Message.user(f"filler turn number {i} with extra words"))

    result = assemble_context(
        base_system="You are an assistant.",
        procedural=procedural,
        short_term=short_term,
        retrieved=[],
        budget=TokenBudget(limit=1),  # far too small to hold any short-term turn
    )
    assert "metric units" in result.system
    assert len(result.messages) == 0  # every short-term turn was evicted, but the rule was never touched


# --- MemGPT paged memory -----------------------------------------------------


def test_memgpt_overflow_triggers_recursive_summarization() -> None:
    provider = MockProvider(script=["x"])
    memory = MemGPTMemory(main_limit=2)

    def summarize(old):
        return provider.complete([Message.user("condense")]).content

    not_yet_compacted = memory.append_main("aaaa bbbb", summarize=summarize)  # exactly at limit
    compacted = memory.append_main("cccc", summarize=summarize)  # pushes over, triggers one pass
    assert not_yet_compacted is False
    assert compacted is True
    assert memory.main == ["x", "cccc"]


def test_memgpt_page_out_then_page_in_sequence_is_deterministic() -> None:
    memory = MemGPTMemory(main_limit=1000)
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="page_out",
            description="page out",
            parameters={"type": "object", "properties": {"key": {"type": "string"}, "text": {"type": "string"}}},
            fn=memory.page_out,
        )
    )
    registry.register(
        Tool(
            name="page_in",
            description="page in",
            parameters={"type": "object", "properties": {"key": {"type": "string"}}},
            fn=memory.page_in,
        )
    )
    provider = MockProvider(
        script=[
            {"tool": "page_out", "args": {"key": "note", "text": "archived note"}},
            {"tool": "page_in", "args": {"key": "note"}},
        ]
    )
    memory.append_main("archived note")
    for _ in range(2):
        completion = provider.complete([Message.user("go")])
        for call in completion.tool_calls:
            registry.execute(call)
    assert memory.page_events == ["page_out(note)", "page_in(note)"]
    assert "archived note" in memory.main  # paged back into main context


# --- episodic memory ---------------------------------------------------------


def test_episodic_memory_records_and_recalls_a_lesson() -> None:
    embedder = HashEmbedder()
    store = VectorStore()
    memory = EpisodicMemory(store, embedder, namespace="user:x")
    memory.record_episode(
        task="terraform apply", outcome="failed: bucket not versioned", lesson="enable bucket versioning first"
    )
    hits = memory.recall_lessons("terraform apply")
    assert len(hits) == 1
    assert hits[0].record.metadata["lesson"] == "enable bucket versioning first"


# --- procedural memory --------------------------------------------------------


def test_procedural_memory_renders_rules_and_dedupes() -> None:
    rules = ProceduralMemory(namespace="user:x")
    rules.add_rule("Always answer in metric units.")
    rules.add_rule("Always answer in metric units.")  # duplicate, ignored
    assert rules.rules == ["Always answer in metric units."]
    assert "Always answer in metric units." in rules.render()


def test_procedural_memory_empty_renders_empty_string() -> None:
    assert ProceduralMemory(namespace="user:x").render() == ""


# --- memory-as-tools ----------------------------------------------------------


def test_memory_tools_store_then_retrieve_round_trip() -> None:
    embedder = HashEmbedder()
    store = VectorStore()
    registry = build_memory_toolset(store, embedder, namespace="user:x")
    store_result = registry.get("memory_store").fn(key="pref", text="drinks dark roast coffee")
    assert "stored" in store_result
    retrieve_result = registry.get("memory_retrieve").fn(query="coffee", top_k=2)
    assert "dark roast coffee" in retrieve_result


def test_memory_tools_update_fails_on_unknown_key() -> None:
    embedder = HashEmbedder()
    registry = build_memory_toolset(VectorStore(), embedder, namespace="user:x")
    result = registry.get("memory_update").fn(key="missing", text="x")
    assert result.startswith("ERROR")


def test_provider_receives_expected_system_prompt_for_extraction() -> None:
    provider = MockProvider(script=["NONE"])
    extract_facts(provider, "hello there")
    assert len(provider.calls) == 1
    assert "durable facts" in provider.calls[0]["system"]
    assert provider.calls[0]["messages"][0].content == "hello there"


# --- file-directory memory backend --------------------------------------------


def test_file_memory_create_read_update_delete() -> None:
    fs = FileMemoryStore(namespace="user:x")
    assert fs.create("notes.md", "v1").startswith("created")
    assert fs.read("notes.md") == "v1"
    assert fs.update("notes.md", "v2").startswith("updated")
    assert fs.read("notes.md") == "v2"
    assert fs.delete("notes.md").startswith("deleted")
    assert fs.read("notes.md").startswith("ERROR")


def test_file_memory_create_twice_errors() -> None:
    fs = FileMemoryStore(namespace="user:x")
    fs.create("notes.md", "v1")
    result = fs.create("notes.md", "v2")
    assert result.startswith("ERROR")
    assert fs.read("notes.md") == "v1"  # first write untouched
