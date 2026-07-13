"""MemGPT-style paged (hierarchical) memory.

An OS-inspired design (Packer et al., "MemGPT: Towards LLMs as Operating
Systems," arXiv:2310.08560; the framework is now called Letta, the pattern
is still called MemGPT) that separates:

- **Main context**: bounded, in-window, like RAM.
- **External context**: unbounded storage on the side, like disk.

Two distinct overflow mechanisms, both implemented here:

1. **Recursive summarization on overflow**: when appending to main context
   would exceed its token limit, the oldest half of main is condensed into
   one summary entry, freeing room without the model deciding anything.
2. **Model-driven paging**: the model itself issues `page_out` / `page_in`
   function calls to move an item between main and external context. Paging
   is a model decision, made through tool calls, not an automatic policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, Tool, ToolRegistry, get_provider, scripted_tool_call

from patterns.memory.short_term import count_tokens

# The demo entry the model pages out, then back in, through function calls.
# Reused for both the seeded main-context entries and the scripted
# `page_out` call below, so the two can never drift apart the way a pair of
# independently hand-typed strings could.
DEMO_ARCHIVE_KEY = "kyoto-trip"
DEMO_ARCHIVE_TEXT = "Kyoto trip: ryokan near Gion, October, under $300/night."


@dataclass
class MemGPTMemory:
    """Two-tier memory: `main` is in-window context; `external` is
    unbounded storage the agent pages items in and out of.

    Attributes:
        main_limit: Token ceiling for `main`. Crossing it on append
            triggers recursive summarization.
        main: The current in-window context, as plain text entries.
        external: Named entries paged out of main, keyed by page name.
        page_events: A log of every page_out/page_in call, in order, for
            asserting the function-call sequence stayed deterministic.
    """

    main_limit: int
    main: list[str] = field(default_factory=list)
    external: dict[str, str] = field(default_factory=dict)
    page_events: list[str] = field(default_factory=list)

    def main_tokens(self) -> int:
        """Approximate token total of everything currently in main context."""
        return sum(count_tokens(t) for t in self.main)

    def page_out(self, key: str, text: str) -> str:
        """Move an item from main context to external storage."""
        self.external[key] = text
        if text in self.main:
            self.main.remove(text)
        self.page_events.append(f"page_out({key})")
        return f"paged out {key!r}"

    def page_in(self, key: str) -> str:
        """Move an item from external storage back into main context."""
        text = self.external.get(key)
        if text is None:
            return f"ERROR: no external memory named {key!r}"
        self.main.append(text)
        self.page_events.append(f"page_in({key})")
        return text

    def append_main(self, text: str, summarize: Callable[[list[str]], str] | None = None) -> bool:
        """Append `text` to main context; if this pushes `main_tokens()`
        over `main_limit`, recursively condense the oldest half of main
        into a single summary entry until it fits again.

        Args:
            text: The entry to append.
            summarize: Called with the oldest half of main when overflow
                triggers, returns the condensed entry to replace it with.

        Returns:
            True if compaction fired at least once during this append.
        """
        self.main.append(text)
        compacted = False
        while self.main_tokens() > self.main_limit and len(self.main) > 1 and summarize is not None:
            half = max(len(self.main) // 2, 1)
            old, recent = self.main[:half], self.main[half:]
            self.main = [summarize(old), *recent]
            compacted = True
        return compacted


def run_memgpt_demo(provider: Provider | None = None) -> MemGPTMemory:
    """Overflow main context to trigger one recursive summarization, then
    have the model page a note out and later page it back in through
    function calls, and return the resulting memory with its page log.
    """
    if provider is None:
        provider = get_provider(
            script=[
                "Condensed: Terraform deployment set up in us-west-2.",
                scripted_tool_call(
                    "page_out",
                    {"key": DEMO_ARCHIVE_KEY, "text": DEMO_ARCHIVE_TEXT},
                ),
                scripted_tool_call("page_in", {"key": DEMO_ARCHIVE_KEY}),
                f"Your Kyoto trip notes: {DEMO_ARCHIVE_TEXT.removeprefix('Kyoto trip: ')}",
            ]
        )

    memory = MemGPTMemory(main_limit=30)

    def summarize(old: list[str]) -> str:
        return provider.complete(
            [Message.user("Condense: " + " | ".join(old))],
            system="Summarize these main-context entries into one short line.",
        ).content.strip()

    for entry in [
        "User set up a us-west-2 Terraform deployment.",
        "User confirmed the Terraform state bucket is versioned.",
        DEMO_ARCHIVE_TEXT,
        "User asked if the ryokan offers a late checkout.",
    ]:
        memory.append_main(entry, summarize=summarize)

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="page_out",
            description="Move an item from main context to external memory.",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}, "text": {"type": "string"}},
                "required": ["key", "text"],
            },
            fn=memory.page_out,
        )
    )
    registry.register(
        Tool(
            name="page_in",
            description="Move an item from external memory back into main context.",
            parameters={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
            fn=memory.page_in,
        )
    )

    # The model decides, through function calls, to archive a note and
    # later recall it; the scripted mock fixes this sequence deterministically.
    archive_call = provider.complete([Message.user("Archive the Kyoto trip note for now, main context is full.")])
    for call in archive_call.tool_calls:
        registry.execute(call)

    recall_call = provider.complete([Message.user("What were my Kyoto trip notes again?")])
    paged_text = ""
    for call in recall_call.tool_calls:
        paged_text = registry.execute(call)

    provider.complete([Message.user(f"Kyoto notes: {paged_text}")])
    return memory
