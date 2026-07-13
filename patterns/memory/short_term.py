"""Short-term memory: the in-window conversation buffer, held under one
interface with three interchangeable modes, plus the budget and
context-editing utilities that keep any of them bounded.

- **Full buffer** (`mode="full"`): append every turn, never evict. Simplest
  and correct until the transcript outgrows the token window.
- **Sliding window** (`mode="window"`): keep only the last `window_turns`
  turns, dropping the oldest on every append. Cheap and deterministic, at
  the cost of forgetting early context.
- **Summarization / compaction** (`mode="summary"`): once the buffer grows
  past `summary_threshold` turns, an LLM call condenses everything except
  the last `window_turns` turns into a running summary, which replaces
  them. Recent turns stay verbatim; only the tail is ever paraphrased.

`TokenBudget` and `evict_to_budget` implement token-budget accounting for
an arbitrary message list, independent of which short-term mode is in use.
`drop_stale_tool_results` implements context editing: deleting old tool
observations outright, as a primitive distinct from summarization, since
nothing is paraphrased, it is simply discarded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from agentic_patterns import Message, Provider, get_provider

Mode = Literal["full", "window", "summary"]


def count_tokens(text: str) -> int:
    """Approximate a token count deterministically as a whitespace word count.

    This is a stand-in for a real tokenizer: good enough to demonstrate
    budget accounting and eviction order without a network call, and stable
    across machines since it does not depend on any specific tokenizer
    vocabulary.
    """
    return len(text.split())


@dataclass
class ShortTermMemory:
    """A conversation buffer that behaves as one of three modes.

    Attributes:
        mode: One of "full", "window", "summary".
        window_turns: For "window", the number of most recent turns kept.
            For "summary", the number of most recent turns always kept
            verbatim alongside the running summary.
        summary_threshold: For "summary", the turn count that must be
            exceeded before `maybe_compact` will condense anything.
        turns: The buffered messages.
        running_summary: The current condensed summary of evicted turns, for
            "summary" mode. Empty until the first compaction fires.
    """

    mode: Mode = "full"
    window_turns: int = 4
    summary_threshold: int = 6
    turns: list[Message] = field(default_factory=list)
    running_summary: str = ""

    def append(self, message: Message) -> None:
        """Add one turn to the buffer, applying the sliding-window rule
        immediately if `mode == "window"`. "full" and "summary" modes just
        grow; "summary" mode requires an explicit `maybe_compact` call to
        evict, since compaction needs an LLM call the buffer cannot make on
        its own.
        """
        self.turns.append(message)
        if self.mode == "window" and len(self.turns) > self.window_turns:
            self.turns = self.turns[-self.window_turns :]

    def maybe_compact(self, summarize: Callable[[list[Message], str], str]) -> bool:
        """For `mode == "summary"`, condense older turns once the buffer
        crosses `summary_threshold`.

        Args:
            summarize: Called with the turns being evicted and the current
                running summary, returns the new running summary.

        Returns:
            True if compaction fired this call, False otherwise (wrong mode,
            or the buffer has not yet crossed the threshold).
        """
        if self.mode != "summary" or len(self.turns) <= self.summary_threshold:
            return False
        keep = self.turns[-self.window_turns :] if self.window_turns else []
        to_summarize = self.turns[: len(self.turns) - len(keep)]
        self.running_summary = summarize(to_summarize, self.running_summary)
        self.turns = keep
        return True

    def render(self) -> list[Message]:
        """Return the messages this buffer contributes to a prompt.

        The running summary, if any, is injected as a `user`-role message
        (never `role="system"`): the top-level `system` string is a separate
        channel, and some provider wire formats drop mid-list system-role
        messages entirely, so anything meant to reach the model as a normal
        turn must use "user" or "assistant".
        """
        if self.running_summary:
            note = Message.user(f"[Summary of earlier conversation] {self.running_summary}")
            return [note, *self.turns]
        return list(self.turns)


@dataclass
class TokenBudget:
    """A simple token ceiling for a message list."""

    limit: int

    def total(self, messages: list[Message]) -> int:
        """Total approximate token count across `messages`."""
        return sum(count_tokens(m.content) for m in messages)

    def fits(self, messages: list[Message]) -> bool:
        """True if `messages` is at or under `limit`."""
        return self.total(messages) <= self.limit


def evict_to_budget(messages: list[Message], budget: TokenBudget, protected: int = 1) -> list[Message]:
    """Drop the oldest evictable messages until `messages` fits `budget`.

    The first `protected` messages (typically a system prompt) are never
    evicted, even if the budget cannot otherwise be met.

    Args:
        messages: Messages to trim, oldest first.
        budget: The token ceiling to trim to.
        protected: Number of leading messages exempt from eviction.
    """
    kept = list(messages)
    while budget.total(kept) > budget.limit and len(kept) > protected:
        del kept[protected]
    return kept


def drop_stale_tool_results(messages: list[Message], keep_last: int = 1) -> list[Message]:
    """Context editing: delete stale tool-result turns outright.

    Distinct from summarization: nothing is paraphrased, older tool
    observations are simply removed, keeping only the most recent
    `keep_last` tool-role messages. Non-tool turns are untouched.

    Args:
        messages: The conversation to edit.
        keep_last: Number of most recent tool-role messages to retain.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.role == "tool"]
    stale = set(tool_indices[:-keep_last]) if keep_last > 0 else set(tool_indices)
    return [m for i, m in enumerate(messages) if i not in stale]


def _make_summarize(provider: Provider) -> Callable[[list[Message], str], str]:
    """Build a `summarize` callback for `maybe_compact` that asks `provider`
    to condense evicted turns and the prior running summary into one line.
    """

    def summarize(old_turns: list[Message], running_summary: str) -> str:
        transcript = "\n".join(f"{m.role}: {m.content}" for m in old_turns if m.content)
        completion = provider.complete(
            [Message.user(f"Previous summary: {running_summary or '(none)'}\nNew turns:\n{transcript}")],
            system=(
                "Condense the previous summary and the new turns into one "
                "updated running summary, 1-2 sentences, keeping only "
                "durable facts."
            ),
        )
        return completion.content.strip()

    return summarize


def run_short_term_demo(provider: Provider | None = None) -> dict[str, Any]:
    """Run the same toy conversation through all three modes, plus token
    budget eviction and context editing, and return what each produced.
    """
    if provider is None:
        provider = get_provider(
            script=["User is trip-planning a Kyoto stay near Gion in October, budget under $300/night."]
        )

    convo = [
        Message.user("I'm planning a trip to Kyoto."),
        Message.assistant("Great, when are you going?"),
        Message.user("October. I'd like to stay somewhere near Gion."),
        Message.assistant("Noted, I'll look for ryokans near Gion for October."),
        Message.user("Also, budget under $300 a night."),
        Message.assistant("Got it, under $300 a night near Gion in October."),
    ]

    full = ShortTermMemory(mode="full")
    for m in convo:
        full.append(m)

    window = ShortTermMemory(mode="window", window_turns=3)
    for m in convo:
        window.append(m)

    summary_mem = ShortTermMemory(mode="summary", summary_threshold=4, window_turns=2)
    summarize = _make_summarize(provider)
    compacted_on_turn = None
    for i, m in enumerate(convo, start=1):
        summary_mem.append(m)
        if summary_mem.maybe_compact(summarize):
            compacted_on_turn = i

    budget = TokenBudget(limit=12)
    trimmed = evict_to_budget([Message.system("system prompt"), *convo], budget, protected=1)

    tool_heavy = [
        Message.user("Look up the weather."),
        Message.tool("call_1", "Kyoto forecast: sunny, 22C"),
        Message.assistant("It'll be sunny and 22C."),
        Message.user("And the exchange rate?"),
        Message.tool("call_2", "1 USD is about 155 JPY"),
        Message.assistant("Right now 1 USD is about 155 JPY."),
    ]
    edited = drop_stale_tool_results(tool_heavy, keep_last=1)

    return {
        "full_turn_count": len(full.turns),
        "window_turn_count": len(window.turns),
        "window_contents": [m.content for m in window.turns],
        "summary_compacted_on_turn": compacted_on_turn,
        "running_summary": summary_mem.running_summary,
        "summary_kept_verbatim": [m.content for m in summary_mem.turns],
        "budget_limit": budget.limit,
        "trimmed_token_total": budget.total(trimmed),
        "trimmed_message_count": len(trimmed),
        "tool_messages_before_edit": sum(1 for m in tool_heavy if m.role == "tool"),
        "tool_messages_after_edit": sum(1 for m in edited if m.role == "tool"),
    }
