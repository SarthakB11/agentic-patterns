"""Group chat / roundtable: one shared thread, a chat manager picks the next speaker.

Every participant reads and writes to the same conversation, unlike the
supervisor variant where workers never see each other's output. A separate
chat manager agent decides, turn by turn, who speaks next or whether the
discussion is done. This suits collaborative ideation and structured review
more than task decomposition, and the brief's warning applies directly:
"keep the roster small to avoid loops and drift," so a hard `max_turns` cap
is not optional here, it is the guard against the chat manager never saying
stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, get_provider

MANAGER_SYSTEM = (
    "You are a chat manager for a small discussion. Read the transcript so far and reply "
    "with exactly one line: either 'NEXT: <participant name>' to let that participant speak "
    "next, or 'STOP' if the discussion has reached a clear conclusion."
)


@dataclass
class ChatTurn:
    """One participant's contribution to a shared group-chat thread."""

    speaker: str
    content: str


@dataclass
class GroupChatResult:
    """The outcome of a group-chat run.

    Attributes:
        turns: Every turn that was spoken, in order.
        stop_reason: "manager_stopped" if the chat manager ended the
            discussion, or "max_turns" if the cap was hit first.
    """

    turns: list[ChatTurn] = field(default_factory=list)
    stop_reason: str = "max_turns"


def run_group_chat(
    manager: Provider,
    participants: dict[str, Provider],
    topic: str,
    *,
    max_turns: int = 6,
) -> GroupChatResult:
    """Run a roundtable discussion with a chat manager choosing speakers.

    Args:
        manager: Provider for the chat manager. Scripted to return
            "NEXT: <name>" or "STOP" each round.
        participants: Participant name mapped to that participant's
            provider, each scripted with its lines for the rounds it speaks.
        topic: The discussion topic, given to every participant as context.
        max_turns: Hard cap on turns, independent of what the manager says,
            so a manager that never stops cannot loop forever.
    """
    turns: list[ChatTurn] = []
    transcript = f"Topic: {topic}"
    for _ in range(max_turns):
        # The placeholder line is only shown while no turns have happened
        # yet; once real turns exist, they speak for themselves.
        visible_transcript = transcript if turns else f"{transcript}\n(no turns yet)"
        decision = manager.complete([Message.user(visible_transcript)], system=MANAGER_SYSTEM).content.strip()
        if decision.upper().startswith("STOP"):
            return GroupChatResult(turns=turns, stop_reason="manager_stopped")
        name = decision.split(":", 1)[1].strip() if ":" in decision else ""
        if name not in participants:
            raise ValueError(f"chat manager named an unknown participant: {decision!r}")
        reply = participants[name].complete(
            [Message.user(f"Topic: {topic}\nTranscript so far:\n{visible_transcript}")],
            system=f"You are {name} in a small working discussion. Speak in 1-2 sentences.",
        ).content
        turns.append(ChatTurn(name, reply))
        transcript += f"\n{name}: {reply}"
    return GroupChatResult(turns=turns, stop_reason="max_turns")


# --- demos -------------------------------------------------------------


def run_group_chat_demo() -> GroupChatResult:
    """A three-person roundtable reaches a conclusion in three turns.

    An engineer raises a concern, a skeptic pressure-tests it, and a product
    manager makes the call; the chat manager then stops the discussion.
    """
    manager = get_provider(
        script=["NEXT: engineer", "NEXT: skeptic", "NEXT: product_manager", "STOP"]
    )
    participants = {
        "engineer": get_provider(
            script=[
                "A Redis cache in front of checkout would cut our p95 read latency, "
                "but adds an operational dependency."
            ]
        ),
        "skeptic": get_provider(
            script=["Before adding Redis, have we ruled out fixing the N+1 query that's causing most of the latency?"]
        ),
        "product_manager": get_provider(
            script=["Let's fix the N+1 query first and revisit Redis only if latency is still a problem after that."]
        ),
    }
    return run_group_chat(
        manager, participants, "Should we adopt a Redis caching layer for the checkout service?"
    )


def run_group_chat_cap_demo() -> GroupChatResult:
    """A manager that never says STOP is cut off by `max_turns`, not left to loop.

    Both scripted participants keep restating their position, and the
    manager keeps alternating between them; the cap stops the run at
    `max_turns` turns regardless.
    """
    manager = get_provider(script=["NEXT: engineer", "NEXT: skeptic", "NEXT: engineer", "NEXT: skeptic"])
    participants = {
        "engineer": get_provider(script=["I still think we should add the cache."] * 2),
        "skeptic": get_provider(script=["I still think we should fix the query first."] * 2),
    }
    return run_group_chat(
        manager,
        participants,
        "Should we adopt a Redis caching layer for the checkout service?",
        max_turns=4,
    )
