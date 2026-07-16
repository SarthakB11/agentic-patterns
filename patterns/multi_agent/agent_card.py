"""A2A Agent Card capability discovery and card-based delegation.

`handoff.py` already models the Agent2Agent protocol's task lifecycle
(`DelegationTask`, `pending -> in_progress -> completed/failed`) but routes
by a hardcoded ROUTE reply naming one of two known specialists. A2A's
distinctive contribution beyond that lifecycle is discovery: each agent
publishes an Agent Card, a machine-readable description of its name,
skills, and modalities, at a well-known path, and a coordinator selects a
delegate by matching a task's needs against advertised cards rather than
against a roster it hardcoded. Google donated A2A to the Linux Foundation on
June 23, 2025 (AWS, Cisco, Microsoft, Salesforce, SAP, and ServiceNow as
founding members); the current spec version past v0.3.0 could not be
confirmed from a primary spec page as of this writing and is left
unverified, but the discovery mechanism itself is stable and is what this
module builds.

This models card-based matching and delegation, not the A2A network
transport. A real deployment fetches a card from
`/.well-known/agent-card.json` over HTTP with OAuth; here `Registry` is an
in-memory stand-in for that fetch, and `match_score` is deterministic
keyword overlap rather than a real capability negotiation. That is enough
to demonstrate the two behaviors A2A's discovery half is for: picking an
un-hardcoded delegate by advertised skill, and failing clearly when no
registered agent can do the job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Message, MockProvider, Provider
from patterns.multi_agent.handoff import DelegationTask


class NoCapableAgentError(ValueError):
    """Raised when no registered `AgentCard` advertises a skill matching the task."""


@dataclass
class Skill:
    """One capability an agent advertises.

    Attributes:
        id: Short skill identifier, e.g. "billing".
        keywords: Words a task's needs are matched against.
    """

    id: str
    keywords: list[str]


@dataclass
class AgentCard:
    """A machine-readable description of one agent's capabilities.

    Attributes:
        name: Unique agent name, used as the delegation target.
        description: Human-readable summary of what the agent does.
        skills: Capabilities this agent advertises.
        endpoint: Where the agent would be reached. Offline this is just a
            label; a real build fetches the card from this endpoint's
            well-known path instead of registering it in-process.
    """

    name: str
    description: str
    skills: list[Skill] = field(default_factory=list)
    endpoint: str = ""


class Registry:
    """An in-memory stand-in for cards fetched from well-known paths."""

    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}

    def register(self, card: AgentCard) -> None:
        """Advertise one agent's card."""
        self._cards[card.name] = card

    def cards(self) -> list[AgentCard]:
        """Every registered card, in registration order."""
        return list(self._cards.values())


def match_score(task_keywords: set[str], card: AgentCard) -> int:
    """Count how many task keywords overlap with a card's advertised skill keywords.

    Deterministic and case-insensitive, so the same task always ranks the
    same cards the same way regardless of call order.
    """
    card_keywords = {kw.lower() for skill in card.skills for kw in skill.keywords}
    return len({k.lower() for k in task_keywords} & card_keywords)


@dataclass
class SelectionResult:
    """The outcome of `select`.

    Attributes:
        card: The winning `AgentCard`.
        score: Its `match_score` against the task.
        tie_broken_by_llm: True if an LLM call broke a tie between two or
            more equally-scored top candidates.
    """

    card: AgentCard
    score: int
    tie_broken_by_llm: bool = False


TIE_BREAK_SYSTEM = "You disambiguate a tie between equally-scored agent cards. Reply with the chosen agent's name only."


def select(
    registry: Registry, task_keywords: set[str], *, tie_break_provider: Provider | None = None
) -> SelectionResult:
    """Rank registered cards by skill match and return the winner.

    Ties are broken by name for determinism; if `tie_break_provider` is
    given and a tie exists among the top scorers, one provider call picks
    between them instead, demonstrating that discovery can end in a model
    call without depending on one.

    Args:
        registry: Cards to rank.
        task_keywords: The incoming task's needs, as a keyword set.
        tie_break_provider: Optional provider for LLM disambiguation.

    Raises:
        NoCapableAgentError: If no card scores above zero.
    """
    scored = [(match_score(task_keywords, c), c) for c in registry.cards()]
    top_score = max((s for s, _ in scored), default=0)
    if top_score == 0:
        raise NoCapableAgentError(f"no registered agent advertises a skill matching {sorted(task_keywords)}")
    top = sorted((c for s, c in scored if s == top_score), key=lambda c: c.name)
    if len(top) == 1 or tie_break_provider is None:
        return SelectionResult(top[0], top_score, tie_broken_by_llm=False)
    names = ", ".join(c.name for c in top)
    prompt = (
        f"Task needs: {sorted(task_keywords)}\n"
        f"Tied candidates (equal skill-match score): {names}\n"
        "Pick exactly one by name."
    )
    chosen_name = tie_break_provider.complete([Message.user(prompt)], system=TIE_BREAK_SYSTEM).content.strip()
    chosen = next((c for c in top if c.name == chosen_name), top[0])
    return SelectionResult(chosen, top_score, tie_broken_by_llm=True)


def delegate(coordinator_name: str, selection: SelectionResult, task: str, provider: Provider) -> DelegationTask:
    """Delegate a task to the selected card's agent through the A2A task lifecycle.

    Reuses `handoff.DelegationTask` rather than a new envelope, so discovery
    feeds the delegation machinery this folder already has.

    Args:
        coordinator_name: Name of the delegating agent.
        selection: The `select` result naming the delegate.
        task: The task text being delegated.
        provider: Provider for the selected agent's own resolution call.
    """
    dtask = DelegationTask(
        task_id=f"a2a-{selection.card.name}",
        from_agent=coordinator_name,
        to_agent=selection.card.name,
        payload=task,
    )
    dtask.transition("in_progress", f"delegated via card match, score={selection.score}")
    delegate_system = f"You are {selection.card.name}. {selection.card.description}"
    answer = provider.complete([Message.user(task)], system=delegate_system).content
    dtask.payload = answer
    dtask.transition("completed", f"resolved by {selection.card.name}")
    return dtask


# --- demo --------------------------------------------------------------


def run_agent_card_demo() -> tuple[SelectionResult, DelegationTask, str]:
    """Register three cards, discover the right one for a task, delegate, and fail on a fourth.

    A billing agent, an export agent, and a translation agent are
    registered. A refund request matches only the billing card and is
    delegated through its full lifecycle. A weather question matches none
    of the three cards, so discovery raises `NoCapableAgentError` instead
    of guessing.
    """
    registry = Registry()
    registry.register(
        AgentCard(
            "billing_agent",
            "Handles invoices, refunds, and payment disputes.",
            [Skill("billing", ["refund", "invoice", "charge", "payment"])],
        )
    )
    registry.register(
        AgentCard(
            "export_agent",
            "Exports account data to CSV or JSON.",
            [Skill("export", ["export", "csv", "json", "download"])],
        )
    )
    registry.register(
        AgentCard(
            "translation_agent",
            "Translates text between languages.",
            [Skill("translate", ["translate", "language", "locale"])],
        )
    )

    task = "Please refund the duplicate charge on my last invoice."
    selection = select(registry, {"refund", "invoice", "charge"})
    provider = MockProvider(script=["Refund issued for the duplicate charge; it will post within 3-5 business days."])
    delegated = delegate("coordinator", selection, task, provider)

    try:
        select(registry, {"weather", "forecast"})
        no_capable_error = ""
    except NoCapableAgentError as exc:
        no_capable_error = str(exc)

    return selection, delegated, no_capable_error
