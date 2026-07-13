"""Procedural memory: the agent's learned rules, skills, and workflows,
how to do things rather than what is true.

Held as a plain ordered list of standing rules, injected directly into the
system prompt and executed rather than re-derived. Unlike semantic and
episodic memory there is nothing to rank by relevance: procedural rules are
not retrieved by similarity to a query, every rule in the set always
applies, so this module has no dependency on the vector store.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, get_provider


@dataclass
class ProceduralMemory:
    """An ordered set of standing rules for one namespace."""

    namespace: str
    rules: list[str] = field(default_factory=list)

    def add_rule(self, rule: str) -> None:
        """Add a rule, ignoring an exact duplicate already present."""
        if rule not in self.rules:
            self.rules.append(rule)

    def render(self) -> str:
        """Render the rule set as a block for the system prompt.

        Returns an empty string when there are no rules, so callers can
        splice this into a system prompt unconditionally.
        """
        if not self.rules:
            return ""
        lines = "\n".join(f"- {rule}" for rule in self.rules)
        return f"Standing rules:\n{lines}"


def run_procedural_demo(provider: Provider | None = None) -> str:
    """Standing rules are injected into the system prompt once and followed
    on every turn, without being restated by the user.
    """
    if provider is None:
        provider = get_provider(
            script=["The deployment runs 3 nodes at 4 vCPUs each, 12 vCPUs total, on Terraform in us-west-2."]
        )
    rules = ProceduralMemory(namespace="user:alex")
    rules.add_rule("Always state infrastructure sizing in vCPUs, not cores.")
    rules.add_rule("Always name the IaC tool and region when describing a deployment.")

    system = f"You are an infrastructure assistant.\n\n{rules.render()}"
    completion = provider.complete([Message.user("How big is my deployment?")], system=system)
    return completion.content
