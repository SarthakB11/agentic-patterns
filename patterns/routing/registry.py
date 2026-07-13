"""Shared route registry and routing metadata, used by every variant module.

`Route` and `RouteRegistry` model the "route registry" the brief calls for:
named routes, each with a description, example utterances (for the semantic
router), a handler, and a model tier (for the cascade and capability-
selection routers). `RouteDecision` is the metadata shape every classifier
in this pattern returns, so a caller can log or test "which route, what
score, how many attempts" the same way regardless of which classifier chose
it.

Kept free of any provider or prompting logic: this module is plain data
plus two small helpers (`validate` and `dispatch`) that implement the
"validate the candidate against the route set" and "run the handler" steps
of the canonical control flow. Classification itself lives in each variant
module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Route:
    """One named destination a router can send an input to.

    Attributes:
        name: Unique route name, e.g. "billing".
        description: What this route is for, shown to an LLM classifier.
        utterances: Example queries that belong on this route, used by the
            semantic router to build per-route vectors. Empty for routes
            that only rule-based or LLM classification ever selects.
        tier: Cost/capability tier this route runs at, e.g. "cheap",
            "strong", or "human". Used by the cascade and capability-
            selection routers; cosmetic for routers that ignore tiers.
        handler: Callable that answers an input once this route is chosen.
            Optional so a route can be declared before its handler exists,
            for example in a test that only checks classification.
    """

    name: str
    description: str
    utterances: list[str] = field(default_factory=list)
    tier: str = "standard"
    handler: Callable[[str], str] | None = None


@dataclass
class RouteDecision:
    """The outcome of a routing decision: which route, how confident, how.

    This is the metadata the brief asks every classifier to produce, so a
    caller (a test, a log line, or the end-to-end demo) can inspect a
    decision the same way no matter which variant produced it.

    Attributes:
        route: The chosen route's name.
        score: A confidence, similarity, or quality score in [0, 1] if the
            classifier produced one, else None (rule-based routing has no
            natural score, for example). This field mixes two different
            things across the folder's variants, and callers should not
            conflate them: a calibrated, genuinely continuous score (the
            semantic router's cosine similarity, a judge's verdict) versus a
            placeholder flag that only ever takes a constant value (`rule_based`
            always reports 1.0 on a match; `reasoning_mode` reports 1.0 or
            0.0 as a binary mode marker, not a probability). Both are valid
            uses of the field, but only the former is safe to threshold-sweep
            or average; `threshold_sweep.py` depends on a score that is
            actually continuous and documents its own stand-in rather than
            reusing one of the placeholder flags.
        method: Which classifier produced this decision, e.g. "rule",
            "semantic", "llm_classifier", "cascade", "fallback",
            "escalation", "reasoning_mode", "handoff".
        attempts: How many handlers or tiers were tried before this
            decision was reached. 1 for a direct classification; higher for
            a cascade that escalated or a fallback chain that retried.
        metadata: Free-form extra detail specific to the method, e.g. which
            keyword matched, or the errors a fallback chain collected.
    """

    route: str
    score: float | None
    method: str
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class RouteRegistry:
    """Collects `Route` objects and looks them up by name."""

    def __init__(self, routes: list[Route] | None = None) -> None:
        self._routes: dict[str, Route] = {}
        for route in routes or []:
            self.register(route)

    def register(self, route: Route) -> None:
        """Add a route to the registry, keyed by its name."""
        self._routes[route.name] = route

    def get(self, name: str) -> Route | None:
        """Look up a route by name, or None if it is not registered."""
        return self._routes.get(name)

    def names(self) -> list[str]:
        """Return every registered route name, in registration order."""
        return list(self._routes)

    def __iter__(self) -> Iterator[Route]:
        return iter(self._routes.values())

    def __contains__(self, name: str) -> bool:
        return name in self._routes

    def validate(self, candidate: str, *, default: str) -> str:
        """Return `candidate` if it names a registered route, else `default`.

        Implements the "validate the candidate against the route set" step
        every classifier needs: an LLM can invent a route name, and even a
        deterministic classifier can be handed a stale route by a caller.
        Treating anything outside the enumerated set as unroutable and
        falling back is safer than dispatching to a route that does not
        exist.

        Args:
            candidate: The route name the classifier produced.
            default: The route to fall back to when `candidate` is unknown.
        """
        return candidate if candidate in self._routes else default

    def dispatch(self, decision: RouteDecision, input_text: str) -> str:
        """Run the handler for `decision.route` against `input_text`.

        Raises:
            KeyError: If the route is not registered.
            ValueError: If the route is registered but has no handler.
        """
        route = self.get(decision.route)
        if route is None:
            known = ", ".join(sorted(self._routes)) or "(none registered)"
            raise KeyError(f"Unknown route {decision.route!r}. Known routes: {known}")
        if route.handler is None:
            raise ValueError(f"Route {decision.route!r} has no handler to dispatch to")
        return route.handler(input_text)
