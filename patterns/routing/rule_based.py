"""Sub-module: rule-based routing.

Deterministic dispatch from keyword matches, standard library only. No
model call, no embedder, nothing to script: the route a given input takes
is a pure function of its text. This is the cheapest and most testable
router in the folder, and the brief's recommended default when a category
set is small and stable. It is brittle to phrasings the rules do not
anticipate, which is exactly what the semantic and LLM-classifier routers
in this folder are for.

Three support routes (billing, technical, account) plus a "general"
default for anything that matches no keyword.
"""

from __future__ import annotations

from patterns.routing.registry import RouteDecision

_KEYWORD_ROUTES: dict[str, tuple[str, ...]] = {
    "billing": ("charge", "invoice", "refund", "payment", "subscription", "billed"),
    "technical": ("crash", "crashes", "error", "bug", "not working", "install", "freezing", "won't load"),
    "account": ("password", "log in", "login", "locked out", "username", "email address"),
}

DEFAULT_ROUTE = "general"


def classify(text: str) -> RouteDecision:
    """Match `text` against each route's keyword list, in registration order.

    The first route with a matching keyword wins. Rule-based routing has no
    natural confidence signal, so a match reports `score=1.0` (fully
    confident) and a miss reports `score=0.0`; `metadata["matched_keyword"]`
    records which keyword fired, for logging and tests.

    Args:
        text: The input to classify.
    """
    lowered = text.lower()
    for route_name, keywords in _KEYWORD_ROUTES.items():
        for keyword in keywords:
            if keyword in lowered:
                return RouteDecision(
                    route=route_name,
                    score=1.0,
                    method="rule",
                    metadata={"matched_keyword": keyword},
                )
    return RouteDecision(route=DEFAULT_ROUTE, score=0.0, method="rule", metadata={"matched_keyword": None})


def run_rule_based_demo() -> list[RouteDecision]:
    """Classify a handful of example support requests and return the decisions.

    Includes one input per keyword route and one that matches nothing, to
    show the default route firing.
    """
    inputs = [
        "I was charged twice for my subscription this month",
        "the app crashes every time I open settings",
        "I forgot my password and I'm locked out of my account",
        "do you offer support in French",
    ]
    return [classify(text) for text in inputs]
