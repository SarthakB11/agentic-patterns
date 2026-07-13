"""Sub-module: LLM-classifier routing.

An LLM is prompted with the route set (name plus description for each) and
asked to return a single structured label naming the best route. This is
strong on nuance and intent that keyword or embedding matching miss, at the
cost of a model call, added latency, and the chance the model returns a
label that is not in the route set at all. The output is always validated
against the registry (`RouteRegistry.validate`); an unknown or malformed
label falls back to a default route rather than dispatching to a route
that does not exist.

The expected reply shape is one line: `ROUTE: <name>`. Real APIs would more
likely use a JSON-mode or tool-call response for this; the plain-text
sentinel is used here so the parser and its failure modes are easy to read
in the scripted transcript.
"""

from __future__ import annotations

import re

from agentic_patterns import Message, Provider, get_provider

from patterns.routing.registry import Route, RouteDecision, RouteRegistry

DEFAULT_ROUTE = "general"

_ROUTES = RouteRegistry(
    [
        Route(name="billing", description="Charges, invoices, refunds, and payment methods."),
        Route(name="technical", description="App or website errors, crashes, and installation problems."),
        Route(name="account", description="Login, password, and account access problems."),
        Route(name=DEFAULT_ROUTE, description="Anything that is not billing, technical, or account related."),
    ]
)

_ROUTE_LINE = re.compile(r"ROUTE:\s*([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)

_CLASSIFIER_SYSTEM = (
    "You are a routing classifier for a customer support system. Read the "
    "customer's message and reply with exactly one line: ROUTE: <name>, "
    "choosing the single best-fitting route name from the list below. "
    "Reply with nothing else.\n\n"
    "Routes:\n"
    + "\n".join(f"- {r.name}: {r.description}" for r in _ROUTES)
)


def parse_route_label(text: str, registry: RouteRegistry, *, default: str = DEFAULT_ROUTE) -> RouteDecision:
    """Parse a classifier reply and validate it against `registry`.

    Args:
        text: The raw model reply, expected to contain a `ROUTE: <name>` line.
        registry: The route set to validate the parsed label against.
        default: Route to fall back to if no line is found, or the parsed
            name is not a registered route.
    """
    match = _ROUTE_LINE.search(text)
    if not match:
        return RouteDecision(
            route=default, score=None, method="llm_classifier", metadata={"raw": text, "valid": False}
        )
    candidate = match.group(1).lower()
    validated = registry.validate(candidate, default=default)
    return RouteDecision(
        route=validated,
        score=None,
        method="llm_classifier",
        metadata={"raw": text, "valid": validated == candidate},
    )


def classify(text: str, provider: Provider, registry: RouteRegistry = _ROUTES, *, default: str = DEFAULT_ROUTE) -> RouteDecision:
    """Ask `provider` to classify `text` and validate its reply.

    Args:
        text: The input to classify.
        provider: The model that plays the classifier.
        registry: The route set the model chooses from and is validated
            against.
        default: Fallback route for an invalid or missing label.
    """
    completion = provider.complete([Message.user(text)], system=_CLASSIFIER_SYSTEM)
    return parse_route_label(completion.content, registry, default=default)


def run_llm_classifier_demo() -> tuple[RouteDecision, RouteDecision]:
    """Classify one clean input and one that yields an invalid label.

    Returns:
        A (valid_decision, fallback_decision) pair: the first from a
        provider scripted to reply with a route the registry recognizes,
        the second from a provider scripted to reply with a route name
        ("shipping") that was never registered, exercising the fallback.
    """
    valid_provider = get_provider(script=["ROUTE: technical"])
    valid_decision = classify("the mobile app keeps crashing when I open it", valid_provider)

    invalid_provider = get_provider(script=["ROUTE: shipping"])
    fallback_decision = classify("where is my package", invalid_provider)

    return valid_decision, fallback_decision
