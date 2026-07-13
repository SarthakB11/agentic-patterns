"""Sub-module: semantic (embedding-similarity) routing.

Each route carries a few example utterances, embedded once and cached on
the route for every later call. An incoming query is embedded with the
same embedder and matched to the route whose utterances it is closest to
by cosine similarity; a query that is not close enough to any route falls
to a "no match" default instead of being forced onto the nearest route
regardless of how far it is. This is faster, cheaper, and more
deterministic than an LLM classifier, and adding a route is just adding
utterances.

Uses `HashEmbedder`, the repo's deterministic stdlib-only stand-in for a
real embedding model (see `agentic_patterns.core.embeddings`): it preserves
the geometry-of-similarity property that matters here (overlapping
vocabulary lands closer together) without a network call.

Classification is split into two functions on purpose: `route_scores`
does the embedding and similarity work, `classify_scores` is a pure
function from a `{route: similarity}` mapping to a `RouteDecision`. Tests
that want to check the threshold boundary can call `classify_scores`
directly with synthetic scores instead of depending on embedder output.
"""

from __future__ import annotations

from agentic_patterns import Embedder, cosine_similarity, get_embedder

from patterns.routing.registry import Route, RouteDecision, RouteRegistry

NO_MATCH_ROUTE = "no_match"
DEFAULT_THRESHOLD = 0.2

_ROUTES = RouteRegistry(
    [
        Route(
            name="billing",
            description="Charges, invoices, refunds, and payment methods.",
            utterances=[
                "I was charged twice for my subscription this month",
                "how do I get a refund for last invoice",
                "please update my credit card payment method",
                "my subscription payment failed to process",
            ],
        ),
        Route(
            name="technical",
            description="App or website errors, crashes, and installation problems.",
            utterances=[
                "the app crashes every time I open settings",
                "I am getting an error installing the latest update",
                "the website will not load on my browser",
                "the software keeps freezing during export",
            ],
        ),
        Route(
            name="account",
            description="Login, password, and account access problems.",
            utterances=[
                "I forgot my account password and cannot log in",
                "how do I change my account email address",
                "my username was changed without my permission",
                "I am locked out of my account after too many login attempts",
            ],
        ),
    ]
)


def _cached_utterance_vectors(route: Route, embedder: Embedder) -> list[list[float]]:
    """Return `route`'s utterance vectors, embedding them only on first use.

    The vectors are cached as a private attribute on `route` itself, so a
    route's utterances are embedded once (the first time any call scores
    against it) and every later call, for that route, reuses the cached
    vectors instead of re-embedding the same utterances.
    """
    cached: list[list[float]] | None = getattr(route, "_utterance_vectors_cache", None)
    if cached is None:
        cached = embedder.embed(route.utterances)
        route._utterance_vectors_cache = cached  # type: ignore[attr-defined]
    return cached


def route_scores(text: str, registry: RouteRegistry = _ROUTES, embedder: Embedder | None = None) -> dict[str, float]:
    """Embed `text` and score it against every route's utterances.

    Args:
        text: The input to score.
        registry: Routes to score against; each must carry `utterances`.
        embedder: Defaults to `get_embedder()`, the hash-based stdlib
            embedder, so this runs with no network call.

    Returns:
        A `{route_name: best_similarity}` mapping, where `best_similarity`
        is the highest cosine similarity between `text` and any one of that
        route's utterances. Each route's utterance vectors are computed
        once and cached on the route (see `_cached_utterance_vectors`); only
        `text` is embedded fresh on every call.
    """
    embedder = embedder or get_embedder()
    query_vector = embedder.embed([text])[0]
    scores: dict[str, float] = {}
    for route in registry:
        if not route.utterances:
            continue
        utterance_vectors = _cached_utterance_vectors(route, embedder)
        scores[route.name] = max(cosine_similarity(query_vector, v) for v in utterance_vectors)
    return scores


def classify_scores(scores: dict[str, float], *, threshold: float = DEFAULT_THRESHOLD) -> RouteDecision:
    """Pick the best-scoring route, or fall back to `NO_MATCH_ROUTE` below threshold.

    Args:
        scores: A `{route_name: similarity}` mapping, as `route_scores` returns.
        threshold: Minimum similarity required to accept the best match.
            Below this, the input is closer to no route than to any of
            them, and is routed to `NO_MATCH_ROUTE` instead.
    """
    if not scores:
        return RouteDecision(route=NO_MATCH_ROUTE, score=None, method="semantic", metadata={"scores": {}})
    best_route = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_route]
    if best_score < threshold:
        return RouteDecision(
            route=NO_MATCH_ROUTE, score=best_score, method="semantic", metadata={"scores": scores, "nearest": best_route}
        )
    return RouteDecision(route=best_route, score=best_score, method="semantic", metadata={"scores": scores})


def classify(text: str, registry: RouteRegistry = _ROUTES, *, threshold: float = DEFAULT_THRESHOLD) -> RouteDecision:
    """Embed and classify `text` in one call: `route_scores` then `classify_scores`."""
    return classify_scores(route_scores(text, registry), threshold=threshold)


def run_semantic_demo() -> list[RouteDecision]:
    """Classify example queries, including one far from every route.

    The last input ("what is the weather like today in Paris") shares no
    real vocabulary with any route's utterances and is expected to land
    below `DEFAULT_THRESHOLD`, demonstrating the no-match path.
    """
    inputs = [
        "why was I billed twice this month",
        "the mobile app keeps crashing on launch",
        "I forgot my password and cannot log into my account",
        "what is the weather like today in Paris",
    ]
    return [classify(text) for text in inputs]
