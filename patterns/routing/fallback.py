"""Sub-module: fallback / resilience routing.

The failure-time complement to primary routing: on error, timeout, or
refusal, re-dispatch to the next handler in an ordered chain instead of
surfacing the failure to the caller. This is the core job the brief assigns
to a model gateway. A wrong route can be worse than no routing, so a
handler here always fails loudly into `HandlerFailure` rather than
returning a silently bad answer; the chain decides what happens next.

Three failure kinds are exercised: a raised error (the handler's dependency
is broken), a timeout (the handler took too long), and a refusal (the
handler responded but declined to help). All three are represented the
same way to the chain, as a `HandlerFailure`, so the chain's logic does not
need to know which kind occurred, only that it needs to move on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agentic_patterns import Message, Provider

from patterns.routing.registry import RouteDecision

_REFUSAL_MARKERS = ("i can't help with that", "i cannot assist", "i'm not able to help")


class HandlerFailure(Exception):
    """Raised by a fallback handler's `call` to signal it produced no answer."""


@dataclass
class FallbackHandler:
    """One link in a fallback chain.

    Attributes:
        name: Handler name, used as the route name if it succeeds.
        call: Produces an answer, or raises `HandlerFailure` describing why
            it could not.
    """

    name: str
    call: Callable[[], str]


def make_provider_handler(name: str, provider: Provider, question: str, *, system: str | None = None) -> FallbackHandler:
    """Build a `FallbackHandler` backed by a `Provider.complete()` call.

    Converts a refusal-shaped reply (one starting with a phrase in
    `_REFUSAL_MARKERS`) into a `HandlerFailure`, so the chain treats a
    refusal the same way it treats a raised error: move to the next
    handler rather than returning the refusal text as if it were an
    answer.
    """

    def call() -> str:
        completion = provider.complete([Message.user(question)], system=system)
        lowered = completion.content.strip().lower()
        if any(lowered.startswith(marker) for marker in _REFUSAL_MARKERS):
            raise HandlerFailure(f"refusal: {completion.content.strip()}")
        return completion.content

    return FallbackHandler(name=name, call=call)


def run_fallback_chain(handlers: list[FallbackHandler]) -> RouteDecision:
    """Try each handler in order until one succeeds.

    Args:
        handlers: Ordered chain of handlers to try.

    Returns:
        A `RouteDecision` naming the handler that succeeded, with
        `attempts` counting every handler tried including failures. If
        every handler fails, the route is "human" (the safe terminal
        fallback) and `metadata["errors"]` lists what each handler reported,
        rather than raising and crashing the caller.
    """
    errors: list[str] = []
    for attempt, handler in enumerate(handlers, start=1):
        try:
            answer = handler.call()
        except HandlerFailure as exc:
            errors.append(f"{handler.name}: {exc}")
            continue
        return RouteDecision(
            route=handler.name,
            score=1.0,
            method="fallback",
            attempts=attempt,
            metadata={"answer": answer, "errors": errors},
        )
    return RouteDecision(
        route="human",
        score=0.0,
        method="fallback",
        attempts=len(handlers),
        metadata={"errors": errors, "reason": "all handlers failed"},
    )


def run_fallback_demo() -> tuple[RouteDecision, RouteDecision]:
    """Run one chain where a later handler recovers and one where all fail.

    Returns:
        A (recovered, exhausted) pair. In the first, the primary handler
        raises a simulated timeout and a secondary handler answers. In the
        second, the primary refuses and the only other handler errors,
        landing on the terminal human route with no crash.
    """

    def timeout_call() -> str:
        raise HandlerFailure("timeout: primary billing service did not respond in 5s")

    def secondary_success_call() -> str:
        return "Your last invoice was $482.10, billed on the 1st for the March subscription period."

    recovered_handlers = [
        FallbackHandler(name="primary_billing_api", call=timeout_call),
        FallbackHandler(name="secondary_billing_cache", call=secondary_success_call),
    ]
    recovered = run_fallback_chain(recovered_handlers)

    def refusal_call() -> str:
        raise HandlerFailure("refusal: I can't help with that request.")

    def broken_call() -> str:
        raise HandlerFailure("error: downstream account service returned HTTP 500")

    exhausted_handlers = [
        FallbackHandler(name="policy_bot", call=refusal_call),
        FallbackHandler(name="account_service", call=broken_call),
    ]
    exhausted = run_fallback_chain(exhausted_handlers)

    return recovered, exhausted
