"""Retrieval-based tool selection (tool search), at flooded catalog scale.

Pasting every tool's schema into every prompt does not scale. RAG-MCP
(arXiv:2505.03275) quantifies why: retrieving relevant servers before the
model ever sees a schema raises selection accuracy from 13.6 to 43.1
percent and cuts prompt tokens over 50 percent on a flooded catalog, and
Anthropic's Tool Search Tool reports roughly 85 percent token reduction on
large catalogs with the same discover-on-demand idea. `search_tools` below
is a runnable, offline stand-in for that runtime step: it embeds each
tool's name and description with the shared `HashEmbedder`, ranks them
against the query by cosine similarity, and only the top few are offered to
the model, exactly like `forced_choice.py`'s `offered_specs` override,
driven by retrieval instead of a hand-picked name.

A ten-tool catalog with an obvious best match cannot show either half of
RAG-MCP's finding: it never floods, so nothing collapses, and retrieval
never misses. `build_flooded_registry` adds near-duplicate distractor
tools, lexically closer to a specific query than the real tool is, so this
module can demonstrate three distinct outcomes instead of asserting numbers
it never exercises: flooding the offered catalog collapses selection to a
plausible-but-wrong distractor (`demo_flood_collapse`), retrieval over the
same flooded catalog restores the right pick at a fraction of the offered
tokens (`demo_tool_search`), and one-shot top-k retrieval can itself miss
the right tool, recoverable only by widening k and re-offering, which is
ScaleMCP's (arXiv:2505.06416) agent-driven retrieval in miniature
(`demo_recall_miss_then_widen`).
"""

from __future__ import annotations

from typing import Any

from agentic_patterns import HashEmbedder, Message, ToolRegistry, cosine_similarity, get_provider, scripted_tool_call

from patterns.tool_use.loop import run_tool_loop
from patterns.tool_use.schema import auto_tool

SYSTEM_PROMPT = "You are an ops assistant with a large tool catalog. Use the tools offered to you to answer."

# Distractor tools for a GBP-to-JPY conversion query: each name and
# description reuses the query's vocabulary (GBP, JPY, "exchange rate",
# "convert") more aggressively than the real convert_currency tool's
# generic description does, which is what pushes convert_currency down in
# a cosine-similarity ranking despite being the tool that actually performs
# the conversion.
_DISTRACTOR_TEMPLATES: list[tuple[str, str]] = [
    ("exchange_rate_convert", "Exchange rate convert GBP to JPY amount lookup tool."),
    ("fx_convert", "FX convert amount GBP JPY exchange rate calculator."),
    ("currency_exchange_lookup", "Currency exchange lookup GBP JPY rate convert amount."),
    ("convert_currency_v2", "Convert an amount between currencies, v2 endpoint, GBP JPY EUR USD exchange rate."),
    ("convert_currency_legacy", "Legacy currency conversion GBP JPY exchange rate, deprecated."),
    ("currency_converter", "Currency converter for GBP to JPY and other pairs, exchange rate lookup."),
]


def build_large_registry() -> ToolRegistry:
    """Build a ten-tool registry: too many to paste into every prompt at full detail."""
    registry = ToolRegistry()

    @auto_tool(registry)
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        """Convert an amount between currencies using fixed exchange rates.

        Args:
            amount: Amount to convert.
            from_currency: Source currency code.
            to_currency: Target currency code.
        """
        return f"{amount} {from_currency} = {amount * 0.92:.2f} {to_currency}"

    @auto_tool(registry)
    def get_weather(city: str) -> str:
        """Look up current weather conditions for a city.

        Args:
            city: City name.
        """
        return f"{city}: 18C, light rain"

    @auto_tool(registry)
    def track_shipment(tracking_number: str) -> str:
        """Track a package shipment by tracking number.

        Args:
            tracking_number: Carrier tracking number.
        """
        return f"{tracking_number}: in transit, arriving in 2 days"

    @auto_tool(registry)
    def get_stock_price(ticker: str) -> str:
        """Look up the latest stock price for a ticker symbol.

        Args:
            ticker: Stock ticker symbol, e.g. "AAPL".
        """
        return f"{ticker}: $189.32"

    @auto_tool(registry)
    def reset_password(account_id: str) -> str:
        """Send a password reset link to the email on file for an account.

        Args:
            account_id: Account identifier.
        """
        return f"password reset link sent for {account_id}"

    @auto_tool(registry)
    def schedule_meeting(topic: str, date: str) -> str:
        """Schedule a calendar meeting on a given date.

        Args:
            topic: Meeting subject.
            date: Meeting date, e.g. "2026-08-01".
        """
        return f"meeting '{topic}' scheduled for {date}"

    @auto_tool(registry)
    def translate_text(text: str, target_language: str) -> str:
        """Translate text into a target language.

        Args:
            text: Source text.
            target_language: Language to translate into, e.g. "French".
        """
        return f"[{target_language}] {text}"

    @auto_tool(registry)
    def lookup_order(order_id: str) -> str:
        """Look up an order's shipping status.

        Args:
            order_id: Order identifier.
        """
        return f"{order_id}: shipped"

    @auto_tool(registry)
    def search_docs(query: str) -> str:
        """Search internal help documentation for a query.

        Args:
            query: Search terms.
        """
        return f"top doc match for {query!r}: 'Refund policy overview'"

    @auto_tool(registry)
    def get_exchange_holiday(country: str) -> str:
        """Look up the next stock exchange holiday for a country.

        Args:
            country: Country name.
        """
        return f"{country}: next exchange holiday is in 12 days"

    return registry


def build_flooded_registry(distractor_count: int = 6) -> ToolRegistry:
    """Extend the ten-tool catalog with near-duplicate distractor tools that flood retrieval.

    Each distractor's name and description reuse a GBP-to-JPY conversion
    query's vocabulary more aggressively than the real `convert_currency`
    tool's generic description does, so a query phrased in those specific
    terms can rank several distractors above the tool that actually
    performs the conversion. This is what RAG-MCP's selection-accuracy
    collapse (13.6 to 43.1 percent, arXiv:2505.03275) looks like
    mechanically: not that the model is bad at picking tools, but that
    flooding the catalog with lexically similar noise buries the right one.

    Args:
        distractor_count: How many of the six scripted distractor tools to add.

    Returns:
        A registry with the original ten tools plus `distractor_count`
        distractors, all sharing `convert_currency`'s argument shape.
    """
    registry = build_large_registry()
    for name, description in _DISTRACTOR_TEMPLATES[:distractor_count]:
        docstring_args = "\n\nArgs:\n    amount: Amount to convert.\n    from_currency: Source currency code.\n    to_currency: Target currency code.\n"

        def make_distractor(tool_name: str, tool_description: str):
            def distractor(amount: float, from_currency: str, to_currency: str) -> str:
                return f"{amount} {from_currency} = {amount * 0.92:.2f} {to_currency} (distractor: {tool_name})"

            distractor.__name__ = tool_name
            distractor.__doc__ = tool_description + docstring_args
            return distractor

        auto_tool(registry)(make_distractor(name, description))
    return registry


def estimate_tokens(specs: list[dict[str, Any]]) -> int:
    """Rough token-count proxy for a list of offered tool specs.

    Not a real tokenizer: a whitespace word count over each spec's name and
    description. Good enough to compare "whole catalog offered" against
    "retrieved subset offered", the quantity RAG-MCP and Anthropic's Tool
    Search Tool report savings on.

    Args:
        specs: Tool specs as offered to a model this turn.

    Returns:
        The total word count across every spec's name and description.
    """
    return sum(len(f"{spec['name']} {spec['description']}".split()) for spec in specs)


def search_tools(query: str, specs: list[dict[str, Any]], embedder: HashEmbedder, top_k: int = 3) -> list[dict[str, Any]]:
    """Rank tool specs against a query by cosine similarity and return the top ones.

    Args:
        query: The user's request, used as the retrieval query.
        specs: Full catalog of provider-neutral tool specs to rank.
        embedder: Embedder used for both the query and each tool's text.
        top_k: Number of top-ranked specs to return.

    Returns:
        The `top_k` specs from `specs` most similar to `query`, most similar
        first.
    """
    tool_texts = [f"{spec['name']}: {spec['description']}" for spec in specs]
    vectors = embedder.embed([*tool_texts, query])
    tool_vectors, query_vector = vectors[:-1], vectors[-1]
    ranked = sorted(zip(specs, tool_vectors), key=lambda pair: cosine_similarity(pair[1], query_vector), reverse=True)
    return [spec for spec, _ in ranked[:top_k]]


def demo_flood_collapse() -> None:
    """Offer the whole flooded catalog with no retrieval; selection collapses to a plausible-but-wrong distractor.

    Sixteen tools (ten real plus six near-duplicate distractors) are all
    offered at once. The distractor `exchange_rate_convert` ranks highest
    by cosine similarity against this query, ahead of the real
    `convert_currency`, so the scripted pick names it: a registered,
    schema-valid, entirely wrong tool.
    """
    registry = build_flooded_registry(distractor_count=6)
    query = "convert 50 GBP to JPY, what's the exchange rate"
    provider = get_provider(
        script=[
            scripted_tool_call("exchange_rate_convert", {"amount": 50, "from_currency": "GBP", "to_currency": "JPY"}),
            "50 GBP converts to about 46.00 JPY using exchange_rate_convert.",
        ]
    )
    messages = [Message.user(query)]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, max_iterations=2)
    picked = result.rounds[0].calls[0].call.name

    print("=== 10a. Tool retrieval at scale: flooded catalog collapses selection ===")
    print(f"user:  {query}")
    print(f"catalog size: {len(registry.specs())} tools, all offered, no retrieval")
    print(f"picked: {picked!r} ({'WRONG' if picked != 'convert_currency' else 'correct'}, a plausible near-duplicate)")
    print()


def demo_tool_search() -> None:
    """Retrieve the top-3 relevant tools out of a flooded catalog and report the token savings.

    A less GBP/JPY-specific phrasing of the same request keeps
    `convert_currency` ranked first even amid the six distractors, so
    top-3 retrieval both picks the right tool and cuts the offered-token
    count against offering the whole flooded catalog.
    """
    registry = build_flooded_registry(distractor_count=6)
    embedder = HashEmbedder()
    query = "please help me convert an amount of money from one currency to another"

    full_specs = registry.specs()
    selected = search_tools(query, full_specs, embedder, top_k=3)
    provider = get_provider(
        script=[
            scripted_tool_call("convert_currency", {"amount": 50, "from_currency": "GBP", "to_currency": "JPY"}),
            "50 GBP converts to about 46.00 JPY at the demo rate.",
        ]
    )
    messages = [Message.user("Convert 50 GBP to JPY for me.")]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, offered_specs=selected, max_iterations=2)

    print("=== 10b. Tool retrieval at scale: top-3 retrieval fixes the pick and cuts tokens ===")
    print(f"retrieval query: {query}")
    print(f"catalog size: {len(full_specs)}, offered after retrieval: {[s['name'] for s in selected]}")
    print(f"offered tokens: full={estimate_tokens(full_specs)}, retrieved={estimate_tokens(selected)}")
    print(f"final: {result.final_answer}")
    print()


def demo_recall_miss_then_widen() -> None:
    """A one-shot top-k retrieval can miss the tool that actually answers the query; widen k and retry.

    Against the GBP/JPY-specific query, `convert_currency` ranks 7th out of
    sixteen tools, well outside a top-3 offer: retrieval "succeeds" in the
    sense that it ran, but the one tool that could answer the request was
    never offered, and no later reasoning step can recover a tool the model
    was never shown. Widening k to 8 brings it back into range.
    """
    registry = build_flooded_registry(distractor_count=6)
    embedder = HashEmbedder()
    query = "convert 50 GBP to JPY, what's the exchange rate"
    specs = registry.specs()

    narrow = search_tools(query, specs, embedder, top_k=3)
    missed = "convert_currency" not in [s["name"] for s in narrow]

    widened = search_tools(query, specs, embedder, top_k=8)
    recovered = "convert_currency" in [s["name"] for s in widened]

    print("=== 10c. Tool retrieval at scale: recall miss, then widen and re-offer ===")
    print(f"user:  {query}")
    print(f"top_k=3 offered: {[s['name'] for s in narrow]}")
    print(f"  convert_currency missed: {missed}")
    print(f"top_k=8 offered: {[s['name'] for s in widened]}")
    print(f"  convert_currency recovered: {recovered}")
    print()


if __name__ == "__main__":
    demo_flood_collapse()
    demo_tool_search()
    demo_recall_miss_then_widen()
