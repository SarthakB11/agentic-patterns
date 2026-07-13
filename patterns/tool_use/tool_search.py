"""Retrieval-based tool selection (tool search).

Pasting every tool's schema into every prompt does not scale: the brief
notes selection accuracy degrades as the catalog grows, and Anthropic's Tool
Search Tool reports roughly 85 percent token reduction on large catalogs by
discovering tools on demand instead. MCP's SEP-1821 standardizes the same
idea server-side. `search_tools` below is a runnable, offline stand-in for
that runtime step: it embeds each tool's name and description with the
shared `HashEmbedder`, ranks them against the query by cosine similarity,
and only the top few are offered to the model, exactly like
`forced_choice.py`'s `offered_specs` override, driven by retrieval instead
of a hand-picked name.
"""

from __future__ import annotations

from typing import Any

from agentic_patterns import HashEmbedder, Message, ToolRegistry, cosine_similarity, get_provider, scripted_tool_call

from patterns.tool_use.loop import run_tool_loop
from patterns.tool_use.schema import auto_tool

SYSTEM_PROMPT = "You are an ops assistant with a large tool catalog. Use the tools offered to you to answer."


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


def demo_tool_search() -> None:
    """Retrieve the top-3 relevant tools out of ten for a currency-conversion request."""
    registry = build_large_registry()
    embedder = HashEmbedder()
    query = "convert 50 GBP to JPY, what's the exchange rate"

    selected = search_tools(query, registry.specs(), embedder, top_k=3)
    provider = get_provider(
        script=[
            scripted_tool_call("convert_currency", {"amount": 50, "from_currency": "GBP", "to_currency": "JPY"}),
            "50 GBP converts to about 46.00 JPY at the demo rate.",
        ]
    )
    messages = [Message.user(query)]

    result = run_tool_loop(provider, registry, messages, system=SYSTEM_PROMPT, offered_specs=selected, max_iterations=2)

    print("=== 10. Retrieval-based tool selection (tool search) ===")
    print(f"user:  {query}")
    print(f"catalog size: {len(registry.specs())}, offered after retrieval: {[s['name'] for s in selected]}")
    print(f"final: {result.final_answer}")
    print()


if __name__ == "__main__":
    demo_tool_search()
