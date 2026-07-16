"""Shared toy tool catalog used by most demo modules in this pattern.

A small, deterministic "ops assistant" domain: weather lookup, currency
conversion, and an internal order and customer directory. Every function is
registered through `schema.auto_tool`, so its JSON Schema is derived from its
type hints and docstring rather than hand-written, doubling as a running
example of schema autogeneration alongside the dedicated `schema.py` demo.
"""

from __future__ import annotations

from agentic_patterns import ToolRegistry
from patterns.tool_use.schema import auto_tool

SYSTEM_PROMPT = (
    "You are an ops assistant for an online retailer. You can look up "
    "weather, convert currencies, and check orders and customer contact "
    "details using the tools available to you. Call a tool whenever you "
    "need information you do not already have. Give short, concrete "
    "final answers."
)

_WEATHER = {
    "tokyo": "18C, light rain",
    "paris": "21C, clear skies",
    "san francisco": "16C, fog",
}

# USD value of one unit of each currency, so converting is one multiply and
# one divide regardless of direction.
_USD_PER_UNIT = {"USD": 1.0, "EUR": 1 / 0.92, "GBP": 1 / 0.79, "JPY": 1 / 149.0}

_ORDERS = {
    "ORD-1001": {"status": "shipped", "customer_id": "CUST-42"},
    "ORD-1002": {"status": "processing", "customer_id": "CUST-77"},
}

_CUSTOMER_EMAILS = {"CUST-42": "priya@example.com", "CUST-77": "sam@example.com"}


def build_registry() -> ToolRegistry:
    """Build the shared registry of read-only ops-assistant tools."""
    registry = ToolRegistry()

    @auto_tool(registry)
    def get_weather(city: str) -> str:
        """Look up current weather conditions for a city.

        Args:
            city: City name, e.g. "Tokyo".
        """
        key = city.strip().lower()
        if key not in _WEATHER:
            raise ValueError(f"no weather data for '{city}'")
        return f"{city}: {_WEATHER[key]}"

    @auto_tool(registry)
    def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
        """Convert an amount between currencies using fixed demo exchange rates.

        Args:
            amount: Amount to convert, denominated in from_currency.
            from_currency: Three-letter source currency code, e.g. "USD".
            to_currency: Three-letter target currency code, e.g. "EUR".
        """
        if from_currency not in _USD_PER_UNIT or to_currency not in _USD_PER_UNIT:
            raise ValueError(f"unsupported currency pair {from_currency!r} -> {to_currency!r}")
        usd = amount * _USD_PER_UNIT[from_currency]
        converted = usd / _USD_PER_UNIT[to_currency]
        return f"{amount} {from_currency} = {converted:.2f} {to_currency}"

    @auto_tool(registry)
    def lookup_order(order_id: str) -> str:
        """Look up an order's shipping status and owning customer id.

        Args:
            order_id: Order identifier, e.g. "ORD-1001".
        """
        if order_id not in _ORDERS:
            raise KeyError(f"no such order '{order_id}'")
        order = _ORDERS[order_id]
        return f"status={order['status']} customer_id={order['customer_id']}"

    @auto_tool(registry)
    def get_customer_email(customer_id: str) -> str:
        """Look up a customer's email address by customer id.

        Args:
            customer_id: Customer identifier, e.g. "CUST-42".
        """
        if customer_id not in _CUSTOMER_EMAILS:
            raise KeyError(f"no such customer '{customer_id}'")
        return _CUSTOMER_EMAILS[customer_id]

    return registry
