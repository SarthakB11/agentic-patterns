"""Shared toy tool registry for the planning pattern's demos.

Every variant module in this pattern plans and executes against the same
small trip-planning domain: weather, attractions, hotel cost, hotel
booking, and an itinerary drafter that combines two upstream outputs. Using
one domain everywhere lets a reader compare the different control-flow
shapes on a single, familiar task instead of relearning a new toy problem
per module.
"""

from __future__ import annotations

from agentic_patterns import Tool, ToolRegistry

_WEATHER = {
    "paris": "Mild and cloudy, high of 18C, light rain expected",
    "lisbon": "Sunny and warm, high of 26C, no rain expected",
    "lyon": "Partly cloudy, high of 20C",
}

_ATTRACTIONS = {
    "paris": "Louvre Museum, Eiffel Tower, Montmartre",
    "lisbon": "Belem Tower, Alfama district, Time Out Market",
    "lyon": "Basilica of Notre-Dame de Fourviere, Old Lyon, Traboules",
}

_HOTEL_RATE_PER_NIGHT = {"paris": 210, "lisbon": 140, "lyon": 150}

# Paris hotels are sold out; used to demo a scripted booking failure and the
# replan it triggers.
_HOTEL_SOLD_OUT = {"paris"}


def get_weather(city: str) -> str:
    """Return the forecast for `city`, or a generic mild forecast if unknown."""
    return _WEATHER.get(city.lower(), "Mild, high of 20C, no rain expected")


def search_attractions(city: str) -> str:
    """Return a comma-separated list of top attractions in `city`."""
    return _ATTRACTIONS.get(city.lower(), "City center walking tour")


def estimate_hotel_cost(city: str, nights: int) -> str:
    """Return an estimated total hotel cost for a stay in `city`."""
    rate = _HOTEL_RATE_PER_NIGHT.get(city.lower(), 160)
    total = rate * int(nights)
    return f"${total} for {nights} night(s) at ${rate}/night in {city}"


def book_hotel(city: str, nights: int) -> str:
    """Book a hotel room in `city`.

    Raises:
        RuntimeError: If `city` has no availability (currently Paris only),
            so replanning modules have a real failure to work around.
    """
    if city.lower() in _HOTEL_SOLD_OUT:
        raise RuntimeError(f"No rooms available in {city} for {nights} night(s)")
    confirmation = f"{city.upper()[:3]}{int(nights):02d}42"
    return f"Booked {nights} night(s) in {city}, confirmation #{confirmation}"


def draft_itinerary(weather: str, attractions: str) -> str:
    """Combine a weather forecast and an attraction list into a short itinerary."""
    return f"Given weather ({weather}), visit: {attractions}."


def build_travel_registry() -> ToolRegistry:
    """Build a fresh `ToolRegistry` with the full trip-planning tool set."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="get_weather",
            description="Get the weather forecast for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            fn=get_weather,
        )
    )
    registry.register(
        Tool(
            name="search_attractions",
            description="List top attractions in a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            fn=search_attractions,
        )
    )
    registry.register(
        Tool(
            name="estimate_hotel_cost",
            description="Estimate total hotel cost for a stay, without booking it.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "nights": {"type": "integer"},
                },
                "required": ["city", "nights"],
            },
            fn=estimate_hotel_cost,
        )
    )
    registry.register(
        Tool(
            name="book_hotel",
            description="Book a hotel room for a stay. Fails if the city has no availability.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "nights": {"type": "integer"},
                },
                "required": ["city", "nights"],
            },
            fn=book_hotel,
        )
    )
    registry.register(
        Tool(
            name="draft_itinerary",
            description="Combine a weather forecast and an attraction list into a short itinerary.",
            parameters={
                "type": "object",
                "properties": {
                    "weather": {"type": "string"},
                    "attractions": {"type": "string"},
                },
                "required": ["weather", "attractions"],
            },
            fn=draft_itinerary,
        )
    )
    return registry
