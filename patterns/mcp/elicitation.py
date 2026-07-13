"""Elicitation: server-initiated structured input, form mode and URL mode.

The package README originally skipped elicitation on the grounds that it
"reuses the sampling nested-request machinery for no new mechanic." The
machinery is shared (a server sends a request back over the same
connection mid-call and waits for the client's answer); the mechanic is
not. Sampling asks the client to run a model. Elicitation asks the client
to ask a *human*: it carries a `requestedSchema` (a small JSON Schema
object) so the client can render a constrained form, and the response is a
three-way `action`, `accept` with `content`, `decline`, or `cancel`, which
sampling's plain `createMessage` result does not have. SEP-1330 standardizes
this `ElicitResult` shape with single- and multi-select enum support;
SEP-1036 adds a URL mode where the server hands the user to a trusted
external page (OAuth, a payment form) and resumes once the client reports
the external step done.

Unlike `sampling.py`, which drives its round trip over a live subprocess
and `StdioServerTransport`, this module runs the same `elicitation/create`
request and `ElicitResult` response shapes in-process: a tool handler
builds the exact JSON-RPC request dict, hands it to a scripted callback
(the same role `on_sampling_request` plays in `client.py`), and processes
the three-way action. Spawning a second subprocess server variant to prove
this would add transport plumbing without teaching a new mechanic;
`sampling.py` already proves reverse-direction requests travel correctly
over a live duplex connection, and elicitation asks a human rather than a
model, which makes it, if anything, more deterministic to script offline
than sampling is.

The URL-mode `requestedSchema` shape below (`{"type": "url", "url": ...}`)
is illustrative rather than a verified wire contract: SEP-1036's exact
field names were not independently confirmed against a shipped spec page
for this module, so treat it as showing the round trip's shape, not a
byte-for-byte implementation of the SEP.

Human approval tiers and escalation policy belong to
`patterns/human_in_the_loop/`; the callbacks here stay minimal, a single
scripted decision per call, matching the same restraint `sampling.py` uses.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from patterns.mcp import jsonrpc

ELICITATION_CLIENT_CAPABILITY = "elicitation"

ElicitationHandler = Callable[[dict[str, Any]], dict[str, Any]]

MEETING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"time_slot": {"type": "string", "enum": ["morning", "afternoon", "evening"]}},
    "required": ["time_slot"],
}

# See the module docstring's note on URL-mode field names being illustrative.
PAYMENT_URL_SCHEMA: dict[str, Any] = {"type": "url", "url": "https://payments.example.invalid/connect"}

_request_ids = itertools.count(1)


def _next_request_id() -> str:
    return f"srv-elicit-{next(_request_ids)}"


def _validate_accept_content(schema: dict[str, Any], content: dict[str, Any]) -> str | None:
    """Return a violation description if `content` does not satisfy `schema`, else `None`.

    A minimal check, matching the mechanism's scope: required keys present,
    and any enum-constrained value falls inside its declared range. This is
    not a general JSON Schema validator.
    """
    for key in schema.get("required", []):
        if key not in content:
            return f"missing required field {key!r}"
    for key, value in content.items():
        prop = schema.get("properties", {}).get(key)
        if prop and "enum" in prop and value not in prop["enum"]:
            return f"{key!r}={value!r} is not one of {prop['enum']}"
    return None


def schedule_meeting(client_capabilities: set[str], handler: ElicitationHandler | None) -> tuple[list[dict[str, Any]], bool]:
    """Tool: schedule a meeting, asking the user for a time slot via form-mode elicitation.

    Args:
        client_capabilities: The capabilities this call's client offered.
            Gated on `elicitation`, exactly as `summarize_note` in
            `server_data.py` gates on `sampling`.
        handler: The scripted callback standing in for a client's elicitation
            UI. Receives the `elicitation/create` params and returns an
            `ElicitResult`-shaped dict: `{"action": "accept", "content": {...}}`,
            `{"action": "decline"}`, or `{"action": "cancel"}`.

    Returns:
        A `(content, is_error)` tuple, the same shape `server_data.py`'s tool
        handlers return.
    """
    if ELICITATION_CLIENT_CAPABILITY not in client_capabilities:
        return [{"type": "text", "text": "client did not offer the elicitation capability; cannot schedule"}], True
    if handler is None:
        return [{"type": "text", "text": "no elicitation handler registered; cannot schedule"}], True

    request = jsonrpc.build_request(
        _next_request_id(), "elicitation/create", {"message": "What time works for your meeting?", "requestedSchema": MEETING_SCHEMA}
    )
    result = handler(request["params"])
    action = result.get("action")

    if action == "accept":
        content = result.get("content", {})
        violation = _validate_accept_content(MEETING_SCHEMA, content)
        if violation:
            return [{"type": "text", "text": f"elicitation response failed schema validation: {violation}"}], True
        return [{"type": "text", "text": f"Meeting scheduled for the {content['time_slot']}."}], False
    if action == "decline":
        return [{"type": "text", "text": "cannot schedule the meeting: the user declined to provide a time"}], True
    if action == "cancel":
        return [{"type": "text", "text": "cannot schedule the meeting: elicitation was cancelled"}], True
    return [{"type": "text", "text": f"unrecognized elicitation action: {action!r}"}], True


def connect_payment_method(client_capabilities: set[str], handler: ElicitationHandler | None) -> tuple[list[dict[str, Any]], bool]:
    """Tool: URL-mode elicitation, handing the user to an external page and resuming after.

    Args:
        client_capabilities: As in `schedule_meeting`.
        handler: Scripted callback. For URL mode, an `accept` result is
            expected to carry `{"completed": True}` once the client reports
            the external step done; any other content is treated as
            incomplete.

    Returns:
        A `(content, is_error)` tuple.
    """
    if ELICITATION_CLIENT_CAPABILITY not in client_capabilities:
        return [{"type": "text", "text": "client did not offer the elicitation capability; cannot connect a payment method"}], True
    if handler is None:
        return [{"type": "text", "text": "no elicitation handler registered; cannot connect a payment method"}], True

    request = jsonrpc.build_request(
        _next_request_id(),
        "elicitation/create",
        {"message": "Complete payment setup at the linked page, then return here.", "requestedSchema": PAYMENT_URL_SCHEMA},
    )
    result = handler(request["params"])
    action = result.get("action")

    if action == "accept" and result.get("content", {}).get("completed") is True:
        return [{"type": "text", "text": "Payment method connected; resuming."}], False
    if action == "accept":
        return [{"type": "text", "text": "payment setup was not reported as completed; cannot resume"}], True
    if action == "decline":
        return [{"type": "text", "text": "cannot connect a payment method: the user declined"}], True
    if action == "cancel":
        return [{"type": "text", "text": "cannot connect a payment method: elicitation was cancelled"}], True
    return [{"type": "text", "text": f"unrecognized elicitation action: {action!r}"}], True


def run_elicitation_demo() -> dict[str, tuple[list[dict[str, Any]], bool]]:
    """Run accept, decline, cancel, schema-violation, no-capability, and URL-mode paths.

    Returns:
        A dict mapping scenario name to `(content, is_error)`, keyed for
        `main.py` to print and `tests/test_mcp.py` to assert against.
    """
    def accept_handler(params: dict[str, Any]) -> dict[str, Any]:
        return {"action": "accept", "content": {"time_slot": "afternoon"}}

    def decline_handler(params: dict[str, Any]) -> dict[str, Any]:
        return {"action": "decline"}

    def cancel_handler(params: dict[str, Any]) -> dict[str, Any]:
        return {"action": "cancel"}

    def bad_schema_handler(params: dict[str, Any]) -> dict[str, Any]:
        return {"action": "accept", "content": {"time_slot": "midnight"}}

    def url_handler(params: dict[str, Any]) -> dict[str, Any]:
        return {"action": "accept", "content": {"completed": True}}

    return {
        "accept": schedule_meeting({"elicitation"}, accept_handler),
        "decline": schedule_meeting({"elicitation"}, decline_handler),
        "cancel": schedule_meeting({"elicitation"}, cancel_handler),
        "schema_violation": schedule_meeting({"elicitation"}, bad_schema_handler),
        "no_capability": schedule_meeting(set(), accept_handler),
        "url_mode": connect_payment_method({"elicitation"}, url_handler),
    }
