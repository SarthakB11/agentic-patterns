"""Output guards: checks on the model's response before it is used.

- `JSONSchemaGuard` parses the response as JSON and validates it against a
  small schema, using only the standard library. It supports the subset of
  JSON Schema this pattern's demos need: `type`, `required`, `enum`,
  `minimum`, and `maximum` on a flat object. It is deterministic and cheap,
  so it runs before any classifier-based guard.
- `ModerationGuard` screens text for blocklisted terms in named categories
  (a stand-in for a trained classifier such as Llama Guard or NemoGuard).

Both guards report `OnFail.RETRY` on failure by default: the caller feeds
the guard's message back to the model and asks it to try again, which is
what `pipeline.run_guarded` implements as the bounded validate-retry-repair
loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from patterns.guardrails.core import GuardResult, OnFail


def validate_schema(obj: Any, schema: dict[str, Any], *, path: str = "$") -> list[str]:
    """Validate `obj` against a flat-object JSON Schema subset.

    Supports `type` ("object", "string", "integer", "number", "boolean",
    "array"), `required`, and, on a property's own schema, `enum`,
    `minimum`, and `maximum`. Returns a list of human-readable violations;
    an empty list means `obj` is valid.
    """
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(obj, dict):
            return [f"{path}: expected object, got {type(obj).__name__}"]
        for req in schema.get("required", []):
            if req not in obj:
                errors.append(f"{path}: missing required field {req!r}")
        properties = schema.get("properties", {})
        for key, sub_schema in properties.items():
            if key in obj:
                errors.extend(_validate_value(obj[key], sub_schema, f"{path}.{key}"))
        return errors
    return _validate_value(obj, schema, path)


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    type_map = {"string": str, "boolean": bool, "array": list}
    if expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path}: expected integer, got {type(value).__name__}")
    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{path}: expected number, got {type(value).__name__}")
    elif expected_type in type_map:
        if not isinstance(value, type_map[expected_type]):
            errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
    if errors:
        return errors
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        errors.append(f"{path}: {value} is below minimum {schema['minimum']}")
    if "maximum" in schema and isinstance(value, (int, float)) and value > schema["maximum"]:
        errors.append(f"{path}: {value} is above maximum {schema['maximum']}")
    return errors


@dataclass
class JSONSchemaGuard:
    """Parses text as JSON and validates it against `schema`.

    Attributes:
        name: Guard name.
        schema: A flat-object JSON Schema, as understood by `validate_schema`.
        on_fail: Action on a parse or validation failure. Defaults to
            `OnFail.RETRY` so the pipeline can reask the model.
    """

    name: str = "json_schema"
    schema: dict[str, Any] = field(default_factory=dict)
    on_fail: OnFail = OnFail.RETRY

    def check(self, value: str) -> GuardResult:
        try:
            obj = json.loads(value)
        except json.JSONDecodeError as exc:
            return GuardResult(
                passed=False,
                action=self.on_fail,
                value=value,
                message=f"response is not valid JSON: {exc.msg}",
            )
        errors = validate_schema(obj, self.schema)
        if errors:
            return GuardResult(
                passed=False,
                action=self.on_fail,
                value=value,
                message="schema violation: " + "; ".join(errors),
            )
        return GuardResult(passed=True, action=OnFail.NOOP, value=obj)


_DEFAULT_BLOCKLIST: dict[str, tuple[str, ...]] = {
    "insult": ("idiot", "moron", "stupid user"),
    "self_harm": ("kill yourself",),
    "violence": ("i will hurt you",),
}


@dataclass
class ModerationGuard:
    """Screens text for blocklisted terms grouped into safety categories.

    A stand-in for a trained safety classifier (Llama Guard, NemoGuard):
    same interface, deterministic keyword matching instead of a model call,
    which keeps this guard's tests reproducible.

    Attributes:
        name: Guard name.
        blocklist: Category name to a tuple of trigger phrases.
        on_fail: Action on a match. Defaults to `OnFail.REFRAIN`: an unsafe
            response is never patched, it is replaced with a safe fallback.
    """

    name: str = "moderation"
    blocklist: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(_DEFAULT_BLOCKLIST))
    on_fail: OnFail = OnFail.REFRAIN

    def check(self, value: str) -> GuardResult:
        lowered = value.lower()
        for category, phrases in self.blocklist.items():
            for phrase in phrases:
                if phrase in lowered:
                    return GuardResult(
                        passed=False,
                        action=self.on_fail,
                        value=value,
                        message=f"blocked category: {category}",
                    )
        return GuardResult(passed=True, action=OnFail.NOOP, value=value)
