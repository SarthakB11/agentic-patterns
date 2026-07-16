"""Constrained decoding: schema-grammar token masking at generation time.

`validation.py` catches a malformed call after generation and pays a repair
round trip. Production systems increasingly prevent the structural half of
that problem before generation ever finishes: OpenAI's `strict: true` and
Anthropic's structured-outputs beta compile a tool's JSON Schema into a
grammar and mask illegal tokens during decoding, so a wrong-typed or
out-of-enum value mostly cannot be produced in the first place. XGrammar
(Dong et al., MLSys 2025, arXiv:2411.15100) made this cheap in production by
splitting the vocabulary into context-independent tokens, always legal or
always illegal for a given grammar state regardless of what came before, and
precomputed once when the grammar is compiled, versus a small
context-dependent set that still needs a runtime check. JSONSchemaBench
(Geng et al., arXiv:2501.10868) measured where masking still breaks: across
9,558 real schemas, coverage collapses from 86 percent on simple schemas to
3 percent on complex ones, which is why `validate_arguments` and the repair
turn in `validation.py` still earn their place for the semantic errors a
grammar cannot rule out (a well-formed but unsupported enum member's cousin:
a well-formed number that is simply the wrong value).

This module is the direct correction to this pattern's former "not
implemented" claim about strict decoding: the grammar and the token mask
below are pure functions of the schema and a scripted preference stream, so
the masking mechanism is fully demonstrable offline with no real model and
no `Provider.complete` call at all. What is not reproduced offline is a real
tokenizer's subword vocabulary and a real model's logit distribution; the
preference stream here stands in for "the model's ranked next-token
choices" the way `MockProvider`'s script stands in for a real completion
everywhere else in this pattern.

Scope: the grammar compiler below supports one schema shape, an object of
required fields each typed `number` or `string` with an `enum`, which is
enough to show masking working field by field. It is not a general JSON
Schema-to-grammar compiler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from patterns.tool_use.loop import validate_arguments

_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")

# The example schema the mechanism section of the research brief uses
# verbatim: {amount: number, currency: enum[...]}.
CONVERT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "amount": {"type": "number"},
        "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
    },
    "required": ["amount", "currency"],
}


@dataclass(frozen=True)
class FieldGrammar:
    """One compiled grammar state: a single required field's legality rule.

    Attributes:
        name: The field name this state accepts a value for.
        kind: "number" (an open-ended, context-dependent check applied at
            generation time) or "enum" (a finite, context-independent set
            precomputed once at compile time).
        legal_values: The finite set of legal tokens, populated only for
            kind="enum". Empty for "number", whose legality is a regex
            predicate rather than an enumerable set.
    """

    name: str
    kind: str
    legal_values: frozenset[str] = frozenset()


@dataclass
class Grammar:
    """A compiled grammar: one `FieldGrammar` state per required field, in order.

    Attributes:
        slots: The per-field states, walked in this order during generation.
        precheck_count: How many finite (context-independent) legal-token
            sets were built during compilation. XGrammar's split precomputes
            this class of check once at compile time; this counter is the
            module's evidence that "once" is literal, not just a claim, no
            matter how many generation steps or demo runs use this same
            compiled grammar afterward.
    """

    slots: list[FieldGrammar]
    precheck_count: int


def compile_grammar(schema: dict[str, Any]) -> Grammar:
    """Compile a JSON Schema object into one grammar state per required field.

    Enum fields get their finite legal-token set built once here, at compile
    time (XGrammar's context-independent precheck). Number fields get no
    precomputed set at all, since the space of legal numbers is unbounded;
    their legality is checked with a regex predicate at generation time
    instead (XGrammar's context-dependent path). That difference in when
    the check happens, not merely what the check is, is the split this
    module demonstrates.

    Args:
        schema: A JSON Schema object with "properties" and "required",
            where every required property is either {"type": "number"} or
            {"type": "string", "enum": [...]}.

    Returns:
        The compiled `Grammar`.
    """
    properties = schema["properties"]
    fields = schema.get("required") or list(properties)
    slots: list[FieldGrammar] = []
    precheck_count = 0
    for name in fields:
        prop = properties[name]
        if prop.get("enum"):
            slots.append(FieldGrammar(name, "enum", frozenset(prop["enum"])))
            precheck_count += 1
        else:
            slots.append(FieldGrammar(name, "number"))
    return Grammar(slots, precheck_count)


def _is_legal(slot: FieldGrammar, token: str) -> bool:
    """Check one candidate token against one grammar state's legality rule.

    For an enum state this is a hash-set membership test against the
    precomputed `legal_values`; for a number state it is a regex match, run
    fresh against this token since the set of legal numbers was never
    enumerated. Neither path re-derives the grammar itself, so calling this
    many times during generation does not grow `Grammar.precheck_count`.
    """
    if slot.kind == "enum":
        return token in slot.legal_values
    return bool(_NUMBER_RE.match(token))


@dataclass
class MaskStep:
    """One generation step's masking record.

    Attributes:
        field_name: The grammar state's field name.
        ranked_candidates: The full ranked preference list offered this step.
        legal: The subset of `ranked_candidates` the mask allowed, in the
            same relative order.
        blocked: The subset the mask rejected; these tokens could not have
            been emitted no matter how highly the model preferred them.
        emitted: The highest-ranked legal candidate, the token actually
            emitted.
    """

    field_name: str
    ranked_candidates: list[str]
    legal: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    emitted: str = ""


def generate_masked(grammar: Grammar, preference_stream: list[list[str]]) -> tuple[dict[str, Any], list[MaskStep]]:
    """Walk a compiled grammar, masking each step's ranked preferences down to the legal set.

    At each field, the ranked candidate list is intersected with that
    state's legal tokens; the highest-ranked legal candidate is emitted,
    every illegal candidate offered that step is recorded as blocked, and
    the FSM advances to the next field. There is no backtracking: masking
    guarantees every emitted token is legal, so the walk always reaches an
    accepting state.

    Args:
        grammar: The compiled grammar to walk.
        preference_stream: One ranked candidate list per field, in the same
            order as `grammar.slots`, standing in for a model's per-step
            ranked next-token preferences, highest-preference first.

    Returns:
        (arguments, steps): the accepted arguments object, values parsed to
        int or float for number fields and kept as-is for enum fields, plus
        the full per-step masking record.

    Raises:
        ValueError: If a step's ranked candidates contain no legal token at
            all, meaning the scripted stream never offered a way forward.
    """
    arguments: dict[str, Any] = {}
    steps: list[MaskStep] = []
    for slot, ranked in zip(grammar.slots, preference_stream):
        legal = [tok for tok in ranked if _is_legal(slot, tok)]
        blocked = [tok for tok in ranked if not _is_legal(slot, tok)]
        if not legal:
            raise ValueError(f"no legal candidate offered for field {slot.name!r} among {ranked!r}")
        emitted = legal[0]
        if slot.kind == "number":
            arguments[slot.name] = float(emitted) if "." in emitted else int(emitted)
        else:
            arguments[slot.name] = emitted
        steps.append(MaskStep(slot.name, list(ranked), legal, blocked, emitted))
    return arguments, steps


def generate_unconstrained(grammar: Grammar, preference_stream: list[list[str]]) -> dict[str, Any]:
    """Take the top-ranked candidate at every step with no legality mask applied.

    Uses the identical `preference_stream` `generate_masked` receives, so
    the only difference between the two paths is whether the mask runs;
    this isolates what masking bought. Values are kept as whatever string
    the top preference was, unparsed, since an unconstrained decode has no
    guarantee the top choice is even a well-formed number.

    Args:
        grammar: The compiled grammar (used only for field order and names).
        preference_stream: Same shape as `generate_masked`'s argument.

    Returns:
        The arguments object an unmasked argmax decode would have produced.
    """
    return {slot.name: ranked[0] for slot, ranked in zip(grammar.slots, preference_stream)}


def demo_constrained_decoding() -> tuple[dict[str, Any], list[MaskStep]]:
    """Mask a scripted preference stream against convert_currency's schema, then contrast with an unmasked decode.

    The stream seeds one illegal top preference per field: a non-numeric
    word ("hundred") ranked above the legal digits for `amount`, and a
    currency outside the enum ("JPY") ranked above the legal "EUR" for
    `currency`. The masked path blocks both and emits {"amount": 100,
    "currency": "EUR"}, an arguments object `validate_arguments` accepts
    with zero repair turns. The unmasked path takes the top preference
    regardless of legality and emits {"amount": "hundred", "currency":
    "JPY"}; `validate_arguments` rejects `amount`'s type on the spot, the
    exact repair round trip `validation.py`'s `demo_structural_repair`
    pays for a structurally invalid call, avoided here by masking generation
    instead of repairing after the fact.
    """
    grammar = compile_grammar(CONVERT_SCHEMA)
    preference_stream = [
        ["hundred", "100"],
        ["JPY", "EUR"],
    ]

    masked_arguments, steps = generate_masked(grammar, preference_stream)
    unmasked_arguments = generate_unconstrained(grammar, preference_stream)

    masked_errors = validate_arguments(CONVERT_SCHEMA, masked_arguments)
    unmasked_errors = validate_arguments(CONVERT_SCHEMA, unmasked_arguments)
    total_blocked = sum(len(step.blocked) for step in steps)

    print("=== 12. Constrained decoding: schema-grammar token masking ===")
    print(f"schema: {CONVERT_SCHEMA['properties']}")
    for step in steps:
        print(
            f"  field={step.field_name}: ranked={step.ranked_candidates} "
            f"blocked={step.blocked} emitted={step.emitted!r}"
        )
    print(f"masked call:   {masked_arguments} (validate_arguments errors: {masked_errors or 'none'})")
    print(f"unmasked call: {unmasked_arguments} (validate_arguments errors: {unmasked_errors})")
    print(f"tokens blocked: {total_blocked}, enum legal-sets precomputed at compile time: {grammar.precheck_count}")
    print()
    return masked_arguments, steps


if __name__ == "__main__":
    demo_constrained_decoding()
