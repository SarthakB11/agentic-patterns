"""Programmable privilege control with monotonic narrowing (Progent-lite).

`pretool_guard.PreToolGuard` hard-codes one policy shape in imperative
Python: a fixed dict of per-tool numeric ranges and an approval threshold,
written once by whoever built the guard. Shi et al.'s Progent, "Securing AI
Agents with Privilege Control" (arXiv:2504.11703), makes the policy
declarative data instead: an ordered list of rules over tool names and
argument predicates, evaluated by a deterministic procedure that never
consults a model. The policy can be generated per task by an LLM and
tightened as a run learns more, but every update is classified as either
narrowing (the new allowed set is a subset of the old one, auto-applied) or
expansion (adds any privilege, and always needs a human), so an agent can
freely restrict its own privileges mid-run but can never grant itself a new
one. `pretool_guard.PreToolGuard` is the simplest point on this design
space, a single fixed policy; this module is the general one: policy as
data, checked by code, narrowed or expanded only by declared updates.

Simplified from the real system: Progent classifies narrowing versus
expansion with an SMT solver, needed for arbitrary predicates. Over the
finite, enumerable predicate set here (equals, one-of, in-range,
prefix-match), subset comparison decides the same classification exactly,
with no solver dependency; `classify_update` says so in its docstring.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentic_patterns import ToolCall

from patterns.guardrails.core import DecisionLog, GuardResult, OnFail, run_guard


@dataclass(frozen=True)
class Predicate:
    """A pure, serializable test on one argument's value.

    Attributes:
        kind: One of "equals", "one_of", "in_range", "prefix".
        params: Parameters for the kind, e.g. `{"low": 0, "high": 100}` for "in_range".
    """

    kind: str
    params: dict[str, Any]

    def matches(self, value: Any) -> bool:
        """Test `value` against this predicate."""
        if self.kind == "equals":
            return value == self.params["value"]
        if self.kind == "one_of":
            return value in self.params["values"]
        if self.kind == "in_range":
            return self.params["low"] <= value <= self.params["high"]
        if self.kind == "prefix":
            return isinstance(value, str) and value.startswith(self.params["prefix"])
        raise ValueError(f"unknown predicate kind: {self.kind!r}")

    def allowed_set_subset_of(self, other: "Predicate") -> bool:
        """Whether every value this predicate allows is also allowed by `other`.

        Decides narrowing versus expansion by pure subset comparison over
        this module's finite predicate kinds, in place of the SMT solver a
        production policy language would need for arbitrary predicates.
        Predicates of different kinds are not comparable (returns False).
        """
        if self.kind == "equals" and other.kind == "equals":
            return self.params["value"] == other.params["value"]
        if self.kind == "equals" and other.kind == "one_of":
            return self.params["value"] in other.params["values"]
        if self.kind == "one_of" and other.kind == "one_of":
            return set(self.params["values"]) <= set(other.params["values"])
        if self.kind == "in_range" and other.kind == "in_range":
            return self.params["low"] >= other.params["low"] and self.params["high"] <= other.params["high"]
        if self.kind == "prefix" and other.kind == "prefix":
            return self.params["prefix"].startswith(other.params["prefix"])
        return False


@dataclass(frozen=True)
class Rule:
    """A tool, constraints on its arguments, and an effect ("allow" or "deny").

    An argument absent from a call is unconstrained by that rule; it does
    not disqualify a match.
    """

    tool_name: str
    arg_constraints: dict[str, Predicate] = field(default_factory=dict)
    effect: str = "allow"

    def matches(self, call: ToolCall) -> bool:
        """Whether `call` satisfies every constraint this rule declares."""
        if call.name != self.tool_name:
            return False
        return all(
            predicate.matches(call.arguments[arg]) for arg, predicate in self.arg_constraints.items() if arg in call.arguments
        )


@dataclass
class Policy:
    """An ordered rule list, evaluated first-match-wins, plus a default-deny effect."""

    rules: list[Rule] = field(default_factory=list)
    default_effect: str = "deny"


def evaluate(policy: Policy, call: ToolCall) -> GuardResult:
    """Decide one tool call: first matching rule wins, else default deny.

    A pure function over data: no model judgment enters enforcement, only
    the ordered rule walk and each rule's predicates.
    """
    for rule in policy.rules:
        if rule.matches(call):
            if rule.effect == "allow":
                return GuardResult(passed=True, action=OnFail.NOOP, value=call, message=f"allowed by rule for {rule.tool_name!r}")
            return GuardResult(passed=False, action=OnFail.REFRAIN, value=call, message=f"denied by rule for {rule.tool_name!r}")
    return GuardResult(passed=False, action=OnFail.REFRAIN, value=call, message=f"no rule allows {call.name!r}: default deny")


@dataclass(frozen=True)
class PolicyUpdate:
    """A proposed rule to add or tighten for one tool."""

    rule: Rule


def classify_update(policy: Policy, update: PolicyUpdate) -> str:
    """Classify a policy update as "narrowing" or "expansion".

    No existing allow rule for the update's tool, or an existing rule whose
    allowed set is not a superset of the new rule's, counts as expansion:
    it grants a privilege the policy did not already have. An update whose
    allowed set is a subset of every existing constraint for that tool
    counts as narrowing. Decided by `Predicate.allowed_set_subset_of`, pure
    subset comparison, not an SMT call; see the module docstring.
    """
    existing = [r for r in policy.rules if r.tool_name == update.rule.tool_name and r.effect == "allow"]
    if not existing:
        return "expansion"
    for arg, new_predicate in update.rule.arg_constraints.items():
        covered = any(
            (old_rule.arg_constraints.get(arg) is not None and new_predicate.allowed_set_subset_of(old_rule.arg_constraints[arg]))
            for old_rule in existing
        )
        if not covered:
            return "expansion"
    return "narrowing"


def _replace_tool_rule(rules: list[Rule], new_rule: Rule) -> list[Rule]:
    """Replace any existing allow rule for `new_rule`'s tool, so the old wider bound cannot still match."""
    kept = [r for r in rules if not (r.tool_name == new_rule.tool_name and r.effect == "allow")]
    return [new_rule, *kept]


def apply_update(
    policy: Policy,
    update: PolicyUpdate,
    log: DecisionLog,
    *,
    human_approve: Callable[[PolicyUpdate, str], bool] | None = None,
) -> Policy:
    """Apply a policy update: auto-apply a narrowing, gate an expansion on human approval.

    Args:
        policy: The current policy. Not mutated; a new `Policy` is returned.
        update: The proposed rule change.
        log: Decision log to record the classification and outcome into.
        human_approve: Called with `(update, classification)` for an
            expansion. Returns True to approve. An expansion with no
            callback, or a denied one, leaves `policy` unchanged.
    """
    classification = classify_update(policy, update)
    replaced = Policy(rules=_replace_tool_rule(policy.rules, update.rule), default_effect=policy.default_effect)

    if classification == "narrowing":
        log.record("policy_engine", GuardResult(passed=True, action=OnFail.FIX, value=update.rule, message="narrowing update auto-applied"))
        return replaced

    approved = human_approve(update, classification) if human_approve is not None else False
    if approved:
        log.record("policy_engine", GuardResult(passed=True, action=OnFail.FIX, value=update.rule, message="expansion update approved by human"))
        return replaced

    log.record("policy_engine", GuardResult(passed=False, action=OnFail.REFRAIN, value=update.rule, message="expansion update denied; policy unchanged"))
    return policy


def generate_policy_from_task(policy_json: str) -> Policy:
    """Parse an LLM-authored policy (a JSON rule list) into a `Policy`.

    This is the "LLM writes the policy, code enforces it" split Progent
    depends on: nothing about how rules are evaluated trusts the model, and
    a rule the model tries to smuggle in still has to pass `evaluate`'s
    ordered walk, including a hard deny placed ahead of it by the caller.

    Args:
        policy_json: A JSON array of objects with "tool", "effect", and
            optionally "arg_constraints" (argument to `{"kind", "params"}`).
    """
    raw_rules = json.loads(policy_json)
    rules = [
        Rule(
            tool_name=entry["tool"],
            arg_constraints={a: Predicate(kind=s["kind"], params=s["params"]) for a, s in entry.get("arg_constraints", {}).items()},
            effect=entry["effect"],
        )
        for entry in raw_rules
    ]
    return Policy(rules=rules)


class PolicyGuard:
    """A `Guard`-shaped wrapper around a `Policy`, for use with `run_guard`."""

    def __init__(self, policy: Policy, name: str = "policy_engine") -> None:
        self.name = name
        self.policy = policy

    def check(self, value: ToolCall) -> GuardResult:
        return evaluate(self.policy, value)


def run_policy_engine_demo() -> tuple[Policy, list[GuardResult]]:
    """Walk a policy through default deny, narrowing, denied expansion, and an LLM-authored policy.

    An unlisted tool is default-denied; `issue_refund` is allowed up to
    $100, so $101 is denied; the bound narrows to $50 without approval and
    immediately enforces; widening to $500 needs approval and is denied,
    leaving the $50 bound in force; an LLM-authored policy that lists an
    "allow" for `delete_account` is still blocked because the caller placed
    a hard "deny" rule for it ahead of the model's own rule.

    Returns:
        The final policy and every `evaluate` result made along the way.
    """
    log = DecisionLog()
    policy = Policy(rules=[Rule(tool_name="issue_refund", arg_constraints={"amount": Predicate("in_range", {"low": 0, "high": 100})})])
    guard = PolicyGuard(policy)

    print("=== Policy engine: declarative privilege control with monotonic narrowing ===")
    steps: list[tuple[str, ToolCall]] = [
        ("delete_account (not in policy)", ToolCall(id="c1", name="delete_account", arguments={"user_id": "u1"})),
        ("issue_refund(amount=100)", ToolCall(id="c2", name="issue_refund", arguments={"amount": 100})),
        ("issue_refund(amount=101)", ToolCall(id="c3", name="issue_refund", arguments={"amount": 101})),
    ]
    results: list[GuardResult] = []
    for label, call in steps:
        result = run_guard(guard, call, log)
        print(f"  {label} -> passed={result.passed}: {result.message}")
        results.append(result)

    narrow_update = PolicyUpdate(rule=Rule(tool_name="issue_refund", arg_constraints={"amount": Predicate("in_range", {"low": 0, "high": 50})}))
    print(f"  update: tighten amount to <=50 -> classified {classify_update(policy, narrow_update)!r}")
    policy = guard.policy = apply_update(policy, narrow_update, log)
    result = run_guard(guard, ToolCall(id="c4", name="issue_refund", arguments={"amount": 75}), log)
    print(f"  issue_refund(amount=75) after narrowing -> passed={result.passed}: {result.message}")
    results.append(result)

    expand_update = PolicyUpdate(rule=Rule(tool_name="issue_refund", arg_constraints={"amount": Predicate("in_range", {"low": 0, "high": 500})}))
    print(f"  update: widen amount to <=500 -> classified {classify_update(policy, expand_update)!r}")
    policy = guard.policy = apply_update(policy, expand_update, log, human_approve=lambda u, c: False)
    result = run_guard(guard, ToolCall(id="c5", name="issue_refund", arguments={"amount": 200}), log)
    print(f"  issue_refund(amount=200) after denied expansion -> passed={result.passed}: {result.message}")
    results.append(result)

    llm_policy_json = json.dumps(
        [{"tool": "delete_account", "effect": "deny"}, {"tool": "issue_refund", "effect": "allow"}, {"tool": "delete_account", "effect": "allow"}]
    )
    llm_guard = PolicyGuard(generate_policy_from_task(llm_policy_json), name="llm_authored_policy")
    result = run_guard(llm_guard, ToolCall(id="c6", name="delete_account", arguments={"user_id": "u2"}), log)
    print(f"  LLM policy tries to allow delete_account, but the hard deny rule is checked first -> passed={result.passed}: {result.message}")
    results.append(result)

    return policy, results
