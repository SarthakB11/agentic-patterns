"""Decision-model judging: a calibrated probability instead of a generated verdict.

Sources: TypeSafe AI, "Introducing System One Models & Jev" (Sep 15, 2026,
https://typesafe.ai/blog/introducing-system-one-models-and-jev) and the
noul primitive docs (https://docs.typesafe.ai/primitives/noul.md); Yubo Li,
Yidi Miao, Ramayya Krishnan, Rema Padman, "JEV-as-a-Judge: Accept When
Confident, Escalate When Unsure," arXiv:2609.26550 (v1 Sep 22, 2026, v6 Oct
29, 2026); Arize, "Jev vs. LLM-as-a-Judge" (Sep 2026,
https://arize.com/blog/jev-as-a-judge.md).

Every other judge in this folder generates text (reasoning, then a SCORE or
VERDICT line) and parses a verdict out of it. A decision model generates no
text. It is asked a yes/no question about a piece of state and returns one
number: the probability that the answer is yes, which TypeSafe calls a
"noul". That changes three things about using it as a judge:

- It is cheap and fast enough to grade everything, at a small fraction of a
  reasoning judge's cost, so the design question moves from "can we afford
  to judge this" to "which cases does it get wrong".
- A probability needs a cutoff. Arize found the default 0.5 cutoff left Jev
  at 76% against Opus 5's 83% on RAGTruth, and tuning the cutoff to 0.80 on
  held-out labels brought it level at 87%. `tune_cutoff` is that step.
- The probability doubles as a confidence for an accept-or-escalate
  cascade. Li et al. accept the decision model's verdict when its
  confidence clears a threshold `tau` and otherwise hand the case to a
  reasoning judge whose verdict replaces it. `run_decision_cascade` is that
  cascade, reusing this folder's pointwise LLM judge as the fallback.

Confidence here is the probability of the label the cutoff picked: `p` for
a pass, `1 - p` for a fail. At the default cutoff of 0.5 that equals the
paper's "largest label probability". With a tuned cutoff it stays
consistent: at cutoff 0.8, `p = 0.75` is a fail held at confidence 0.25,
which escalates, rather than a fail held at a misleading 0.75. One
consequence worth knowing: whenever `1 - tau < cutoff <= tau`, the cascade
accepts exactly the cases with `p >= tau` (pass) or `p <= 1 - tau` (fail),
so its final verdicts do not depend on the cutoff at all. A tuned cutoff
changes the decision model's own verdict on the unsure middle band, which
the cascade escalates anyway; it matters when the decision model runs
alone, or when `tau` is low enough to accept part of that band.

Boundary notes: `selective.py` gets confidence by resampling an LLM judge;
this module gets it from the decision model in one call, so the two are
two ways to fill the same `tau` slot. The paper's prospective threshold
rule (a one-sided lower confidence bound against the fallback) is left out
to keep the module readable in one sitting; `tune_cutoff` picks a cutoff by
plain accuracy on a labeled slice. Routing a case across model tiers in
general is `patterns/routing/`; here the fallback is an abstract judge
callable.

Offline, `ScriptedDecisionModel` replays fixed probabilities, the way
`MockProvider` replays completions. Set `AGENTIC_PATTERNS_DECISION_MODEL=
typesafe` with `TYPESAFE_API_KEY` to call Jev for real through
`TypeSafeDecisionModel` (base URL and model id are overridable, so the same
class reaches the OpenRouter route).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agentic_patterns import MockProvider
from patterns.evaluation.eval_set import EvalCase, get_case
from patterns.evaluation.meta import auroc, brier_score, expected_calibration_error
from patterns.evaluation.pointwise import build_pointwise_judge
from patterns.evaluation.verdict import Verdict

DEFAULT_QUESTION = "Does the reply correctly and directly answer the user's question?"
DEFAULT_CUTOFF = 0.5
DEFAULT_TAU = 0.9  # the paper's offline operating point
TYPESAFE_BASE_URL = "https://api.typesafe.ai"
TYPESAFE_MODEL = "jev-1.13.0"  # pinned: an alias like jev-latest moves under you
_ANSWER_KEY = "verdict"


def _as_probability(value: object) -> float:
    """Return `value` as a float in [0, 1], or raise.

    Rejects what `float()` would quietly accept: booleans (`True` would
    become a confident 1.0), strings, None, and NaN (which slips past a
    `p < 0 or p > 1` check because every comparison with NaN is False).

    Raises:
        ValueError: If `value` is not a real number in [0, 1].
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"probability must be a number, got {value!r}")
    p = float(value)
    if not (0.0 <= p <= 1.0):  # also False for NaN
        raise ValueError(f"probability {p} is outside [0, 1]")
    return p


class DecisionModel(Protocol):
    """Anything that answers a yes/no question about some state with P(yes)."""

    def noul(self, state: str, instructions: str) -> float:
        """Return the probability, in [0, 1], that the answer to `instructions` about `state` is yes."""
        ...


class DecisionScriptExhausted(RuntimeError):
    """Raised when a scripted decision model is asked more questions than it was scripted for."""


class ScriptedDecisionModel:
    """Offline decision model: replays a fixed list of probabilities and records every question.

    The decision-model counterpart of `MockProvider`. `calls` keeps each
    `(state, instructions)` pair so tests can assert on what was asked.
    """

    def __init__(self, probabilities: Sequence[float]) -> None:
        self._script = [_as_probability(p) for p in probabilities]
        self._index = 0
        self.calls: list[tuple[str, str]] = []

    def noul(self, state: str, instructions: str) -> float:
        """Pop the next scripted probability."""
        if self._index >= len(self._script):
            raise DecisionScriptExhausted(
                f"Decision script exhausted after {len(self._script)} answer(s); "
                "the example asked more questions than expected."
            )
        self.calls.append((state, instructions))
        p = self._script[self._index]
        self._index += 1
        return p


def build_noul_request(model: str, state: str, instructions: str) -> dict[str, Any]:
    """Build the JSON body for one noul question on TypeSafe's `/v1/systemone` endpoint.

    Pure function, unit-tested without a network call, like the wire-format
    converters for the chat providers in `agentic_patterns.core`.
    """
    return {
        "model": model,
        "state": state,
        "questions": {_ANSWER_KEY: {"type": "noul", "instructions": instructions}},
    }


def parse_noul_response(payload: dict[str, Any]) -> float:
    """Extract P(yes) from a `/v1/systemone` response body.

    Fails closed: a missing answer, a wrong answer type, or a value that is
    not a real number in [0, 1] (a string, a boolean, null, NaN) raises
    rather than silently becoming a pass or a fail.

    Raises:
        ValueError: If the payload does not carry a valid noul answer.
    """
    answers = payload.get("answers")
    answer = answers.get(_ANSWER_KEY) if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("type") != "noul" or "noul" not in answer:
        raise ValueError(f"response carries no noul answer under {_ANSWER_KEY!r}: {payload!r}")
    return _as_probability(answer["noul"])


class TypeSafeDecisionModel:
    """Live decision model over TypeSafe's System One HTTP API.

    One POST per question to `{base_url}/v1/systemone` with a bearer key.
    `httpx` is imported lazily so the offline path never needs it. Token
    usage reported by the API accumulates in `input_tokens` and
    `output_tokens` so a caller can price a run; a response with no usage
    field is counted in `calls_without_usage` instead of being priced as
    free without notice. There is no retry: a 429 or 5xx raises, which is
    the right default for teaching code and the wrong one for a batch job.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = TYPESAFE_MODEL,
        base_url: str = TYPESAFE_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.calls_without_usage = 0

    def noul(self, state: str, instructions: str) -> float:
        """Ask one yes/no question and return P(yes)."""
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                'TypeSafeDecisionModel requires httpx: pip install "agentic-patterns[providers]"'
            ) from exc
        response = httpx.post(
            f"{self.base_url}/v1/systemone",
            json=build_noul_request(self.model, state, instructions),
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        p_yes = parse_noul_response(payload)
        usage = payload.get("usage")
        if isinstance(usage, dict) and "input_tokens" in usage:
            self.input_tokens += int(usage.get("input_tokens") or 0)
            self.output_tokens += int(usage.get("output_tokens") or 0)
        else:
            self.calls_without_usage += 1
        self.calls += 1
        return p_yes


def get_decision_model(script: Sequence[float] | None = None, name: str | None = None) -> DecisionModel:
    """Build a decision model selected by name, mirroring `agentic_patterns.get_provider`.

    Args:
        script: Probabilities for `ScriptedDecisionModel`; required when the
            resolved name is "mock".
        name: "mock" or "typesafe". Defaults to the
            `AGENTIC_PATTERNS_DECISION_MODEL` environment variable, else "mock".

    Raises:
        ValueError: On an unknown name, a mock with no script, or "typesafe"
            without `TYPESAFE_API_KEY` set.
    """
    resolved = name or os.environ.get("AGENTIC_PATTERNS_DECISION_MODEL", "mock")
    if resolved == "mock":
        if script is None:
            raise ValueError("the mock decision model requires a script of probabilities")
        return ScriptedDecisionModel(script)
    if resolved == "typesafe":
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise ValueError("AGENTIC_PATTERNS_DECISION_MODEL=typesafe needs TYPESAFE_API_KEY in the environment")
        return TypeSafeDecisionModel(
            key,
            model=os.environ.get("TYPESAFE_MODEL", TYPESAFE_MODEL),
            base_url=os.environ.get("TYPESAFE_BASE_URL", TYPESAFE_BASE_URL),
        )
    raise ValueError(f"unknown decision model {resolved!r}; expected 'mock' or 'typesafe'")


def format_state(case: EvalCase, output: str) -> str:
    """Render the case and candidate reply as the state a decision model reads.

    The reference answer is included when the case has one, the same
    reference-based mode `pointwise.py` uses: Arize and OpenRouter both
    report decision models doing best when the evidence is in the state.
    """
    parts = [f"Question:\n{case.input}"]
    if case.reference:
        parts.append(f"Reference answer:\n{case.reference}")
    parts.append(f"Reply:\n{output}")
    return "\n\n".join(parts)


@dataclass
class DecisionVerdict:
    """One case judged by a decision model, possibly escalated.

    Attributes:
        case_id: The `EvalCase.id` this verdict belongs to.
        p_yes: The decision model's probability that the reply passes.
        decision_passed: `p_yes >= cutoff`, the decision model's own verdict.
        confidence: Probability of the label the cutoff picked.
        accepted: True if `confidence >= tau`, so the decision model's
            verdict stands.
        escalated: True if the case went to the fallback judge.
        passed: The final verdict: the decision model's if accepted, the
            fallback's if escalated, None if neither was available.
        fallback: The fallback judge's verdict, when escalated.
    """

    case_id: str
    p_yes: float
    decision_passed: bool
    confidence: float
    accepted: bool
    escalated: bool
    passed: bool | None
    fallback: Verdict | None = None


def judge_with_decision_model(
    model: DecisionModel,
    case: EvalCase,
    output: str,
    *,
    cutoff: float = DEFAULT_CUTOFF,
    tau: float = DEFAULT_TAU,
    question: str = DEFAULT_QUESTION,
    escalate: Callable[[EvalCase, str], Verdict] | None = None,
) -> DecisionVerdict:
    """Judge one case: one decision-model call, then accept or escalate.

    Args:
        model: The decision model.
        case: The eval case.
        output: The candidate reply.
        cutoff: P(yes) at or above this is a pass.
        tau: Confidence at or above this accepts the decision model's verdict.
        question: The yes/no instruction the decision model answers.
        escalate: Fallback judge for unaccepted cases. If None, an
            unaccepted case keeps no final verdict.
    """
    p_yes = model.noul(format_state(case, output), question)
    decision_passed = p_yes >= cutoff
    confidence = p_yes if decision_passed else 1.0 - p_yes
    if confidence >= tau:
        return DecisionVerdict(case.id, p_yes, decision_passed, confidence, True, False, decision_passed)
    if escalate is None:
        return DecisionVerdict(case.id, p_yes, decision_passed, confidence, False, False, None)
    fallback = escalate(case, output)
    return DecisionVerdict(case.id, p_yes, decision_passed, confidence, False, True, fallback.passed, fallback)


@dataclass
class CascadeResult:
    """A batch judged by the accept-or-escalate cascade, scored against human labels.

    Attributes:
        verdicts: One `DecisionVerdict` per case, in input order.
        coverage: Fraction of cases the decision model settled alone.
        escalated_count: Cases sent to the fallback judge.
        decision_accuracy: The decision model alone at `cutoff`, every case.
        cascade_accuracy: Final verdicts against human labels; an unresolved
            case counts as wrong.
        auroc: Ranking quality of `p_yes` (None if labels hold one class).
        brier: Brier score of `p_yes`.
        ece: Expected calibration error of `p_yes`.
        cutoff: The pass cutoff used.
        tau: The acceptance threshold used.
    """

    verdicts: list[DecisionVerdict]
    coverage: float
    escalated_count: int
    decision_accuracy: float
    cascade_accuracy: float
    auroc: float | None
    brier: float
    ece: float
    cutoff: float
    tau: float


def run_decision_cascade(
    model: DecisionModel,
    cases_and_outputs: list[tuple[EvalCase, str]],
    human_labels: list[bool],
    *,
    cutoff: float = DEFAULT_CUTOFF,
    tau: float = DEFAULT_TAU,
    question: str = DEFAULT_QUESTION,
    escalate: Callable[[EvalCase, str], Verdict] | None = None,
) -> CascadeResult:
    """Judge a batch with the cascade and report accuracy, coverage, and calibration.

    Raises:
        ValueError: If `cases_and_outputs` and `human_labels` differ in length or are empty.
    """
    if not cases_and_outputs or len(cases_and_outputs) != len(human_labels):
        raise ValueError("cases_and_outputs and human_labels must be non-empty and the same length")
    verdicts = [
        judge_with_decision_model(model, case, output, cutoff=cutoff, tau=tau, question=question, escalate=escalate)
        for case, output in cases_and_outputs
    ]
    n = len(verdicts)
    probabilities = [v.p_yes for v in verdicts]
    ranking = auroc(probabilities, human_labels) if len(set(human_labels)) == 2 else None
    return CascadeResult(
        verdicts=verdicts,
        coverage=sum(v.accepted for v in verdicts) / n,
        escalated_count=sum(v.escalated for v in verdicts),
        decision_accuracy=sum(v.decision_passed == h for v, h in zip(verdicts, human_labels)) / n,
        cascade_accuracy=sum(v.passed is not None and v.passed == h for v, h in zip(verdicts, human_labels)) / n,
        auroc=ranking,
        brier=brier_score(probabilities, human_labels),
        ece=expected_calibration_error(probabilities, human_labels),
        cutoff=cutoff,
        tau=tau,
    )


def tune_cutoff(
    probabilities: list[float], labels: list[bool], candidates: Sequence[float] | None = None
) -> tuple[float, float]:
    """Pick the cutoff that maximizes accuracy on a labeled tuning slice.

    Candidates default to 0.05 through 0.95 in steps of 0.05. Ties go to
    the candidate nearest 0.5, so a well-calibrated model keeps the default;
    two candidates equally far from 0.5 (0.45 and 0.55) go to the one listed
    first. Tune on one slice and report on another: a cutoff scored on the
    slice it was picked from is an in-sample upper bound, not a result.

    Returns:
        `(cutoff, accuracy_on_the_tuning_slice)`.

    Raises:
        ValueError: If the lists are empty or differ in length, or
            `candidates` is empty.
    """
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("tune_cutoff requires non-empty probabilities and labels of the same length")
    grid = list(candidates) if candidates is not None else [round(0.05 * i, 2) for i in range(1, 20)]
    if not grid:
        raise ValueError("tune_cutoff requires at least one candidate cutoff")

    def accuracy(cut: float) -> float:
        return sum((p >= cut) == y for p, y in zip(probabilities, labels)) / len(labels)

    best = max(grid, key=lambda cut: (accuracy(cut), -abs(cut - 0.5)))
    return best, accuracy(best)


# --- demos -------------------------------------------------------------------


def run_decision_cascade_demo() -> CascadeResult:
    """Six replies judged by a scripted decision model, two escalated to the LLM judge.

    Four probabilities are decisive (0.97, 0.03, 0.95, 0.06) and accepted at
    `tau=0.9`. Two sit in the middle (0.62, 0.41) and escalate to the
    pointwise LLM judge, which is scripted for exactly those two calls, so
    the run proves the fallback is called only when the decision model is
    unsure. Both middle cases are ones the decision model leaned the wrong
    way on; the fallback corrects them, so the cascade reaches 6/6 where
    the decision model alone gets 4/6.
    """
    cases_and_outputs = [
        (get_case("refund_policy"), "Refunds are available within 30 days if you have your order number."),
        (get_case("refund_policy"), "We are open Monday through Friday."),
        (get_case("cancel_subscription"), "Go to Account Settings, select Subscription, and click Cancel."),
        (get_case("cancel_subscription"), "Subscriptions cannot be cancelled."),
        (get_case("order_status_lookup"), "Your order is on its way, please check back later."),
        (get_case("refund_investigation"), "I've opened a refund review; you'll hear back within 2 business days."),
    ]
    human_labels = [True, False, True, False, False, True]
    # Scripted directly, not via `get_decision_model`: the fallback below is
    # scripted for exactly two calls, so this demo is offline by design even
    # when AGENTIC_PATTERNS_DECISION_MODEL selects the live model.
    model = ScriptedDecisionModel([0.97, 0.03, 0.95, 0.06, 0.62, 0.41])
    fallback_provider = MockProvider(
        [
            "Does not name the order or a concrete status.\nSCORE: 3\nVERDICT: fail",
            "Acknowledges the request and gives a concrete next step and timeline.\nSCORE: 8\nVERDICT: pass",
        ]
    )
    # Same evidence as the decision model: `format_state` includes the
    # reference when a case has one, so the fallback judge does too.
    fallback = build_pointwise_judge(fallback_provider, reference_mode=True)
    return run_decision_cascade(model, cases_and_outputs, human_labels, escalate=fallback)


def run_cutoff_tuning_demo() -> tuple[CascadeResult, CascadeResult, float]:
    """Right ranking, wrong cutoff: tuning on labels fixes accuracy while AUROC stays put.

    The model ranks every good reply above every bad one (AUROC 1.0), but
    it scores bad replies around 0.6, so at the default 0.5 cutoff it
    passes them. Tuning on a four-case labeled slice moves the cutoff up;
    applied to four held-out cases with the same shape, accuracy goes from
    0.5 to 1.0 while AUROC does not move. AUROC said the model could tell
    good from bad; the cutoff was the problem, and Brier and ECE flag the
    miscalibration behind it.

    Returns:
        `(held_out_at_default, held_out_at_tuned, tuned_cutoff)`.
    """
    tuning_probabilities = [0.93, 0.62, 0.88, 0.58]
    tuning_labels = [True, False, True, False]
    cutoff, _ = tune_cutoff(tuning_probabilities, tuning_labels)

    held_out = [
        (get_case("refund_policy"), "Refunds within 30 days with an order number."),
        (get_case("refund_policy"), "Refunds are handled by the billing team."),
        (get_case("cancel_subscription"), "Account Settings, then Subscription, then Cancel."),
        (get_case("cancel_subscription"), "Please contact support to discuss your plan."),
    ]
    held_out_labels = [True, False, True, False]
    held_out_probabilities = [0.91, 0.64, 0.86, 0.6]
    at_default = run_decision_cascade(
        ScriptedDecisionModel(held_out_probabilities), held_out, held_out_labels, tau=0.0
    )
    at_tuned = run_decision_cascade(
        ScriptedDecisionModel(held_out_probabilities), held_out, held_out_labels, cutoff=cutoff, tau=0.0
    )
    return at_default, at_tuned, cutoff
