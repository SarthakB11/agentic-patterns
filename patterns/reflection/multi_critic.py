"""Sub-module: parallel specialist critics with an aggregation policy.

Every other variant in this folder runs exactly one critic. This one runs
several critics with independent lenses (correctness, style, safety) on the
same draft, in parallel with no cross-talk, and aggregates their verdicts
into one `Critique` before a single refine step. One critic blurs
independent quality axes; the aggregation policy is a real design choice:

- ``veto``: approved only if every critic approves; the combined score is
  the minimum score present, so one harsh lens drags the verdict down.
- ``mean``: the combined score is the average of the scores present,
  approved once that average clears a threshold.
- ``weighted``: the combined score is a weight-normalized sum, so a
  heavily weighted lens (for example safety) can drag the aggregate below
  threshold even when the other lenses score high.

Each per-lens critique is kept verbatim, tagged with its lens name, and
concatenated into the merged critique the refine step reads, so one rewrite
addresses every axis at once.

See N-Critics (Mousavi et al., arXiv:2310.18679) for the ensemble idea and
its reported plateau beyond about four critics, which is why the ensembles
here stay small; and Generative Verifiers (Zhang et al., arXiv:2408.15240)
for aggregating several verdicts instead of trusting one. Distinct from
multi-agent debate (`patterns/multi_agent/`): these critics never see each
other's output or argue, they are independent judgments combined by fixed
arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agentic_patterns import Provider, get_provider

from patterns.reflection.loop import Critique, ReflectionResult, run_reflection_loop
from patterns.reflection.prompting import make_critique, make_generate, make_refine

_TASK = (
    "Write a two-sentence changelog entry announcing that the export "
    "feature now supports CSV, for a public-facing release notes page."
)

_GENERATOR_SYSTEM = "You write terse, accurate release notes. Reply with only the entry."
_CORRECTNESS_SYSTEM = (
    "You are a correctness reviewer with no attachment to this draft; you "
    "did not write it. Check only whether every claim is checkable (no "
    "vague adjectives). SCORE out of 10 and comments; start with APPROVED "
    "if nothing needs fixing."
)
_STYLE_SYSTEM = (
    "You are a style reviewer with no attachment to this draft; you did "
    "not write it. Check only for terseness and release-notes tone. SCORE "
    "out of 10 and comments; start with APPROVED if nothing needs fixing."
)
_SAFETY_SYSTEM = (
    "You are a safety and policy reviewer with no attachment to this "
    "draft; you did not write it. Check only whether the entry implies an "
    "unverified security or compliance claim. SCORE out of 10 and "
    "comments; start with APPROVED if nothing needs fixing."
)

_GENERATOR_SCRIPT = [
    "The export feature now securely exports your data to CSV. Try it today!",
    "The export feature now supports exporting data to CSV, in addition to the existing formats.",
]
_UNVERIFIED_CLAIM_NOTE = "Unverified security implication in 'securely exports'."


@dataclass
class CriticLens:
    """One specialist critic in the ensemble.

    Attributes:
        name: Short label tagging this lens's comments in the merged critique.
        provider: The `Provider` playing this critic, scripted independently.
        system: System prompt describing this lens's persona and standard.
        weight: Relative importance under the "weighted" policy; ignored by
            "veto" and "mean".
    """

    name: str
    provider: Provider
    system: str
    weight: float = 1.0


def _aggregate(per_lens: list[tuple[str, Critique]], *, policy: str, threshold: float) -> Critique:
    """Combine per-lens critiques into one aggregate `Critique`.

    Args:
        per_lens: (lens name, weight, that lens's `Critique`) triples.
        policy: One of "veto", "mean", "weighted".
        threshold: Score threshold "mean" and "weighted" use to set
            `approved`. Ignored by "veto", which derives `approved` from
            the per-critic votes instead.
    """
    comments = "\n".join(f"[{name}] {crit.comments}" for name, _w, crit in per_lens)
    scored = [(w, crit.score) for _n, w, crit in per_lens if crit.score is not None]

    if policy == "veto":
        approved = all(crit.approved for _n, _w, crit in per_lens)
        score = min(s for _w, s in scored) if scored else None
    elif policy == "mean":
        score = sum(s for _w, s in scored) / len(scored) if scored else None
        approved = score is not None and score >= threshold
    elif policy == "weighted":
        weight_total = sum(w for w, _s in scored)
        score = sum(w * s for w, s in scored) / weight_total if weight_total else None
        approved = score is not None and score >= threshold
    else:
        raise ValueError(f"Unknown aggregation policy: {policy!r}")

    return Critique(comments=comments, score=score, approved=approved)


def make_multi_critic(
    lenses: list[CriticLens], task: str, *, policy: str = "veto", threshold: float = 8.0
) -> Callable[[str], Critique]:
    """Build a `critique` callable that fans a draft out to every lens and aggregates.

    Args:
        lenses: Specialist critics run independently on every draft, each
            shown the draft as an external submission (same self-correction
            blind-spot workaround as `generator_critic.py`).
        task: The original task description, given to every critic.
        policy: Aggregation policy: "veto", "mean", or "weighted".
        threshold: Score threshold "mean" and "weighted" use to set the
            aggregate's `approved` flag.

    Returns:
        A callable usable as `run_reflection_loop`'s `critique` argument;
        nothing else about the loop changes.
    """

    def critique(draft: str) -> Critique:
        per_lens: list[tuple[str, float, Critique]] = []
        for lens in lenses:
            lens_critique = make_critique(lens.provider, task, system=lens.system, external_framing=True)
            per_lens.append((lens.name, lens.weight, lens_critique(draft)))
        return _aggregate(per_lens, policy=policy, threshold=threshold)

    return critique


def _build_lenses(safety_provider: Provider, safety_weight: float) -> list[CriticLens]:
    """Build the correctness/style/safety ensemble for both demos below."""
    correctness = get_provider(
        script=["SCORE: 9\nThe CSV claim is concrete and checkable.", "APPROVED: yes\nSCORE: 9\nStill checkable."]
    )
    style = get_provider(script=["SCORE: 8\nTerse and on-tone.", "APPROVED: yes\nSCORE: 9\nTerse, no marketing."])
    return [
        CriticLens("correctness", correctness, _CORRECTNESS_SYSTEM),
        CriticLens("style", style, _STYLE_SYSTEM),
        CriticLens("safety", safety_provider, _SAFETY_SYSTEM, weight=safety_weight),
    ]


def run_multi_critic_demo(
    generator_provider: Provider | None = None, lens_providers: dict[str, Provider] | None = None
) -> ReflectionResult:
    """Run a veto-policy ensemble: one lens's rejection blocks approval.

    Round one: correctness and style approve, but safety flags an
    unverified security implication and rejects; under "veto" the aggregate
    is not approved and the score is the minimum present (safety's 3),
    regardless of the other two scoring 9 and 8. Round two: the revision
    drops the claim, all three approve, and the loop stops.

    Args:
        generator_provider: Drives generate and refine. Defaults to a
            `MockProvider` scripted with a draft that oversells security
            and a revision that does not.
        lens_providers: Maps lens name to a `Provider`, overriding the
            default scripted ensemble. Passing this skips the default
            construction entirely.
    """
    if generator_provider is None:
        generator_provider = get_provider(script=_GENERATOR_SCRIPT)
    if lens_providers is None:
        safety = get_provider(
            script=[f"SCORE: 3\n{_UNVERIFIED_CLAIM_NOTE}", "APPROVED: yes\nSCORE: 9\nNo unverified claim remains."]
        )
        lenses = _build_lenses(safety, safety_weight=1.0)
    else:
        lenses = [
            CriticLens("correctness", lens_providers["correctness"], _CORRECTNESS_SYSTEM),
            CriticLens("style", lens_providers["style"], _STYLE_SYSTEM),
            CriticLens("safety", lens_providers["safety"], _SAFETY_SYSTEM),
        ]

    generate = make_generate(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    critique = make_multi_critic(lenses, _TASK, policy="veto")
    refine = make_refine(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    return run_reflection_loop(generate, critique, refine, max_iterations=3)


def run_weighted_flip_demo(generator_provider: Provider | None = None) -> ReflectionResult:
    """Run a weighted-policy ensemble where a heavily weighted lens flips the verdict.

    Correctness scores 9 and style scores 8, which would clear an 8.0
    threshold under "mean" (average 8.5). But safety is weighted 3x the
    others and scores 3, so the weight-normalized aggregate
    ``(9*1 + 8*1 + 3*3) / 5 = 5.2`` falls below threshold: the loop keeps
    refining instead of stopping approved, even though two of three lenses
    were satisfied. This is the design point of a weighted policy: a
    safety-critical lens can veto by weight without an explicit veto rule.

    Args:
        generator_provider: Drives generate and refine. Defaults to a
            `MockProvider` scripted with an oversold draft and a corrected
            revision, matching `run_multi_critic_demo`.
    """
    if generator_provider is None:
        generator_provider = get_provider(script=_GENERATOR_SCRIPT)
    safety = get_provider(
        script=[f"SCORE: 3\n{_UNVERIFIED_CLAIM_NOTE}", "APPROVED: yes\nSCORE: 9\nNo unverified claim remains."]
    )
    lenses = _build_lenses(safety, safety_weight=3.0)

    generate = make_generate(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    critique = make_multi_critic(lenses, _TASK, policy="weighted", threshold=8.0)
    refine = make_refine(generator_provider, _TASK, system=_GENERATOR_SYSTEM)
    return run_reflection_loop(generate, critique, refine, max_iterations=3)
