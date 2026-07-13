"""Semantic similarity evaluator: reference comparison by embedding distance.

Catches paraphrases that exact match misses, at the cost of still being
reference-dependent and blind to open-ended quality it has no reference for.
Uses the repo's `HashEmbedder` and `cosine_similarity`, the same stand-in
embedder the RAG and memory patterns use, so this module needs no network
call and no trained model to run offline.
"""

from __future__ import annotations

from agentic_patterns import Embedder, cosine_similarity, get_embedder

from patterns.evaluation.eval_set import EvalCase
from patterns.evaluation.exact import Score

DEFAULT_SIMILARITY_THRESHOLD = 0.3


def semantic_similarity_evaluator(
    case: EvalCase,
    output: str,
    *,
    embedder: Embedder | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Score:
    """Score `output` by cosine similarity to `case.reference`.

    Args:
        case: The eval case. `case.reference` must be set.
        output: The candidate output to compare against the reference.
        embedder: Embedder to use. Defaults to `get_embedder()`, which
            resolves to the deterministic `HashEmbedder` unless
            `AGENTIC_PATTERNS_EMBEDDER` selects a real one.
        threshold: Minimum cosine similarity to count as a pass.

    Raises:
        ValueError: If `case.reference` is None.
    """
    if case.reference is None:
        raise ValueError(f"Case {case.id!r} has no reference to compare against")
    if embedder is None:
        embedder = get_embedder()

    output_vec, reference_vec = embedder.embed([output, case.reference])
    similarity = cosine_similarity(output_vec, reference_vec)
    passed = similarity >= threshold
    detail = f"cosine similarity {similarity:.3f} {'>=' if passed else '<'} threshold {threshold:.3f}"
    return Score(case_id=case.id, evaluator="semantic_similarity", passed=passed, detail=detail)
