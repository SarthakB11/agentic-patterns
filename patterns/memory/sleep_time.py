"""Sleep-time compute: offline pre-derivation amortized across many queries.

`write_policy.consolidate` already makes one offline model call to reconcile
a contradiction, and used to call itself "sleep-time consolidation." Per
Sleep-time Compute (Lin, Snell, Wang, Packer, Wooders, Stoica, Gonzalez,
arXiv:2504.13171), that name belongs here instead: sleep-time compute is a
background pass that pre-derives a compact "learned context" of inferences
from raw memory once, before any query arrives, so that many later, unseen
queries answer from the derived context with less online work than deriving
per query from scratch. `consolidate`'s one-off contradiction reconcile
never amortizes across queries; this module is the multi-query accounting
`consolidate` never carried.

Two online paths are compared over the same raw context and the same list
of queries:

- **Path A, test-time only**: for each query, derive what it needs from raw
  context on the spot (one call), then answer from that derivation (one
  call). 2 online calls per query, `2n` total, no offline phase.
- **Path B, sleep-time**: one offline sleep pass pre-derives a learned
  context before any query arrives. Each query then answers directly from
  the learned context (one call), unless the sleep pass did not anticipate
  what it needs, in which case it falls back to Path A's per-query
  derive-then-answer for that one query (a fallback the model recognizes
  in text, since a scripted mock cannot introspect what a completion
  omitted).

The offline sleep pass is real compute, not free: it is counted once and
folded into Path B's total so a single query (`n=1`) shows no advantage,
matching the paper's efficacy-tracks-predictability caveat, i.e. amortization
requires that queries actually share a context and that the sleep pass
anticipates them, both properties this module makes visible rather than
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Message, Provider, get_provider

DERIVE_SYSTEM_PROMPT = "Derive only the specific fact(s) this query needs from the context. Be concise."
ANSWER_SYSTEM_PROMPT = "Answer the query using only the given context, in one sentence."
SLEEP_SYSTEM_PROMPT = (
    "Offline pre-derivation: read this raw context and derive a compact "
    "block of inferences (totals, resolved contradictions, entity "
    "summaries) that a later, unseen query is likely to need. Be concise."
)


@dataclass
class SleepTimeReport:
    """Outcome of comparing test-time-only and sleep-time paths.

    Attributes:
        learned_context: The offline sleep pass's pre-derived context.
        path_a_answers: Path A's answer per query, in query order.
        path_a_online_calls: Path A's total online call count, `2 * n`.
        path_b_answers: Path B's answer per query, in query order.
        path_b_online_calls: Path B's online call count: 1 per query the
            learned context covered, 2 per query that fell back.
        path_b_total_calls: `path_b_online_calls` plus the one offline
            sleep pass, so a fair comparison against Path A includes the
            cost Path B paid to build the learned context.
        fallback_queries: Queries the learned context did not cover, which
            fell back to on-the-spot derivation.
    """

    learned_context: str
    path_a_answers: list[str] = field(default_factory=list)
    path_a_online_calls: int = 0
    path_b_answers: list[str] = field(default_factory=list)
    path_b_online_calls: int = 0
    path_b_total_calls: int = 0
    fallback_queries: list[str] = field(default_factory=list)


def sleep_pass(provider: Provider, raw_context: str) -> str:
    """Run the offline sleep pass once: derive a learned context from raw
    memory before any query arrives.
    """
    completion = provider.complete([Message.user(raw_context)], system=SLEEP_SYSTEM_PROMPT)
    return completion.content.strip()


def _derive(provider: Provider, context: str, query: str) -> str:
    completion = provider.complete(
        [Message.user(f"Context: {context}\nQuery: {query}")], system=DERIVE_SYSTEM_PROMPT
    )
    return completion.content.strip()


def _answer(provider: Provider, context: str, query: str) -> str:
    completion = provider.complete(
        [Message.user(f"{context}\n\nQuery: {query}")], system=ANSWER_SYSTEM_PROMPT
    )
    return completion.content.strip()


def answer_test_time_only(provider: Provider, raw_context: str, query: str) -> str:
    """Path A for one query: derive on the spot, then answer. 2 online calls."""
    derived = _derive(provider, raw_context, query)
    return _answer(provider, derived, query)


def run_sleep_time_pipeline(
    provider: Provider,
    raw_context: str,
    queries: list[str],
    covered: dict[str, bool],
) -> SleepTimeReport:
    """Compare Path A and Path B over the same context and query list.

    Args:
        provider: Model to run the sleep pass, derivations, and answers with.
        raw_context: The raw memory both paths ultimately derive from.
        queries: Questions to answer, sharing `raw_context`.
        covered: Per-query flag: True if the learned context anticipated
            this query (Path B answers directly from it), False if it did
            not (Path B falls back to Path A's derive-then-answer for that
            query). Missing queries default to not covered.

    Returns:
        A `SleepTimeReport` with both paths' answers and call counts.
    """
    learned_context = sleep_pass(provider, raw_context)

    path_a_answers = [answer_test_time_only(provider, raw_context, q) for q in queries]
    path_a_online_calls = 2 * len(queries)

    path_b_answers: list[str] = []
    path_b_online_calls = 0
    fallback_queries: list[str] = []
    for query in queries:
        if covered.get(query, False):
            path_b_answers.append(_answer(provider, learned_context, query))
            path_b_online_calls += 1
        else:
            path_b_answers.append(answer_test_time_only(provider, raw_context, query))
            path_b_online_calls += 2
            fallback_queries.append(query)

    return SleepTimeReport(
        learned_context=learned_context,
        path_a_answers=path_a_answers,
        path_a_online_calls=path_a_online_calls,
        path_b_answers=path_b_answers,
        path_b_online_calls=path_b_online_calls,
        path_b_total_calls=path_b_online_calls + 1,  # + the one offline sleep pass
        fallback_queries=fallback_queries,
    )


def run_sleep_time_demo(provider: Provider | None = None) -> dict[str, object]:
    """Three queries share one raw context: two are the kind of thing a
    sleep pass would anticipate (order totals, shipping status), one asks
    about something the sleep pass never derived (a return policy detail),
    so it falls back. Compares the crossover against a single-query run
    where sleep-time has no advantage.
    """
    raw_context = (
        "Order #4471: 2x wireless mouse ($25 each), 1x USB-C hub ($40). "
        "Shipped 2026-07-10, arriving 2026-07-14. Customer asked about a "
        "possible return of the hub if it doesn't fit their laptop."
    )
    queries = [
        "What is the order total?",
        "When does the order arrive?",
        "Can the hub be returned if it doesn't fit?",
    ]
    covered = {queries[0]: True, queries[1]: True, queries[2]: False}

    total_answer = "The order total is $90."
    arrival_answer = "The order arrives on 2026-07-14."
    return_derived = "Return policy is not in the order context; needs the returns policy doc."
    return_answer = "The hub's return eligibility isn't in this order's memory; check the returns policy."

    if provider is None:
        provider = get_provider(
            script=[
                # Sleep pass (offline, once): pre-derive what a plausible
                # query set needs from the raw order context.
                "Order #4471 total: $90 (2 mice at $25 + 1 hub at $40). Ships arrive 2026-07-14.",
                # Path A: derive-then-answer for all 3 queries, on the spot.
                "Order total: 2*$25 + $40 = $90.",
                total_answer,
                "Arrival date: 2026-07-14.",
                arrival_answer,
                return_derived,
                return_answer,
                # Path B: queries 1-2 answer straight from the learned
                # context (1 call each, same answer as path A); query 3
                # falls back to the same derive-then-answer as path A.
                total_answer,
                arrival_answer,
                return_derived,
                return_answer,
            ]
        )
    report = run_sleep_time_pipeline(provider, raw_context, queries, covered)

    single_provider = get_provider(
        script=[
            "Order #4471 total: $90.",  # sleep pass
            "Order total: 2*$25 + $40 = $90.",  # path A derive
            total_answer,  # path A answer
            total_answer,  # path B answer, same content, from the learned context
        ]
    )
    single = run_sleep_time_pipeline(single_provider, raw_context, [queries[0]], {queries[0]: True})

    return {
        "learned_context": report.learned_context,
        "path_a_answers": report.path_a_answers,
        "path_a_online_calls": report.path_a_online_calls,
        "path_b_answers": report.path_b_answers,
        "path_b_online_calls": report.path_b_online_calls,
        "path_b_total_calls": report.path_b_total_calls,
        "fallback_queries": report.fallback_queries,
        "covered_answers_match_path_a": report.path_a_answers[:2] == report.path_b_answers[:2],
        "single_query_path_a_calls": single.path_a_online_calls,
        "single_query_path_b_total_calls": single.path_b_total_calls,
        "single_query_no_advantage": single.path_a_online_calls == single.path_b_total_calls,
    }
