"""Deep research: an iterative decompose-retrieve-synthesize loop, instead of
one flat retrieval pass or one round of query rewriting.

`agentic.py` gives a model a single search tool and lets it re-search when a
result feels thin, but it keeps no record of what any earlier search
established. `query_transform.py`'s multi-query expansion splits a question
into sub-queries and fuses their results in one shot, with no reading step
and no follow-up. This module is the shape induced by the 2025 RL-for-search
line (Search-R1, Jin et al., arXiv:2503.09516; ReSearch, Chen et al.,
arXiv:2503.19470; DeepResearcher, Zheng et al., arXiv:2504.03160): those
papers train a model to interleave reasoning with retrieval, and the
inference loop that training produces is plan, decompose, retrieve, read,
check coverage, and synthesize with cross-source citations. The RL training
itself is out of scope here; only the inference loop is built.

The loop keeps an evidence notebook, one `Finding` per sub-question, each
citing the chunk ids it was read from. After a round of reads, a coverage
check asks the model whether the notebook can already answer the original
question or whether a gap remains; a gap spawns a follow-up sub-question for
the next round. A round budget caps the loop, so a coverage check that keeps
finding gaps cannot run forever: once the budget is spent, the module stops
asking and synthesizes a report from whatever the notebook holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Embedder, Message, Provider, get_embedder, get_provider

from patterns.rag.chunking import Chunk
from patterns.rag.corpus import default_chunks
from patterns.rag.dense import DenseIndex, build_dense_index, dense_retrieve
from patterns.rag.generation import ABSTAIN_ANSWER, GroundedAnswer, extract_citations
from patterns.rag.query_transform import parse_multi_queries

_DECOMPOSE_SYSTEM = (
    "Split the user's question into 2-3 focused sub-questions that together would let you fully "
    "answer it through separate research steps. Reply with one sub-question per line, no "
    "numbering, no other text."
)

_READ_SYSTEM = (
    "Read the sub-question and the retrieved chunks. If the chunks answer it, reply with one "
    "sentence stating the finding, citing the chunk id of every claim in square brackets, like "
    "[chunk-id]. If the chunks do not answer the sub-question, reply with exactly: NOT FOUND."
)

_COVERAGE_SYSTEM = (
    "Given the original question and the research notebook of findings gathered so far, decide "
    "whether the notebook can fully answer the original question. If something is still missing, "
    "reply with the missing sub-questions, one per line, no numbering. If nothing is missing, "
    "reply with exactly: DONE."
)

_SYNTHESIS_SYSTEM = (
    "Write a short report synthesizing the research notebook below into a full answer to the "
    "original question. Cite chunk ids in square brackets for every claim, using only ids that "
    "appear in the notebook. If the notebook leaves part of the question unanswered, say so "
    "instead of guessing."
)

_DEMO_QUERY = (
    "Walk me through what happens once a SEV1 incident is declared, all the way through the "
    "postmortem, and who takes over if the on-call primary does not respond to a page?"
)


@dataclass
class Finding:
    """One evidence-notebook entry produced by reading chunks for a sub-question.

    Attributes:
        sub_question: The sub-question this finding answers.
        claim: The model's one-sentence finding, with inline citations.
        chunk_ids: Chunk ids the finding actually cited, validated against
            the chunks it was shown.
    """

    sub_question: str
    claim: str
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class DeepResearchResult:
    """The full record of one deep-research run.

    Attributes:
        query: The original question.
        sub_question_tree: Every sub-question asked, in the order it was
            asked: the initial decomposition followed by any gap-driven
            follow-ups, across every round.
        notebook: Every finding collected, in the order it was added.
        rounds_used: Number of read rounds run (a round budget stops this
            short of the sub-question tree exhausting itself).
        answer: The synthesized grounded answer, or an abstain result if the
            notebook ended up empty.
    """

    query: str
    sub_question_tree: list[str]
    notebook: list[Finding]
    rounds_used: int
    answer: GroundedAnswer


def build_decompose_prompt(query: str) -> str:
    """Build the prompt asking the model to decompose a question into sub-questions."""
    return f"Question: {query}"


def decompose_question(query: str, provider: Provider) -> list[str]:
    """Ask the model to split a question into an initial sub-question list.

    Args:
        query: The original research question.
        provider: `Provider` used for the decomposition call.

    Returns:
        The initial sub-questions, reusing `query_transform.parse_multi_queries`
        for the one-per-line parse.
    """
    completion = provider.complete([Message.user(build_decompose_prompt(query))], system=_DECOMPOSE_SYSTEM)
    return parse_multi_queries(completion.content)


def build_read_prompt(sub_question: str, chunks: list[Chunk]) -> str:
    """Build the prompt asking the model to extract a finding from retrieved chunks."""
    context = "\n\n".join(f"[{chunk.id}] {chunk.text}" for chunk in chunks)
    return f"Sub-question: {sub_question}\n\nRetrieved chunks:\n{context}"


def read_for_finding(sub_question: str, chunks: list[Chunk], provider: Provider) -> Finding | None:
    """Read retrieved chunks for one sub-question and extract a cited finding.

    Args:
        sub_question: The sub-question being researched this step.
        chunks: Chunks retrieved for `sub_question`. An empty list is read
            as "not found" without a model call.
        provider: `Provider` used for the read call.

    Returns:
        A `Finding` citing the chunk ids the model actually used, or `None`
        if the chunks did not answer the sub-question ("not found").
    """
    if not chunks:
        return None
    completion = provider.complete([Message.user(build_read_prompt(sub_question, chunks))], system=_READ_SYSTEM)
    text = completion.content.strip()
    if text.upper().startswith("NOT FOUND"):
        return None
    valid_ids = {chunk.id for chunk in chunks}
    cited = extract_citations(text, valid_ids)
    return Finding(sub_question=sub_question, claim=text, chunk_ids=cited)


def build_coverage_prompt(query: str, notebook: list[Finding]) -> str:
    """Build the prompt asking the model to grade notebook coverage and name gaps."""
    lines = [f"Original question: {query}", "", "Notebook so far:"]
    for finding in notebook:
        lines.append(f"- [{finding.sub_question}] {finding.claim}")
    return "\n".join(lines)


def parse_coverage(text: str) -> list[str]:
    """Parse a coverage reply into a gap list, empty when the reply is DONE."""
    stripped = text.strip()
    if stripped.upper() == "DONE":
        return []
    return parse_multi_queries(text)


def check_coverage(query: str, notebook: list[Finding], provider: Provider) -> list[str]:
    """Ask the model whether the notebook covers the original question.

    Args:
        query: The original research question.
        notebook: Findings collected so far.
        provider: `Provider` used for the coverage call.

    Returns:
        Missing sub-questions to research next, or an empty list when the
        model replies DONE.
    """
    completion = provider.complete([Message.user(build_coverage_prompt(query, notebook))], system=_COVERAGE_SYSTEM)
    return parse_coverage(completion.content)


def build_synthesis_prompt(query: str, notebook: list[Finding]) -> str:
    """Build the prompt asking the model to synthesize the notebook into one report."""
    lines = [f"Original question: {query}", "", "Research notebook:"]
    for finding in notebook:
        lines.append(f"- Sub-question: {finding.sub_question}")
        lines.append(f"  Finding: {finding.claim}")
    return "\n".join(lines)


def synthesize_report(query: str, notebook: list[Finding], provider: Provider) -> GroundedAnswer:
    """Synthesize the evidence notebook into one grounded, cited report.

    Args:
        query: The original research question.
        notebook: Findings collected across every round. An empty notebook
            abstains without a model call, the same failure-closed behavior
            `generation.generate_grounded_answer` uses for empty context.
        provider: `Provider` used for the synthesis call.

    Returns:
        A `GroundedAnswer` whose citations are validated against the union
        of chunk ids actually present in the notebook, not against every id
        ever retrieved, so a "not found" sub-question can never leak a
        citation into the final report.
    """
    if not notebook:
        return GroundedAnswer(answer=ABSTAIN_ANSWER, citations=[], abstained=True)
    completion = provider.complete([Message.user(build_synthesis_prompt(query, notebook))], system=_SYNTHESIS_SYSTEM)
    valid_ids = {chunk_id for finding in notebook for chunk_id in finding.chunk_ids}
    citations = extract_citations(completion.content, valid_ids)
    return GroundedAnswer(answer=completion.content, citations=citations, abstained=False)


def run_deep_research(
    query: str,
    dense_index: DenseIndex,
    embedder: Embedder,
    provider: Provider,
    *,
    top_k: int = 2,
    max_rounds: int = 3,
) -> DeepResearchResult:
    """Run the decompose-retrieve-read-check-synthesize loop to completion.

    Each round retrieves and reads for every currently open sub-question,
    then, if the round budget allows another round, asks a coverage check
    for gaps. The coverage check is skipped on the round that hits the
    budget: there would be nowhere left to send a gap, so the call would
    only spend a scripted turn for no effect. This is what keeps an
    over-eager coverage script (one that always reports a gap) from
    running forever: the budget check happens before the call that could
    extend the loop, not after.

    Args:
        query: The research question.
        dense_index: A `DenseIndex` over the corpus each sub-question retrieves from.
        embedder: Embedder used to embed each sub-question.
        provider: `Provider` used for decomposition, reads, coverage checks,
            and synthesis.
        top_k: Chunks retrieved per sub-question, per round.
        max_rounds: Maximum read rounds before the loop stops asking for
            more sub-questions and synthesizes from what it has.

    Returns:
        A `DeepResearchResult` with the full sub-question tree, the evidence
        notebook, the round count, and the synthesized answer.
    """
    sub_question_tree = decompose_question(query, provider)
    open_questions = list(sub_question_tree)
    notebook: list[Finding] = []
    round_number = 0

    while open_questions and round_number < max_rounds:
        round_number += 1
        for sub_question in open_questions:
            results = dense_retrieve(sub_question, dense_index, embedder, top_k=top_k)
            finding = read_for_finding(sub_question, [sc.chunk for sc in results], provider)
            if finding is not None:
                notebook.append(finding)

        open_questions = []
        if round_number < max_rounds:
            gaps = check_coverage(query, notebook, provider)
            if gaps:
                sub_question_tree.extend(gaps)
                open_questions = gaps

    answer = synthesize_report(query, notebook, provider)
    return DeepResearchResult(
        query=query,
        sub_question_tree=sub_question_tree,
        notebook=notebook,
        rounds_used=round_number,
        answer=answer,
    )


def run_deep_research_demo(
    provider: Provider | None = None,
    *,
    dense_index: DenseIndex | None = None,
    embedder: Embedder | None = None,
) -> DeepResearchResult:
    """Demonstrate a gap-driven follow-up round completing a research notebook.

    The demo question spans two documents plus a detail a narrow first fetch
    misses. Decomposition splits it into a SEV1-to-postmortem sub-question
    and an on-call-escalation sub-question. Round 1's top-2 fetch for the
    first sub-question surfaces the SEV1 declaration and rollback mitigation
    but, at `top_k=2`, misses the chunk stating the postmortem deadline; the
    second sub-question is answered in full. The coverage check catches the
    gap and proposes a narrower follow-up sub-question; round 2 retrieves
    and reads it, landing the postmortem deadline in the notebook. The round
    budget (2) is now spent, so the loop skips a third coverage check and
    synthesizes a three-citation report from all three findings.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with the decomposition, three reads, one
            coverage check, and the final synthesis.
        dense_index: A prebuilt `DenseIndex` over the sample corpus. Built
            fresh with `embedder` when omitted, so the demo still runs
            standalone with no arguments.
        embedder: Embedder for sub-question encoding, and for building
            `dense_index` when it is not supplied. Defaults to
            `agentic_patterns.get_embedder`.

    Returns:
        The `DeepResearchResult` for the demo query.
    """
    if embedder is None:
        embedder = get_embedder()
    if dense_index is None:
        dense_index = build_dense_index(default_chunks(), embedder)
    if provider is None:
        provider = get_provider(
            script=[
                "What is the process from a SEV1 incident being declared through to the postmortem "
                "being filed?\n"
                "Who takes over escalation if the on-call primary does not respond to a page?",
                "The on-call engineer declares severity within five minutes of the first alert, and "
                "the first mitigation step for a deploy-caused SEV1 is always a rollback rather than "
                "a forward fix [incident-runbook#0] [incident-runbook#1].",
                "If the primary does not acknowledge a page within fifteen minutes, PagerDuty "
                "escalates automatically to the secondary and then to the engineering manager "
                "[oncall-rotation#0].",
                "postmortem due within forty eight hours of incident resolution",
                "A postmortem is due within forty eight hours of resolution [incident-runbook#2].",
                "Once a SEV1 is declared within five minutes of the first alert, the on-call engineer "
                "performs a rollback as the first mitigation step for a deploy-caused incident "
                "[incident-runbook#0] [incident-runbook#1]. If the primary does not acknowledge a "
                "page within fifteen minutes, PagerDuty escalates first to the secondary and then to "
                "the engineering manager [oncall-rotation#0]. The postmortem itself is due within "
                "forty eight hours of resolution [incident-runbook#2].",
            ]
        )
    return run_deep_research(_DEMO_QUERY, dense_index, embedder, provider, top_k=2, max_rounds=2)
