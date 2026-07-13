"""Agentic RAG: expose retrieval as a tool the model calls in a loop, instead
of running retrieval once as a fixed pre-step.

The model receives a `search_knowledge_base` tool and decides for itself
when to call it, what to search for, and whether the results are enough to
answer or need a narrower follow-up search. This mirrors the LangGraph
custom-RAG-agent tutorial's grade-then-rewrite-or-generate graph, and
production equivalents include the OpenAI Agents SDK's hosted `file_search`
tool and Anthropic's client-side `memory` tool (memory_20250818): retrieval
becomes something the agent invokes, not a step that always runs before the
model sees the question.

This is the flat, single-tool version of agentic retrieval: one search tool,
re-search on a hunch, and no memory of what an earlier search established
beyond the raw chunks it returned. `deep_research.py` is the multi-step
evolution the 2025 search-agent line (Search-R1, ReSearch, DeepResearcher)
induced: decompose into sub-questions, retrieve and read per sub-question
into an evidence notebook, check coverage, and synthesize with citations
traced back to the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_patterns import Embedder, Message, Provider, Tool, ToolRegistry, get_embedder, get_provider

from patterns.rag.chunking import Chunk, ScoredChunk
from patterns.rag.corpus import default_chunks
from patterns.rag.dense import DenseIndex, build_dense_index, dense_retrieve
from patterns.rag.generation import ABSTAIN_ANSWER, GroundedAnswer, extract_citations

_AGENTIC_DEMO_QUERY = (
    "What is the exact rollback command an on-call engineer runs for a deploy-caused SEV1, "
    "and how quickly must the rollback complete?"
)

_AGENT_SYSTEM = (
    "You answer questions about Aurora Cloud's internal policies. Call "
    "search_knowledge_base to find evidence before answering; never answer "
    "from memory alone. If the results do not fully cover the question, "
    "call the tool again with a narrower or different query before giving "
    "up. When you have enough evidence, reply with a final answer that "
    "cites chunk ids in square brackets and makes no further tool call."
)

_SEARCH_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "Search query for the knowledge base."}},
    "required": ["query"],
}


@dataclass
class AgenticRagResult:
    """The outcome of one agentic RAG run.

    Attributes:
        transcript: Human-readable log of tool calls, observations, and the
            final answer, in order.
        rounds_used: Number of provider calls the loop made.
        answer: The final `GroundedAnswer`, or an abstain result if the loop
            ran out of tool-call rounds without producing a final answer.
    """

    transcript: list[str] = field(default_factory=list)
    rounds_used: int = 0
    answer: GroundedAnswer = field(default_factory=lambda: GroundedAnswer(answer=ABSTAIN_ANSWER, abstained=True))


def build_search_tool(dense_index: DenseIndex, embedder: Embedder, *, top_k: int = 3) -> tuple[Tool, list[list[ScoredChunk]]]:
    """Build a `search_knowledge_base` tool backed by dense retrieval.

    Args:
        dense_index: A `DenseIndex` over the corpus the tool searches.
        embedder: Embedder used to embed each search query.
        top_k: Chunks returned per search call.

    Returns:
        A tuple of the `Tool` and a log list that accumulates each search
        call's results, in call order, so the caller can recover which
        chunks were actually seen across the whole run.
    """
    search_log: list[list[ScoredChunk]] = []

    def search_knowledge_base(query: str) -> str:
        results = dense_retrieve(query, dense_index, embedder, top_k=top_k)
        search_log.append(results)
        if not results:
            return "No matching chunks found."
        return "\n".join(f"[{sc.chunk.id}] score={sc.score:.3f} {sc.chunk.text}" for sc in results)

    tool = Tool(
        name="search_knowledge_base",
        description="Search the Aurora Cloud internal knowledge base and return the most relevant chunks.",
        parameters=_SEARCH_TOOL_PARAMETERS,
        fn=search_knowledge_base,
    )
    return tool, search_log


def run_agentic_rag(
    query: str, provider: Provider, dense_index: DenseIndex, embedder: Embedder, *, max_rounds: int = 4
) -> AgenticRagResult:
    """Run the agentic retrieval loop: search, grade implicitly, maybe re-search, answer.

    Args:
        query: The user's question.
        provider: `Provider` driving the loop; must be scripted (for
            `MockProvider`) to eventually stop making tool calls.
        dense_index: A `DenseIndex` the search tool retrieves from.
        embedder: Embedder used for each search query.
        max_rounds: Maximum provider calls before the loop gives up and
            abstains, guarding against a model that never stops searching.

    Returns:
        An `AgenticRagResult` with the full transcript and the final answer.
    """
    tool, search_log = build_search_tool(dense_index, embedder)
    registry = ToolRegistry()
    registry.register(tool)

    messages: list[Message] = [Message.user(query)]
    transcript: list[str] = []
    seen_chunks: dict[str, Chunk] = {}

    for round_number in range(1, max_rounds + 1):
        completion = provider.complete(messages, tools=registry.specs(), system=_AGENT_SYSTEM)

        if not completion.tool_calls:
            transcript.append(f"round {round_number}: final answer")
            valid_ids = set(seen_chunks)
            citations = extract_citations(completion.content, valid_ids)
            answer = GroundedAnswer(answer=completion.content, citations=citations, abstained=False)
            return AgenticRagResult(transcript=transcript, rounds_used=round_number, answer=answer)

        messages.append(Message.assistant(completion.content, tool_calls=completion.tool_calls))
        for call in completion.tool_calls:
            transcript.append(f"round {round_number}: tool call search_knowledge_base(query={call.arguments.get('query')!r})")
            observation = registry.execute(call)
            transcript.append(f"round {round_number}: observation -> {observation}")
            messages.append(Message.tool(call.id, observation))
        if search_log:
            for scored in search_log[-1]:
                seen_chunks[scored.chunk.id] = scored.chunk

    transcript.append(f"stop: {max_rounds} rounds used without a final answer, abstaining")
    return AgenticRagResult(transcript=transcript, rounds_used=max_rounds, answer=GroundedAnswer(answer=ABSTAIN_ANSWER, abstained=True))


def run_agentic_rag_demo(
    provider: Provider | None = None,
    *,
    dense_index: DenseIndex | None = None,
    embedder: Embedder | None = None,
) -> AgenticRagResult:
    """Demonstrate the agentic loop broadening a search that came back incomplete.

    The demo question needs two facts that live in two different documents:
    that the first mitigation step is a rollback (`incident-runbook`), and
    the exact rollback command and its two-minute deadline
    (`deploy-policy`). A first, broad search surfaces the mitigation step but
    not the command; the model narrows its second search to find it, then
    answers citing both.

    Args:
        provider: A `Provider` to drive the demo. Defaults to a
            `MockProvider` scripted with a broad search, a narrower
            follow-up search, and a final two-citation answer.
        dense_index: A prebuilt `DenseIndex` over the sample corpus. Built
            fresh with `embedder` when omitted, so the demo still runs
            standalone with no arguments.
        embedder: Embedder the search tool uses, and for building
            `dense_index` when it is not supplied. Defaults to
            `agentic_patterns.get_embedder`.

    Returns:
        The `AgenticRagResult` for the demo query.
    """
    if embedder is None:
        embedder = get_embedder()
    if dense_index is None:
        dense_index = build_dense_index(default_chunks(), embedder)
    if provider is None:
        provider = get_provider(
            script=[
                {"tool": "search_knowledge_base", "args": {"query": "rollback procedure for a deploy-caused SEV1 incident"}},
                {"tool": "search_knowledge_base", "args": {"query": "aurora rollback command revert release stable minutes"}},
                "The first mitigation step for a deploy-caused SEV1 is always a rollback, not a "
                "forward fix [incident-runbook#1]. The on-call engineer runs `aurora rollback "
                "release-id` to revert to the previous stable release, and this must complete "
                "within two minutes [deploy-policy#1].",
            ]
        )
    return run_agentic_rag(_AGENTIC_DEMO_QUERY, provider, dense_index, embedder)
