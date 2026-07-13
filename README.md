# agentic-patterns

[![CI](https://github.com/SarthakB11/agentic-patterns/actions/workflows/ci.yml/badge.svg)](https://github.com/SarthakB11/agentic-patterns/actions/workflows/ci.yml)

Twelve core agentic AI patterns, each implemented as a small, runnable, tested example in plain Python. Every example runs offline against a deterministic scripted mock provider: clone the repo and run any pattern with no API key, no install, and no network.

## Why this repo

Agent frameworks change monthly; the patterns underneath them do not. This repo implements the patterns themselves, without a framework, so you can see exactly what a supervisor, a reflection loop, or an MCP handshake does at the level of messages and control flow. Each folder is teaching code: typed, documented, tested, and small enough to read in one sitting. Each folder README also says which sub-variants were left out and why, so the coverage claims are checkable.

## Quickstart

```bash
git clone https://github.com/SarthakB11/agentic-patterns.git
cd agentic-patterns
python3 -m patterns.react.main
```

That already works. There is nothing to install and no key to set: the mock provider replays a scripted, coherent conversation, so each pattern's control flow (tool calls, critiques, routing decisions, handshakes) completes exactly as it would against a live model, deterministically.

To run the test suite:

```bash
python3 -m pip install -e ".[dev]"
pytest -q
```

To run any example against a real API instead of the mock, set environment variables and run the same command:

| Provider                                 | Variables                                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------ |
| OpenAI or any OpenAI-compatible endpoint | `AGENTIC_PATTERNS_PROVIDER=openai`, `OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `OPENAI_MODEL` |
| Anthropic                                | `AGENTIC_PATTERNS_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL`           |
| Real embeddings (memory, RAG, routing)   | `AGENTIC_PATTERNS_EMBEDDER=openai`, `OPENAI_API_KEY`                                             |

## The twelve patterns

| Pattern                                          | What it does                                                                                           | Reach for it when                                                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| [ReAct](patterns/react/)                         | Interleaves reasoning and tool calls in one loop: thought, action, observation, repeat                 | The number of steps is unknown upfront and each action depends on the last observation                        |
| [Planning](patterns/planning/)                   | Produces an explicit plan first, then executes it: sequential, DAG-parallel, replanning, ReWOO         | The task structure is predictable, steps can run in parallel, or a plan needs review before anything executes |
| [Reflection](patterns/reflection/)               | Generates, critiques, and refines with explicit stop conditions and best-so-far tracking               | Output quality matters more than latency and a checkable signal (tests, a rubric) exists                      |
| [Tool use](patterns/tool_use/)                   | Function calling from a single shot to parallel calls, forced choice, self-repair, and code-as-action  | The model must act on the world, not just describe it                                                         |
| [Memory](patterns/memory/)                       | Short-term windows and summarization up to vector stores, write policies, and MemGPT-style paging      | State must survive past the context window, across turns or across sessions                                   |
| [RAG](patterns/rag/)                             | Retrieval-augmented generation: naive dense, hybrid BM25 + dense with RRF, reranking, grading, abstain | Answers must be grounded in a corpus the model was not trained on, with citations                             |
| [Multi-agent](patterns/multi_agent/)             | Supervisor and workers, handoffs, debate, maker-checker, hierarchies over shared state                 | Work genuinely splits into roles with separate contexts, and coordination overhead pays for itself            |
| [Evaluation](patterns/evaluation/)               | Exact checks, LLM-as-judge with bias controls, juries, trajectory scoring, regression gates            | You change a prompt or model and need to know nothing broke                                                   |
| [MCP](patterns/mcp/)                             | A Model Context Protocol client and server built from scratch over JSON-RPC 2.0 and stdio              | A tool integration should be reusable across hosts and run behind a process boundary                          |
| [Guardrails](patterns/guardrails/)               | Fail-closed validation at every trust boundary: input, retrieval, output, and pre-tool                 | Anything untrusted flows in or consequential actions flow out                                                 |
| [Human-in-the-loop](patterns/human_in_the_loop/) | Approval gates with four decisions, audit logs, durable interrupt and resume, risk tiers               | An action is irreversible or expensive enough that a person must stay in the chain                            |
| [Routing](patterns/routing/)                     | Sends each request to the right model, mode, or handler: semantic, classifier, cascades, fallbacks     | One model or one configuration should not serve every request                                                 |

## A reading order

If you are new to agents, read the folders in this order. Each group builds on the one before it.

```mermaid
flowchart LR
    subgraph A[Acting]
        tool_use --> react --> planning --> reflection
    end
    subgraph B[Knowledge and state]
        memory --> rag
    end
    subgraph C[Scale and control]
        routing --> multi_agent --> evaluation
    end
    subgraph D[Safety and interop]
        guardrails --> human_in_the_loop --> mcp
    end
    A --> B --> C --> D
```

Start with tool use, since every other pattern assumes a model that can call functions. ReAct turns tool calls into a loop; planning front-loads the loop's decisions; reflection closes the loop on quality. Memory and RAG give the loop state and knowledge. Routing, multi-agent, and evaluation are what you add when one loop is not enough. Guardrails and human-in-the-loop are what you add before you trust any of it, and MCP is how tools outgrow a single codebase.

## Repo layout

```
agentic_patterns/core/   shared harness: Provider abstraction (mock, OpenAI-compatible,
                         Anthropic), ToolRegistry, deterministic hash embedder, env config
patterns/<name>/         one folder per pattern: runnable main.py, one module per
                         sub-variant, a README with a flowchart and sources
tests/                   one test file per pattern, plus core tests and a smoke test
                         that runs every entrypoint offline
```

Design choices worth knowing about:

- The offline path imports only the standard library. `httpx` is needed only for real providers and is imported lazily.
- Mock scripts live next to the demo functions, so a reader sees the whole scripted conversation next to the loop that consumes it. The pattern logic itself never special-cases the mock; swapping providers is a config change.
- Wire-format conversions for the real providers are pure functions with their own unit tests, so provider correctness is tested without a network call.
- Tests assert on mechanics (what was sent to the model, how loops stop, what gets refused), not on prose.

## License

MIT. Built by [Sarthak Bhardwaj](https://github.com/SarthakB11).
