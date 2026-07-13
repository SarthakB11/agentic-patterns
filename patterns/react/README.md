# ReAct (reason + act loop)

ReAct is an agent control pattern that interleaves natural-language reasoning with tool use inside a single loop. At each step the model emits a Thought, then an Action (a tool call with arguments), and the runtime returns an Observation; the Thought/Action/Observation triple is appended to a running scratchpad and the model is prompted again with the accumulated history. The loop repeats until the model emits a terminal Finish action or a stop condition fires.

## When to use it

Use ReAct when a task needs several tool calls whose arguments depend on earlier results, when the number of steps is not known ahead of time, or when you want a readable trace for debugging and trust. It fits multi-hop lookups, retrieval-augmented question answering, and any workflow where the model must decide dynamically what to do next.

Avoid it for a single tool call or a fixed pipeline, where a plain function-calling round trip is faster and cheaper. Avoid it when latency and token budget are tight, since every iteration is a separate model call over a growing transcript. If the full plan is knowable upfront, a planner-executor or ReWOO-style approach avoids paying for reasoning between every action.

## How this example works

Every variant in this folder shares the same control flow: build a prompt from the goal and the history so far, call the model, decide whether the response is a terminal answer or a tool request, execute the tool if any, append the result, and repeat until Finish, a repeat guard, or the iteration cap fires.

```mermaid
flowchart TD
    A[Build prompt: goal + scratchpad] --> B[Call model]
    B --> C{Response is Finish or has no tool call?}
    C -->|yes| D[Return final answer]
    C -->|no| E{Action parses and tool is known?}
    E -->|no| F[Append error Observation]
    E -->|yes| G[Execute tool, catch exceptions]
    G --> H[Append Thought/Action/Observation]
    F --> I{Last steps repeat?}
    H --> I
    I -->|yes| J[Stop: loop detected]
    I -->|no| K{Iteration cap reached?}
    K -->|yes| L[Stop: force message or one generate call]
    K -->|no| A
```

## Variants implemented

- `parser.py`: the `Thought: ... / Action: Tool[args]` text grammar and its strict parse-failure behavior, unit-tested on its own.
- `scratchpad.py`: the Thought/Action/Observation history, its rendering back into a prompt, observation truncation, and repeat detection.
- `text_loop.py`: few-shot and zero-shot text-parsing ReAct, the canonical form from the original paper, with force and generate early-stop policies.
- `native_loop.py`: native tool-calling ReAct using structured `ToolCall` objects instead of parsed text; carries reasoning content across turns unmodified, which is also the right handling for a reasoning model's thinking output.
- `programmatic.py`: batched tool calling, where one model turn requests several independent tool calls at once instead of one call per round trip.
- `reflexion.py`: ReAct plus Reflexion, an outer loop that retries a failed episode after the agent writes a self-critique of its own trajectory.
- `world.py`: the toy knowledge base and the two tools (`search`, `lookup`) every demo above answers questions against.

Not implemented, with reasons: Plan-then-execute and ReWOO decouple planning from execution entirely rather than adding a ReAct variant, and belong in `patterns/planning/`, not here. Tree-search ReAct (LATS) needs a value estimate over candidate trajectories that a scripted mock cannot meaningfully produce offline. An MCP tool adapter would only wrap `ToolRegistry` in a transport layer with no new offline-testable loop behavior, and subagent delegation is a multi-agent concern better suited to `patterns/multi_agent/`. An explicit verify phase is covered by `reflexion.py`'s self-critique step rather than duplicated as a separate always-on stage.

## Run it

```
python3 -m patterns.react.main
```

Expected output (abridged):

```
ReAct pattern demo: reason + act loop, driven against a scripted MockProvider.

=== 1. Few-shot text-parsing ReAct (canonical, two-hop lookup) ===
Step 1
  Thought:     I need to find which country the Great Wall is located in.
  Action:      search[Great Wall]
  Observation: The Great Wall is located in China.
...
Answer:  Beijing
```

## Real providers

Every demo calls `get_provider(script=...)`, which defaults to `MockProvider`. Set one of these to run the identical loop code against a real API instead:

- `AGENTIC_PATTERNS_PROVIDER=openai` plus `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`, `OPENAI_BASE_URL`).
- `AGENTIC_PATTERNS_PROVIDER=anthropic` plus `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`).

## Sources

- Shunyu Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," 2022. https://arxiv.org/abs/2210.03629
- Noah Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning," 2023. https://arxiv.org/abs/2303.11366
- Binfeng Xu et al., "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models," 2023. https://arxiv.org/abs/2305.18323
- Lilian Weng, "LLM Powered Autonomous Agents," Lil'Log, 2023. https://lilianweng.github.io/posts/2023-06-23-agent/
- Anthropic, "Building Effective Agents," 2024. https://www.anthropic.com/research/building-effective-agents
- LangGraph docs, `create_agent` (successor to the deprecated `create_react_agent`). https://langchain-ai.github.io/langgraph/
