"""Conceptual notes: variants with no runnable demo in this pattern.

Two taxonomy entries from docs/research/tool_use.md stay as text-only notes
because reproducing them offline would need a trained model or a genuine
multi-module runtime, not a scripted mock. The other two entries the brief
originally scoped this way, retrieval-based selection and code-as-action,
are runnable elsewhere in this pattern (`tool_search.py`, `code_execution.py`
respectively) since Anthropic and MCP turned both into first-party runtime
primitives after the brief's base sources; this module notes that promotion
rather than repeating the concept.
"""

from __future__ import annotations

LEARNED_TOOL_USE = """\
Learned / trained tool use (Toolformer, Schick et al. 2023)

Every module in this pattern prompts a general-purpose model with a tool
catalog at inference time; the model was never trained specifically to call
these tools. Toolformer takes the opposite approach: it fine-tunes the model
itself to insert API calls into its own generated text, learned in a
self-supervised way. The model samples candidate call insertions, keeps the
ones that reduce its own loss on the surrounding text (a proxy for "this
call was useful here"), and fine-tunes on the filtered examples. The
resulting model decides which API to call, with what arguments, and where
to splice the result, without a catalog being pasted into its prompt at all.

This repo cannot reproduce that offline: it needs a real fine-tuning run
over self-generated data, not a scripted response. The tradeoff worth
naming is prompted versus learned tool use: prompting is instant to add a
new tool (just extend the catalog) but pays a per-call context and selection
cost; a fine-tuned model pays that cost once during training and calls
tools "for free" at inference, at the price of needing to retrain the model
whenever the tool catalog changes.
"""

MRKL_ROUTING = """\
Neuro-symbolic routing (MRKL, Karpas et al. 2022)

Every loop in this pattern offers one flat tool catalog to one model, which
picks whichever tool it judges relevant. MRKL (Modular Reasoning, Knowledge
and Language) instead puts a router in front of a set of expert modules,
some symbolic (a calculator, a database query engine, a rule-based
component) and some neural (an LLM specialized for a domain), and the
router's job is to classify a query and dispatch it to the right expert
rather than let one model attempt everything through a shared tool
interface. Routing is a distinct decision from tool selection within a
single model's context: a router can send arithmetic straight to a
calculator module that never sees a prompt at all, sidestepping the
"language models are bad at exact arithmetic" failure mode entirely instead
of hoping the model remembers to call a calculator tool.

This repo's loop.py is closer to the flat-catalog side of that spectrum by
design, matching the shared core's single-Provider, single-ToolRegistry
contract; a MRKL-style router would sit a layer above it, choosing which
provider or which registry a query goes to before `run_tool_loop` ever
runs. Model and route selection is `patterns/routing/` territory, not a
tool-use loop concern.
"""

PROMOTED_TO_RUNNABLE = """\
Two variants the brief listed as concept-only are runnable elsewhere in
this pattern: retrieval-based tool selection is `tool_search.py`, and
code-as-action / programmatic tool calling is `code_execution.py`. Both
became first-party API primitives after the brief's base sources (Anthropic
Tool Search Tool, MCP SEP-1821 for the former; Anthropic Programmatic Tool
Calling and "Code execution with MCP" for the latter), which is why this
pattern treats them as runnable modules instead of notes.
"""


def _first_paragraph(note: str) -> str:
    """Return a note constant's title line plus its first sentence, for a compact transcript excerpt."""
    lines = [line for line in note.strip().splitlines() if line.strip()]
    return f"{lines[0]}\n  {lines[1].split('. ')[0]}."


def print_concept_notes() -> None:
    """Print a short excerpt of each conceptual, non-runnable note; see this module's constants for the full text."""
    print("=== 11. Conceptual notes: not runnable offline (excerpted; full text in concepts.py) ===")
    print(_first_paragraph(LEARNED_TOOL_USE))
    print(_first_paragraph(MRKL_ROUTING))
    print(_first_paragraph(PROMOTED_TO_RUNNABLE))


if __name__ == "__main__":
    print_concept_notes()
