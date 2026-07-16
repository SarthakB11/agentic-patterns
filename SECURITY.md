# Security Policy

## Scope

agentic-patterns is educational reference code. It exists to show how agentic
patterns work at the level of messages and control flow, not to serve as a
hardened library you deploy as-is.

A few folders implement patterns that are themselves about security:

- **`patterns/guardrails/`**: input/output checkpoints, PII masking, a
  dual-LLM quarantine design, a capability-based policy engine, and a
  prompt-injection attack/defense harness.
- **`patterns/mcp/`**: a from-scratch MCP client and server, including a
  tool-integrity pin ledger that detects the rug-pull (TOCTOU) failure mode.
- **`patterns/human_in_the_loop/`**: approval gates, risk tiers, and audit
  trails that stop a side effect until a human decision is recorded.

These modules demonstrate the _shape_ of a defense so the mechanism is
legible and testable: what a capability check looks like, what a pinned tool
definition looks like, what a fail-closed gate looks like. They are not
audited security libraries, and they have not been red-teamed by anyone but
the author. Several say so directly in their own README.
`patterns/guardrails/README.md` states plainly that checkpoint detection
(regex-based prompt-injection screens, moderation blocklists) is "a cheap
first filter and an audit signal," not a guarantee, and that adaptive
attacks have bypassed eight published detection defenses in the literature
it cites. `patterns/mcp/README.md` states that the integrity guard is not
wired into the client's own dispatch by default; a reader has to apply it.

Do not drop any module from this repo into a production system unmodified
and treat it as your security boundary. Use it to understand the pattern,
then build or adopt something audited for the actual guarantee you need.

## Runtime and network behavior

On its default path, every example in this repo runs fully offline. Each
pattern is driven by a deterministic, scripted mock provider defined in the
codebase; there are no outbound network calls, no telemetry, and no secrets
required to run any demo, including the MCP client/server pair, which talks
to a real subprocess over stdio or loopback HTTP but never leaves the
machine.

Real network calls only happen if you opt in, by setting
`AGENTIC_PATTERNS_PROVIDER=openai` or `AGENTIC_PATTERNS_PROVIDER=anthropic`
(or `AGENTIC_PATTERNS_EMBEDDER=openai`) and supplying an API key as an
environment variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). Those keys are
read from the environment at call time through
`agentic_patterns.get_provider`; the repo never writes a key to disk, logs
it, or sends it anywhere but the provider you selected. If you don't set
these variables, no credential of any kind is needed.

## Reporting a vulnerability

If you find an actual security issue in this repo's own code (a real code
execution path, a credential leak, a path traversal, or a way the mock or
transport code behaves unsafely, not a limitation of the teaching pattern
itself), please report it privately instead of opening a public issue:

- Preferred: open a [GitHub private security advisory](https://github.com/SarthakB11/agentic-patterns/security/advisories/new) on this repository.
- Alternative: email sarthak.bhardwaj21b@iiitg.ac.in with `SECURITY` in the subject line.

Please include the affected file(s) and a commit hash if you have one, steps
to reproduce or a minimal script, what you expected versus what happened,
and your read on impact.

This is a solo-maintained portfolio project, not a funded security team. I
will acknowledge a report within a few days and aim to have a fix or a
public response within two to three weeks, faster for anything with a clear
exploit path. If you don't hear back in that window, a follow-up email is
welcome. There is no bug bounty.

## About the citations in the security-pattern modules

Several modules cite the specific research, benchmarks, or CVEs their
mechanism is modeled on. `patterns/mcp/integrity.py` is written against the
rug-pull scenario in **CVE-2025-54136** ("MCPoison"), a real remote-code-
execution report in a previously approved MCP configuration file swapped
without re-approval. `patterns/guardrails/` cites AgentDojo, CaMeL, the
Design Patterns for Securing LLM Agents paper, Progent, and the
adaptive-attacks paper that broke eight published defenses, among others;
the full list with arXiv IDs is in each folder's README.

These citations teach the failure mode accurately, so a reader can verify
the mechanism against its primary source rather than take the code's word
for it. They are not a claim that this repo's implementation covers the
referenced CVE, resists the referenced attack in production, or matches the
referenced paper's full system. Treat every citation as "this is the
real-world problem the following code illustrates," not as evidence of
production-grade coverage.
