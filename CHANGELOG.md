# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-16

First public release: twelve agentic AI patterns as runnable, tested reference
code that runs offline with no API key.

### Added

- Shared core harness (`agentic_patterns/core`): a `Provider` abstraction with
  a deterministic scripted `MockProvider`, thin OpenAI-compatible and Anthropic
  HTTP clients behind a lazy import, a `ToolRegistry`, a stdlib-only
  `HashEmbedder`, an opaque reasoning channel on `Completion` and `Message`, and
  environment-driven provider and embedder selection.
- Twelve pattern folders, each covering its variants from the canonical form
  through 2025-2026 research: ReAct, Planning, Reflection, Tool use, Memory,
  RAG, Multi-agent orchestration, Evaluation, MCP, Guardrails,
  Human-in-the-loop, and Routing. Every module cites the paper or system it
  implements, and every citation was verified against its primary source.
- A from-scratch MCP client and server over JSON-RPC 2.0 (stdio and a stateless
  variant), tool-integrity pinning, async tasks, elicitation, and discovery.
- Terminal demo recordings for every pattern, reproducible from
  `tools/record_demos.py`.
- A test suite of 750-plus deterministic tests that runs offline with no network
  or API key, plus a smoke test that runs every pattern demo.

### Tooling

- Continuous integration running ruff, pyright, the test suite with a coverage
  gate on Python 3.11 and 3.12, and a check that every demo runs on a bare
  interpreter.
- Ruff and pyright configuration, a pre-commit config, an `.editorconfig`, and a
  `Makefile` for the common tasks.

[0.1.0]: https://github.com/SarthakB11/agentic-patterns/releases/tag/v0.1.0
