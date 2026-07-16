# Contributing

This repo is a portfolio of reference implementations for twelve agentic AI patterns. The bar for a contribution is the same as the bar the existing code already holds itself to: runnable offline, deterministic, typed, small, and honest about what it does and does not cover. Read `README.md` and one folder README (`patterns/react/README.md` is a good model) before opening a PR; they set the tone this file assumes.

## Project shape

```
agentic_patterns/core/   frozen shared harness: Provider abstraction (mock, OpenAI-compatible,
                          Anthropic), ToolRegistry, deterministic hash embedder, env config
patterns/<name>/          one folder per pattern: main.py, one module per sub-variant,
                          a README with a flowchart, variant list, and sources
tests/                    one test file per pattern, plus core tests and a smoke test
                          that runs every entrypoint offline
```

`agentic_patterns/core/` is frozen: it's the shared surface every pattern imports, and a change there ripples into all twelve folders. Do not extend it for a single pattern's convenience. If a pattern needs something the core does not provide, put it in that pattern's own module first; propose a core change only if at least two patterns need the same thing, and open an issue before the PR.

`patterns/<name>/` is where nearly all contributions land: a new variant module, a README update, or a test.

## Local setup

```bash
python3 -m pip install -e ".[dev]"
```

Before opening a PR, all three of these must pass locally:

```bash
pytest -q          # whole suite, offline, no API key, no network
ruff check .
pyright
```

To run any pattern's demo directly:

```bash
python3 -m patterns.<name>.main
```

for example `python3 -m patterns.react.main`. Every demo runs against the scripted `MockProvider` by default, so this works with nothing installed beyond the base package and produces the same output every time.

## Non-negotiable principles

These are enforced in review, not suggestions:

- **Offline by default.** Every example must run under the scripted `MockProvider` with no API key and no network access. Real providers (OpenAI-compatible, Anthropic) are opt-in via environment variables and `httpx`, imported lazily so the offline path never needs it. If your variant cannot be scripted deterministically, it does not belong in `main.py`'s default run.
- **Deterministic tests.** No sleeps, no wall-clock assertions, no reliance on dict or set ordering, no live network calls. A test that is flaky once is a test that gets deleted, not quarantined.
- **Cited and verified.** Any module that implements a specific paper, system, or named technique carries that citation in its docstring or the README's Sources section, and the citation is checked against the primary source (the actual paper, blog post, or docs page, not a secondary summary) before the PR ships. If you cannot find or verify the primary source, do not cite it, and say in the PR description what's uncited and why.
- **Prose discipline.** No em-dashes anywhere in this repo: not in code comments, not in docstrings, not in READMEs, not in commit messages. Avoid the banned-vocab list in any prose you write: delve, leverage, comprehensive, robust, seamless, foster, utilize, landscape, crucial, pivotal. If you need to name the rule itself (as this file does), that's the only exception.

## Adding a pattern variant

A new variant is a new module inside an existing `patterns/<name>/` folder (or a new pattern folder, if you're proposing a genuinely new pattern; open an issue first). A complete contribution has all of the following, not a subset:

1. **The module itself** (`patterns/<name>/your_variant.py`), with a module or class docstring that names the paper, system, or post it implements and links to it. Teaching code: typed, small, readable in one sitting. No unexplained abstraction for its own sake.
2. **A demo wired into `main.py`**, so `python3 -m patterns.<name>.main` exercises the new variant the same way it exercises every other one, offline and deterministically.
3. **Tests** in `tests/test_<name>.py` that assert on mechanics, not prose: what was sent to the model, how the loop stops, what gets rejected. Mirror the depth of coverage the rest of that test file already has for sibling variants.
4. **A README entry**: add the variant to the "Variants implemented" list with the same one-line-then-detail style the existing entries use, and update the flowchart or sources section if your variant changes control flow or introduces a new citation. If your variant deliberately leaves out a sub-case (a stochastic sub-mode, a training-time technique, a piece that belongs in a different pattern folder), say so explicitly in a skipped-with-reasons note, the same way `patterns/react/README.md` explains why full MCTS and ReWOO aren't in that folder. Silent omission is not acceptable; the coverage claims in this repo are meant to be checkable.

If you're unsure whether an idea is in scope for an existing folder, read that folder's README first, especially its skipped-with-reasons notes: several already discuss and reject specific extensions (for example why `patterns/react/` leaves out full MCTS), and yours may be covered there.

## Commits

Sign your commits and sign off under the DCO:

```bash
git commit -S -s
```

`-S` gives a GPG/SSH signature, `-s` adds a `Signed-off-by` trailer. Both are encouraged on every commit.

Do not add AI-attribution trailers (`Co-Authored-By: Claude`, `Generated with...`, or similar) to commit messages in this repo. If a tool assisted you, that's fine, the commit should just read like any other commit: what changed and why.

## Code style

Full type hints on every function signature. Google-style docstrings. Line length 120 (enforced by `ruff`). Keep modules small: if a variant module is doing two things, it's probably two modules. Match the surrounding pattern folder's existing style before introducing a new one.
