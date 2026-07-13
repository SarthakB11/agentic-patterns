"""Smoke test: every pattern's entrypoint runs offline and exits cleanly.

This locks in the repo's core promise: any pattern can be run straight from
a clone with no API key, no network, and no configuration. Each entrypoint
is run in a subprocess with the provider selection variables stripped, so a
key present in the developer's environment cannot mask a broken mock path.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PATTERNS = sorted(
    p.name
    for p in (REPO_ROOT / "patterns").iterdir()
    if p.is_dir() and not p.name.startswith("__")
)


def _offline_env() -> dict[str, str]:
    """Return an environment with provider and key variables removed."""
    env = dict(os.environ)
    for var in (
        "AGENTIC_PATTERNS_PROVIDER",
        "AGENTIC_PATTERNS_EMBEDDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        env.pop(var, None)
    return env


def test_all_twelve_patterns_are_present() -> None:
    assert len(PATTERNS) == 12, f"Expected 12 pattern folders, found {len(PATTERNS)}: {PATTERNS}"


@pytest.mark.parametrize("name", PATTERNS)
def test_entrypoint_runs_offline(name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", f"patterns.{name}.main"],
        cwd=REPO_ROOT,
        env=_offline_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"patterns.{name}.main exited {result.returncode}\n"
        f"stdout tail: {result.stdout[-500:]}\nstderr tail: {result.stderr[-500:]}"
    )
    assert result.stdout.strip(), f"patterns.{name}.main printed nothing"
