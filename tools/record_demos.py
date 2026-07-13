#!/usr/bin/env python3
"""Record and render terminal demo GIFs for each agentic pattern.

Every pattern's `main.py` runs offline and prints instantly, so a raw
terminal recording is a one-frame dump. This tool captures that output,
replays it back line by line with a small delay, and records *that*
replay with asciinema, so the resulting GIF reads like someone typing
the command and watching the walkthrough scroll by.

Pipeline: capture output -> replay with paced delays -> asciinema records
the replay -> agg renders the asciicast to a GIF.

Requires two external tools on PATH:
  - asciinema (`uv tool install asciinema` or `pip install --user asciinema`)
  - agg, the asciicast-to-GIF renderer (a single binary from
    https://github.com/asciinema/agg/releases)

Usage:
    python3 tools/record_demos.py replay <pattern> [--target-seconds N]
        [--min-delay-ms N] [--max-delay-ms N]
        Run the pattern's main module, capture its output, and re-print
        it line by line with a computed delay. This is the command
        asciinema wraps (`asciinema rec -c "python3 tools/record_demos.py
        replay <pattern>" out.cast`); it is not meant to be piped into
        anything else.

    python3 tools/record_demos.py record <pattern> [options]
        Record one pattern to an asciicast, then render it to a GIF at
        docs/demos/<pattern>.gif. If the rendered GIF is over the size
        budget, retries with a lower fps cap before giving up.

    python3 tools/record_demos.py record-all [options]
        Record every pattern under patterns/ that has a main.py. This
        is the one command that reproduces every GIF in the repo; the
        top-level README's hero GIF reuses docs/demos/react.gif rather
        than a separate recording.

Options common to record and record-all:
    --cast-dir DIR       where .cast files are written (default: a temp
                          dir outside the repo; .cast files are never
                          committed)
    --out-dir DIR        where GIFs are written (default: docs/demos)
    --cols N             terminal width (default: 100)
    --rows N             terminal height (default: 30)
    --theme NAME         agg color theme (default: dracula)
    --font-size N        agg font size in px (default: 12)
    --fps-cap N          agg starting fps cap (default: 6)
    --idle-time-limit S  agg idle time cap in seconds (default: 2)
    --size-budget-mb N   per-GIF size budget in MB (default: 1.5)
    --target-seconds N   override the paced replay duration for every
                          pattern (default: per-pattern, see
                          TARGET_SECONDS_OVERRIDES)

Example:
    python3 tools/record_demos.py record-all
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PATTERNS_DIR = REPO_ROOT / "patterns"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "demos"
DEFAULT_CAST_DIR = Path(tempfile.gettempdir()) / "agentic-patterns-demos"

DEFAULT_TARGET_SECONDS = 20.0
DEFAULT_MIN_DELAY_MS = 30
DEFAULT_MAX_DELAY_MS = 120
DEFAULT_COLS = 100
DEFAULT_ROWS = 30
DEFAULT_THEME = "dracula"
DEFAULT_FONT_SIZE = 12
DEFAULT_FPS_CAP = 6
DEFAULT_IDLE_TIME_LIMIT = 2.0
DEFAULT_SIZE_BUDGET_MB = 1.5
MAX_RENDER_ATTEMPTS = 8

# Patterns whose output has enough lines that the default pacing would
# push the GIF over the size budget. Paced faster, not truncated.
TARGET_SECONDS_OVERRIDES: dict[str, float] = {
    "planning": 12.0,
    "reflection": 14.0,
    "routing": 14.0,
    "tool_use": 14.0,
    "rag": 14.0,
    "guardrails": 16.0,
    "human_in_the_loop": 16.0,
    "multi_agent": 16.0,
}


def discover_patterns() -> list[str]:
    """Return every pattern name under patterns/ that has a main.py, sorted."""
    names = []
    for entry in sorted(PATTERNS_DIR.iterdir()):
        if entry.is_dir() and (entry / "main.py").exists():
            names.append(entry.name)
    return names


def capture_output(pattern: str, cols: int) -> list[str]:
    """Run the pattern's main module offline and return its stdout lines."""
    env = dict(os.environ)
    env["COLUMNS"] = str(cols)
    result = subprocess.run(
        [sys.executable, "-m", f"patterns.{pattern}.main"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout.splitlines()


def compute_delay_seconds(
    num_lines: int, target_seconds: float, min_delay_ms: int, max_delay_ms: int
) -> float:
    """Pick one delay per printed line so total playback lands near the target."""
    if num_lines <= 0:
        return 0.0
    raw_ms = (target_seconds * 1000) / num_lines
    clamped_ms = max(min_delay_ms, min(max_delay_ms, raw_ms))
    return clamped_ms / 1000


def cmd_replay(args: argparse.Namespace) -> None:
    """Print the pattern's captured output line by line with a paced delay."""
    lines = capture_output(args.pattern, args.cols)
    delay = compute_delay_seconds(
        len(lines), args.target_seconds, args.min_delay_ms, args.max_delay_ms
    )
    for line in lines:
        print(line)
        sys.stdout.flush()
        time.sleep(delay)


def target_seconds_for(pattern: str, override: float | None) -> float:
    if override is not None:
        return override
    return TARGET_SECONDS_OVERRIDES.get(pattern, DEFAULT_TARGET_SECONDS)


def record_cast(pattern: str, cast_dir: Path, args: argparse.Namespace) -> Path:
    """Run asciinema over the paced replay and return the .cast path."""
    cast_dir.mkdir(parents=True, exist_ok=True)
    cast_path = cast_dir / f"{pattern}.cast"
    if cast_path.exists():
        cast_path.unlink()

    target_seconds = target_seconds_for(pattern, args.target_seconds)
    replay_cmd = " ".join(
        shlex.quote(part)
        for part in [
            sys.executable,
            str(Path(__file__).resolve()),
            "replay",
            pattern,
            "--target-seconds",
            str(target_seconds),
            "--min-delay-ms",
            str(args.min_delay_ms),
            "--max-delay-ms",
            str(args.max_delay_ms),
            "--cols",
            str(args.cols),
        ]
    )
    subprocess.run(
        [
            "asciinema",
            "rec",
            "-c",
            replay_cmd,
            "--cols",
            str(args.cols),
            "--rows",
            str(args.rows),
            "--overwrite",
            str(cast_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return cast_path


MIN_FPS_CAP = 4
MIN_FONT_SIZE = 9


def render_gif(cast_path: Path, pattern: str, out_dir: Path, args: argparse.Namespace) -> Path:
    """Render a .cast to a GIF, backing off fps cap then font size until it fits the budget.

    Longer demos (guardrails, planning, tool_use, ...) have enough lines that
    the fps cap alone cannot bring them under budget once it bottoms out; the
    second lever, a smaller font size, shrinks every frame's pixel area
    instead of truncating content.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pattern}.gif"
    budget_bytes = int(args.size_budget_mb * 1_000_000)
    fps_cap = args.fps_cap
    font_size = args.font_size

    for attempt in range(1, MAX_RENDER_ATTEMPTS + 1):
        subprocess.run(
            [
                "agg",
                "--cols",
                str(args.cols),
                "--rows",
                str(args.rows),
                "--theme",
                args.theme,
                "--font-size",
                str(font_size),
                "--fps-cap",
                str(fps_cap),
                "--idle-time-limit",
                str(args.idle_time_limit),
                str(cast_path),
                str(out_path),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        size = out_path.stat().st_size
        if size <= budget_bytes or attempt == MAX_RENDER_ATTEMPTS:
            print(
                f"{pattern}: {size / 1_000_000:.2f} MB "
                f"(fps_cap={fps_cap}, font_size={font_size}, attempt {attempt})"
            )
            return out_path
        print(f"{pattern}: {size / 1_000_000:.2f} MB over budget, retrying smaller")
        if fps_cap > MIN_FPS_CAP:
            fps_cap = max(MIN_FPS_CAP, fps_cap - 2)
        else:
            font_size = max(MIN_FONT_SIZE, font_size - 1)

    return out_path


def cmd_record(args: argparse.Namespace) -> None:
    cast_path = record_cast(args.pattern, Path(args.cast_dir), args)
    render_gif(cast_path, args.pattern, Path(args.out_dir), args)


def cmd_record_all(args: argparse.Namespace) -> None:
    patterns = discover_patterns()
    total_bytes = 0
    for pattern in patterns:
        cast_path = record_cast(pattern, Path(args.cast_dir), args)
        gif_path = render_gif(cast_path, pattern, Path(args.out_dir), args)
        total_bytes += gif_path.stat().st_size
    print(f"\n{len(patterns)} GIFs, {total_bytes / 1_000_000:.2f} MB total")


def add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cast-dir", default=str(DEFAULT_CAST_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--theme", default=DEFAULT_THEME)
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    parser.add_argument("--fps-cap", type=int, default=DEFAULT_FPS_CAP)
    parser.add_argument("--idle-time-limit", type=float, default=DEFAULT_IDLE_TIME_LIMIT)
    parser.add_argument("--size-budget-mb", type=float, default=DEFAULT_SIZE_BUDGET_MB)
    parser.add_argument("--target-seconds", type=float, default=None)
    parser.add_argument("--min-delay-ms", type=int, default=DEFAULT_MIN_DELAY_MS)
    parser.add_argument("--max-delay-ms", type=int, default=DEFAULT_MAX_DELAY_MS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay_parser = subparsers.add_parser("replay", help="print paced output for asciinema to record")
    replay_parser.add_argument("pattern")
    replay_parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    replay_parser.add_argument("--target-seconds", type=float, default=DEFAULT_TARGET_SECONDS)
    replay_parser.add_argument("--min-delay-ms", type=int, default=DEFAULT_MIN_DELAY_MS)
    replay_parser.add_argument("--max-delay-ms", type=int, default=DEFAULT_MAX_DELAY_MS)
    replay_parser.set_defaults(func=cmd_replay)

    record_parser = subparsers.add_parser("record", help="record and render one pattern's GIF")
    record_parser.add_argument("pattern", choices=discover_patterns())
    add_render_args(record_parser)
    record_parser.set_defaults(func=cmd_record)

    record_all_parser = subparsers.add_parser("record-all", help="record and render every pattern's GIF")
    add_render_args(record_all_parser)
    record_all_parser.set_defaults(func=cmd_record_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
