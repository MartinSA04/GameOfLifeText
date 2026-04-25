"""Run kernprof on the generator and render a readable text report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

PROFILE_OUTPUT = Path("profiling/generator_lineprofile.txt")
TARGET_MODULE = "game_of_life_text.profile_generator_target"
KERNPROF_TARGETS = (
    "game_of_life_text.text.render_text_block_construction.__wrapped__",
    "game_of_life_text.text._pack_block_plans",
    "game_of_life_text.text._base_construction_footprint.__wrapped__",
    "game_of_life_text.construction.plan_block",
    "game_of_life_text.construction.combine_plans",
    "game_of_life_text.construction.evolve_construction",
    "game_of_life_text.construction.block_synthesis_variant.__wrapped__",
    "game_of_life_text.simulator._step_array_n",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the profiling run."""

    parser = argparse.ArgumentParser(
        description=(
            "Run kernprof on render_text_block_construction and save a plain text report."
        )
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--text",
        default="GAME OF LIFE",
        help="Text payload to feed into the block-text generator.",
    )
    source_group.add_argument(
        "--text-file",
        type=Path,
        help="Read the generator input text from a UTF-8 file.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many profiling runs to execute.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep generator caches warm across runs instead of clearing them each time.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROFILE_OUTPUT,
        help="Rendered text output path.",
    )
    parser.add_argument(
        "--lprof-output",
        type=Path,
        help="Optional raw kernprof output path. Defaults to the text output stem with .lprof.",
    )
    parser.add_argument(
        "--unit",
        type=float,
        default=1e-6,
        help="Display timing units passed to python -m line_profiler.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Profile the generator with kernprof and render the results to text."""

    args = parse_args(argv)
    if args.repeat <= 0:
        msg = "--repeat must be greater than 0"
        raise ValueError(msg)

    text = _load_text(args)
    output_path = args.output
    lprof_output = args.lprof_output or output_path.with_suffix(".lprof")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lprof_output.parent.mkdir(parents=True, exist_ok=True)

    kernprof_command = _build_kernprof_command(args, lprof_output)
    line_profiler_command = _build_line_profiler_command(lprof_output, args.unit)

    wall_start = perf_counter()
    _run_command(kernprof_command)
    rendered_output = _run_command(line_profiler_command)
    wall_seconds = perf_counter() - wall_start

    report = _build_report(
        text=text,
        repeat=args.repeat,
        keep_cache=args.keep_cache,
        wall_seconds=wall_seconds,
        rendered_output=rendered_output,
        lprof_output=lprof_output,
        unit=args.unit,
    )
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote rendered line profile to {output_path}")
    print(f"Wrote raw kernprof stats to {lprof_output}")
    return 0


def _load_text(args: argparse.Namespace) -> str:
    """Load input text from the CLI arguments."""

    if args.text_file is None:
        return args.text
    return args.text_file.read_text(encoding="utf-8").rstrip("\n")


def _build_kernprof_command(args: argparse.Namespace, lprof_output: Path) -> list[str]:
    """Build the kernprof command used to capture the raw line profile."""

    command = [
        sys.executable,
        "-m",
        "kernprof",
        "--no-config",
        "-l",
        "-q",
        "-o",
        str(lprof_output),
    ]
    for target in KERNPROF_TARGETS:
        command.extend(["-p", target])
    command.extend(["-m", TARGET_MODULE])
    if args.text_file is not None:
        command.extend(["--text-file", str(args.text_file)])
    else:
        command.extend(["--text", args.text])
    command.extend(["--repeat", str(args.repeat)])
    if args.keep_cache:
        command.append("--keep-cache")
    return command


def _build_line_profiler_command(lprof_output: Path, unit: float) -> list[str]:
    """Build the line_profiler rendering command."""

    return [
        sys.executable,
        "-m",
        "line_profiler",
        "--no-config",
        "--skip-zero",
        "--summarize",
        "-u",
        str(unit),
        str(lprof_output),
    ]


def _run_command(command: list[str]) -> str:
    """Run one subprocess command and return its stdout."""

    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        details = "\n".join(part for part in (stdout, stderr) if part)
        msg = f"command failed: {' '.join(command)}"
        if details:
            msg = f"{msg}\n{details}"
        raise RuntimeError(msg)
    return completed.stdout


def _build_report(
    *,
    text: str,
    repeat: int,
    keep_cache: bool,
    wall_seconds: float,
    rendered_output: str,
    lprof_output: Path,
    unit: float,
) -> str:
    """Build a readable plain-text report around the rendered line-profiler output."""

    header = [
        "Generator Line Profile",
        "======================",
        "Target: game_of_life_text.text.render_text_block_construction",
        f"Repeat: {repeat}",
        f"Cache mode: {'warm' if keep_cache else 'cold'}",
        f"Wall time: {wall_seconds:.6f} s",
        f"Display unit: {unit:g} s",
        f"Raw profile: {lprof_output}",
        f"Input text: {_format_text_value(text)}",
        "",
        "line_profiler output",
        "--------------------",
        rendered_output.rstrip(),
    ]
    return "\n".join(header) + "\n"


def _format_text_value(text: str) -> str:
    """Render the input text as one readable plain-text value."""

    return json.dumps(text)


if __name__ == "__main__":
    raise SystemExit(main())
