"""Workload module executed under kernprof for generator profiling."""

from __future__ import annotations

import argparse
from pathlib import Path

from .construction import block_synthesis_variant
from .text import _base_block_data, render_text_block_construction


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the profiled workload."""

    parser = argparse.ArgumentParser(
        description="Run render_text_block_construction as a kernprof workload.",
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
        help="How many workload runs to execute.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep generator caches warm across runs instead of clearing them each time.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the profiled generator workload."""

    args = parse_args(argv)
    if args.repeat <= 0:
        msg = "--repeat must be greater than 0"
        raise ValueError(msg)

    text = _load_text(args)
    for _ in range(args.repeat):
        if not args.keep_cache:
            _clear_generator_caches()
        render_text_block_construction(text)
    return 0


def _load_text(args: argparse.Namespace) -> str:
    """Load input text from the CLI arguments."""

    if args.text_file is None:
        return args.text
    return args.text_file.read_text(encoding="utf-8").rstrip("\n")


def _clear_generator_caches() -> None:
    """Clear memoized generator helpers for cold-path profiling."""

    render_text_block_construction.cache_clear()
    _base_block_data.cache_clear()
    block_synthesis_variant.cache_clear()


if __name__ == "__main__":
    raise SystemExit(main())
