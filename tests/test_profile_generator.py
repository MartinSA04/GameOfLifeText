"""Tests for the kernprof-based generator profiler wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path

from game_of_life_text.profile_generator import (
    TARGET_MODULE,
    _build_kernprof_command,
    _build_report,
)


def test_build_kernprof_command_profiles_wrapped_generator_targets() -> None:
    """The kernprof command should point at the generator workload module and targets."""

    args = argparse.Namespace(
        text="HI",
        text_file=None,
        repeat=2,
        keep_cache=True,
    )

    command = _build_kernprof_command(args, Path("profiling/generator_lineprofile.lprof"))

    assert command[:7] == [
        command[0],
        "-m",
        "kernprof",
        "--no-config",
        "-l",
        "-q",
        "-o",
    ]
    assert "game_of_life_text.text.render_text_block_construction.__wrapped__" in command
    assert "game_of_life_text.text._base_construction_footprint.__wrapped__" in command
    assert "game_of_life_text.construction.block_synthesis_variant.__wrapped__" in command
    assert command[-7:] == [
        "-m",
        TARGET_MODULE,
        "--text",
        "HI",
        "--repeat",
        "2",
        "--keep-cache",
    ]


def test_build_report_includes_header_and_rendered_output() -> None:
    """The rendered report should include metadata plus the raw line_profiler text."""

    report = _build_report(
        text="HI",
        repeat=3,
        keep_cache=False,
        wall_seconds=1.25,
        rendered_output="Timer unit: 1e-06 s\n\nTotal time: 0.1 s",
        lprof_output=Path("profiling/generator_lineprofile.lprof"),
        unit=1e-6,
    )

    assert "Generator Line Profile" in report
    assert "Repeat: 3" in report
    assert "Cache mode: cold" in report
    assert 'Input text: "HI"' in report
    assert "line_profiler output" in report
    assert "Timer unit: 1e-06 s" in report
