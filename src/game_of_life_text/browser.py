"""Bridge between the browser worker and the construction planner.

The web app runs this module inside Pyodide: ``web/engine-worker.js`` copies
the package into the Pyodide filesystem and calls the two functions below.
Everything crossing the boundary is JSON, so the worker never handles Python
objects and never has to convert them.
"""

from __future__ import annotations

import json

from .construction import center_construction, minimum_centered_board_size
from .font import FONT_5X7
from .simulator import SimulationConfig, centered_cells
from .text import (
    ProgressCallback,
    render_text_block_construction,
    render_text_block_construction_with_progress,
)

BOARD_PADDING = 8


def generate_text(text: str, on_progress: ProgressCallback | None = None) -> str:
    """Plan a centered glider construction for ``text`` and return it as JSON.

    Planning is the slow half of a generate, and it is synchronous, so the
    worker passes a JS callback in as ``on_progress`` to drive the page's bar.
    It is called with ``(blocks_placed, blocks_total)``.
    """

    normalized = text.rstrip("\n")
    if not normalized.strip():
        msg = "Type at least one visible character."
        raise ValueError(msg)

    plan = (
        render_text_block_construction(normalized)
        if on_progress is None
        else render_text_block_construction_with_progress(normalized, on_progress)
    )
    width, height = minimum_centered_board_size(plan, padding=BOARD_PADDING)
    config = SimulationConfig(width=width, height=height, wrap=False)

    return json.dumps(
        {
            "width": width,
            "height": height,
            "generations": plan.generations,
            "initial": center_construction(config, plan).points.tolist(),
            "target": centered_cells(config, plan.target_cells).points.tolist(),
        },
        separators=(",", ":"),
    )


def supported_characters() -> str:
    """Return the characters the bundled bitmap font can render."""

    return "".join(FONT_5X7)
