"""Render ASCII text as stable Game of Life patterns built from 2x2 blocks."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from typing import Final

import numpy as np

from .construction import (
    BLOCK_PATTERN,
    ConstructionPlan,
    combine_plans,
    evolve_construction,
    plan_block,
)
from .simulator import (
    Pattern,
    Point,
    _step_array,  # type: ignore[reportPrivateUsage]
)

Glyph = tuple[str, ...]
BLOCK_TEXT_STRIDE: Final[int] = 6
BLOCK_TEXT_LETTER_SPACING: Final[int] = BLOCK_TEXT_STRIDE
BLOCK_TEXT_LINE_SPACING: Final[int] = BLOCK_TEXT_STRIDE * 2
_MAX_PACK_SLOT: Final[int] = 256

FONT_5X7: Final[dict[str, Glyph]] = {
    " ": (
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
    ),
    "!": (
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        ".....",
        "..#..",
    ),
    "-": (
        ".....",
        ".....",
        ".....",
        "#####",
        ".....",
        ".....",
        ".....",
    ),
    ".": (
        ".....",
        ".....",
        ".....",
        ".....",
        ".....",
        "..#..",
        "..#..",
    ),
    "?": (
        ".###.",
        "#...#",
        "....#",
        "...#.",
        "..#..",
        ".....",
        "..#..",
    ),
    "0": (
        ".###.",
        "#...#",
        "#..##",
        "#.#.#",
        "##..#",
        "#...#",
        ".###.",
    ),
    "1": (
        "..#..",
        ".##..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        ".###.",
    ),
    "2": (
        ".###.",
        "#...#",
        "....#",
        "...#.",
        "..#..",
        ".#...",
        "#####",
    ),
    "3": (
        "####.",
        "....#",
        "....#",
        ".###.",
        "....#",
        "....#",
        "####.",
    ),
    "4": (
        "...#.",
        "..##.",
        ".#.#.",
        "#..#.",
        "#####",
        "...#.",
        "...#.",
    ),
    "5": (
        "#####",
        "#....",
        "#....",
        "####.",
        "....#",
        "....#",
        "####.",
    ),
    "6": (
        ".###.",
        "#....",
        "#....",
        "####.",
        "#...#",
        "#...#",
        ".###.",
    ),
    "7": (
        "#####",
        "....#",
        "...#.",
        "..#..",
        ".#...",
        ".#...",
        ".#...",
    ),
    "8": (
        ".###.",
        "#...#",
        "#...#",
        ".###.",
        "#...#",
        "#...#",
        ".###.",
    ),
    "9": (
        ".###.",
        "#...#",
        "#...#",
        ".####",
        "....#",
        "....#",
        ".###.",
    ),
    "A": (
        ".###.",
        "#...#",
        "#...#",
        "#####",
        "#...#",
        "#...#",
        "#...#",
    ),
    "B": (
        "####.",
        "#...#",
        "#...#",
        "####.",
        "#...#",
        "#...#",
        "####.",
    ),
    "C": (
        ".###.",
        "#...#",
        "#....",
        "#....",
        "#....",
        "#...#",
        ".###.",
    ),
    "D": (
        "####.",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "####.",
    ),
    "E": (
        "#####",
        "#....",
        "#....",
        "####.",
        "#....",
        "#....",
        "#####",
    ),
    "F": (
        "#####",
        "#....",
        "#....",
        "####.",
        "#....",
        "#....",
        "#....",
    ),
    "G": (
        ".###.",
        "#...#",
        "#....",
        "#.###",
        "#...#",
        "#...#",
        ".###.",
    ),
    "H": (
        "#...#",
        "#...#",
        "#...#",
        "#####",
        "#...#",
        "#...#",
        "#...#",
    ),
    "I": (
        "#####",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "#####",
    ),
    "J": (
        "..###",
        "...#.",
        "...#.",
        "...#.",
        "...#.",
        "#..#.",
        ".##..",
    ),
    "K": (
        "#...#",
        "#..#.",
        "#.#..",
        "##...",
        "#.#..",
        "#..#.",
        "#...#",
    ),
    "L": (
        "#....",
        "#....",
        "#....",
        "#....",
        "#....",
        "#....",
        "#####",
    ),
    "M": (
        "#...#",
        "##.##",
        "#.#.#",
        "#.#.#",
        "#...#",
        "#...#",
        "#...#",
    ),
    "N": (
        "#...#",
        "##..#",
        "#.#.#",
        "#..##",
        "#...#",
        "#...#",
        "#...#",
    ),
    "O": (
        ".###.",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        ".###.",
    ),
    "P": (
        "####.",
        "#...#",
        "#...#",
        "####.",
        "#....",
        "#....",
        "#....",
    ),
    "Q": (
        ".###.",
        "#...#",
        "#...#",
        "#...#",
        "#.#.#",
        "#..#.",
        ".##.#",
    ),
    "R": (
        "####.",
        "#...#",
        "#...#",
        "####.",
        "#.#..",
        "#..#.",
        "#...#",
    ),
    "S": (
        ".####",
        "#....",
        "#....",
        ".###.",
        "....#",
        "....#",
        "####.",
    ),
    "T": (
        "#####",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
    ),
    "U": (
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        ".###.",
    ),
    "V": (
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        "#...#",
        ".#.#.",
        "..#..",
    ),
    "W": (
        "#...#",
        "#...#",
        "#...#",
        "#.#.#",
        "#.#.#",
        "##.##",
        "#...#",
    ),
    "X": (
        "#...#",
        "#...#",
        ".#.#.",
        "..#..",
        ".#.#.",
        "#...#",
        "#...#",
    ),
    "Y": (
        "#...#",
        "#...#",
        ".#.#.",
        "..#..",
        "..#..",
        "..#..",
        "..#..",
    ),
    "Z": (
        "#####",
        "....#",
        "...#.",
        "..#..",
        ".#...",
        "#....",
        "#####",
    ),
}


def render_text_block_pattern(text: str) -> Pattern:
    """Render text into a stable still-life pattern made from 2x2 blocks."""

    normalized_text = _normalize_text(text)
    origins = _text_pixel_origins(normalized_text)
    if not origins:
        return Pattern.empty()

    origin_array = np.asarray(origins, dtype=np.int32)
    block_array = BLOCK_PATTERN.points
    tiled = origin_array[:, None, :] + block_array[None, :, :]
    return Pattern(tiled.reshape(-1, 2))


@cache
def render_text_block_construction(text: str) -> ConstructionPlan:
    """Return a deterministic glider seed plan that settles into block text.

    Builds blocks outward from the text center using the same outward launch
    direction as the original sequential planner. Each block claims the smallest
    launch period whose glider trajectory does not collide with any already-placed
    block, so distant blocks share a slot and settle in parallel.
    """

    normalized_text = _normalize_text(text)
    target_cells = render_text_block_pattern(normalized_text)
    block_origins = _extract_block_origins(target_cells)
    if not block_origins:
        msg = "block text construction requires at least one block"
        raise ValueError(msg)

    center_x = sum(x + 0.5 for x, _ in block_origins) / len(block_origins)
    center_y = sum(y + 0.5 for _, y in block_origins) / len(block_origins)
    ordered_origins = sorted(
        block_origins,
        key=lambda origin: (
            _block_distance_squared(origin, center=(center_x, center_y)),
            origin[1],
            origin[0],
        ),
    )

    plan = combine_plans(_pack_block_plans(ordered_origins, center=(center_x, center_y)))
    if evolve_construction(plan) != target_cells:
        msg = "deterministic block-text construction verification failed; try shorter text"
        raise ValueError(msg)
    return ConstructionPlan(
        initial_cells=plan.initial_cells,
        target_cells=target_cells,
        generations=plan.generations,
    )


def glyph_for_character(character: str) -> Glyph:
    """Return the 5x7 glyph for one supported character."""

    normalized = character.upper()
    if normalized not in FONT_5X7:
        msg = f"unsupported character {character!r}"
        raise ValueError(msg)
    return FONT_5X7[normalized]


def _pack_block_plans(
    ordered_origins: Sequence[Point],
    *,
    center: tuple[float, float],
) -> list[ConstructionPlan]:
    """Place each block at the smallest launch period that keeps all glider paths apart.

    Blocks are placed in distance-from-center order with the same outward launch
    direction the original sequential planner picked. For each block we sweep
    extra_periods upward and accept the first value whose Moore-expanded swept
    cells are spatially disjoint from every already-placed block — those blocks
    are guaranteed independent and run in parallel. When a candidate's
    footprint touches an existing block we verify by simulating the candidate
    together with the touching subset; non-touching blocks cannot disturb the
    result and can be ignored, keeping the verification small.
    """

    placed: list[tuple[ConstructionPlan, Pattern]] = []
    for origin in ordered_origins:
        orientation = _block_orientation(origin, center=center)
        for extra_periods in range(_MAX_PACK_SLOT + 1):
            candidate = plan_block(origin, orientation=orientation, extra_periods=extra_periods)
            candidate_footprint = _block_construction_footprint(orientation, extra_periods, origin)

            touching = [
                placed_plan
                for placed_plan, placed_fp in placed
                if not placed_fp.isdisjoint(candidate_footprint)
            ]
            if not touching:
                placed.append((candidate, candidate_footprint))
                break

            tentative = combine_plans([*touching, candidate])
            expected_patterns = [plan.target_cells for plan in touching]
            expected_patterns.append(candidate.target_cells)
            expected = Pattern.merge(*expected_patterns)
            if evolve_construction(tentative) == expected:
                placed.append((candidate, candidate_footprint))
                break
        else:
            msg = f"could not pack block at {origin!r} within {_MAX_PACK_SLOT} slots"
            raise ValueError(msg)
    return [plan for plan, _ in placed]


@cache
def _base_construction_footprint(orientation: str, extra_periods: int) -> Pattern:
    """Return the Moore-expanded swept cells for one block construction at origin (0, 0)."""

    plan = plan_block((0, 0), orientation=orientation, extra_periods=extra_periods)
    if plan.initial_cells:
        min_x, min_y, max_x, max_y = plan.initial_cells.bounds()
        padding = plan.generations + 4
        height = max_y - min_y + 1 + padding * 2
        width = max_x - min_x + 1 + padding * 2
        grid = np.zeros((height, width), dtype=np.bool_)
        points = plan.initial_cells.points
        grid[points[:, 1] - min_y + padding, points[:, 0] - min_x + padding] = True
        swept_arr = grid.copy()
        for _ in range(plan.generations):
            grid = _step_array(grid, wrap=False)
            swept_arr |= grid
        swept = Pattern.from_grid(swept_arr).translated(min_x - padding, min_y - padding)
    else:
        swept = Pattern.empty()
    swept = Pattern.merge(swept, plan.target_cells)
    if not swept:
        return Pattern.empty()

    offsets = np.asarray(
        [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
        dtype=np.int32,
    )
    expanded = swept.points[:, None, :] + offsets[None, :, :]
    return Pattern(expanded.reshape(-1, 2))


def _block_construction_footprint(
    orientation: str, extra_periods: int, origin: Point
) -> Pattern:
    """Translate the cached base footprint to ``origin``."""

    base = _base_construction_footprint(orientation, extra_periods)
    return base.translated(origin[0], origin[1])


def _normalize_text(text: str) -> str:
    """Normalize text input and reject empty payloads."""

    normalized_text = text.upper()
    if not normalized_text:
        msg = "text cannot be empty"
        raise ValueError(msg)
    return normalized_text


def _text_pixel_origins(text: str) -> tuple[Point, ...]:
    """Return the top-left origin of each live glyph pixel in the text layout."""

    glyph_height = len(next(iter(FONT_5X7.values())))
    origins: list[Point] = []
    line_y = 0
    for line in text.splitlines():
        cursor_x = 0
        for character in line:
            glyph = glyph_for_character(character)
            for glyph_y, row in enumerate(glyph):
                for glyph_x, token in enumerate(row):
                    if token != "#":
                        continue
                    origins.append(
                        (
                            cursor_x + glyph_x * BLOCK_TEXT_STRIDE,
                            line_y + glyph_y * BLOCK_TEXT_STRIDE,
                        )
                    )
            cursor_x += len(glyph[0]) * BLOCK_TEXT_STRIDE + BLOCK_TEXT_LETTER_SPACING
        line_y += glyph_height * BLOCK_TEXT_STRIDE + BLOCK_TEXT_LINE_SPACING
    return tuple(origins)


def _extract_block_origins(pattern: Pattern) -> tuple[Point, ...]:
    """Return the top-left cell for each 2x2 block in a pattern."""

    if not pattern:
        return ()

    min_x, min_y, max_x, max_y = pattern.bounds()
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    mask = np.zeros((height, width), dtype=np.bool_)
    rel = pattern.points - np.array((min_x, min_y), dtype=np.int32)
    mask[rel[:, 1], rel[:, 0]] = True

    origins: list[Point] = []
    for x, y in pattern:
        rel_x = x - min_x
        rel_y = y - min_y
        if rel_x + 1 >= width or rel_y + 1 >= height:
            continue
        if not (
            mask[rel_y, rel_x]
            and mask[rel_y, rel_x + 1]
            and mask[rel_y + 1, rel_x]
            and mask[rel_y + 1, rel_x + 1]
        ):
            continue
        if (rel_x > 0 and mask[rel_y, rel_x - 1]) or (rel_y > 0 and mask[rel_y - 1, rel_x]):
            continue
        origins.append((x, y))
    return tuple(origins)


def _block_orientation(origin: Point, *, center: tuple[float, float]) -> str:
    """Return the deterministic outward launch direction for one block."""

    center_x, center_y = center
    delta_x = origin[0] + 0.5 - center_x
    delta_y = origin[1] + 0.5 - center_y
    if abs(delta_x) >= abs(delta_y):
        return "east" if delta_x >= 0 else "west"
    return "south" if delta_y >= 0 else "north"


def _block_distance_squared(origin: Point, *, center: tuple[float, float]) -> float:
    """Return the squared distance from the shared text center."""

    center_x, center_y = center
    delta_x = origin[0] + 0.5 - center_x
    delta_y = origin[1] + 0.5 - center_y
    return delta_x * delta_x + delta_y * delta_y
