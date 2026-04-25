"""Render ASCII text as stable Game of Life patterns built from 2x2 blocks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import cache
from typing import Final, NamedTuple

from .construction import (
    BLOCK_PATTERN,
    ConstructionPlan,
    combine_plans,
    evolve_construction,
    plan_block,
)
from .simulator import Board, Pattern, Point, SimulationConfig

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
    live_cells: set[Point] = set()
    for origin in _text_pixel_origins(normalized_text):
        live_cells.update(
            (origin[0] + tile_x, origin[1] + tile_y) for tile_x, tile_y in BLOCK_PATTERN
        )
    return frozenset(live_cells)


@cache
def render_text_block_construction(text: str) -> ConstructionPlan:
    """Return a deterministic glider seed plan that settles into block text.

    Each block is packed into the smallest launch period whose (cell, generation)
    glider footprint does not collide with any block already placed, so blocks
    in disjoint regions of the text settle in parallel rather than in series.
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

    plan = combine_plans(
        _pack_block_plans(ordered_origins, center=(center_x, center_y))
    )
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


class _BlockTimeline(NamedTuple):
    """The (cell, generation) footprint of one block construction with adjacency margin."""

    cells_at_gen: tuple[frozenset[Point], ...]
    settled_cells: frozenset[Point]


def _pack_block_plans(
    ordered_origins: Sequence[Point],
    *,
    center: tuple[float, float],
) -> list[ConstructionPlan]:
    """Pack each block into the smallest non-conflicting (slot, orientation) pair.

    Adjacent blocks often have a chain of constraints where one block's choice cuts
    off every option for a later block. We use bounded depth-first backtracking:
    when no (slot, orientation) fits the current block, we undo the most recent
    placement and try its next option.
    """

    preferences_by_origin = {
        origin: _block_orientation_preferences(origin, center=center)
        for origin in ordered_origins
    }
    placed: list[tuple[ConstructionPlan, _BlockTimeline]] = []
    budget = [len(ordered_origins) * 64]

    def fits(timeline: _BlockTimeline) -> bool:
        return all(not _timelines_conflict(timeline, other) for _, other in placed)

    def backtrack(idx: int) -> bool:
        if idx == len(ordered_origins):
            return True
        if budget[0] <= 0:
            return False
        budget[0] -= 1

        origin = ordered_origins[idx]
        for extra_periods in range(_MAX_PACK_SLOT + 1):
            for orientation in preferences_by_origin[origin]:
                candidate_timeline = _translate_timeline(
                    _base_block_timeline(orientation, extra_periods),
                    origin,
                )
                if not fits(candidate_timeline):
                    continue
                candidate_plan = plan_block(
                    origin, orientation=orientation, extra_periods=extra_periods
                )
                placed.append((candidate_plan, candidate_timeline))
                if backtrack(idx + 1):
                    return True
                placed.pop()
                if budget[0] <= 0:
                    return False
        return False

    if not backtrack(0):
        msg = "could not pack block-text construction within search budget"
        raise ValueError(msg)
    return [plan for plan, _ in placed]


@cache
def _base_block_timeline(orientation: str, extra_periods: int) -> _BlockTimeline:
    """Return the per-generation footprint of one block construction at origin (0, 0)."""

    plan = plan_block((0, 0), orientation=orientation, extra_periods=extra_periods)
    settled_cells = _expand_with_adjacency(plan.target_cells)
    if not plan.initial_cells:
        return _BlockTimeline(cells_at_gen=(), settled_cells=settled_cells)

    min_x = min(x for x, _ in plan.initial_cells)
    min_y = min(y for _, y in plan.initial_cells)
    max_x = max(x for x, _ in plan.initial_cells)
    max_y = max(y for _, y in plan.initial_cells)
    padding = plan.generations + 4
    config = SimulationConfig(
        width=max_x - min_x + 1 + padding * 2,
        height=max_y - min_y + 1 + padding * 2,
        wrap=False,
    )
    shifted = {(x - min_x + padding, y - min_y + padding) for x, y in plan.initial_cells}
    board = Board.from_points(config, shifted)

    cells_at_gen: list[frozenset[Point]] = []
    for _ in range(plan.generations + 1):
        live_real = ((x + min_x - padding, y + min_y - padding) for x, y in board.live_cells)
        cells_at_gen.append(_expand_with_adjacency(live_real))
        board = board.step()

    return _BlockTimeline(cells_at_gen=tuple(cells_at_gen), settled_cells=settled_cells)


def _translate_timeline(timeline: _BlockTimeline, origin: Point) -> _BlockTimeline:
    """Shift a base timeline by ``origin``."""

    offset_x, offset_y = origin
    return _BlockTimeline(
        cells_at_gen=tuple(
            frozenset((x + offset_x, y + offset_y) for x, y in cells)
            for cells in timeline.cells_at_gen
        ),
        settled_cells=frozenset(
            (x + offset_x, y + offset_y) for x, y in timeline.settled_cells
        ),
    )


def _expand_with_adjacency(cells: Iterable[Point]) -> frozenset[Point]:
    """Return ``cells`` plus every Moore neighbor of each cell."""

    expanded: set[Point] = set()
    for x, y in cells:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                expanded.add((x + dx, y + dy))
    return frozenset(expanded)


def _timelines_conflict(a: _BlockTimeline, b: _BlockTimeline) -> bool:
    """Return whether two block timelines have an active cell in common at the same gen."""

    horizon = max(len(a.cells_at_gen), len(b.cells_at_gen))
    for t in range(horizon):
        a_cells = a.cells_at_gen[t] if t < len(a.cells_at_gen) else a.settled_cells
        b_cells = b.cells_at_gen[t] if t < len(b.cells_at_gen) else b.settled_cells
        if not a_cells.isdisjoint(b_cells):
            return True
    return not a.settled_cells.isdisjoint(b.settled_cells)


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

    origins: list[Point] = []
    for x, y in sorted(pattern):
        block_cells = {(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)}
        if not block_cells.issubset(pattern):
            continue
        if (x - 1, y) in pattern or (x, y - 1) in pattern:
            continue
        origins.append((x, y))
    return tuple(origins)


def _block_orientation_preferences(
    origin: Point,
    *,
    center: tuple[float, float],
) -> tuple[str, ...]:
    """Return the four launch directions, outward-first, for one block."""

    center_x, center_y = center
    delta_x = origin[0] + 0.5 - center_x
    delta_y = origin[1] + 0.5 - center_y
    horizontal = "east" if delta_x >= 0 else "west"
    vertical = "south" if delta_y >= 0 else "north"
    primary, secondary = (
        (horizontal, vertical) if abs(delta_x) >= abs(delta_y) else (vertical, horizontal)
    )
    rest = tuple(d for d in ("east", "west", "north", "south") if d not in {primary, secondary})
    return (primary, secondary, *rest)


def _block_distance_squared(origin: Point, *, center: tuple[float, float]) -> float:
    """Return the squared distance from the shared text center."""

    center_x, center_y = center
    delta_x = origin[0] + 0.5 - center_x
    delta_y = origin[1] + 0.5 - center_y
    return delta_x * delta_x + delta_y * delta_y
