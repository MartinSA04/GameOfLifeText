"""Render ASCII text as stable Game of Life patterns built from 2x2 blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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

    Builds blocks outward from the text center. Each block tries the two outward
    launch directions implied by its position relative to that center, and
    claims the smallest launch period whose glider trajectory does not collide
    with any already-placed block, so distant blocks share a slot and settle
    in parallel.
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

    Blocks are placed in distance-from-center order. For each block we sweep
    extra_periods upward and try both outward launch directions implied by the
    block's position relative to the shared center, accepting the first
    candidate whose Moore-expanded swept cells are spatially disjoint from every
    already-placed block. Those blocks are guaranteed independent and run in
    parallel. When a candidate's footprint touches an existing block we verify
    by simulating the candidate together with the touching subset; non-touching
    blocks cannot disturb the result and can be ignored, keeping the
    verification small.
    """

    placed: list[_PlacedBlock] = []
    for origin in ordered_origins:
        orientations = _block_orientations(origin, center=center)
        accepted = False
        for extra_periods in range(_MAX_PACK_SLOT + 1):
            for orientation in orientations:
                base_data = _base_block_data(orientation, extra_periods)
                candidate_footprint = base_data.footprint.translated(origin[0], origin[1])

                touching = [
                    placed_block
                    for placed_block in placed
                    if not placed_block.footprint.isdisjoint(candidate_footprint)
                ]
                if not touching:
                    candidate = plan_block(
                        origin, orientation=orientation, extra_periods=extra_periods
                    )
                    placed.append(
                        _PlacedBlock(
                            plan=candidate,
                            footprint=candidate_footprint,
                            origin=origin,
                            orientation=orientation,
                            extra_periods=extra_periods,
                        )
                    )
                    accepted = True
                    break

                # Cheap precise pairwise filter: if any single touching block
                # collides with the candidate at the same generation, the slot
                # is unreachable. The check is cached on relative geometry, so
                # the regular text grid produces heavy cache hits.
                if any(
                    _pair_conflicts_cached(
                        tb.orientation,
                        tb.extra_periods,
                        orientation,
                        extra_periods,
                        origin[0] - tb.origin[0],
                        origin[1] - tb.origin[1],
                    )
                    for tb in touching
                ):
                    continue

                candidate = plan_block(origin, orientation=orientation, extra_periods=extra_periods)
                tentative = combine_plans([tb.plan for tb in touching] + [candidate])
                expected_patterns = [tb.plan.target_cells for tb in touching]
                expected_patterns.append(candidate.target_cells)
                expected = Pattern.merge(*expected_patterns)
                if evolve_construction(tentative) == expected:
                    placed.append(
                        _PlacedBlock(
                            plan=candidate,
                            footprint=candidate_footprint,
                            origin=origin,
                            orientation=orientation,
                            extra_periods=extra_periods,
                        )
                    )
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            msg = f"could not pack block at {origin!r} within {_MAX_PACK_SLOT} slots"
            raise ValueError(msg)
    return [block.plan for block in placed]


@dataclass(frozen=True, slots=True)
class _PlacedBlock:
    """One placed block tagged with the metadata pair-checks need."""

    plan: ConstructionPlan
    footprint: Pattern
    origin: Point
    orientation: str
    extra_periods: int


@cache
def _pair_conflicts_cached(
    orientation_a: str,
    extra_periods_a: int,
    orientation_b: str,
    extra_periods_b: int,
    dx: int,
    dy: int,
) -> bool:
    """Pairwise conflict result for one relative geometry, cached forever.

    Two block constructions either interact at some generation or they do not;
    the answer is a deterministic function of the launch directions, the relative
    offset, and the two extra_periods values. Text glyphs reuse the same
    relative geometry over and over, so this cache becomes the hot path.
    """

    a_data = _base_block_data(orientation_a, extra_periods_a)
    b_data = _base_block_data(orientation_b, extra_periods_b)
    return _pair_conflicts_timeline(a_data, (0, 0), b_data, (dx, dy))


@dataclass(frozen=True, slots=True)
class _BaseBlockData:
    """Per-(orientation, extra_periods) precomputed data for one block at origin (0, 0).

    ``cells_per_gen`` and ``shadow_per_gen`` store cells as bit-packed integers
    rather than ``(x, y)`` tuples so the pairwise-conflict check can shift a
    timeline by one ``int`` offset and use native ``frozenset`` operations.
    """

    footprint: Pattern
    cells_per_gen: tuple[frozenset[int], ...]
    shadow_per_gen: tuple[frozenset[int], ...]
    settled: frozenset[int]
    settled_shadow: frozenset[int]


# Bit-packing constants for cell coordinates. With offset 2^14 each axis we can
# represent any point inside ±16K, far beyond the largest text we render.
_COORD_OFFSET: Final[int] = 1 << 14
_COORD_SCALE: Final[int] = 1 << 16


def _pack_point(x: int, y: int) -> int:
    return ((x + _COORD_OFFSET) * _COORD_SCALE) + (y + _COORD_OFFSET)


def _shift_constant(dx: int, dy: int) -> int:
    return dx * _COORD_SCALE + dy


def _pair_conflicts_timeline(
    a: _BaseBlockData,
    a_origin: Point,
    b: _BaseBlockData,
    b_origin: Point,
) -> bool:
    """Return True if A and B have live cells within Moore distance 1 at any same gen.

    Pair conflicts are precisely captured here: two cells that are one apart
    affect each other's next state directly. Cells two apart cannot pairwise
    interact (they only meet at a shared neighbor cell that, in the pair,
    gets a single contribution from each side and stays dead). Multi-block
    collusion on a shared neighbor still requires a full simulation.
    """

    shift = _shift_constant(b_origin[0] - a_origin[0], b_origin[1] - a_origin[1])
    horizon = max(len(a.cells_per_gen), len(b.cells_per_gen))
    a_settled = a.settled
    b_settled_shadow = b.settled_shadow
    a_cells_per_gen = a.cells_per_gen
    b_shadow_per_gen = b.shadow_per_gen
    a_len = len(a_cells_per_gen)
    b_len = len(b_shadow_per_gen)
    if shift == 0:
        for t in range(horizon):
            a_cells = a_cells_per_gen[t] if t < a_len else a_settled
            b_shadow = b_shadow_per_gen[t] if t < b_len else b_settled_shadow
            if not a_cells.isdisjoint(b_shadow):
                return True
        return not a_settled.isdisjoint(b_settled_shadow)
    for t in range(horizon):
        a_cells = a_cells_per_gen[t] if t < a_len else a_settled
        b_shadow = b_shadow_per_gen[t] if t < b_len else b_settled_shadow
        for p in a_cells:
            if (p - shift) in b_shadow:
                return True
    return any(p + shift in a_settled for p in b_settled_shadow)


@cache
def _base_block_data(orientation: str, extra_periods: int) -> _BaseBlockData:
    """Compute footprint and per-generation cell sets for one block at origin (0, 0).

    The simulation is shared between the spatial footprint (used as the fast
    pattern-disjoint pre-filter to find touching neighbors) and the per-gen
    bit-packed cell sets (used by the precise pairwise-conflict filter that
    rejects bad extra_periods values without ever running a combined-plan
    simulation).
    """

    plan = plan_block((0, 0), orientation=orientation, extra_periods=extra_periods)
    settled = _pack_pattern(plan.target_cells)
    settled_shadow = _expand_packed_with_adjacency(settled)

    if not plan.initial_cells:
        return _BaseBlockData(
            footprint=Pattern.empty(),
            cells_per_gen=(),
            shadow_per_gen=(),
            settled=settled,
            settled_shadow=settled_shadow,
        )

    min_x, min_y, max_x, max_y = plan.initial_cells.bounds()
    if plan.target_cells:
        target_min_x, target_min_y, target_max_x, target_max_y = plan.target_cells.bounds()
        min_x = min(min_x, target_min_x)
        min_y = min(min_y, target_min_y)
        max_x = max(max_x, target_max_x)
        max_y = max(max_y, target_max_y)
    padding = 4
    height = max_y - min_y + 1 + padding * 2
    width = max_x - min_x + 1 + padding * 2
    grid = np.zeros((height, width), dtype=np.bool_)
    points = plan.initial_cells.points
    grid[points[:, 1] - min_y + padding, points[:, 0] - min_x + padding] = True

    swept_arr = grid.copy()
    cells_per_gen: list[frozenset[int]] = []
    shadow_per_gen: list[frozenset[int]] = []
    for _ in range(plan.generations + 1):
        ys, xs = np.nonzero(grid)
        packed = frozenset(
            _pack_point(int(x) + min_x - padding, int(y) + min_y - padding)
            for x, y in zip(xs, ys, strict=True)
        )
        cells_per_gen.append(packed)
        shadow_per_gen.append(_expand_packed_with_adjacency(packed))
        swept_arr |= grid
        grid = _step_array(grid, wrap=False)

    swept_pattern = Pattern.from_grid(swept_arr).translated(min_x - padding, min_y - padding)
    swept_with_target = Pattern.merge(swept_pattern, plan.target_cells)
    footprint = _moore_expand_pattern(swept_with_target)
    return _BaseBlockData(
        footprint=footprint,
        cells_per_gen=tuple(cells_per_gen),
        shadow_per_gen=tuple(shadow_per_gen),
        settled=settled,
        settled_shadow=settled_shadow,
    )


def _pack_pattern(pattern: Pattern) -> frozenset[int]:
    if not pattern:
        return frozenset()
    return frozenset(_pack_point(int(p[0]), int(p[1])) for p in pattern.points)


_NEIGHBOR_SHIFTS: Final[tuple[int, ...]] = tuple(
    _shift_constant(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
)


def _expand_packed_with_adjacency(cells: frozenset[int]) -> frozenset[int]:
    if not cells:
        return cells
    expanded: set[int] = set()
    for shift in _NEIGHBOR_SHIFTS:
        expanded.update(p + shift for p in cells)
    return frozenset(expanded)


def _moore_expand_pattern(pattern: Pattern) -> Pattern:
    if not pattern:
        return Pattern.empty()
    offsets = np.asarray(
        [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
        dtype=np.int32,
    )
    expanded = pattern.points[:, None, :] + offsets[None, :, :]
    return Pattern(expanded.reshape(-1, 2))


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


def _block_orientations(origin: Point, *, center: tuple[float, float]) -> tuple[str, str]:
    """Return the two outward launch directions to try for one block.

    The dominant axis is tried first to preserve the previous deterministic
    preference, and the secondary axis is used as a fallback during search.
    """

    center_x, center_y = center
    delta_x = origin[0] + 0.5 - center_x
    delta_y = origin[1] + 0.5 - center_y
    horizontal = "east" if delta_x >= 0 else "west"
    vertical = "south" if delta_y >= 0 else "north"
    if abs(delta_x) >= abs(delta_y):
        return (horizontal, vertical)
    return (vertical, horizontal)


def _block_distance_squared(origin: Point, *, center: tuple[float, float]) -> float:
    """Return the squared distance from the shared text center."""

    center_x, center_y = center
    delta_x = origin[0] + 0.5 - center_x
    delta_y = origin[1] + 0.5 - center_y
    return delta_x * delta_x + delta_y * delta_y
