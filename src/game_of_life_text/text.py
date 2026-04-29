"""Render ASCII text as stable Game of Life patterns built from 2x2 blocks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
from .font import DEFAULT_ASCENT, DESCENT, FONT_5X7, Glyph
from .simulator import (
    Pattern,
    Point,
    _step_array,  # type: ignore[reportPrivateUsage]
)

__all__ = [
    "FONT_5X7",
    "Glyph",
    "glyph_for_character",
    "render_text_block_construction",
    "render_text_block_construction_with_progress",
    "render_text_block_pattern",
]

ProgressCallback = Callable[[int, int], None]
BLOCK_TEXT_STRIDE: Final[int] = 6
BLOCK_TEXT_LETTER_SPACING: Final[int] = BLOCK_TEXT_STRIDE
BLOCK_TEXT_LINE_SPACING: Final[int] = BLOCK_TEXT_STRIDE * 2
_MAX_PACK_SLOT: Final[int] = 256


def render_text_block_pattern(text: str) -> Pattern:
    """Render text into a stable still-life pattern made from 2x2 blocks."""

    origins = _text_pixel_origins(text)
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

    return _render_text_block_construction(text, progress_callback=None)


def render_text_block_construction_with_progress(
    text: str,
    progress_callback: ProgressCallback,
) -> ConstructionPlan:
    """Return a text construction while reporting ``current, total`` build progress."""

    return _render_text_block_construction(text, progress_callback=progress_callback)


def _render_text_block_construction(
    text: str,
    *,
    progress_callback: ProgressCallback | None,
) -> ConstructionPlan:
    """Return a text construction with optional block-packing progress reporting."""

    target_cells = render_text_block_pattern(text)
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

    total_steps = len(ordered_origins) + 1
    if progress_callback is not None:
        progress_callback(0, total_steps)
    plan = combine_plans(
        _pack_block_plans(
            ordered_origins,
            center=(center_x, center_y),
            progress_callback=progress_callback,
            total_steps=total_steps,
        )
    )
    if evolve_construction(plan) != target_cells:
        msg = "deterministic block-text construction verification failed; try shorter text"
        raise ValueError(msg)
    if progress_callback is not None:
        progress_callback(total_steps, total_steps)
    return ConstructionPlan(
        initial_cells=plan.initial_cells,
        target_cells=target_cells,
        generations=plan.generations,
    )


def glyph_for_character(character: str) -> Glyph:
    """Return the 5x7 glyph for one supported character."""

    if character not in FONT_5X7:
        msg = f"unsupported character {character!r}"
        raise ValueError(msg)
    return FONT_5X7[character]


def _pack_block_plans(
    ordered_origins: Sequence[Point],
    *,
    center: tuple[float, float],
    progress_callback: ProgressCallback | None = None,
    total_steps: int | None = None,
) -> list[ConstructionPlan]:
    """Place each block at the smallest launch period whose evolution composes
    cleanly with every placed block's evolution at every generation.

    Two checks together let us accept a slot without ever simulating the
    combined plan. First, every candidate cell must be at Moore distance >= 2
    from every master cell at the same generation, so no live cell ever sees a
    neighbor it would not see when its block runs alone. Second, at every dead
    cell receiving live neighbors from both the candidate and the master, the
    combined neighbor count must give the same next state as the master and the
    candidate individually — the only way the planar union can diverge from the
    union of planar evolutions is if two blocks together flip a dead cell into
    a state neither of them flips alone, and that flip is fully determined by
    the per-cell neighbor counts the master tracks across generations.
    """

    placed_plans: list[ConstructionPlan] = []
    master_footprint: set[int] = set()
    master_shadow_per_gen: list[set[int]] = []
    master_settled_shadow: set[int] = set()
    master_neighbor_count_per_gen: list[dict[int, int]] = []
    master_settled_neighbor_count: dict[int, int] = {}

    for origin in ordered_origins:
        orientations = _block_orientations(origin, center=center)
        shift = _shift_constant(origin[0], origin[1])
        accepted = False
        for extra_periods in range(_MAX_PACK_SLOT + 1):
            for orientation in orientations:
                base_data = _base_block_data(orientation, extra_periods)
                if not _candidate_safe(
                    base_data,
                    shift,
                    master_footprint,
                    master_shadow_per_gen,
                    master_settled_shadow,
                    master_neighbor_count_per_gen,
                    master_settled_neighbor_count,
                ):
                    continue

                candidate = plan_block(
                    origin, orientation=orientation, extra_periods=extra_periods
                )
                placed_plans.append(candidate)
                _master_absorb(
                    base_data,
                    shift,
                    master_footprint,
                    master_shadow_per_gen,
                    master_settled_shadow,
                    master_neighbor_count_per_gen,
                    master_settled_neighbor_count,
                )
                accepted = True
                break
            if accepted:
                break
        if not accepted:
            msg = f"could not pack block at {origin!r} within {_MAX_PACK_SLOT} slots"
            raise ValueError(msg)
        if progress_callback is not None:
            progress_callback(len(placed_plans), total_steps or len(ordered_origins))
    return placed_plans


def _candidate_safe(
    base_data: _BaseBlockData,
    shift: int,
    master_footprint: set[int],
    master_shadow_per_gen: list[set[int]],
    master_settled_shadow: set[int],
    master_neighbor_count_per_gen: list[dict[int, int]],
    master_settled_neighbor_count: dict[int, int],
) -> bool:
    """Return True iff the shifted candidate composes cleanly with the master."""

    if not master_footprint:
        return True
    # Spacetime-union pre-filter: if no cell ever touched by the candidate is
    # ever touched by any placed block, both checks below are vacuous.
    overlapping = False
    for p in base_data.footprint:
        if (p + shift) in master_footprint:
            overlapping = True
            break
    if not overlapping:
        return True

    cand_cells_per_gen = base_data.cells_per_gen
    cand_settled = base_data.settled
    cand_len = len(cand_cells_per_gen)
    master_len = len(master_shadow_per_gen)
    horizon = max(cand_len, master_len)

    # Pair check: every candidate cell must be at Moore distance >= 2 from
    # every master cell at the same generation.
    for t in range(horizon):
        cand_cells = cand_cells_per_gen[t] if t < cand_len else cand_settled
        master_shadow = master_shadow_per_gen[t] if t < master_len else master_settled_shadow
        if not cand_cells or not master_shadow:
            continue
        for p in cand_cells:
            if (p + shift) in master_shadow:
                return False
    if cand_settled and master_settled_shadow:
        for p in cand_settled:
            if (p + shift) in master_settled_shadow:
                return False

    # Shared-dead-neighbor check: at any dead cell receiving neighbors from
    # both, the combined count's next state must match the per-side next state.
    for t in range(horizon):
        cand_cells = cand_cells_per_gen[t] if t < cand_len else cand_settled
        master_nbr = (
            master_neighbor_count_per_gen[t] if t < master_len else master_settled_neighbor_count
        )
        if not cand_cells or not master_nbr:
            continue
        if not _shared_neighbors_consistent(cand_cells, shift, master_nbr):
            return False
    return (
        not cand_settled
        or not master_settled_neighbor_count
        or _shared_neighbors_consistent(cand_settled, shift, master_settled_neighbor_count)
    )


def _shared_neighbors_consistent(
    cand_cells: frozenset[int],
    shift: int,
    master_nbr: dict[int, int],
) -> bool:
    """Return True iff combined neighbor counts at shared dead cells match per-side next states.

    A dead cell becomes alive on the next step iff it has exactly three live
    neighbors. If both the candidate and the master contribute neighbors at a
    dead cell, the combined count can cross the 3 threshold even when neither
    side reaches it alone (or vice versa), which means the union evolves
    differently from the union of individual evolutions. Catching that here
    lets us skip the combined-plan simulation that the original packer used as
    a safety net for this exact case.
    """

    cand_nbr: dict[int, int] = {}
    for cell in cand_cells:
        shifted_cell = cell + shift
        for nbr_shift in _PURE_NEIGHBOR_SHIFTS:
            key = shifted_cell + nbr_shift
            cand_nbr[key] = cand_nbr.get(key, 0) + 1
    for c, c_n in cand_nbr.items():
        m_n = master_nbr.get(c, 0)
        if m_n == 0:
            continue
        combined = m_n + c_n
        if (combined == 3) != (m_n == 3 or c_n == 3):
            return False
    return True


def _master_absorb(
    base_data: _BaseBlockData,
    shift: int,
    master_footprint: set[int],
    master_shadow_per_gen: list[set[int]],
    master_settled_shadow: set[int],
    master_neighbor_count_per_gen: list[dict[int, int]],
    master_settled_neighbor_count: dict[int, int],
) -> None:
    """Union the shifted candidate's per-generation state into the master."""

    cand_cells_per_gen = base_data.cells_per_gen
    cand_shadow_per_gen = base_data.shadow_per_gen
    cand_settled = base_data.settled
    cand_settled_shadow = base_data.settled_shadow
    cand_len = len(cand_shadow_per_gen)
    master_len = len(master_shadow_per_gen)

    shifted_settled_shadow = {p + shift for p in cand_settled_shadow}
    shifted_settled_nbr_contrib: dict[int, int] = {}
    for cell in cand_settled:
        shifted_cell = cell + shift
        for nbr_shift in _PURE_NEIGHBOR_SHIFTS:
            key = shifted_cell + nbr_shift
            shifted_settled_nbr_contrib[key] = shifted_settled_nbr_contrib.get(key, 0) + 1

    if cand_len > master_len:
        for _ in range(cand_len - master_len):
            master_shadow_per_gen.append(set(master_settled_shadow))
            master_neighbor_count_per_gen.append(dict(master_settled_neighbor_count))
        master_len = cand_len

    for t in range(cand_len):
        master_shadow_per_gen[t].update(p + shift for p in cand_shadow_per_gen[t])
        nbr_t = master_neighbor_count_per_gen[t]
        for cell in cand_cells_per_gen[t]:
            shifted_cell = cell + shift
            for nbr_shift in _PURE_NEIGHBOR_SHIFTS:
                key = shifted_cell + nbr_shift
                nbr_t[key] = nbr_t.get(key, 0) + 1
    for t in range(cand_len, master_len):
        master_shadow_per_gen[t].update(shifted_settled_shadow)
        nbr_t = master_neighbor_count_per_gen[t]
        for c, n in shifted_settled_nbr_contrib.items():
            nbr_t[c] = nbr_t.get(c, 0) + n

    master_settled_shadow.update(shifted_settled_shadow)
    for c, n in shifted_settled_nbr_contrib.items():
        master_settled_neighbor_count[c] = master_settled_neighbor_count.get(c, 0) + n
    master_footprint.update(p + shift for p in base_data.footprint)


@dataclass(frozen=True, slots=True)
class _BaseBlockData:
    """Per-(orientation, extra_periods) precomputed data for one block at origin (0, 0).

    Cell sets are bit-packed integers so a translation becomes a single integer
    add and the disjointness check can use native ``set`` operations.
    """

    footprint: frozenset[int]
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


@cache
def _base_block_data(orientation: str, extra_periods: int) -> _BaseBlockData:
    """Compute the per-generation Moore-shadow timeline for one block at origin (0, 0).

    ``footprint`` is the spacetime union of all per-generation shadows plus the
    settled-state shadow. It is the cheap pre-filter for the master-disjoint
    check: if no cell the candidate ever touches is ever touched by any placed
    block, the per-generation check is guaranteed to pass.
    """

    plan = plan_block((0, 0), orientation=orientation, extra_periods=extra_periods)
    settled = _pack_pattern(plan.target_cells)
    settled_shadow = _expand_packed_with_adjacency(settled)

    if not plan.initial_cells:
        return _BaseBlockData(
            footprint=settled_shadow,
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
        grid = _step_array(grid, wrap=False)

    footprint_cells: set[int] = set(settled_shadow)
    for shadow in shadow_per_gen:
        footprint_cells.update(shadow)

    return _BaseBlockData(
        footprint=frozenset(footprint_cells),
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
_PURE_NEIGHBOR_SHIFTS: Final[tuple[int, ...]] = tuple(
    _shift_constant(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)
)


def _expand_packed_with_adjacency(cells: frozenset[int]) -> frozenset[int]:
    if not cells:
        return cells
    expanded: set[int] = set()
    for shift in _NEIGHBOR_SHIFTS:
        expanded.update(p + shift for p in cells)
    return frozenset(expanded)


def _text_pixel_origins(text: str) -> tuple[Point, ...]:
    """Return the top-left origin of each live glyph pixel in the text layout.

    Glyphs on a line are baseline-aligned: each line picks its ascent and
    descent from the tallest of each on the row, so a 9-row ``Å`` and a
    7-row ``A`` end at the same bottom edge while a 9-row ``g`` extends two
    rows further down for its descender.
    """

    origins: list[Point] = []
    line_y = 0
    for line in text.splitlines():
        line_glyphs: list[tuple[Glyph, int, int]] = []
        max_ascent = 0
        max_descent = 0
        for character in line:
            glyph = glyph_for_character(character)
            descent = DESCENT.get(character, 0)
            ascent = len(glyph) - descent
            line_glyphs.append((glyph, ascent, descent))
            max_ascent = max(max_ascent, ascent)
            max_descent = max(max_descent, descent)
        if not line_glyphs:
            max_ascent = DEFAULT_ASCENT
        cursor_x = 0
        for glyph, ascent, _descent in line_glyphs:
            glyph_top_y = line_y + (max_ascent - ascent) * BLOCK_TEXT_STRIDE
            for glyph_y, row in enumerate(glyph):
                for glyph_x, token in enumerate(row):
                    if token != "#":
                        continue
                    origins.append(
                        (
                            cursor_x + glyph_x * BLOCK_TEXT_STRIDE,
                            glyph_top_y + glyph_y * BLOCK_TEXT_STRIDE,
                        )
                    )
            cursor_x += len(glyph[0]) * BLOCK_TEXT_STRIDE + BLOCK_TEXT_LETTER_SPACING
        line_y += (max_ascent + max_descent) * BLOCK_TEXT_STRIDE + BLOCK_TEXT_LINE_SPACING
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
