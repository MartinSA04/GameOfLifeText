"""Reusable glider constructions for stable block targets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from typing import Final

import numpy as np

from .simulator import Board, Pattern, Point, SimulationConfig

type Direction = Point
type _Component = tuple[Pattern, Direction]

BLOCK_PATTERN: Final[Pattern] = Pattern.from_points(((0, 0), (1, 0), (0, 1), (1, 1)))
BLOCK_SYNTHESIS_BASE_GENERATIONS: Final[int] = 4
_BLOCK_SYNTHESIS_BASE_TARGET: Final[Pattern] = Pattern.from_points(
    (
        (-4, -2),
        (-4, -1),
        (-3, -2),
        (-3, -1),
    )
)
_BLOCK_SYNTHESIS_BASE_COMPONENTS: Final[tuple[_Component, ...]] = (
    (
        Pattern.from_points(
            (
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (2, 1),
            )
        ),
        (-1, -1),
    ),
    (
        Pattern.from_points(
            (
                (-3, -3),
                (-3, -2),
                (-3, -1),
                (-2, -1),
                (-1, -2),
            )
        ),
        (-1, 1),
    ),
)
_ORIENTATION_TRANSFORMS: Final[dict[str, tuple[int, bool]]] = {
    "east": (0, False),
    "west": (0, True),
    "north": (1, False),
    "south": (3, False),
}


@dataclass(frozen=True, slots=True, eq=False)
class ConstructionPlan:
    """A seed pattern that settles into a deterministic target."""

    initial_cells: Pattern
    target_cells: Pattern
    generations: int

    def __post_init__(self) -> None:
        if self.generations < 0:
            msg = "generations must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class _BlockSynthesisVariant:
    """One oriented two-glider block construction."""

    orientation: str
    target_top_left: Point
    components: tuple[_Component, ...]


def block_at(top_left: Point) -> Pattern:
    """Return the exact cells for a 2x2 block at a requested top-left point."""

    return BLOCK_PATTERN.translated(top_left[0], top_left[1])


def plan_block(
    top_left: Point,
    *,
    orientation: str = "east",
    extra_periods: int = 0,
) -> ConstructionPlan:
    """Return a two-glider synthesis plan for one block."""

    if extra_periods < 0:
        msg = "extra_periods cannot be negative"
        raise ValueError(msg)

    target_cells = block_at(top_left)
    return ConstructionPlan(
        initial_cells=gliders_for_block(
            top_left,
            orientation=orientation,
            extra_periods=extra_periods,
        ),
        target_cells=target_cells,
        generations=BLOCK_SYNTHESIS_BASE_GENERATIONS + extra_periods * 4,
    )


def gliders_for_block(
    top_left: Point,
    *,
    orientation: str = "east",
    extra_periods: int = 0,
) -> Pattern:
    """Return a two-glider seed that settles into a block at ``top_left``."""

    if extra_periods < 0:
        msg = "extra_periods cannot be negative"
        raise ValueError(msg)

    variant = block_synthesis_variant(orientation)
    shift_x = top_left[0] - variant.target_top_left[0]
    shift_y = top_left[1] - variant.target_top_left[1]

    translated_components: list[Pattern] = []
    for cells, direction in variant.components:
        direction_x, direction_y = direction
        backstep_x = shift_x - direction_x * extra_periods
        backstep_y = shift_y - direction_y * extra_periods
        translated_components.append(cells.translated(backstep_x, backstep_y))
    return Pattern.merge(*translated_components)


def combine_plans(plans: Iterable[ConstructionPlan]) -> ConstructionPlan:
    """Merge several independently planned syntheses into one board seed."""

    initial_patterns: list[Pattern] = []
    target_patterns: list[Pattern] = []
    generations = 0

    for plan in plans:
        if plan.initial_cells:
            initial_patterns.append(plan.initial_cells)
        if plan.target_cells:
            target_patterns.append(plan.target_cells)
        generations = max(generations, plan.generations)

    return ConstructionPlan(
        initial_cells=Pattern.merge(*initial_patterns),
        target_cells=Pattern.merge(*target_patterns),
        generations=generations,
    )


def center_construction(config: SimulationConfig, plan: ConstructionPlan) -> Pattern:
    """Translate a construction so its finished target is centered on the board."""

    if not plan.target_cells:
        return Pattern.empty()

    min_x, min_y, max_x, max_y = plan.target_cells.bounds()
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    offset_x = (config.width - width) // 2 - min_x
    offset_y = (config.height - height) // 2 - min_y

    translated = plan.initial_cells.translated(offset_x, offset_y)
    if translated:
        points = translated.points
        invalid_mask = (
            (points[:, 0] < 0)
            | (points[:, 0] >= config.width)
            | (points[:, 1] < 0)
            | (points[:, 1] >= config.height)
        )
        if bool(np.any(invalid_mask)):
            msg = "board is too small for this glider construction; increase width or height"
            raise ValueError(msg)
    return translated


def minimum_centered_board_size(
    plan: ConstructionPlan,
    *,
    padding: int = 0,
) -> tuple[int, int]:
    """Return the minimum board size needed to center a construction safely."""

    if not plan.target_cells:
        side = max(1, padding * 2 + 1)
        return (side, side)
    if not plan.initial_cells:
        target_min_x, target_min_y, target_max_x, target_max_y = plan.target_cells.bounds()
        return (
            target_max_x - target_min_x + 1 + padding * 2,
            target_max_y - target_min_y + 1 + padding * 2,
        )

    min_initial_x, min_initial_y, max_initial_x, max_initial_y = plan.initial_cells.bounds()
    min_target_x, min_target_y, max_target_x, max_target_y = plan.target_cells.bounds()
    target_width = max_target_x - min_target_x + 1
    target_height = max_target_y - min_target_y + 1
    left_need = max(0, min_target_x - min_initial_x)
    right_need = max(0, max_initial_x - max_target_x)
    top_need = max(0, min_target_y - min_initial_y)
    bottom_need = max(0, max_initial_y - max_target_y)
    return (
        target_width + (max(left_need, right_need) + padding) * 2,
        target_height + (max(top_need, bottom_need) + padding) * 2,
    )


def evolve_construction(plan: ConstructionPlan) -> Pattern:
    """Simulate a construction plan and return its final settled cells.

    Each glider in a block synthesis launches inside the initial-cells bounding
    box and converges to a target cell that is also inside the same bounding
    box of (initial ``union`` target). The trajectory therefore never escapes
    that union. We use a tight bounding box plus a small safety margin instead
    of the previous ``generations + 12`` padding, which kept the simulation
    board growing quadratically with extra_periods.
    """

    if not plan.initial_cells:
        return Pattern.empty()

    init_min_x, init_min_y, init_max_x, init_max_y = plan.initial_cells.bounds()
    if plan.target_cells:
        target_min_x, target_min_y, target_max_x, target_max_y = plan.target_cells.bounds()
        min_x = min(init_min_x, target_min_x)
        min_y = min(init_min_y, target_min_y)
        max_x = max(init_max_x, target_max_x)
        max_y = max(init_max_y, target_max_y)
    else:
        min_x, min_y, max_x, max_y = init_min_x, init_min_y, init_max_x, init_max_y

    padding = 8
    config = SimulationConfig(
        width=max_x - min_x + 1 + padding * 2,
        height=max_y - min_y + 1 + padding * 2,
        wrap=False,
    )
    shifted = plan.initial_cells.translated(-min_x + padding, -min_y + padding)
    board = Board.from_points(config, shifted).step_n(plan.generations)
    return board.live_cells.translated(min_x - padding, min_y - padding)


@cache
def block_synthesis_variant(orientation: str) -> _BlockSynthesisVariant:
    """Return one oriented two-glider block synthesis."""

    if orientation not in _ORIENTATION_TRANSFORMS:
        msg = f"unknown block synthesis orientation {orientation!r}"
        raise ValueError(msg)

    rotation_count, reflect = _ORIENTATION_TRANSFORMS[orientation]
    target = _transform_pattern(_BLOCK_SYNTHESIS_BASE_TARGET, rotation_count, reflect)
    components = tuple(
        (
            _transform_pattern(cells, rotation_count, reflect),
            _transform_direction(direction, rotation_count, reflect),
        )
        for cells, direction in _BLOCK_SYNTHESIS_BASE_COMPONENTS
    )
    min_x, min_y, _, _ = target.bounds()
    return _BlockSynthesisVariant(
        orientation=orientation,
        target_top_left=(min_x, min_y),
        components=components,
    )


def _transform_pattern(pattern: Pattern, rotation_count: int, reflect: bool) -> Pattern:
    """Apply the synthesis symmetry transform to every cell."""

    if not pattern:
        return Pattern.empty()
    transformed = np.array(pattern.points, copy=True)
    if reflect:
        transformed[:, 0] *= -1
    for _ in range(rotation_count % 4):
        transformed = np.column_stack((transformed[:, 1], -transformed[:, 0]))
    return Pattern(transformed.astype(np.int32, copy=False))


def _transform_direction(direction: Direction, rotation_count: int, reflect: bool) -> Direction:
    """Apply the synthesis symmetry transform to one glider direction."""

    direction_x, direction_y = direction
    if reflect:
        direction_x = -direction_x
    for _ in range(rotation_count % 4):
        direction_x, direction_y = direction_y, -direction_x
    return (direction_x, direction_y)
