"""Reusable glider constructions for stable block targets."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from .simulator import Board, LiveCells, Pattern, Point, SimulationConfig

type Direction = Point
type _Component = tuple[Pattern, Direction]

BLOCK_PATTERN: Final[Pattern] = frozenset({(0, 0), (1, 0), (0, 1), (1, 1)})
BLOCK_SYNTHESIS_BASE_GENERATIONS: Final[int] = 4
_BLOCK_SYNTHESIS_BASE_TARGET: Final[Pattern] = frozenset(
    {
        (-4, -2),
        (-4, -1),
        (-3, -2),
        (-3, -1),
    }
)
_BLOCK_SYNTHESIS_BASE_COMPONENTS: Final[tuple[_Component, ...]] = (
    (
        frozenset(
            {
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (2, 1),
            }
        ),
        (-1, -1),
    ),
    (
        frozenset(
            {
                (-3, -3),
                (-3, -2),
                (-3, -1),
                (-2, -1),
                (-1, -2),
            }
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


class ConstructionPlan(BaseModel):
    """A seed pattern that settles into a deterministic target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_cells: Pattern
    target_cells: Pattern
    generations: int = Field(ge=0)


class _BlockSynthesisVariant(BaseModel):
    """One oriented two-glider block construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    orientation: str
    target_top_left: Point
    components: tuple[_Component, ...]


def block_at(top_left: Point) -> Pattern:
    """Return the exact cells for a 2x2 block at a requested top-left point."""

    origin_x, origin_y = top_left
    return frozenset((origin_x + x, origin_y + y) for x, y in BLOCK_PATTERN)


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

    initial_cells: set[Point] = set()
    for cells, direction in variant.components:
        direction_x, direction_y = direction
        backstep_x = shift_x - direction_x * extra_periods
        backstep_y = shift_y - direction_y * extra_periods
        initial_cells.update((backstep_x + x, backstep_y + y) for x, y in cells)
    return frozenset(initial_cells)


def combine_plans(plans: Iterable[ConstructionPlan]) -> ConstructionPlan:
    """Merge several independently planned syntheses into one board seed."""

    initial_cells: set[Point] = set()
    target_cells: set[Point] = set()
    generations = 0

    for plan in plans:
        initial_cells.update(plan.initial_cells)
        target_cells.update(plan.target_cells)
        generations = max(generations, plan.generations)

    return ConstructionPlan(
        initial_cells=frozenset(initial_cells),
        target_cells=frozenset(target_cells),
        generations=generations,
    )


def center_construction(config: SimulationConfig, plan: ConstructionPlan) -> LiveCells:
    """Translate a construction so its finished target is centered on the board."""

    if not plan.target_cells:
        return frozenset()

    min_x = min(x for x, _ in plan.target_cells)
    min_y = min(y for _, y in plan.target_cells)
    max_x = max(x for x, _ in plan.target_cells)
    max_y = max(y for _, y in plan.target_cells)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    offset_x = (config.width - width) // 2 - min_x
    offset_y = (config.height - height) // 2 - min_y

    translated = frozenset((x + offset_x, y + offset_y) for x, y in plan.initial_cells)
    invalid_point = next(
        (
            point
            for point in translated
            if not 0 <= point[0] < config.width or not 0 <= point[1] < config.height
        ),
        None,
    )
    if invalid_point is not None:
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
        return (max(1, padding * 2 + 1), max(1, padding * 2 + 1))
    if not plan.initial_cells:
        target_min_x = min(x for x, _ in plan.target_cells)
        target_min_y = min(y for _, y in plan.target_cells)
        target_max_x = max(x for x, _ in plan.target_cells)
        target_max_y = max(y for _, y in plan.target_cells)
        return (
            target_max_x - target_min_x + 1 + padding * 2,
            target_max_y - target_min_y + 1 + padding * 2,
        )

    min_initial_x = min(x for x, _ in plan.initial_cells)
    min_initial_y = min(y for _, y in plan.initial_cells)
    max_initial_x = max(x for x, _ in plan.initial_cells)
    max_initial_y = max(y for _, y in plan.initial_cells)
    min_target_x = min(x for x, _ in plan.target_cells)
    min_target_y = min(y for _, y in plan.target_cells)
    max_target_x = max(x for x, _ in plan.target_cells)
    max_target_y = max(y for _, y in plan.target_cells)
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
    """Simulate a construction plan and return its final settled cells."""

    if not plan.initial_cells:
        return frozenset()

    min_x = min(x for x, _ in plan.initial_cells)
    min_y = min(y for _, y in plan.initial_cells)
    max_x = max(x for x, _ in plan.initial_cells)
    max_y = max(y for _, y in plan.initial_cells)
    padding = plan.generations + 12
    config = SimulationConfig(
        width=max_x - min_x + 1 + padding * 2,
        height=max_y - min_y + 1 + padding * 2,
        wrap=False,
    )
    shifted = {(x - min_x + padding, y - min_y + padding) for x, y in plan.initial_cells}
    board = Board.from_points(config, shifted)
    for _ in range(plan.generations):
        board = board.step()
    return frozenset((x + min_x - padding, y + min_y - padding) for x, y in board.live_cells)


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
    return _BlockSynthesisVariant(
        orientation=orientation,
        target_top_left=(min(x for x, _ in target), min(y for _, y in target)),
        components=components,
    )


def _transform_pattern(pattern: Pattern, rotation_count: int, reflect: bool) -> Pattern:
    """Apply the synthesis symmetry transform to every cell."""

    transformed = set(pattern)
    if reflect:
        transformed = {(-x, y) for x, y in transformed}
    for _ in range(rotation_count % 4):
        transformed = {(y, -x) for x, y in transformed}
    return frozenset(transformed)


def _transform_direction(direction: Direction, rotation_count: int, reflect: bool) -> Direction:
    """Apply the synthesis symmetry transform to one glider direction."""

    direction_x, direction_y = direction
    if reflect:
        direction_x = -direction_x
    for _ in range(rotation_count % 4):
        direction_x, direction_y = direction_y, -direction_x
    return (direction_x, direction_y)
