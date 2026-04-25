"""Core simulation types and helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from random import Random
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

type Point = tuple[int, int]
type Pattern = frozenset[Point]
type LiveCells = frozenset[Point]

NEIGHBOR_OFFSETS: Final[tuple[Point, ...]] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


class SimulationConfig(BaseModel):
    """Size and edge behavior for a board."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    wrap: bool = True


class Board(BaseModel):
    """An immutable Game of Life board state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config: SimulationConfig
    live_cells: LiveCells
    generation: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_live_cells(self) -> Self:
        """Ensure the board state is internally consistent."""

        invalid_point = next(
            (point for point in self.live_cells if not _is_in_bounds(self.config, point)),
            None,
        )
        if invalid_point is not None:
            msg = f"point {invalid_point!r} is outside the board"
            raise ValueError(msg)
        return self

    @property
    def alive_count(self) -> int:
        """Return the number of live cells."""

        return len(self.live_cells)

    @classmethod
    def from_points(
        cls,
        config: SimulationConfig,
        points: Iterable[Point],
        *,
        generation: int = 0,
    ) -> Self:
        """Create a board from any iterable of points."""

        return cls(config=config, live_cells=frozenset(points), generation=generation)

    def is_alive(self, point: Point) -> bool:
        """Return whether a point is alive on the board."""

        return point in self.live_cells

    def step(self) -> Board:
        """Advance the board by one generation."""

        neighbor_counts: Counter[Point] = Counter()
        for cell in self.live_cells:
            for neighbor in self._iter_neighbors(cell):
                neighbor_counts[neighbor] += 1

        next_cells = frozenset(
            cell
            for cell, count in neighbor_counts.items()
            if count == 3 or (count == 2 and cell in self.live_cells)
        )
        return Board(
            config=self.config,
            live_cells=next_cells,
            generation=self.generation + 1,
        )

    def _iter_neighbors(self, point: Point) -> Iterable[Point]:
        """Yield all valid neighboring points for a cell."""

        x, y = point
        for dx, dy in NEIGHBOR_OFFSETS:
            normalized = _normalize_point(self.config, (x + dx, y + dy))
            if normalized is not None:
                yield normalized


def centered_cells(config: SimulationConfig, pattern: Pattern) -> LiveCells:
    """Return a centered copy of a preset pattern, clipped to the board."""

    if not pattern:
        return frozenset()

    min_x = min(x for x, _ in pattern)
    min_y = min(y for _, y in pattern)
    normalized_pattern = frozenset((x - min_x, y - min_y) for x, y in pattern)
    width = max(x for x, _ in normalized_pattern) + 1
    height = max(y for _, y in normalized_pattern) + 1
    offset_x = (config.width - width) // 2
    offset_y = (config.height - height) // 2

    return frozenset(
        (offset_x + x, offset_y + y)
        for x, y in normalized_pattern
        if _is_in_bounds(config, (offset_x + x, offset_y + y))
    )


def random_cells(config: SimulationConfig, density: float, *, seed: int | None = None) -> LiveCells:
    """Generate a reproducible random board using the requested density."""

    if not 0.0 <= density <= 1.0:
        msg = "density must be between 0.0 and 1.0"
        raise ValueError(msg)

    rng = Random(seed)
    return frozenset(
        (x, y) for y in range(config.height) for x in range(config.width) if rng.random() < density
    )


def _normalize_point(config: SimulationConfig, point: Point) -> Point | None:
    """Translate a point to a valid board coordinate."""

    x, y = point
    if config.wrap:
        return (x % config.width, y % config.height)
    if _is_in_bounds(config, point):
        return point
    return None


def _is_in_bounds(config: SimulationConfig, point: Point) -> bool:
    """Return whether a point is inside the board."""

    x, y = point
    return 0 <= x < config.width and 0 <= y < config.height
