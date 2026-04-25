"""Core simulation types and helpers backed by a NumPy stepper."""

from __future__ import annotations

from collections.abc import Iterable
from random import Random
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

type Point = tuple[int, int]
type Pattern = frozenset[Point]
type LiveCells = frozenset[Point]


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

        return self.step_n(1)

    def step_n(self, generations: int) -> Board:
        """Advance the board by ``generations`` generations using vectorized stepping."""

        if generations < 0:
            msg = "generations must be non-negative"
            raise ValueError(msg)
        if generations == 0:
            return self

        grid = _cells_to_array(self.config, self.live_cells)
        grid = _step_array_n(grid, generations, wrap=self.config.wrap)
        return Board(
            config=self.config,
            live_cells=_array_to_cells(grid),
            generation=self.generation + generations,
        )


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


def _cells_to_array(config: SimulationConfig, cells: Iterable[Point]) -> np.ndarray:
    """Convert a sparse cell set into a (height, width) bool array."""

    grid = np.zeros((config.height, config.width), dtype=np.bool_)
    for x, y in cells:
        grid[y, x] = True
    return grid


def _array_to_cells(grid: np.ndarray) -> LiveCells:
    """Convert a (height, width) bool array back to a frozenset of (x, y) points."""

    ys, xs = np.nonzero(grid)
    return frozenset(zip(xs.tolist(), ys.tolist(), strict=True))


def _step_array(grid: np.ndarray, *, wrap: bool) -> np.ndarray:
    """Apply one Game of Life generation to a 2D bool array."""

    return _step_array_n(grid, 1, wrap=wrap)


def _step_array_n(grid: np.ndarray, generations: int, *, wrap: bool) -> np.ndarray:
    """Apply ``generations`` Game of Life steps using a single set of pre-allocated buffers."""

    if generations <= 0:
        return grid
    height, width = grid.shape
    padded = np.zeros((height + 2, width + 2), dtype=np.int8)
    neighbors = np.empty((height, width), dtype=np.int8)
    current = grid
    for _ in range(generations):
        if wrap:
            padded[1:-1, 1:-1] = current
            padded[0, 1:-1] = current[-1, :]
            padded[-1, 1:-1] = current[0, :]
            padded[1:-1, 0] = current[:, -1]
            padded[1:-1, -1] = current[:, 0]
            padded[0, 0] = current[-1, -1]
            padded[0, -1] = current[-1, 0]
            padded[-1, 0] = current[0, -1]
            padded[-1, -1] = current[0, 0]
        else:
            padded[1:-1, 1:-1] = current

        np.copyto(neighbors, padded[0:height, 0:width])
        neighbors += padded[0:height, 1 : width + 1]
        neighbors += padded[0:height, 2 : width + 2]
        neighbors += padded[1 : height + 1, 0:width]
        neighbors += padded[1 : height + 1, 2 : width + 2]
        neighbors += padded[2 : height + 2, 0:width]
        neighbors += padded[2 : height + 2, 1 : width + 1]
        neighbors += padded[2 : height + 2, 2 : width + 2]
        current = (neighbors == 3) | ((neighbors == 2) & current)
    return current


def _is_in_bounds(config: SimulationConfig, point: Point) -> bool:
    """Return whether a point is inside the board."""

    x, y = point
    return 0 <= x < config.width and 0 <= y < config.height
