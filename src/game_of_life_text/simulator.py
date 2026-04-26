"""Core simulation types and helpers backed directly by NumPy arrays."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

type Point = tuple[int, int]
type BoolGrid = npt.NDArray[np.bool_]
type PointArray = npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Size and edge behavior for a board."""

    width: int
    height: int
    wrap: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0:
            msg = "width must be greater than 0"
            raise ValueError(msg)
        if self.height <= 0:
            msg = "height must be greater than 0"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True, eq=False)
class Pattern:
    """A canonical NumPy-backed collection of unique ``(x, y)`` points."""

    _points: PointArray
    _bbox: tuple[int, int, int, int] | None = None
    _point_set: frozenset[Point] | None = None

    def __post_init__(self) -> None:
        normalized = _normalize_points(self._points)
        object.__setattr__(self, "_points", normalized)
        object.__setattr__(self, "_bbox", _bbox_of(normalized))
        object.__setattr__(self, "_point_set", None)

    @classmethod
    def empty(cls) -> Pattern:
        """Return an empty pattern."""

        return cls(_empty_points())

    @classmethod
    def from_points(cls, points: Iterable[Point]) -> Pattern:
        """Build a pattern from any iterable of points."""

        return cls(_points_from_iterable(points))

    @classmethod
    def from_grid(cls, grid: BoolGrid) -> Pattern:
        """Build a pattern from a boolean grid (cells are inherently unique)."""

        ys, xs = np.nonzero(grid)
        if xs.size == 0:
            return cls.empty()
        points = np.column_stack((xs, ys)).astype(np.int32, copy=False)
        return _from_unique_sorted_points(_lex_sort_points(points))

    @classmethod
    def merge(cls, *patterns: Pattern) -> Pattern:
        """Union several patterns into one canonical pattern."""

        arrays = [pattern.points for pattern in patterns if pattern]
        if not arrays:
            return cls.empty()
        return cls(np.concatenate(arrays, axis=0))

    @property
    def points(self) -> PointArray:
        """Return the underlying read-only ``(x, y)`` array."""

        return self._points

    def __bool__(self) -> bool:
        return self._points.shape[0] > 0

    def __len__(self) -> int:
        return int(self._points.shape[0])

    def __iter__(self) -> Iterator[Point]:
        for point in self._points:
            yield (int(point[0]), int(point[1]))

    def __contains__(self, point: object) -> bool:
        if not isinstance(point, tuple) or len(point) != 2:
            return False
        if not self:
            return False
        x, y = point
        return bool(np.any((self._points[:, 0] == x) & (self._points[:, 1] == y)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pattern):
            return NotImplemented
        return np.array_equal(self._points, other._points)

    def bounds(self) -> tuple[int, int, int, int]:
        """Return ``(min_x, min_y, max_x, max_y)`` for the pattern."""

        if self._bbox is None:
            msg = "pattern is empty"
            raise ValueError(msg)
        return self._bbox

    def translated(self, offset_x: int, offset_y: int) -> Pattern:
        """Return a translated copy of the pattern (shift preserves uniqueness)."""

        if not self:
            return Pattern.empty()
        offset = np.array((offset_x, offset_y), dtype=np.int32)
        return _from_unique_sorted_points(self._points + offset)

    def clipped_to(self, config: SimulationConfig) -> Pattern:
        """Return only the points inside ``config`` (subset preserves uniqueness)."""

        if not self:
            return self
        mask = (
            (self._points[:, 0] >= 0)
            & (self._points[:, 0] < config.width)
            & (self._points[:, 1] >= 0)
            & (self._points[:, 1] < config.height)
        )
        if bool(np.all(mask)):
            return self
        return _from_unique_sorted_points(self._points[mask])

    def isdisjoint(self, other: Pattern) -> bool:
        """Return whether two patterns share no points.

        Spatially disjoint patterns are extremely common in the planner, so
        we short-circuit on bounding boxes first. When bboxes do overlap we
        compare a (lazily cached) Python frozenset of points, iterating the
        smaller side, which beats ``np.intersect1d`` on the small-to-medium
        patterns the planner actually produces.
        """

        if self._bbox is None or other._bbox is None:
            return True
        a_min_x, a_min_y, a_max_x, a_max_y = self._bbox
        b_min_x, b_min_y, b_max_x, b_max_y = other._bbox
        if a_max_x < b_min_x or b_max_x < a_min_x:
            return True
        if a_max_y < b_min_y or b_max_y < a_min_y:
            return True
        smaller, larger = (self, other) if len(self) <= len(other) else (other, self)
        return smaller._as_set().isdisjoint(larger._as_set())

    def _as_set(self) -> frozenset[Point]:
        """Return a (cached) Python frozenset view of the points for fast lookup."""

        if self._point_set is None:
            object.__setattr__(
                self,
                "_point_set",
                frozenset(
                    zip(
                        self._points[:, 0].tolist(),
                        self._points[:, 1].tolist(),
                        strict=True,
                    )
                ),
            )
        assert self._point_set is not None
        return self._point_set


@dataclass(frozen=True, slots=True, eq=False)
class Board:
    """An immutable Game of Life board state backed by a boolean grid."""

    config: SimulationConfig
    grid: BoolGrid
    generation: int = 0

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=np.bool_)
        expected_shape = (self.config.height, self.config.width)
        if grid.shape != expected_shape:
            msg = f"grid shape {grid.shape!r} does not match expected {expected_shape!r}"
            raise ValueError(msg)
        if self.generation < 0:
            msg = "generation must be non-negative"
            raise ValueError(msg)
        normalized = np.array(grid, dtype=np.bool_, copy=True)
        normalized.setflags(write=False)
        object.__setattr__(self, "grid", normalized)

    @property
    def live_cells(self) -> Pattern:
        """Return the current live cells as a canonical point pattern."""

        return Pattern.from_grid(self.grid)

    @property
    def alive_count(self) -> int:
        """Return the number of live cells."""

        return int(np.count_nonzero(self.grid))

    @classmethod
    def from_points(
        cls,
        config: SimulationConfig,
        points: Pattern | Iterable[Point],
        *,
        generation: int = 0,
    ) -> Board:
        """Create a board from any iterable of points."""

        return cls(config=config, grid=_cells_to_array(config, points), generation=generation)

    def is_alive(self, point: Point) -> bool:
        """Return whether a point is alive on the board."""

        x, y = point
        if not _is_in_bounds(self.config, point):
            return False
        return bool(self.grid[y, x])

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

        grid = _step_array_n(self.grid, generations, wrap=self.config.wrap)
        return Board(config=self.config, grid=grid, generation=self.generation + generations)


def centered_cells(config: SimulationConfig, pattern: Pattern) -> Pattern:
    """Return a centered copy of a preset pattern, clipped to the board."""

    if not pattern:
        return Pattern.empty()

    min_x, min_y, max_x, max_y = pattern.bounds()
    normalized_pattern = pattern.translated(-min_x, -min_y)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    offset_x = (config.width - width) // 2
    offset_y = (config.height - height) // 2
    return normalized_pattern.translated(offset_x, offset_y).clipped_to(config)


def random_cells(config: SimulationConfig, density: float, *, seed: int | None = None) -> Pattern:
    """Generate a reproducible random board using the requested density."""

    if not 0.0 <= density <= 1.0:
        msg = "density must be between 0.0 and 1.0"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    grid = rng.random((config.height, config.width)) < density
    return Pattern.from_grid(grid)


def _cells_to_array(config: SimulationConfig, cells: Pattern | Iterable[Point]) -> BoolGrid:
    """Convert sparse live cells into a ``(height, width)`` bool array."""

    grid = np.zeros((config.height, config.width), dtype=np.bool_)
    points = cells.points if isinstance(cells, Pattern) else _points_from_iterable(cells)
    if points.size == 0:
        return grid

    invalid_mask = (
        (points[:, 0] < 0)
        | (points[:, 0] >= config.width)
        | (points[:, 1] < 0)
        | (points[:, 1] >= config.height)
    )
    if bool(np.any(invalid_mask)):
        invalid_point = points[np.flatnonzero(invalid_mask)[0]]
        msg = f"point {(int(invalid_point[0]), int(invalid_point[1]))!r} is outside the board"
        raise ValueError(msg)
    grid[points[:, 1], points[:, 0]] = True
    return grid


def _step_array(grid: BoolGrid, *, wrap: bool) -> BoolGrid:
    """Apply one Game of Life generation to a 2D bool array."""

    return _step_array_n(grid, 1, wrap=wrap)


def _step_array_n(grid: BoolGrid, generations: int, *, wrap: bool) -> BoolGrid:
    """Apply ``generations`` Game of Life steps using pre-allocated buffers."""

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


def _empty_points() -> PointArray:
    points = np.empty((0, 2), dtype=np.int32)
    points.setflags(write=False)
    return points


def _normalize_points(points: PointArray) -> PointArray:
    array = np.asarray(points, dtype=np.int32)
    if array.size == 0:
        return _empty_points()
    if array.ndim != 2 or array.shape[1] != 2:
        msg = "points must have shape (n, 2)"
        raise ValueError(msg)
    normalized = np.unique(array, axis=0)
    normalized.setflags(write=False)
    return normalized


def _from_unique_sorted_points(points: PointArray) -> Pattern:
    """Build a pattern from points that are already unique, skipping ``np.unique``.

    Translation, grid masking, and grid extraction all yield unique cells, so
    we can avoid the ``np.unique(axis=0)`` sort that dominates the planner's
    Pattern construction cost.
    """

    pattern = Pattern.__new__(Pattern)
    array = np.ascontiguousarray(points, dtype=np.int32)
    array.setflags(write=False)
    object.__setattr__(pattern, "_points", array)
    object.__setattr__(pattern, "_bbox", _bbox_of(array))
    object.__setattr__(pattern, "_point_set", None)
    return pattern


def _bbox_of(points: PointArray) -> tuple[int, int, int, int] | None:
    if points.shape[0] == 0:
        return None
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1]))


def _lex_sort_points(points: PointArray) -> PointArray:
    """Lexicographically sort an (n, 2) point array by (x, y)."""

    if points.shape[0] <= 1:
        return points
    order = np.lexsort((points[:, 1], points[:, 0]))
    return points[order]


def _points_from_iterable(points: Iterable[Point]) -> PointArray:
    materialized = tuple(points)
    if not materialized:
        return _empty_points()
    return np.asarray(materialized, dtype=np.int32)


def _structured_points_view(points: PointArray) -> npt.NDArray[np.void]:
    dtype = np.dtype([("x", np.int32), ("y", np.int32)])
    return np.ascontiguousarray(points).view(dtype).reshape(-1)
