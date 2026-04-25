"""Tests for the Game of Life core."""

from game_of_life_text.simulator import Board, SimulationConfig, centered_cells, random_cells


def test_block_is_stable() -> None:
    """A block should remain unchanged across generations."""

    config = SimulationConfig(width=6, height=6, wrap=False)
    board = Board.from_points(config, {(2, 2), (3, 2), (2, 3), (3, 3)})

    next_board = board.step()

    assert next_board.live_cells == board.live_cells
    assert next_board.generation == 1


def test_blinker_oscillates() -> None:
    """A blinker should flip orientation every step."""

    config = SimulationConfig(width=5, height=5, wrap=False)
    board = Board.from_points(config, {(1, 2), (2, 2), (3, 2)})

    next_board = board.step()
    final_board = next_board.step()

    assert next_board.live_cells == frozenset({(2, 1), (2, 2), (2, 3)})
    assert final_board.live_cells == board.live_cells


def test_wrapping_births_cells_across_edges() -> None:
    """Wrapping should let opposite edges interact."""

    config = SimulationConfig(width=5, height=5, wrap=True)
    board = Board.from_points(config, {(0, 0), (4, 0), (0, 4)})

    next_board = board.step()

    assert next_board.is_alive((4, 4))


def test_centered_cells_places_patterns_in_the_middle() -> None:
    """Patterns should be centered on the board."""

    config = SimulationConfig(width=7, height=7, wrap=False)

    centered = centered_cells(config, frozenset({(0, 0), (1, 0), (2, 0)}))

    assert centered == frozenset({(2, 3), (3, 3), (4, 3)})


def test_random_cells_are_seeded() -> None:
    """The same seed should generate the same board."""

    config = SimulationConfig(width=8, height=4, wrap=True)

    first = random_cells(config, 0.25, seed=7)
    second = random_cells(config, 0.25, seed=7)

    assert first == second
