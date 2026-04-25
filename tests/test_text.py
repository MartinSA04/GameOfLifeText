"""Tests for stable block-text rendering."""

import pytest

from game_of_life_text.construction import evolve_construction
from game_of_life_text.simulator import Board, SimulationConfig
from game_of_life_text.text import (
    FONT_5X7,
    render_text_block_construction,
    render_text_block_pattern,
)


def test_rendered_block_text_is_stable() -> None:
    """Text rendered from blocks should remain unchanged."""

    pattern = render_text_block_pattern("HI")
    config = SimulationConfig(width=80, height=80, wrap=False)
    board = Board.from_points(config, {(x + 10, y + 10) for x, y in pattern})

    assert board.step().live_cells == board.live_cells


def test_block_text_construction_matches_dense_stable_text() -> None:
    """The deterministic glider construction should settle into block text."""

    plan = render_text_block_construction("I")

    assert plan.target_cells == render_text_block_pattern("I")
    assert evolve_construction(plan) == plan.target_cells


def test_all_single_glyph_block_constructions_settle_correctly() -> None:
    """Every supported single glyph should work with the fixed block constructor."""

    for character in FONT_5X7:
        if character == " ":
            continue
        plan = render_text_block_construction(character)
        assert evolve_construction(plan) == plan.target_cells


def test_multi_character_block_construction_settles_correctly() -> None:
    """A representative multi-character block text should also settle correctly."""

    plan = render_text_block_construction("OF")
    assert evolve_construction(plan) == plan.target_cells


def test_render_text_block_pattern_rejects_unsupported_characters() -> None:
    """Unsupported glyphs should fail with a clear error."""

    with pytest.raises(ValueError, match="unsupported character"):
        render_text_block_pattern("HI$")


def test_render_text_block_pattern_rejects_empty_input() -> None:
    """Empty text should fail before construction."""

    with pytest.raises(ValueError, match="text cannot be empty"):
        render_text_block_pattern("")
