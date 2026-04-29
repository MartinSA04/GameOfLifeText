"""Tests for stable block-text rendering."""

import pytest

from game_of_life_text import text as text_module
from game_of_life_text.construction import ConstructionPlan, evolve_construction
from game_of_life_text.simulator import Board, Pattern, Point, SimulationConfig
from game_of_life_text.text import (
    FONT_5X7,
    render_text_block_construction,
    render_text_block_construction_with_progress,
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


def test_lowercase_letters_render_as_distinct_small_glyphs() -> None:
    """Lowercase input should preserve case instead of rendering as uppercase."""

    assert render_text_block_pattern("a") != render_text_block_pattern("A")
    assert render_text_block_pattern("az") != render_text_block_pattern("AZ")


def test_text_construction_reports_block_build_progress() -> None:
    """Progress callbacks should report determinate construction progress."""

    calls: list[tuple[int, int]] = []

    render_text_block_construction_with_progress(
        "I",
        lambda current, total: calls.append((current, total)),
    )

    assert calls[0][0] == 0
    assert calls[-1][0] == calls[-1][1]
    assert calls[-1][1] > 1
    assert [current for current, _ in calls] == sorted(current for current, _ in calls)
    assert len({total for _, total in calls}) == 1


def test_pack_block_plans_tries_fallback_outward_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The packing search should try the secondary outward direction if needed."""

    first_origin = (0, 0)
    second_origin = (10, -1)
    attempts: list[tuple[Point, str, int]] = []

    def fake_plan_block(
        origin: Point,
        *,
        orientation: str = "east",
        extra_periods: int = 0,
    ) -> ConstructionPlan:
        attempts.append((origin, orientation, extra_periods))
        offset = 0 if origin == first_origin else 100
        pattern = Pattern.from_points(((offset, 0),))
        return ConstructionPlan(initial_cells=pattern, target_cells=pattern, generations=0)

    def fake_base_block_data(orientation: str, extra_periods: int) -> text_module._BaseBlockData:
        _ = extra_periods
        if orientation == "east":
            # The second origin's translated cells coincide with the first
            # origin's master cell, so the pair-check fails and the planner
            # falls back to the secondary orientation.
            cells = frozenset(
                {text_module._pack_point(0, 0), text_module._pack_point(10, -1)}
            )
        else:
            cells = frozenset({text_module._pack_point(200, 0)})
        shadow = text_module._expand_packed_with_adjacency(cells)
        return text_module._BaseBlockData(
            footprint=shadow,
            cells_per_gen=(cells,),
            shadow_per_gen=(shadow,),
            settled=cells,
            settled_shadow=shadow,
        )

    monkeypatch.setattr(text_module, "plan_block", fake_plan_block)
    monkeypatch.setattr(text_module, "_base_block_data", fake_base_block_data)
    monkeypatch.setattr(text_module, "evolve_construction", lambda plan: Pattern.empty())

    plans = text_module._pack_block_plans((first_origin, second_origin), center=(0.0, 0.0))

    assert len(plans) == 2
    second_attempts = [attempt for attempt in attempts if attempt[0] == second_origin]
    # The east candidate is rejected by the pair check before reaching plan_block,
    # so only the accepted north fallback is logged.
    assert second_attempts == [(second_origin, "north", 0)]


def test_render_text_block_pattern_rejects_unsupported_characters() -> None:
    """Unsupported glyphs should fail with a clear error."""

    with pytest.raises(ValueError, match="unsupported character"):
        render_text_block_pattern("HI$")
