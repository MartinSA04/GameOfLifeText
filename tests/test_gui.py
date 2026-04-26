"""Tests for the PySide6 GUI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from game_of_life_text import gui as gui_module
from game_of_life_text.construction import center_construction
from game_of_life_text.gui import GameOfLifeWindow, parse_optional_int
from game_of_life_text.simulator import centered_cells
from game_of_life_text.text import render_text_block_construction


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Provide one QApplication for GUI tests."""

    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


def test_gui_can_build_block_text_construction_board(qapp: QApplication) -> None:
    """The GUI should build glider-based block text boards from form controls."""

    _ = qapp
    window = GameOfLifeWindow()
    try:
        window.source_combo.setCurrentText("Stable text")
        window.text_input.setPlainText("I")
        window.width_spin.setValue(220)
        window.height_spin.setValue(140)

        window.apply_simulation_settings()

        assert window.current_board is not None
        plan = render_text_block_construction("I")
        initial_cells = window.current_board.live_cells
        board = window.current_board
        for _ in range(plan.generations):
            board = board.step()

        assert board.live_cells != initial_cells
        assert board.step().live_cells == board.live_cells
    finally:
        window.close()


def test_gui_can_build_random_board(qapp: QApplication) -> None:
    """The GUI should build a random board with the chosen seed."""

    _ = qapp
    window = GameOfLifeWindow()
    try:
        window.source_combo.setCurrentText("Random board")
        window.random_density_spin.setValue(0.3)
        window.seed_input.setText("42")
        window.width_spin.setValue(20)
        window.height_spin.setValue(10)

        window.apply_simulation_settings()

        assert window.current_board is not None
        assert window.current_board.alive_count > 0
    finally:
        window.close()


def test_gui_can_draw_and_erase_cells_on_the_canvas(qapp: QApplication) -> None:
    """Interactive draw mode should add and remove cells on the current board."""

    _ = qapp
    window = GameOfLifeWindow()
    try:
        window.show()
        window.source_combo.setCurrentText("Random board")
        window.random_density_spin.setValue(0.0)
        window.width_spin.setValue(12)
        window.height_spin.setValue(12)
        window.wrap_checkbox.setChecked(False)
        window.apply_simulation_settings()
        qapp.processEvents()

        assert window.current_board is not None
        assert not window.current_board.is_alive((0, 0))

        window.draw_button.setChecked(True)
        point = window.board_canvas.cell_center((0, 0))
        assert point is not None

        QTest.mouseClick(window.board_canvas, Qt.MouseButton.LeftButton, pos=point)
        assert window.current_board is not None
        assert window.current_board.is_alive((0, 0))

        QTest.mouseClick(window.board_canvas, Qt.MouseButton.RightButton, pos=point)
        assert window.current_board is not None
        assert not window.current_board.is_alive((0, 0))
    finally:
        window.close()


def test_gui_zoom_buttons_change_canvas_scale(qapp: QApplication) -> None:
    """Zoom controls should increase the visible size of board cells."""

    _ = qapp
    window = GameOfLifeWindow()
    try:
        window.show()
        window.source_combo.setCurrentText("Random board")
        window.random_density_spin.setValue(0.0)
        window.width_spin.setValue(12)
        window.height_spin.setValue(12)
        window.wrap_checkbox.setChecked(False)
        window.apply_simulation_settings()
        qapp.processEvents()

        first = window.board_canvas.cell_center((1, 1))
        second = window.board_canvas.cell_center((2, 1))
        assert first is not None
        assert second is not None
        span_before = second.x() - first.x()

        QTest.mouseClick(window.zoom_in_button, Qt.MouseButton.LeftButton)
        qapp.processEvents()

        first = window.board_canvas.cell_center((1, 1))
        second = window.board_canvas.cell_center((2, 1))
        assert first is not None
        assert second is not None
        span_after = second.x() - first.x()

        assert span_after > span_before
    finally:
        window.close()


def test_gui_text_focus_view_fills_preview_by_default(qapp: QApplication) -> None:
    """Stable-text boards should default to a content-focused preview."""

    _ = qapp
    window = GameOfLifeWindow()
    try:
        window.show()
        window.source_combo.setCurrentText("Stable text")
        window.text_input.setPlainText("HI")
        window.width_spin.setValue(260)
        window.height_spin.setValue(180)
        window.apply_simulation_settings()
        qapp.processEvents()

        assert window.current_board is not None
        assert window.board_canvas.is_showing_focus_region()
        target = centered_cells(
            window.current_board.config,
            render_text_block_construction("HI").target_cells,
        )
        min_x = min(x for x, _ in target)
        min_y = min(y for _, y in target)

        first = window.board_canvas.cell_center((min_x, min_y))
        second = window.board_canvas.cell_center((min_x + 1, min_y))
        assert first is not None
        assert second is not None
        focus_span = second.x() - first.x()

        QTest.mouseClick(window.full_view_button, Qt.MouseButton.LeftButton)
        qapp.processEvents()

        first = window.board_canvas.cell_center((min_x, min_y))
        second = window.board_canvas.cell_center((min_x + 1, min_y))
        assert first is not None
        assert second is not None
        full_span = second.x() - first.x()

        assert focus_span > full_span
    finally:
        window.close()


def test_gui_can_export_generated_plan_as_xy_pairs(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The export button should write board size plus centered ``x,y`` seed lines."""

    _ = qapp
    export_path = tmp_path / "stable_text_plan.txt"
    window = GameOfLifeWindow()
    try:
        window.show()
        window.source_combo.setCurrentText("Stable text")
        window.text_input.setPlainText("I")
        window.width_spin.setValue(220)
        window.height_spin.setValue(140)
        window.apply_simulation_settings()
        qapp.processEvents()

        monkeypatch.setattr(
            gui_module.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(export_path), "Text Files (*.txt)"),
        )

        QTest.mouseClick(window.export_button, Qt.MouseButton.LeftButton)

        plan = render_text_block_construction("I")
        expected_cells = center_construction(window.current_board.config, plan)
        expected_lines = [
            f"# board_size={window.current_board.config.width},{window.current_board.config.height}",
            *(f"{x},{y}" for x, y in expected_cells),
        ]

        assert export_path.read_text(encoding="utf-8").splitlines() == expected_lines
    finally:
        window.close()


def test_parse_optional_int_helper() -> None:
    """The optional int helper should accept blanks, ints, and reject garbage."""

    assert parse_optional_int("") is None
    assert parse_optional_int("7", minimum=0) == 7

    with pytest.raises(ValueError, match="whole number"):
        parse_optional_int("nope")
