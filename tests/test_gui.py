"""Tests for the PySide6 GUI."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from game_of_life_text import gui as gui_module
from game_of_life_text.construction import center_construction
from game_of_life_text.gui import GameOfLifeWindow, inspect_text_construction, parse_optional_int
from game_of_life_text.simulator import centered_cells
from game_of_life_text.text import render_text_block_construction


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Provide one QApplication for GUI tests."""

    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication([])


def wait_for_gui(
    qapp: QApplication,
    predicate: Callable[[], bool],
    *,
    timeout_ms: int = 5000,
) -> None:
    """Process Qt events until a GUI predicate becomes true."""

    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    qapp.processEvents()
    assert predicate()


def test_gui_can_build_block_text_construction_board(qapp: QApplication) -> None:
    """The GUI should auto-size and build glider-based block text boards."""

    _ = qapp
    window = GameOfLifeWindow()
    try:
        window.source_combo.setCurrentText("Stable text")
        window.text_input.setPlainText("I")

        window.apply_simulation_settings()
        wait_for_gui(qapp, lambda: window.current_board is not None)

        assert window.current_board is not None
        insight = inspect_text_construction("I")
        assert window.current_board.config.width == insight.recommended_width
        assert window.current_board.config.height == insight.recommended_height
        plan = render_text_block_construction("I")
        initial_cells = window.current_board.live_cells
        board = window.current_board
        for _ in range(plan.generations):
            board = board.step()

        assert board.live_cells != initial_cells
        assert board.step().live_cells == board.live_cells
    finally:
        window.close()


def test_gui_defaults_to_text_generation_workflow(qapp: QApplication) -> None:
    """The window should open in text mode without instructional filler copy."""

    _ = qapp
    window = GameOfLifeWindow()
    try:
        assert window.source_combo.currentText() == "Stable text"
        assert window.rebuild_button.text() == "Generate"
        assert window.text_input.toPlainText() == ""
        assert window.text_summary_label.text() == ""
        assert window.preview_summary_label.text() == ""
        assert window.generation_progress_bar.isHidden()
        assert window.progress_bar.isHidden()
        assert not window.text_frame.isHidden()
        assert window.board_group.isHidden()
        assert not window.rebuild_button.isEnabled()
        assert window.current_text_insight is None
    finally:
        window.close()


def test_gui_only_runs_text_generation_when_generate_is_used(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editing text should not invoke the planner before the generate action."""

    _ = qapp
    calls: list[str] = []
    original_inspect = gui_module.inspect_text_construction

    def tracked_inspect(
        text: str,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> object:
        _ = progress_callback
        calls.append(text)
        return original_inspect(text)

    monkeypatch.setattr(gui_module, "inspect_text_construction", tracked_inspect)
    window = GameOfLifeWindow()
    try:
        window.text_input.setPlainText("I")
        window.width_spin.setValue(260)
        window.height_spin.setValue(180)
        qapp.processEvents()

        assert calls == []
        assert window.text_summary_label.text() == ""
        assert window.rebuild_button.isEnabled()
        assert window.current_board is None

        QTest.mouseClick(window.play_button, Qt.MouseButton.LeftButton)

        assert calls == []
        assert window.current_board is None

        QTest.mouseClick(window.rebuild_button, Qt.MouseButton.LeftButton)
        wait_for_gui(qapp, lambda: window.current_text_insight is not None)

        assert calls == ["I"]
        assert window.text_summary_label.text() == ""
        assert window.current_text_insight is not None
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
        assert window.progress_bar.isHidden()
    finally:
        window.close()


def test_gui_can_build_blank_board(qapp: QApplication) -> None:
    """Blank mode should create an empty board using manual board dimensions."""

    _ = qapp
    window = GameOfLifeWindow()
    try:
        window.source_combo.setCurrentText("Blank board")
        window.width_spin.setValue(30)
        window.height_spin.setValue(20)

        window.apply_simulation_settings()

        assert window.current_board is not None
        assert window.current_board.config.width == 30
        assert window.current_board.config.height == 20
        assert window.current_board.alive_count == 0
    finally:
        window.close()


def test_gui_only_shows_board_controls_for_manual_board_modes(qapp: QApplication) -> None:
    """Board size belongs to random and blank modes, not text generation."""

    _ = qapp
    window = GameOfLifeWindow()
    try:
        window.show()
        qapp.processEvents()

        assert not window.board_group.isVisibleTo(window)

        window.source_combo.setCurrentText("Random board")
        qapp.processEvents()

        assert window.board_group.isVisibleTo(window)
        assert window.random_frame.isVisibleTo(window)

        window.source_combo.setCurrentText("Blank board")
        qapp.processEvents()

        assert window.board_group.isVisibleTo(window)
        assert not window.random_frame.isVisibleTo(window)

        window.source_combo.setCurrentText("Stable text")
        qapp.processEvents()

        assert not window.board_group.isVisibleTo(window)
    finally:
        window.close()


def test_gui_text_boards_use_recommended_dimensions(qapp: QApplication) -> None:
    """Stable text board size should come from the generated construction."""

    _ = qapp
    window = GameOfLifeWindow()
    try:
        window.source_combo.setCurrentText("Stable text")
        window.text_input.setPlainText("I")
        window.apply_simulation_settings()
        wait_for_gui(qapp, lambda: window.current_board is not None)
        insight = inspect_text_construction("I")

        assert window.current_board is not None
        assert not window.fit_text_button.isVisible()
        assert window.width_spin.value() == insight.recommended_width
        assert window.height_spin.value() == insight.recommended_height
    finally:
        window.close()


def test_gui_progress_bar_tracks_stable_text_settle_generation(qapp: QApplication) -> None:
    """Stable text progress should track generations until the construction settles."""

    _ = qapp
    window = GameOfLifeWindow()
    try:
        window.source_combo.setCurrentText("Stable text")
        window.text_input.setPlainText("I")
        window.apply_simulation_settings()
        wait_for_gui(qapp, lambda: window.current_board is not None)
        plan = render_text_block_construction("I")

        assert not window.progress_bar.isHidden()
        assert window.progress_bar.minimum() == 0
        assert window.progress_bar.maximum() == plan.generations
        assert window.progress_bar.value() == 0

        window.advance_generation()

        assert window.progress_bar.value() == 1
        assert window.progress_bar.format() == f"1 / {plan.generations}"
    finally:
        window.close()


def test_gui_generation_progress_bar_shows_while_text_board_is_built(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text generation should show determinate progress while building."""

    _ = qapp
    original_inspect = gui_module.inspect_text_construction
    started = Event()
    release = Event()
    window = GameOfLifeWindow()

    def tracked_inspect(
        text: str,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> object:
        assert progress_callback is not None
        progress_callback(1, 4)
        started.set()
        assert release.wait(timeout=2)
        return original_inspect(text, progress_callback=progress_callback)

    monkeypatch.setattr(gui_module, "inspect_text_construction", tracked_inspect)
    try:
        window.source_combo.setCurrentText("Stable text")
        window.text_input.setPlainText("I")

        window.apply_simulation_settings()

        assert started.wait(timeout=2)
        wait_for_gui(qapp, lambda: window.generation_progress_bar.value() == 1)
        assert not window.generation_progress_bar.isHidden()
        assert window.generation_progress_bar.minimum() == 0
        assert window.generation_progress_bar.maximum() == 4
        assert window.generation_progress_bar.format() == "Generating 1 / 4"
        assert not window.rebuild_button.isEnabled()

        release.set()
        wait_for_gui(qapp, lambda: window.current_board is not None)
        assert window.generation_progress_bar.isHidden()
        assert window.rebuild_button.isEnabled()
    finally:
        release.set()
        window.close()


def test_gui_playback_controls_live_under_preview_grid(qapp: QApplication) -> None:
    """Simulation buttons should be grouped with the board preview, not the form."""

    _ = qapp
    window = GameOfLifeWindow()
    try:
        assert window.play_button.parentWidget() is window.simulation_controls_bar
        assert window.step_button.parentWidget() is window.simulation_controls_bar
        assert window.draw_button.parentWidget() is window.simulation_controls_bar
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
        window.text_input.setPlainText("I")
        window.width_spin.setValue(260)
        window.height_spin.setValue(180)
        window.apply_simulation_settings()
        wait_for_gui(qapp, lambda: window.current_board is not None)

        assert window.current_board is not None
        assert window.board_canvas.is_showing_focus_region()
        target = centered_cells(
            window.current_board.config,
            render_text_block_construction("I").target_cells,
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
        wait_for_gui(qapp, lambda: window.current_board is not None)

        monkeypatch.setattr(
            gui_module.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(export_path), "Text Files (*.txt)"),
        )

        QTest.mouseClick(window.export_button, Qt.MouseButton.LeftButton)

        plan = render_text_block_construction("I")
        expected_cells = center_construction(window.current_board.config, plan)
        expected_lines = [
            (
                "# board_size="
                f"{window.current_board.config.width},{window.current_board.config.height}"
            ),
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
