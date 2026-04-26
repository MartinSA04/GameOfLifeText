"""PySide6 GUI for the Game of Life simulator."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPaintEvent, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .construction import center_construction, minimum_centered_board_size
from .simulator import Board, Pattern, SimulationConfig, centered_cells, random_cells
from .text import render_text_block_construction

APP_STYLESHEET = """
QMainWindow {
    background: #f4efe7;
    color: #1f2822;
}
QWidget {
    color: #1f2822;
}
QGroupBox {
    border: 1px solid #d8cfc3;
    border-radius: 12px;
    background: #fbf8f3;
    margin-top: 12px;
    padding: 14px 14px 12px 14px;
    font-weight: 600;
    color: #243029;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: #2a362f;
}
QPushButton {
    background: #e4d8c6;
    color: #233129;
    border: 1px solid #c4b59f;
    border-radius: 10px;
    padding: 8px 14px;
    min-height: 18px;
}
QPushButton:checked {
    background: #1e3d31;
    color: #f7f3ea;
    border-color: #1e3d31;
}
QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #fffdf9;
    color: #18231e;
    border: 1px solid #d6ccbf;
    border-radius: 8px;
    padding: 6px;
    selection-background-color: #c7dbc8;
    selection-color: #18231e;
}
QLineEdit:disabled,
QPlainTextEdit:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled,
QComboBox:disabled {
    color: #6b756f;
    background: #f2ede4;
}
QComboBox QAbstractItemView {
    background: #fffdf9;
    color: #18231e;
    selection-background-color: #d8e7d8;
    selection-color: #18231e;
}
QPlainTextEdit {
    selection-background-color: #c7dbc8;
}
QLabel {
    color: #233129;
}
QCheckBox {
    color: #233129;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
QCheckBox::indicator:unchecked {
    border: 1px solid #9aa394;
    background: #fffdf9;
    border-radius: 4px;
}
QCheckBox::indicator:checked {
    border: 1px solid #1e3d31;
    background: #1e3d31;
    border-radius: 4px;
}
QLabel#headline {
    font-size: 20px;
    font-weight: 700;
    color: #1f2822;
}
QLabel#subhead {
    color: #57655d;
}
"""


class SimulationSettings(BaseModel):
    """Validated settings for building a simulation board."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    source: str
    density: float = Field(ge=0.0, le=1.0)
    seed: int | None = None
    fps: float = Field(gt=0)
    steps: int | None = Field(default=None, ge=0)
    text: str | None = None
    wrap: bool = True


class BoardCanvas(QWidget):
    """Large board preview canvas with interactive drawing and zoom."""

    cell_painted = Signal(int, int, bool)
    view_changed = Signal()

    _MIN_ZOOM: float = 1.0
    _MAX_ZOOM: float = 12.0
    _ZOOM_STEP: float = 1.25
    _FOCUS_MARGIN_CELLS: int = 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._board: Board | None = None
        self._draw_enabled = False
        self._drag_alive_state: bool | None = None
        self._focus_pattern: Pattern | None = None
        self._show_focus_region = False
        self._last_drag_cell: tuple[int, int] | None = None
        self._message = "No board loaded"
        self._zoom_factor = 1.0
        self.setMinimumSize(720, 720)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def board(self) -> Board | None:
        return self._board

    def set_board(self, board: Board | None, *, message: str) -> None:
        self._board = board
        self._last_drag_cell = None
        self._message = message
        self.view_changed.emit()
        self.update()

    def set_focus_pattern(self, pattern: Pattern | None, *, show: bool = False) -> None:
        self._focus_pattern = pattern
        if pattern is None:
            self._show_focus_region = False
        elif show:
            self._show_focus_region = True
        self.view_changed.emit()
        self.update()

    def set_draw_enabled(self, enabled: bool) -> None:
        self._draw_enabled = enabled
        self._drag_alive_state = None
        self._last_drag_cell = None
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        self.unsetCursor()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(840, 840)

    def zoom_factor(self) -> float:
        return self._zoom_factor

    def has_focus_pattern(self) -> bool:
        return self._focus_pattern is not None and bool(self._focus_pattern)

    def is_showing_focus_region(self) -> bool:
        return self._show_focus_region and self.has_focus_pattern()

    def zoom_in(self) -> None:
        self._set_zoom(self._zoom_factor * self._ZOOM_STEP)

    def zoom_out(self) -> None:
        self._set_zoom(self._zoom_factor / self._ZOOM_STEP)

    def reset_zoom(self) -> None:
        self._set_zoom(1.0)

    def show_focus_region(self) -> None:
        if not self.has_focus_pattern():
            return
        self._show_focus_region = True
        self.reset_zoom()
        self.view_changed.emit()
        self.update()

    def show_full_board(self) -> None:
        self._show_focus_region = False
        self.reset_zoom()
        self.view_changed.emit()
        self.update()

    def cell_center(self, point: tuple[int, int]) -> QPoint | None:
        if self._board is None:
            return None
        geometry = self._board_geometry()
        if geometry is None:
            return None
        origin_x, origin_y, cell_size, board_width, board_height, min_x, min_y = geometry
        x, y = point
        if not 0 <= x < self._board.config.width or not 0 <= y < self._board.config.height:
            return None
        if not (
            min_x <= x < min_x + board_width // cell_size
            and min_y <= y < min_y + board_height // cell_size
        ):
            return None
        if board_width <= 0 or board_height <= 0:
            return None
        return QPoint(
            origin_x + (x - min_x) * cell_size + cell_size // 2,
            origin_y + (y - min_y) * cell_size + cell_size // 2,
        )

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        _ = event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f4efe7"))

        panel_rect = self.rect().adjusted(18, 18, -18, -18)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#fbf8f3"))
        painter.drawRoundedRect(panel_rect, 20, 20)

        caption_rect = panel_rect.adjusted(18, 12, -18, -18)
        painter.setPen(QColor("#43524b"))
        painter.setFont(QFont("Sans Serif", 11))
        painter.drawText(
            caption_rect,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            self._message,
        )

        if self._board is None:
            painter.setPen(QColor("#7a867f"))
            painter.setFont(QFont("Sans Serif", 14))
            painter.drawText(
                panel_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Choose settings on the left,\nthen build a board.",
            )
            return

        preview_rect = panel_rect.adjusted(28, 52, -28, -28)
        self._paint_board(painter, preview_rect, self._board)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._board is None:
            super().wheelEvent(event)
            return
        if event.angleDelta().y() > 0:
            self.zoom_in()
        elif event.angleDelta().y() < 0:
            self.zoom_out()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        alive_state = self._alive_state_for_button(event.button())
        if not self._draw_enabled or alive_state is None:
            super().mousePressEvent(event)
            return
        self._drag_alive_state = alive_state
        self._paint_at_event(event, alive=alive_state)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._draw_enabled or self._drag_alive_state is None:
            super().mouseMoveEvent(event)
            return
        self._paint_at_event(event, alive=self._drag_alive_state)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag_alive_state = None
        self._last_drag_cell = None
        super().mouseReleaseEvent(event)

    def _paint_board(self, painter: QPainter, rect: QRect, board: Board) -> None:
        _ = rect
        geometry = self._board_geometry()
        if geometry is None:
            return
        origin_x, origin_y, cell_size, board_width, board_height, min_x, min_y = geometry
        width = board_width // cell_size
        height = board_height // cell_size

        painter.fillRect(origin_x, origin_y, board_width, board_height, QColor("#ebe1d2"))
        painter.setPen(QPen(QColor("#d2c4b1"), 1))
        painter.drawRect(origin_x, origin_y, board_width, board_height)

        if cell_size >= 9:
            grid_pen = QPen(QColor("#ddcfbd"), 1)
            painter.setPen(grid_pen)
            for x in range(1, width):
                painter.drawLine(
                    origin_x + x * cell_size,
                    origin_y,
                    origin_x + x * cell_size,
                    origin_y + board_height,
                )
            for y in range(1, height):
                painter.drawLine(
                    origin_x,
                    origin_y + y * cell_size,
                    origin_x + board_width,
                    origin_y + y * cell_size,
                )

        live_margin = 1 if cell_size >= 6 else 0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#18332a"))
        for x, y in board.live_cells:
            if not (min_x <= x < min_x + width and min_y <= y < min_y + height):
                continue
            painter.drawRect(
                origin_x + (x - min_x) * cell_size + live_margin,
                origin_y + (y - min_y) * cell_size + live_margin,
                max(1, cell_size - live_margin * 2),
                max(1, cell_size - live_margin * 2),
            )

    def _paint_at_event(self, event: QMouseEvent, *, alive: bool) -> None:
        cell = self._cell_from_position(event.position().toPoint())
        if cell is None or cell == self._last_drag_cell:
            return
        self._last_drag_cell = cell
        self.cell_painted.emit(cell[0], cell[1], alive)

    def _cell_from_position(self, position: QPoint) -> tuple[int, int] | None:
        if self._board is None:
            return None
        geometry = self._board_geometry()
        if geometry is None:
            return None
        origin_x, origin_y, cell_size, board_width, board_height, min_x, min_y = geometry
        if not (
            origin_x <= position.x() < origin_x + board_width
            and origin_y <= position.y() < origin_y + board_height
        ):
            return None
        return (
            min_x + (position.x() - origin_x) // cell_size,
            min_y + (position.y() - origin_y) // cell_size,
        )

    def _board_geometry(self) -> tuple[int, int, int, int, int, int, int] | None:
        if self._board is None:
            return None
        panel_rect = self.rect().adjusted(18, 18, -18, -18)
        preview_rect = panel_rect.adjusted(28, 52, -28, -28)
        min_x, min_y, max_x, max_y = self._view_bounds()
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        base_cell_size = max(1, min(preview_rect.width() // width, preview_rect.height() // height))
        cell_size = max(1, int(base_cell_size * self._zoom_factor))
        board_width = width * cell_size
        board_height = height * cell_size
        origin_x = preview_rect.x() + (preview_rect.width() - board_width) // 2
        origin_y = preview_rect.y() + (preview_rect.height() - board_height) // 2
        return (origin_x, origin_y, cell_size, board_width, board_height, min_x, min_y)

    def _alive_state_for_button(self, button: Qt.MouseButton) -> bool | None:
        if button == Qt.MouseButton.LeftButton:
            return True
        if button == Qt.MouseButton.RightButton:
            return False
        return None

    def _set_zoom(self, zoom_factor: float) -> None:
        normalized = min(max(zoom_factor, self._MIN_ZOOM), self._MAX_ZOOM)
        if abs(normalized - self._zoom_factor) < 1e-9:
            return
        self._zoom_factor = normalized
        self.view_changed.emit()
        self.update()

    def _view_bounds(self) -> tuple[int, int, int, int]:
        if self._board is None:
            return (0, 0, 0, 0)
        if self.is_showing_focus_region():
            focus_pattern = self._focus_pattern
            if focus_pattern is None:
                return (0, 0, self._board.config.width - 1, self._board.config.height - 1)
            min_x, min_y, max_x, max_y = focus_pattern.bounds()
            return (
                max(0, min_x - self._FOCUS_MARGIN_CELLS),
                max(0, min_y - self._FOCUS_MARGIN_CELLS),
                min(self._board.config.width - 1, max_x + self._FOCUS_MARGIN_CELLS),
                min(self._board.config.height - 1, max_y + self._FOCUS_MARGIN_CELLS),
            )
        return (0, 0, self._board.config.width - 1, self._board.config.height - 1)


class GameOfLifeWindow(QMainWindow):
    """Main Qt window for the simulator."""

    def __init__(self) -> None:
        super().__init__()
        self.current_board: Board | None = None
        self.current_export_pattern: Pattern | None = None
        self.current_export_name: str | None = None
        self._current_step_limit: int | None = None
        self._steps_taken = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.advance_generation)
        self.setWindowTitle("Game of Life Studio")
        self.resize(1280, 920)
        self.setStyleSheet(APP_STYLESHEET)
        self._build_ui()
        self.board_canvas.cell_painted.connect(self.paint_simulation_cell)
        self.board_canvas.view_changed.connect(self._sync_view_controls)
        self._sync_source_controls()
        self._sync_view_controls()
        self.apply_simulation_settings()

    def apply_simulation_settings(self) -> None:
        """Build a new board from the current controls."""

        try:
            settings = self.build_simulation_settings()
            board, export_pattern = build_initial_board(settings)
            focus_pattern, show_focus = self._simulation_focus_pattern(settings, board)
        except ValueError as exc:
            self._show_error(str(exc))
            return

        self._stop_animation()
        self.current_board = board
        self.current_export_pattern = export_pattern
        self.current_export_name = (
            _suggest_export_name(settings.text) if settings.text is not None else None
        )
        self._current_step_limit = settings.steps
        self._steps_taken = 0
        self.board_canvas.set_board(
            self.current_board,
            message=self._board_message("Simulation ready"),
        )
        self._set_canvas_focus(focus_pattern, show=show_focus)
        self._sync_export_button()
        self._set_status("Simulation board rebuilt.")

    def advance_generation(self) -> None:
        """Advance the current board by one generation."""

        if self.current_board is None:
            return
        if self._current_step_limit is not None and self._steps_taken >= self._current_step_limit:
            self._stop_animation()
            self._set_status("Configured step limit reached.")
            return

        self.current_board = self.current_board.step()
        self._steps_taken += 1
        self.board_canvas.set_board(
            self.current_board,
            message=self._board_message("Simulation running"),
        )
        self._set_status("Advanced one generation.")

    def paint_simulation_cell(self, x: int, y: int, alive: bool) -> None:
        """Apply one interactive cell edit to the current board."""

        if self.current_board is None:
            return
        self._stop_animation()
        grid = self.current_board.grid.copy()
        grid[y, x] = alive
        self.current_board = Board(
            config=self.current_board.config,
            grid=grid,
            generation=self.current_board.generation,
        )
        self.board_canvas.set_board(
            self.current_board,
            message=self._board_message("Simulation edited"),
        )
        self._set_status(
            f"{'Added' if alive else 'Removed'} cell at ({x}, {y}). "
            "Left-drag adds, right-drag erases."
        )

    def build_simulation_settings(self) -> SimulationSettings:
        """Collect validated settings from the form."""

        source = self.source_combo.currentText()
        density = 0.18
        seed: int | None = None
        text: str | None = None

        if source == "Random board":
            density = self.random_density_spin.value()
            seed = parse_optional_int(self.seed_input.text())
        else:
            text = self.text_input.toPlainText().rstrip("\n")
            render_text_block_construction(text)

        return SimulationSettings(
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            source=source,
            density=density,
            seed=seed,
            fps=self.fps_spin.value(),
            steps=parse_optional_int(self.steps_input.text(), minimum=0),
            text=text,
            wrap=self.wrap_checkbox.isChecked(),
        )

    def _build_ui(self) -> None:
        """Build the complete window layout."""

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(4, 2, 4, 2)
        headline = QLabel("Game of Life Studio")
        headline.setObjectName("headline")
        subhead = QLabel(
            "Random boards, interactive drawing, and stable text from glider syntheses."
        )
        subhead.setObjectName("subhead")
        header_layout.addWidget(headline)
        header_layout.addWidget(subhead)
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        controls = self._build_simulation_panel()
        controls.setMinimumWidth(380)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(10)
        self.status_label = QLabel("Ready.")
        preview_layout.addWidget(self.status_label)
        view_bar = QWidget()
        view_layout = QHBoxLayout(view_bar)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(8)
        self.full_view_button = QPushButton("Full Board")
        self.focus_view_button = QPushButton("Focus Content")
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.setFixedWidth(40)
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setFixedWidth(40)
        self.view_state_label = QLabel("View: board | Zoom 1.00x")
        view_layout.addWidget(self.full_view_button)
        view_layout.addWidget(self.focus_view_button)
        view_layout.addStretch(1)
        view_layout.addWidget(self.zoom_out_button)
        view_layout.addWidget(self.zoom_in_button)
        view_layout.addWidget(self.view_state_label)
        preview_layout.addWidget(view_bar)
        self.board_canvas = BoardCanvas()
        self.full_view_button.clicked.connect(self.board_canvas.show_full_board)
        self.focus_view_button.clicked.connect(self.board_canvas.show_focus_region)
        self.zoom_out_button.clicked.connect(self.board_canvas.zoom_out)
        self.zoom_in_button.clicked.connect(self.board_canvas.zoom_in)
        preview_layout.addWidget(self.board_canvas)
        self.draw_button.toggled.connect(self.board_canvas.set_draw_enabled)

        splitter.addWidget(controls)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

    def _build_simulation_panel(self) -> QWidget:
        """Build the simulation controls panel."""

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        seed_group = QGroupBox("Seed")
        seed_layout = QVBoxLayout(seed_group)
        seed_form = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(("Random board", "Stable text"))
        self.source_combo.currentIndexChanged.connect(self._sync_source_controls)
        seed_form.addRow("Source", self.source_combo)
        seed_layout.addLayout(seed_form)

        self.random_frame = self._build_random_seed_controls()
        self.text_frame = self._build_text_seed_controls()
        seed_layout.addWidget(self.random_frame)
        seed_layout.addWidget(self.text_frame)
        layout.addWidget(seed_group)

        board_group = QGroupBox("Board")
        board_form = QFormLayout(board_group)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 400)
        self.width_spin.setValue(40)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 300)
        self.height_spin.setValue(20)
        self.wrap_checkbox = QCheckBox("Toroidal wrapping")
        self.wrap_checkbox.setChecked(True)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 120.0)
        self.fps_spin.setValue(12.0)
        self.fps_spin.setSingleStep(1.0)
        self.fps_spin.valueChanged.connect(self._sync_timer_interval)
        self.steps_input = QLineEdit()
        self.steps_input.setPlaceholderText("Blank = continuous")
        board_form.addRow("Width", self.width_spin)
        board_form.addRow("Height", self.height_spin)
        board_form.addRow("FPS", self.fps_spin)
        board_form.addRow("Steps", self.steps_input)
        board_form.addRow("", self.wrap_checkbox)
        layout.addWidget(board_group)

        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)
        self.rebuild_button = QPushButton("Build Board")
        self.rebuild_button.clicked.connect(self.apply_simulation_settings)
        self.draw_button = QPushButton("Draw Cells")
        self.draw_button.setCheckable(True)
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggle_animation)
        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self.advance_generation)
        self.export_button = QPushButton("Export Plan")
        self.export_button.clicked.connect(self.export_plan)
        actions_layout.addWidget(self.rebuild_button)
        actions_layout.addWidget(self.draw_button)
        actions_layout.addWidget(self.play_button)
        actions_layout.addWidget(self.step_button)
        actions_layout.addWidget(self.export_button)
        layout.addWidget(actions_group)
        layout.addStretch(1)
        return widget

    def _build_random_seed_controls(self) -> QFrame:
        frame = QFrame()
        form = QFormLayout(frame)
        self.random_density_spin = QDoubleSpinBox()
        self.random_density_spin.setRange(0.0, 1.0)
        self.random_density_spin.setSingleStep(0.01)
        self.random_density_spin.setValue(0.18)
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Blank = random")
        form.addRow("Density", self.random_density_spin)
        form.addRow("Seed", self.seed_input)
        return frame

    def _build_text_seed_controls(self) -> QFrame:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        self.text_input = QPlainTextEdit()
        self.text_input.setPlaceholderText("Enter ASCII text")
        self.text_input.setFixedHeight(110)
        layout.addWidget(self.text_input)
        return frame

    def _toggle_animation(self, checked: bool) -> None:
        if checked:
            if self.current_board is None:
                self.apply_simulation_settings()
            if self.current_board is None:
                self.play_button.setChecked(False)
                return
            self.play_button.setText("Pause")
            self._sync_timer_interval()
            self._timer.start()
            self._set_status("Animation running.")
            return

        self._timer.stop()
        self.play_button.setText("Play")

    def _stop_animation(self) -> None:
        self._timer.stop()
        if self.play_button.isChecked():
            self.play_button.blockSignals(True)
            self.play_button.setChecked(False)
            self.play_button.blockSignals(False)
        self.play_button.setText("Play")

    def _sync_timer_interval(self) -> None:
        interval_ms = max(1, int(1000 / self.fps_spin.value()))
        self._timer.setInterval(interval_ms)

    def _simulation_focus_pattern(
        self,
        settings: SimulationSettings,
        board: Board,
    ) -> tuple[Pattern | None, bool]:
        if settings.text is not None:
            target = render_text_block_construction(settings.text).target_cells
            return (centered_cells(board.config, target), True)
        return (None, False)

    def _set_canvas_focus(self, pattern: Pattern | None, *, show: bool) -> None:
        self.board_canvas.set_focus_pattern(pattern, show=show)

    def _sync_view_controls(self) -> None:
        has_board = self.board_canvas.board() is not None
        has_focus = self.board_canvas.has_focus_pattern()
        showing_focus = self.board_canvas.is_showing_focus_region()
        self.full_view_button.setEnabled(has_board)
        self.focus_view_button.setEnabled(has_board and has_focus)
        self.zoom_in_button.setEnabled(has_board)
        self.zoom_out_button.setEnabled(has_board)
        mode = "content" if showing_focus else "board"
        if not has_board:
            mode = "none"
        self.view_state_label.setText(f"View: {mode} | Zoom {self.board_canvas.zoom_factor():.2f}x")

    def _sync_source_controls(self) -> None:
        source = self.source_combo.currentText()
        self.random_frame.setVisible(source == "Random board")
        self.text_frame.setVisible(source == "Stable text")
        if source == "Stable text":
            self.width_spin.setValue(max(self.width_spin.value(), 220))
            self.height_spin.setValue(max(self.height_spin.value(), 140))
            self.wrap_checkbox.setChecked(False)
        self._sync_export_button()

    def _sync_export_button(self) -> None:
        source = self.source_combo.currentText()
        has_export_pattern = self.current_export_pattern is not None and bool(self.current_export_pattern)
        self.export_button.setEnabled(source == "Stable text" and has_export_pattern)

    def export_plan(self) -> None:
        """Export the latest stable-text seed plus its board size."""

        if (
            self.current_board is None
            or self.current_export_pattern is None
            or not self.current_export_pattern
        ):
            self._show_error("Build a stable-text board before exporting a plan.")
            return

        suggested_name = self.current_export_name or "stable_text_plan.txt"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Generated Plan",
            suggested_name,
            "Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return

        lines = [
            f"# board_size={self.current_board.config.width},{self.current_board.config.height}",
            *(f"{x},{y}" for x, y in self.current_export_pattern),
        ]
        try:
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            self._show_error(f"could not export plan: {exc}")
            return
        self._set_status(f"Exported plan to {path}.")

    def _board_message(self, prefix: str) -> str:
        if self.current_board is None:
            return prefix
        return (
            f"{prefix} | generation {self.current_board.generation} | "
            f"alive {self.current_board.alive_count} | "
            f"wrap {'on' if self.current_board.config.wrap else 'off'}"
        )

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Game of Life Studio", message)


def parse_optional_int(text: str, *, minimum: int | None = None) -> int | None:
    """Parse an optional integer field from a line edit."""

    normalized = text.strip()
    if normalized == "":
        return None
    try:
        value = int(normalized)
    except ValueError as exc:
        msg = "enter a whole number or leave the field blank"
        raise ValueError(msg) from exc
    if minimum is not None and value < minimum:
        msg = f"enter a value greater than or equal to {minimum}"
        raise ValueError(msg)
    return value


def build_initial_board(settings: SimulationSettings) -> tuple[Board, Pattern | None]:
    """Construct the initial board from validated settings."""

    config = SimulationConfig(width=settings.width, height=settings.height, wrap=settings.wrap)
    export_pattern: Pattern | None = None
    if settings.text is not None:
        plan = render_text_block_construction(settings.text)
        width, height = minimum_centered_board_size(plan, padding=4)
        config = SimulationConfig(
            width=max(settings.width, width),
            height=max(settings.height, height),
            wrap=settings.wrap,
        )
        live_cells = center_construction(config, plan)
        export_pattern = live_cells
    else:
        live_cells = random_cells(config, settings.density, seed=settings.seed)
    return (Board.from_points(config, live_cells), export_pattern)


def _suggest_export_name(text: str | None) -> str:
    """Return a readable default filename for a stable-text export."""

    if text is None:
        return "stable_text_plan.txt"
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in text.strip())
    normalized = "_".join(part for part in cleaned.split("_") if part)
    if not normalized:
        normalized = "stable_text"
    return f"{normalized}_plan.txt"


def create_application() -> QApplication:
    """Return a QApplication instance."""

    app = QApplication.instance()
    if isinstance(app, QApplication):
        return app
    return QApplication(sys.argv)


def main() -> int:
    """Launch the PySide6 GUI."""

    app = create_application()
    window = GameOfLifeWindow()
    window.show()
    return app.exec()
