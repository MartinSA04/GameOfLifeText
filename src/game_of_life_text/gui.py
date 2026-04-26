"""PySide6 GUI for the Game of Life simulator."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPainter, QPaintEvent, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
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

from .construction import ConstructionPlan, center_construction, minimum_centered_board_size
from .simulator import Board, Pattern, SimulationConfig, centered_cells, random_cells
from .text import render_text_block_construction

TEXT_SOURCE = "Stable text"
RANDOM_SOURCE = "Random board"
BLANK_SOURCE = "Blank board"
MANUAL_BOARD_SOURCES = frozenset((RANDOM_SOURCE, BLANK_SOURCE))

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
    font-weight: 700;
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
QLabel#insightLabel {
    background: #f1eadf;
    border: 1px solid #d8c8b3;
    border-radius: 10px;
    color: #314039;
    padding: 8px 10px;
}
QPushButton#primaryAction {
    background: #1e3d31;
    color: #f7f3ea;
    border-color: #1e3d31;
    font-weight: 700;
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


@dataclass(frozen=True, slots=True)
class TextConstructionInsight:
    """Readable metadata about the current text construction."""

    plan: ConstructionPlan
    line_count: int
    character_count: int
    block_count: int
    target_width: int
    target_height: int
    recommended_width: int
    recommended_height: int


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
        self._message = ""
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
                "No board",
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
        self.current_settings: SimulationSettings | None = None
        self.current_export_pattern: Pattern | None = None
        self.current_export_name: str | None = None
        self.current_text_insight: TextConstructionInsight | None = None
        self._current_step_limit: int | None = None
        self._steps_taken = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.advance_generation)
        self.setWindowTitle("Game of Life Text Studio")
        self.resize(1280, 920)
        self.setStyleSheet(APP_STYLESHEET)
        self._build_ui()
        self.board_canvas.cell_painted.connect(self.paint_simulation_cell)
        self.board_canvas.view_changed.connect(self._sync_view_controls)
        self._sync_source_controls()
        self._sync_view_controls()
        self._refresh_preview_summary()

    def apply_simulation_settings(self) -> None:
        """Build a new board from the current controls."""

        text_insight: TextConstructionInsight | None = None
        try:
            settings = self.build_simulation_settings()
            if settings.text is not None:
                text_insight = inspect_text_construction(settings.text)
            board, export_pattern = build_initial_board(
                settings,
                text_plan=text_insight.plan if text_insight is not None else None,
            )
            focus_pattern, show_focus = self._simulation_focus_pattern(
                settings,
                board,
                text_insight=text_insight,
            )
        except ValueError as exc:
            self._show_error(str(exc))
            return

        self._stop_animation()
        self.current_board = board
        self.current_settings = settings
        self.current_export_pattern = export_pattern
        self.current_export_name = (
            _suggest_export_name(settings.text) if settings.text is not None else None
        )
        self.current_text_insight = text_insight
        self._current_step_limit = settings.steps
        self._steps_taken = 0
        self._sync_board_size_controls(board)
        self.board_canvas.set_board(
            self.current_board,
            message=self._board_message(),
        )
        self._set_canvas_focus(focus_pattern, show=show_focus)
        self._sync_export_button()
        self._refresh_preview_summary()
        self._set_status("Generated." if settings.text is not None else "Ready.")

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
            message=self._board_message(),
        )
        self._refresh_preview_summary()
        self._set_status(f"Generation {self.current_board.generation}.")

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
            message=self._board_message(),
        )
        self._refresh_preview_summary()
        self._set_status(f"{'Added' if alive else 'Removed'} ({x}, {y}).")

    def build_simulation_settings(self) -> SimulationSettings:
        """Collect validated settings from the form."""

        source = self.source_combo.currentText()
        density = 0.18
        seed: int | None = None
        text: str | None = None

        if source == RANDOM_SOURCE:
            density = self.random_density_spin.value()
            seed = parse_optional_int(self.seed_input.text())
        elif source == BLANK_SOURCE:
            density = 0.0
        else:
            text = self.text_input.toPlainText().rstrip("\n")
            if text == "":
                msg = "text cannot be empty"
                raise ValueError(msg)

        return SimulationSettings(
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            source=source,
            density=density,
            seed=seed,
            fps=self.fps_spin.value(),
            steps=parse_optional_int(self.steps_input.text(), minimum=0),
            text=text,
            wrap=self.wrap_checkbox.isChecked() if source in MANUAL_BOARD_SOURCES else False,
        )

    def _build_ui(self) -> None:
        """Build the complete window layout."""

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        controls = self._build_simulation_panel()
        controls.setMinimumWidth(380)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(10)
        self.status_label = QLabel()
        self.status_label.setVisible(False)
        preview_layout.addWidget(self.status_label)
        self.preview_summary_label = QLabel()
        self.preview_summary_label.setObjectName("insightLabel")
        self.preview_summary_label.setWordWrap(True)
        self.preview_summary_label.setVisible(False)
        view_bar = QWidget()
        view_layout = QHBoxLayout(view_bar)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(8)
        self.full_view_button = QPushButton("Board")
        self.full_view_button.setCheckable(True)
        self.focus_view_button = QPushButton("Text")
        self.focus_view_button.setCheckable(True)
        self.view_button_group = QButtonGroup(self)
        self.view_button_group.setExclusive(True)
        self.view_button_group.addButton(self.full_view_button)
        self.view_button_group.addButton(self.focus_view_button)
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.setFixedWidth(40)
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setFixedWidth(40)
        self.view_state_label = QLabel("Preview: board | Zoom 1.00x")
        self.view_state_label.setVisible(False)
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

        self.simulation_controls_bar = QWidget()
        simulation_controls_layout = QHBoxLayout(self.simulation_controls_bar)
        simulation_controls_layout.setContentsMargins(0, 0, 0, 0)
        simulation_controls_layout.setSpacing(8)
        self.draw_button = QCheckBox("Draw mode")
        self.play_button = QPushButton("Run")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._toggle_animation)
        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self.advance_generation)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.1, 120.0)
        self.fps_spin.setValue(12.0)
        self.fps_spin.setSingleStep(1.0)
        self.fps_spin.valueChanged.connect(self._sync_timer_interval)
        self.steps_input = QLineEdit()
        self.steps_input.setFixedWidth(84)
        self.steps_input.setVisible(False)
        simulation_controls_layout.addWidget(self.draw_button)
        simulation_controls_layout.addWidget(self.play_button)
        simulation_controls_layout.addWidget(self.step_button)
        simulation_controls_layout.addStretch(1)
        simulation_controls_layout.addWidget(QLabel("Speed"))
        simulation_controls_layout.addWidget(self.fps_spin)
        preview_layout.addWidget(self.simulation_controls_bar)
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

        generator_group = QGroupBox("Setup")
        generator_layout = QVBoxLayout(generator_group)
        seed_form = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems((TEXT_SOURCE, RANDOM_SOURCE, BLANK_SOURCE))
        self.source_combo.currentIndexChanged.connect(self._sync_source_controls)
        seed_form.addRow("Mode", self.source_combo)
        generator_layout.addLayout(seed_form)
        self.text_frame = self._build_text_seed_controls()
        self.random_frame = self._build_random_seed_controls()
        generator_layout.addWidget(self.text_frame)
        generator_layout.addWidget(self.random_frame)
        layout.addWidget(generator_group)

        primary_row = QHBoxLayout()
        primary_row.setContentsMargins(0, 0, 0, 0)
        self.rebuild_button = QPushButton("Generate")
        self.rebuild_button.setObjectName("primaryAction")
        self.rebuild_button.clicked.connect(self.apply_simulation_settings)
        self.export_button = QPushButton("Export Plan")
        self.export_button.clicked.connect(self.export_plan)
        primary_row.addWidget(self.rebuild_button)
        primary_row.addWidget(self.export_button)
        layout.addLayout(primary_row)

        self.board_group = QGroupBox("Board")
        board_layout = QVBoxLayout(self.board_group)
        board_form = QFormLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 400)
        self.width_spin.setValue(240)
        self.width_spin.valueChanged.connect(self._sync_generation_controls)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 300)
        self.height_spin.setValue(160)
        self.height_spin.valueChanged.connect(self._sync_generation_controls)
        self.wrap_checkbox = QCheckBox("Toroidal wrapping")
        self.wrap_checkbox.setChecked(False)
        board_form.addRow("Width", self.width_spin)
        board_form.addRow("Height", self.height_spin)
        board_form.addRow("", self.wrap_checkbox)
        board_layout.addLayout(board_form)
        layout.addWidget(self.board_group)
        layout.addStretch(1)
        return widget

    def _build_random_seed_controls(self) -> QFrame:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.random_density_spin = QDoubleSpinBox()
        self.random_density_spin.setRange(0.0, 1.0)
        self.random_density_spin.setSingleStep(0.01)
        self.random_density_spin.setValue(0.18)
        self.seed_input = QLineEdit()
        form.addRow("Density", self.random_density_spin)
        form.addRow("Seed", self.seed_input)
        layout.addLayout(form)
        return frame

    def _build_text_seed_controls(self) -> QFrame:
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        self.text_input = QPlainTextEdit()
        self.text_input.setFixedHeight(150)
        self.text_input.textChanged.connect(self._sync_generation_controls)
        layout.addWidget(self.text_input)
        self.text_summary_label = QLabel()
        self.text_summary_label.setObjectName("insightLabel")
        self.text_summary_label.setWordWrap(True)
        self.text_summary_label.setVisible(False)
        self.fit_text_button = QPushButton("Fit Board")
        self.fit_text_button.clicked.connect(self.apply_recommended_text_board_size)
        self.fit_text_button.setVisible(False)
        self.fit_text_button.setEnabled(False)
        return frame

    def _toggle_animation(self, checked: bool) -> None:
        if checked:
            if self.current_board is None:
                self.play_button.setChecked(False)
                self._set_status("Build a board first.")
                return
            self.play_button.setText("Pause")
            self._sync_timer_interval()
            self._timer.start()
            self._set_status("Running.")
            return

        self._timer.stop()
        self.play_button.setText("Run")

    def _stop_animation(self) -> None:
        self._timer.stop()
        if self.play_button.isChecked():
            self.play_button.blockSignals(True)
            self.play_button.setChecked(False)
            self.play_button.blockSignals(False)
        self.play_button.setText("Run")

    def _sync_timer_interval(self) -> None:
        interval_ms = max(1, int(1000 / self.fps_spin.value()))
        self._timer.setInterval(interval_ms)

    def _simulation_focus_pattern(
        self,
        settings: SimulationSettings,
        board: Board,
        *,
        text_insight: TextConstructionInsight | None = None,
    ) -> tuple[Pattern | None, bool]:
        if settings.text is not None:
            target = (
                text_insight.plan.target_cells
                if text_insight is not None
                else render_text_block_construction(settings.text).target_cells
            )
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
        self.full_view_button.setVisible(has_focus)
        self.focus_view_button.setVisible(has_focus)
        self.full_view_button.blockSignals(True)
        self.focus_view_button.blockSignals(True)
        self.full_view_button.setChecked(has_board and not showing_focus)
        self.focus_view_button.setChecked(has_board and showing_focus)
        self.full_view_button.blockSignals(False)
        self.focus_view_button.blockSignals(False)
        self.zoom_in_button.setEnabled(has_board)
        self.zoom_out_button.setEnabled(has_board)
        mode = "content" if showing_focus else "board"
        if not has_board:
            mode = "none"
        self.view_state_label.setText(
            f"Preview: {mode} | Zoom {self.board_canvas.zoom_factor():.2f}x"
        )

    def _sync_source_controls(self) -> None:
        source = self.source_combo.currentText()
        text_mode = source == TEXT_SOURCE
        manual_board_mode = source in MANUAL_BOARD_SOURCES
        self.random_frame.setVisible(source == RANDOM_SOURCE)
        self.text_frame.setVisible(text_mode)
        self.board_group.setVisible(manual_board_mode)
        if text_mode:
            self.rebuild_button.setText("Generate")
        elif source == BLANK_SOURCE:
            self.rebuild_button.setText("Blank")
        else:
            self.rebuild_button.setText("Randomize")
        self._sync_generation_controls()
        self._sync_export_button()

    def _sync_export_button(self) -> None:
        source = self.source_combo.currentText()
        has_export_pattern = self.current_export_pattern is not None and bool(
            self.current_export_pattern
        )
        can_export = (
            source == TEXT_SOURCE and has_export_pattern and self._generated_text_matches_editor()
        )
        self.export_button.setVisible(can_export)
        self.export_button.setEnabled(can_export)

    def _sync_board_size_controls(self, board: Board) -> None:
        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.width_spin.setValue(board.config.width)
        self.height_spin.setValue(board.config.height)
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)
        self._sync_generation_controls()

    def _sync_generation_controls(self) -> None:
        if self.source_combo.currentText() != TEXT_SOURCE:
            self.text_summary_label.clear()
            self.fit_text_button.setEnabled(False)
            self.rebuild_button.setEnabled(True)
            self._sync_export_button()
            return

        text = self._editor_text()
        if text == "":
            self.text_summary_label.clear()
            self.fit_text_button.setEnabled(False)
            self.rebuild_button.setEnabled(False)
            self._sync_export_button()
            return

        self.rebuild_button.setEnabled(True)
        if self._generated_text_matches_editor():
            assert self.current_text_insight is not None
            self.text_summary_label.clear()
            self.fit_text_button.setEnabled(True)
            self._sync_export_button()
            return

        self.text_summary_label.clear()
        self.fit_text_button.setEnabled(False)
        self._sync_export_button()

    def apply_recommended_text_board_size(self) -> None:
        """Apply the current text construction's recommended board size."""

        if not self._generated_text_matches_editor():
            return
        assert self.current_text_insight is not None
        self.width_spin.setValue(self.current_text_insight.recommended_width)
        self.height_spin.setValue(self.current_text_insight.recommended_height)
        self.wrap_checkbox.setChecked(False)
        self._set_status("Fit.")

    def _editor_text(self) -> str:
        return self.text_input.toPlainText().rstrip("\n")

    def _generated_text_matches_editor(self) -> bool:
        return (
            self.current_settings is not None
            and self.current_settings.text is not None
            and self.current_text_insight is not None
            and self.current_settings.text == self._editor_text()
        )

    def _refresh_preview_summary(self) -> None:
        self.preview_summary_label.clear()

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

    def _board_message(self) -> str:
        if self.current_board is None:
            return ""
        if self.current_settings is not None and self.current_settings.text is not None:
            if self.current_text_insight is not None:
                settle = self.current_text_insight.plan.generations
                generation = (
                    f"{self.current_board.generation}/{settle}"
                    if self.current_board.generation < settle
                    else str(self.current_board.generation)
                )
            else:
                generation = str(self.current_board.generation)
            return f"Gen {generation} | Live {self.current_board.alive_count}"
        return (
            f"Gen {self.current_board.generation} | "
            f"Live {self.current_board.alive_count} | "
            f"Wrap {'on' if self.current_board.config.wrap else 'off'}"
        )

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_label.setVisible(message != "")

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Game of Life Text Studio", message)


def inspect_text_construction(text: str) -> TextConstructionInsight:
    """Return readable metadata for one text construction request."""

    plan = render_text_block_construction(text)
    target_min_x, target_min_y, target_max_x, target_max_y = plan.target_cells.bounds()
    recommended_width, recommended_height = minimum_centered_board_size(plan, padding=4)
    return TextConstructionInsight(
        plan=plan,
        line_count=max(1, len(text.splitlines())),
        character_count=sum(1 for character in text if character != "\n"),
        block_count=len(plan.target_cells) // 4,
        target_width=target_max_x - target_min_x + 1,
        target_height=target_max_y - target_min_y + 1,
        recommended_width=recommended_width,
        recommended_height=recommended_height,
    )


def _format_text_insight_message(
    insight: TextConstructionInsight,
    *,
    selected_width: int,
    selected_height: int,
) -> str:
    """Render a compact user-facing summary of a text construction."""

    if (
        selected_width >= insight.recommended_width
        and selected_height >= insight.recommended_height
    ):
        fit_status = "fits"
    else:
        fit_status = "expand"
    return (
        f"{insight.character_count} chars"
        f"  |  {insight.line_count} line(s)"
        f"  |  {insight.block_count} blocks"
        f"  |  settle {insight.plan.generations}"
        f"  |  final {insight.target_width} x {insight.target_height}"
        f"  |  min {insight.recommended_width} x {insight.recommended_height}"
        f"  |  {fit_status}"
    )


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


def build_initial_board(
    settings: SimulationSettings,
    *,
    text_plan: ConstructionPlan | None = None,
) -> tuple[Board, Pattern | None]:
    """Construct the initial board from validated settings."""

    config = SimulationConfig(width=settings.width, height=settings.height, wrap=settings.wrap)
    export_pattern: Pattern | None = None
    if settings.text is not None:
        plan = text_plan if text_plan is not None else render_text_block_construction(settings.text)
        width, height = minimum_centered_board_size(plan, padding=4)
        config = SimulationConfig(width=width, height=height, wrap=False)
        live_cells = center_construction(config, plan)
        export_pattern = live_cells
    else:
        live_cells = random_cells(config, settings.density, seed=settings.seed)
    return (Board.from_points(config, live_cells), export_pattern)


def _suggest_export_name(text: str | None) -> str:
    """Return a readable default filename for a stable-text export."""

    if text is None:
        return "stable_text_plan.txt"
    cleaned = "".join(
        character.lower() if character.isalnum() else "_" for character in text.strip()
    )
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
