# Game of Life

A small, typed Conway's Game of Life desktop app built with PySide6.

## Requirements

- Python 3.12+
- `uv` by Astral

## Setup

```bash
uv sync
```

## Run

```bash
uv run game-of-life-gui
```

## Features

- A large board canvas with full-board and content-focused views, plus zoom
- **Random board** seeding with an adjustable density and reproducible seed
- **Stable text** seeding that synthesizes ASCII text from a deterministic
  multi-glider construction whose blocks settle into the requested still life
- **Drawing**: toggle "Draw Cells" and left-drag to add cells, right-drag to
  erase
- Play/pause and step controls with adjustable FPS and an optional step limit

The block-text constructor schedules each 2x2 block from the center outward,
launching its two-glider synthesis on a farther ring with a larger delay so
all blocks land at the same time. Unsupported layouts fail with a clear error
instead of drawing the wrong pattern.

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```
