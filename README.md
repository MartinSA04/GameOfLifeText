# Game of Life Text Studio

A typed Conway's Game of Life desktop app built with PySide6, centered on deterministic text generation from glider syntheses.

## Requirements

- Python 3.14+ (`.python-version` pins the current 3.14.4 release)
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

- A text-first GUI that opens directly into stable-text generation
- Auto-sized stable-text boards, with manual board size shown only for random and blank boards
- A large board canvas with text/board view toggles and zoom
- A generation progress bar plus a stable-text settle progress bar
- **Stable text** seeding that synthesizes ASCII text from a deterministic
  multi-glider construction whose blocks settle into the requested still life
- **Random board** seeding with adjustable density and reproducible seed
- **Blank board** seeding for drawing from scratch
- **Draw mode** for left-drag add and right-drag erase
- Run/pause and step controls with adjustable speed

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
