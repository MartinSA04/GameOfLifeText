# Game of Life Text Studio

A typed PySide6 desktop app for Conway's Game of Life, focused on generating
stable text from deterministic glider syntheses.

![Game of Life Text Studio demo](./GOL.GIF)

## Highlights

- Text-first workflow for stable block-text generation, including uppercase and lowercase letters.
- Determinate generation progress while the glider construction is built.
- Settle progress while the board evolves into the final text.
- Auto-sized text boards, with manual board size only shown for random and blank boards.
- Focused text/board preview modes, zoom, run/pause, stepping, and draw mode.
- Random and blank board modes for sandboxing and drawing from scratch.
- Exportable stable-text seed plans as centered `x,y` cells.

## Requirements

- Python 3.14+ (`.python-version` pins the current 3.14.4 release)
- [`uv`](https://docs.astral.sh/uv/)

## Quick Start

```bash
uv python install 3.14.4
uv sync --python 3.14.4
uv run game-of-life-gui
```

If Python 3.14.4 is already installed, `uv sync` is enough.

## Workflow

1. Open the app in stable-text mode.
2. Type supported ASCII text, including uppercase and lowercase letters.
3. Select `Generate`.
4. Watch the generation progress, then run or step the board until the settle bar completes.
5. Use `Text` and `Board` preview toggles to switch between the finished text and the full seed.

## Controls

- `Generate`: builds a stable-text glider seed.
- `Randomize`: creates a random board using density and optional seed.
- `Blank`: creates an empty board with the chosen dimensions.
- `Draw mode`: left-drag adds cells, right-drag erases cells.
- `Run` / `Pause`: starts and stops evolution.
- `Step`: advances one generation.
- `+` / `-`: zooms the board preview.

## How It Works

The text renderer turns glyph pixels into 2x2 still-life blocks. The planner
places two-glider syntheses from the text center outward, choosing launch
directions and delays that avoid collisions. Every generated construction is
verified before it is shown in the GUI.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```
