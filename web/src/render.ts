/**
 * Canvas painter. Colors are read from the stylesheet's custom properties, so
 * styles.css stays the only place the palette is defined.
 */

import type { Camera, CellRange, Size } from "./camera.js";
import { visibleRange } from "./camera.js";
import type { Grid } from "./life.js";

export interface Frame {
  board: Size;
  camera: Camera;
  cells: Grid | null;
  ghost: Grid | null;
}

export interface Renderer {
  /** Match the backing store to the element size; returns the CSS viewport. */
  resize(viewport: Size): Size;
  draw(frame: Frame): void;
  /** Forget the palette, so the next draw reads the stylesheet again. */
  reread(): void;
}

interface Palette {
  canvas: string;
  cell: string;
  ghost: string;
  grid: string;
  /** How much of the halo under each cell to keep; a lit board wants more. */
  bloom: number;
}

interface PaintOptions {
  color: string;
  alpha?: number;
  inflate?: number;
}

const GRID_SCALE = 9;
const BLOOM_SCALE = 3;
const CELL_GAP_SCALE = 5;
const MAX_DEVICE_RATIO = 2;

/** Null until the stylesheet has applied and the properties resolve. */
function palette(): Palette | null {
  const styles = getComputedStyle(document.documentElement);
  const read = (name: string) => styles.getPropertyValue(name).trim();
  const colors = {
    canvas: read("--canvas"),
    cell: read("--cell"),
    ghost: read("--cell-ghost"),
    grid: read("--canvas-grid"),
    bloom: Number(read("--cell-bloom"))
  };
  return colors.cell ? colors : null;
}

function context2d(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) throw new Error("this browser has no 2d canvas context");
  return context;
}

export function createRenderer(canvas: HTMLCanvasElement): Renderer {
  const context = context2d(canvas);
  let viewport: Size = { width: 0, height: 0 };
  let colors: Palette | null = null;

  function paint(
    grid: Grid,
    board: Size,
    camera: Camera,
    range: CellRange,
    { color, alpha = 1, inflate = 0 }: PaintOptions
  ) {
    const { scale } = camera;
    const gap = scale >= CELL_GAP_SCALE ? 1 : 0;
    const size = Math.max(1, scale - gap) + inflate * 2;
    // Below a few pixels per cell, snapping merges neighbors into blobs.
    const snap = scale >= BLOOM_SCALE;

    context.globalAlpha = alpha;
    context.fillStyle = color;
    for (let y = range.y0; y < range.y1; y += 1) {
      const row = y * board.width;
      const top = camera.y + y * scale - inflate;
      for (let x = range.x0; x < range.x1; x += 1) {
        if (!grid[row + x]) continue;
        const left = camera.x + x * scale - inflate;
        if (snap) context.fillRect(Math.round(left), Math.round(top), size, size);
        else context.fillRect(left, top, size, size);
      }
    }
    context.globalAlpha = 1;
  }

  function paintGrid(camera: Camera, range: CellRange, color: string) {
    context.strokeStyle = color;
    context.lineWidth = 1;
    context.beginPath();
    for (let x = range.x0; x <= range.x1; x += 1) {
      const left = Math.round(camera.x + x * camera.scale) + 0.5;
      context.moveTo(left, camera.y + range.y0 * camera.scale);
      context.lineTo(left, camera.y + range.y1 * camera.scale);
    }
    for (let y = range.y0; y <= range.y1; y += 1) {
      const top = Math.round(camera.y + y * camera.scale) + 0.5;
      context.moveTo(camera.x + range.x0 * camera.scale, top);
      context.lineTo(camera.x + range.x1 * camera.scale, top);
    }
    context.stroke();
  }

  return {
    reread() {
      colors = null;
    },

    resize(next) {
      viewport = next;
      const ratio = Math.min(window.devicePixelRatio || 1, MAX_DEVICE_RATIO);
      const width = Math.max(1, Math.round(viewport.width * ratio));
      const height = Math.max(1, Math.round(viewport.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      return viewport;
    },

    draw({ board, camera, cells, ghost }) {
      colors ??= palette();
      // Nothing sensible to paint with yet; the next draw picks the colors up.
      if (!colors) return;

      context.fillStyle = colors.canvas;
      context.fillRect(0, 0, viewport.width, viewport.height);
      if (!board.width || !board.height || !cells) return;

      const range = visibleRange(camera, board, viewport);
      if (camera.scale >= GRID_SCALE) paintGrid(camera, range, colors.grid);
      if (ghost) paint(ghost, board, camera, range, { color: colors.ghost });
      if (camera.scale >= BLOOM_SCALE && colors.bloom > 0) {
        paint(cells, board, camera, range, {
          color: colors.cell,
          alpha: colors.bloom,
          inflate: 2
        });
      }
      paint(cells, board, camera, range, { color: colors.cell });
    }
  };
}
