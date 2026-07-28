/**
 * Game of Life kernel over flat grids: one byte per cell, row-major,
 * `width * height` long. The planner runs in Python; this is what the browser
 * uses to actually evolve a board.
 */

import type { Point } from "../types/protocol.js";

export type Grid = Uint8Array;

export interface StepOptions {
  wrap?: boolean;
}

/** Writes the next generation into `out` and returns its live-cell count. */
export type Stepper = (grid: Grid, out: Grid) => number;

export function createGrid(width: number, height: number): Grid {
  return new Uint8Array(width * height);
}

export function pointsToGrid(points: readonly Point[], width: number, height: number): Grid {
  const grid = createGrid(width, height);
  for (const [x, y] of points) {
    if (x >= 0 && x < width && y >= 0 && y < height) grid[y * width + x] = 1;
  }
  return grid;
}

export function gridToPoints(grid: Grid, width: number): Point[] {
  const points: Point[] = [];
  for (let index = 0; index < grid.length; index += 1) {
    if (grid[index]) points.push([index % width, Math.floor(index / width)]);
  }
  return points;
}

export function countLive(grid: Grid): number {
  let live = 0;
  for (let index = 0; index < grid.length; index += 1) live += grid[index];
  return live;
}

/**
 * Build a stepper bound to one board size.
 *
 * Neighbor counts come from column sums over three rows: `sums[x + 1]` holds
 * the live cells of column x in the row above, the row itself and the row
 * below, so a cell's neighbor count is three consecutive sums minus itself.
 * The sums array is padded by one on each side and that padding holds the
 * wrapped column (or zero, on a bounded board), which keeps the inner loop
 * free of edge checks.
 *
 * Returning the live-cell count saves the caller a second pass over the grid.
 */
export function createStepper(
  width: number,
  height: number,
  { wrap = false }: StepOptions = {}
): Stepper {
  const sums = new Uint8Array(width + 2);

  const addRow = (grid: Grid, start: number) => {
    for (let x = 0; x < width; x += 1) sums[x + 1] += grid[start + x];
  };

  return function step(grid, out) {
    let live = 0;
    for (let y = 0; y < height; y += 1) {
      const row = y * width;
      const above = y > 0 ? y - 1 : wrap ? height - 1 : -1;
      const below = y < height - 1 ? y + 1 : wrap ? 0 : -1;

      sums.fill(0);
      addRow(grid, row);
      if (above >= 0) addRow(grid, above * width);
      if (below >= 0) addRow(grid, below * width);
      if (wrap) {
        sums[0] = sums[width];
        sums[width + 1] = sums[1];
      }

      for (let x = 0; x < width; x += 1) {
        const index = row + x;
        const self = grid[index];
        const neighbors = sums[x] + sums[x + 1] + sums[x + 2] - self;
        const alive = neighbors === 3 || (neighbors === 2 && self === 1) ? 1 : 0;
        out[index] = alive;
        live += alive;
      }
    }
    return live;
  };
}

/** Advance a grid once and return the next generation as a new grid. */
export function stepGrid(grid: Grid, width: number, height: number, options?: StepOptions): Grid {
  const out = createGrid(width, height);
  createStepper(width, height, options)(grid, out);
  return out;
}

/** Small deterministic PRNG, so a seed always rebuilds the same board. */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return function random() {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomGrid(width: number, height: number, density: number, seed: number): Grid {
  const grid = createGrid(width, height);
  const random = mulberry32(seed);
  for (let index = 0; index < grid.length; index += 1) {
    grid[index] = random() < density ? 1 : 0;
  }
  return grid;
}
