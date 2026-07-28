import { describe, expect, it } from "vitest";

import type { Point } from "../../web/types/protocol.js";
import type { StepOptions } from "../../web/src/life.js";
import {
  countLive,
  createGrid,
  createStepper,
  gridToPoints,
  mulberry32,
  pointsToGrid,
  randomGrid,
  stepGrid
} from "../../web/src/life.js";

const WIDTH = 6;
const HEIGHT = 6;

const keys = (points: readonly Point[]) => points.map(([x, y]) => `${x},${y}`).sort();

const step = (points: Point[], options?: StepOptions) =>
  keys(gridToPoints(stepGrid(pointsToGrid(points, WIDTH, HEIGHT), WIDTH, HEIGHT, options), WIDTH));

describe("stepGrid", () => {
  it("keeps a block still", () => {
    const block: Point[] = [
      [1, 1],
      [2, 1],
      [1, 2],
      [2, 2]
    ];
    expect(step(block)).toEqual(keys(block));
  });

  it("flips a blinker between phases", () => {
    const horizontal: Point[] = [
      [1, 2],
      [2, 2],
      [3, 2]
    ];
    const vertical: Point[] = [
      [2, 1],
      [2, 2],
      [2, 3]
    ];
    expect(step(horizontal)).toEqual(keys(vertical));
    expect(step(vertical)).toEqual(keys(horizontal));
  });

  it("moves a glider one cell diagonally every four generations", () => {
    const glider: Point[] = [
      [1, 0],
      [2, 1],
      [0, 2],
      [1, 2],
      [2, 2]
    ];
    let grid = pointsToGrid(glider, 10, 10);
    for (let generation = 0; generation < 4; generation += 1) {
      grid = stepGrid(grid, 10, 10);
    }
    const moved: Point[] = glider.map(([x, y]) => [x + 1, y + 1]);
    expect(keys(gridToPoints(grid, 10))).toEqual(keys(moved));
  });

  it("only sees across an edge when the board wraps", () => {
    // A vertical blinker split across the top and bottom edges: those three
    // cells are only in a row when the board is a torus.
    const split: Point[] = [
      [3, 0],
      [3, 4],
      [3, 5]
    ];
    expect(step(split, { wrap: true })).toContain("3,5");
    expect(step(split, { wrap: false })).toEqual([]);
  });

  it("kills everything on an empty board", () => {
    expect(step([])).toEqual([]);
  });
});

describe("createStepper", () => {
  const blinker: Point[] = [
    [1, 2],
    [2, 2],
    [3, 2]
  ];

  it("returns the live-cell count of the generation it wrote", () => {
    const out = createGrid(WIDTH, HEIGHT);
    const live = createStepper(WIDTH, HEIGHT)(pointsToGrid(blinker, WIDTH, HEIGHT), out);

    expect(live).toBe(3);
    expect(live).toBe(countLive(out));
  });

  it("can be reused across generations without leaking state", () => {
    const stepper = createStepper(WIDTH, HEIGHT);
    const first = pointsToGrid(blinker, WIDTH, HEIGHT);
    const second = createGrid(WIDTH, HEIGHT);

    expect(stepper(first, second)).toBe(3);
    expect(stepper(second, first)).toBe(3);
    expect(keys(gridToPoints(first, WIDTH))).toEqual(keys(blinker));
  });
});

describe("grid conversion", () => {
  it("round-trips points", () => {
    const points: Point[] = [
      [0, 0],
      [5, 4],
      [2, 3]
    ];
    expect(keys(gridToPoints(pointsToGrid(points, WIDTH, HEIGHT), WIDTH))).toEqual(keys(points));
  });

  it("drops points outside the board", () => {
    const outside: Point[] = [
      [-1, 0],
      [0, -1],
      [WIDTH, 0],
      [0, HEIGHT]
    ];
    expect(countLive(pointsToGrid(outside, WIDTH, HEIGHT))).toBe(0);
  });
});

describe("randomGrid", () => {
  it("is reproducible for a seed and differs between seeds", () => {
    const first = Array.from(randomGrid(20, 20, 0.3, 7));
    expect(Array.from(randomGrid(20, 20, 0.3, 7))).toEqual(first);
    expect(Array.from(randomGrid(20, 20, 0.3, 8))).not.toEqual(first);
  });

  it("respects the extremes of density", () => {
    expect(countLive(randomGrid(12, 12, 0, 1))).toBe(0);
    expect(countLive(randomGrid(12, 12, 1, 1))).toBe(144);
  });

  it("lands near the requested density", () => {
    expect(countLive(randomGrid(200, 200, 0.25, 99)) / 40000).toBeCloseTo(0.25, 2);
  });
});

describe("mulberry32", () => {
  it("stays inside the unit interval", () => {
    const random = mulberry32(1234);
    for (let draw = 0; draw < 1000; draw += 1) {
      const value = random();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});
