import { describe, expect, it } from "vitest";

import type { Camera, Size } from "../../web/src/camera.js";
import { cellAt, center, fit, pan, visibleRange, zoomAt } from "../../web/src/camera.js";

const BOARD: Size = { width: 100, height: 50 };
const VIEWPORT: Size = { width: 800, height: 600 };

const pixels = (scale: number): Camera => ({ scale, x: 0, y: 0, fitted: scale });

describe("fit", () => {
  it("centers the whole board inside the viewport", () => {
    const camera = fit(BOARD, VIEWPORT);

    expect(camera.scale * BOARD.width).toBeLessThanOrEqual(VIEWPORT.width);
    expect(camera.scale * BOARD.height).toBeLessThanOrEqual(VIEWPORT.height);
    expect(camera.x * 2 + BOARD.width * camera.scale).toBeCloseTo(VIEWPORT.width);
    expect(camera.y * 2 + BOARD.height * camera.scale).toBeCloseTo(VIEWPORT.height);
  });

  it("records the fitted scale, which the zoom limits follow", () => {
    const wide = fit({ width: 4000, height: 20 }, VIEWPORT);
    const small = fit({ width: 10, height: 10 }, VIEWPORT);

    expect(wide.fitted).toBe(wide.scale);
    expect(wide.scale).toBeLessThan(small.scale);
  });

  it("fits the tighter axis", () => {
    const camera = fit({ width: 10, height: 1000 }, VIEWPORT);

    expect(camera.scale * 1000).toBeLessThanOrEqual(VIEWPORT.height);
  });
});

describe("zoomAt", () => {
  it("keeps the board point under the anchor fixed", () => {
    const camera = fit(BOARD, VIEWPORT);
    const anchor = { x: 320, y: 210 };

    expect(cellAt(zoomAt(camera, 2, anchor), anchor, BOARD)).toEqual(cellAt(camera, anchor, BOARD));
  });

  it("clamps zooming out and in", () => {
    const camera = fit(BOARD, VIEWPORT);
    let out = camera;
    let into = camera;
    for (let index = 0; index < 40; index += 1) {
      out = zoomAt(out, 0.5, { x: 0, y: 0 });
      into = zoomAt(into, 2, { x: 0, y: 0 });
    }

    expect(out.scale).toBeCloseTo(camera.fitted * 0.35);
    expect(into.scale).toBeCloseTo(camera.fitted * 18);
  });
});

describe("pan and center", () => {
  it("shifts the origin by the drag delta", () => {
    const camera = fit(BOARD, VIEWPORT);

    expect(pan(camera, 30, -12)).toMatchObject({ x: camera.x + 30, y: camera.y - 12 });
  });

  it("re-centers without changing the scale", () => {
    const dragged = pan(fit(BOARD, VIEWPORT), 400, 400);
    const recentered = center(dragged, BOARD, VIEWPORT);

    expect(recentered.scale).toBe(dragged.scale);
    expect(recentered).toEqual(fit(BOARD, VIEWPORT));
  });
});

describe("cellAt", () => {
  it("maps a viewport point to a cell index", () => {
    expect(cellAt(pixels(10), { x: 25, y: 5 }, BOARD)).toEqual({ x: 2, y: 0, index: 2 });
    expect(cellAt(pixels(10), { x: 5, y: 25 }, BOARD)).toEqual({
      x: 0,
      y: 2,
      index: 2 * BOARD.width
    });
  });

  it("returns null outside the board", () => {
    const camera = pixels(10);

    expect(cellAt(camera, { x: -1, y: 5 }, BOARD)).toBeNull();
    expect(cellAt(camera, { x: 5, y: -1 }, BOARD)).toBeNull();
    expect(cellAt(camera, { x: BOARD.width * 10 + 1, y: 5 }, BOARD)).toBeNull();
    expect(cellAt(camera, { x: 5, y: BOARD.height * 10 + 1 }, BOARD)).toBeNull();
  });
});

describe("visibleRange", () => {
  it("covers the whole board when it fits", () => {
    expect(visibleRange(fit(BOARD, VIEWPORT), BOARD, VIEWPORT)).toEqual({
      x0: 0,
      x1: BOARD.width,
      y0: 0,
      y1: BOARD.height
    });
  });

  it("narrows to what the viewport shows when zoomed in", () => {
    const camera: Camera = { scale: 20, x: -400, y: -200, fitted: 4 };

    expect(visibleRange(camera, BOARD, { width: 200, height: 100 })).toEqual({
      x0: 20,
      x1: 30,
      y0: 10,
      y1: 15
    });
  });

  it("stays inside the board when panned away", () => {
    const camera: Camera = { scale: 20, x: -100000, y: -100000, fitted: 4 };

    expect(visibleRange(camera, BOARD, VIEWPORT)).toEqual({
      x0: BOARD.width,
      x1: BOARD.width,
      y0: BOARD.height,
      y1: BOARD.height
    });
  });
});
