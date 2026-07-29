/**
 * Board-to-viewport mapping. A camera is cell size in CSS pixels, the board
 * origin's offset inside the viewport, and the scale `fit` picked — which is
 * what the zoom limits are derived from. Everything here is pure, so the
 * interaction math can be tested without a canvas.
 */

export interface Camera {
  scale: number;
  x: number;
  y: number;
  fitted: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface Vector {
  x: number;
  y: number;
}

export interface Cell {
  x: number;
  y: number;
  index: number;
}

/** Half-open cell range: `x0 <= x < x1`. */
export interface CellRange {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/** A rectangle of cells on the board. */
export interface Box extends Size {
  x: number;
  y: number;
}

const MIN_PADDING = 16;
const MAX_PADDING = 42;
const MIN_SCALE = 0.05;
// A handful of cells should not fill the screen, however small the region is.
const MAX_FIT_SCALE = 24;
const ZOOM_OUT_LIMIT = 0.35;
const ZOOM_IN_LIMIT = 18;

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** Center the board in the viewport at the camera's current scale. */
export function center(camera: Camera, board: Size, viewport: Size): Camera {
  return {
    ...camera,
    x: (viewport.width - board.width * camera.scale) / 2,
    y: (viewport.height - board.height * camera.scale) / 2
  };
}

/** Scale a region of the board into the viewport with a margin, and center it. */
export function fitBox(box: Box, viewport: Size): Camera {
  const padding = clamp(viewport.width * 0.04, MIN_PADDING, MAX_PADDING);
  const scale = clamp(
    Math.min(
      (viewport.width - padding * 2) / box.width,
      (viewport.height - padding * 2) / box.height
    ),
    MIN_SCALE,
    MAX_FIT_SCALE
  );
  return {
    scale,
    fitted: scale,
    x: (viewport.width - box.width * scale) / 2 - box.x * scale,
    y: (viewport.height - box.height * scale) / 2 - box.y * scale
  };
}

/** Scale the whole board into the viewport with a margin, and center it. */
export function fit(board: Size, viewport: Size): Camera {
  return fitBox({ x: 0, y: 0, width: board.width, height: board.height }, viewport);
}

/** Zoom about a viewport point, keeping the cell under it in place. */
export function zoomAt(camera: Camera, factor: number, anchor: Vector): Camera {
  const scale = clamp(
    camera.scale * factor,
    Math.max(0.03, camera.fitted * ZOOM_OUT_LIMIT),
    Math.max(24, camera.fitted * ZOOM_IN_LIMIT)
  );
  const boardX = (anchor.x - camera.x) / camera.scale;
  const boardY = (anchor.y - camera.y) / camera.scale;
  return { ...camera, scale, x: anchor.x - boardX * scale, y: anchor.y - boardY * scale };
}

export function pan(camera: Camera, dx: number, dy: number): Camera {
  return { ...camera, x: camera.x + dx, y: camera.y + dy };
}

/** The cell under a viewport point, or null when that is off the board. */
export function cellAt(camera: Camera, point: Vector, board: Size): Cell | null {
  const x = Math.floor((point.x - camera.x) / camera.scale);
  const y = Math.floor((point.y - camera.y) / camera.scale);
  if (x < 0 || x >= board.width || y < 0 || y >= board.height) return null;
  return { x, y, index: y * board.width + x };
}

/** The cells the viewport covers, so drawing can skip the rest of the board. */
export function visibleRange(camera: Camera, board: Size, viewport: Size): CellRange {
  const span = (offset: number, extent: number, limit: number) =>
    [
      clamp(Math.floor(-offset / camera.scale), 0, limit),
      clamp(Math.ceil((extent - offset) / camera.scale), 0, limit)
    ] as const;
  const [x0, x1] = span(camera.x, viewport.width, board.width);
  const [y0, y1] = span(camera.y, viewport.height, board.height);
  return { x0, x1, y0, y1 };
}
