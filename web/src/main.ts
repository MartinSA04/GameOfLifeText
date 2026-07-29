import type { Box, Camera, Size, Vector } from "./camera.js";
import { cellAt, center, fitBox, pan, zoomAt } from "./camera.js";
import type { Engine } from "./engine.js";
import { createEngine } from "./engine.js";
import type { Grid } from "./life.js";
import {
  countLive,
  createGrid,
  createStepper,
  gridToPoints,
  liveBounds,
  pointsToGrid,
  randomGrid,
  type Stepper
} from "./life.js";
import { createRenderer } from "./render.js";
import { toRle } from "./rle.js";

type Mode = "text" | "random" | "blank";
type View = "board" | "target";
type Theme = "dark" | "light";

const MIN_DIMENSION = 8;
const MAX_DIMENSION = 500;
const TEXT_LIMIT = 80;
const ZOOM_STEP = 1.25;
const WHEEL_STEP = 1.12;

const element = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element: #${id}`);
  return node as T;
};

const ui = {
  theme: element<HTMLButtonElement>("theme"),
  boot: element("boot"),
  bootLabel: element("boot-label"),
  canvas: element<HTMLCanvasElement>("canvas"),
  canvasWrap: element("canvas-wrap"),
  textPanel: element("text-panel"),
  textInput: element<HTMLTextAreaElement>("text-input"),
  textCount: element("text-count"),
  generate: element<HTMLButtonElement>("generate"),
  boardPanel: element("board-panel"),
  width: element<HTMLInputElement>("width"),
  height: element<HTMLInputElement>("height"),
  density: element<HTMLInputElement>("density"),
  densityOut: element("density-out"),
  seed: element<HTMLInputElement>("seed"),
  wrap: element<HTMLInputElement>("wrap"),
  build: element<HTMLButtonElement>("build"),
  run: element<HTMLButtonElement>("run"),
  step: element<HTMLButtonElement>("step"),
  reset: element<HTMLButtonElement>("reset"),
  speed: element<HTMLInputElement>("speed"),
  speedOut: element("speed-out"),
  progress: element("progress"),
  progressName: element("progress-name"),
  progressBar: element<HTMLProgressElement>("progress-bar"),
  progressOut: element("progress-out"),
  error: element("error"),
  views: element("views"),
  viewBoard: element<HTMLButtonElement>("view-board"),
  viewTarget: element<HTMLButtonElement>("view-target"),
  ghost: element<HTMLInputElement>("ghost"),
  ghostToggle: element("ghost-toggle"),
  draw: element<HTMLButtonElement>("draw"),
  zoomIn: element<HTMLButtonElement>("zoom-in"),
  zoomOut: element<HTMLButtonElement>("zoom-out"),
  fit: element<HTMLButtonElement>("fit"),
  export: element<HTMLButtonElement>("export"),
  statGen: element("stat-gen"),
  statLive: element("stat-live"),
  statSize: element("stat-size"),
  status: element("status")
};

const renderer = createRenderer(ui.canvas);

interface Pointer extends Vector {
  camera: Camera;
  lastCell: number | null;
}

interface State {
  mode: Mode;
  view: View;
  text: string;
  board: Size;
  cells: Grid | null;
  seed: Grid | null;
  target: Grid | null;
  scratch: Grid | null;
  stepper: Stepper | null;
  generation: number;
  settle: number;
  live: number;
  planning: boolean;
  running: boolean;
  wrap: boolean;
  ghost: boolean;
  drawMode: boolean;
  paintAlive: boolean;
  pointer: Pointer | null;
  camera: Camera;
  engine: Engine | null;
  lastFrame: number;
}

const state: State = {
  mode: "text",
  view: "board",
  text: "",
  board: { width: 0, height: 0 },
  cells: null,
  seed: null,
  target: null,
  scratch: null,
  stepper: null,
  generation: 0,
  settle: 0,
  live: 0,
  planning: false,
  running: false,
  wrap: false,
  ghost: true,
  drawMode: false,
  paintAlive: true,
  pointer: null,
  camera: { scale: 1, x: 0, y: 0, fitted: 1 },
  engine: null,
  lastFrame: 0
};

const viewport = (): Size => {
  const rect = ui.canvasWrap.getBoundingClientRect();
  return { width: rect.width, height: rect.height };
};

const visibleGrid = () => (state.view === "target" ? state.target : state.cells);

const clampInt = (value: string, low: number, high: number) => {
  const parsed = Math.round(Number(value));
  if (!Number.isFinite(parsed)) return low;
  return Math.min(high, Math.max(low, parsed));
};

const slug = (value: string) =>
  value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^\w-]/g, "");

const setStatus = (text: string) => {
  ui.status.textContent = text;
};

function setError(text = "") {
  ui.error.textContent = text;
  ui.error.hidden = !text;
}

function draw() {
  renderer.draw({
    board: state.board,
    camera: state.camera,
    cells: visibleGrid(),
    ghost: state.ghost && state.view === "board" ? state.target : null
  });
}

/**
 * Switch theme. The board's colors live in the stylesheet, so the renderer has
 * to be told to read them again; `remember` is false for the system following
 * the OS, which must not look like a choice the user made.
 */
function setTheme(theme: Theme, remember = true) {
  document.documentElement.dataset.theme = theme;
  const other: Theme = theme === "dark" ? "light" : "dark";
  ui.theme.textContent = other;
  ui.theme.setAttribute("aria-label", `switch to the ${other} theme`);
  if (remember) {
    try {
      localStorage.setItem("theme", theme);
    } catch {
      // Private browsing can refuse storage; the switch still works this visit.
    }
  }
  renderer.reread();
  draw();
}

const currentTheme = (): Theme =>
  document.documentElement.dataset.theme === "light" ? "light" : "dark";

/** Everything worth looking at: what is on screen, plus the ghost under it. */
function contentBox(): Box | null {
  const { width, height } = state.board;
  const shown = visibleGrid();
  const ghost = state.ghost && state.view === "board" ? state.target : null;
  const boxes = [shown, ghost]
    .filter((grid): grid is Grid => grid !== null)
    .map((grid) => liveBounds(grid, width, height))
    .filter((box): box is Box => box !== null);
  if (!boxes.length) return null;

  const left = Math.min(...boxes.map((box) => box.x));
  const top = Math.min(...boxes.map((box) => box.y));
  const right = Math.max(...boxes.map((box) => box.x + box.width));
  const bottom = Math.max(...boxes.map((box) => box.y + box.height));
  // A cell of air on each side, so nothing sits against the edge.
  return { x: left - 1, y: top - 1, width: right - left + 2, height: bottom - top + 2 };
}

function fitBoard() {
  const { width, height } = state.board;
  if (!width) return;
  state.camera = fitBox(contentBox() ?? { x: 0, y: 0, width, height }, viewport());
}

/** Drive the one progress bar, which planning and settling take turns owning. */
function setProgress(name: string, done: number, total: number) {
  ui.progressName.textContent = name;
  ui.progressBar.value = total ? (done / total) * 100 : 0;
  ui.progressOut.textContent = `${done}/${total}`;
}

function syncStats() {
  const onTarget = state.view === "target";
  const live = onTarget && state.target ? countLive(state.target) : state.live;
  ui.statGen.textContent = String(onTarget ? state.settle : state.generation);
  ui.statLive.textContent = live.toLocaleString();
  ui.statSize.textContent = `${state.board.width}×${state.board.height}`;

  if (state.settle && !state.planning) {
    setProgress("settle", Math.min(state.generation, state.settle), state.settle);
  }
}

function syncControls() {
  const isText = state.mode === "text";
  const hasBoard = state.board.width > 0;
  const hasTarget = state.target !== null;
  const onTarget = state.view === "target";

  ui.textPanel.hidden = !isText;
  ui.boardPanel.hidden = isText;
  ui.build.textContent = state.mode === "random" ? "new random board" : "new blank board";
  ui.views.hidden = !hasTarget;
  ui.progress.hidden = !state.planning && state.settle === 0;
  ui.ghostToggle.hidden = !hasTarget;
  ui.export.disabled = !hasBoard;
  ui.draw.disabled = onTarget;
  ui.run.disabled = !hasBoard || onTarget;
  ui.step.disabled = ui.run.disabled;
  ui.reset.disabled = !hasBoard;
  ui.run.textContent = state.running ? "pause" : "run";
  ui.viewBoard.setAttribute("aria-pressed", String(!onTarget));
  ui.viewTarget.setAttribute("aria-pressed", String(onTarget));
  syncStats();
}

interface BoardInit {
  width: number;
  height: number;
  cells: Grid;
  target?: Grid | null;
  settle?: number;
  wrap?: boolean;
}

function setBoard({ width, height, cells, target = null, settle = 0, wrap = false }: BoardInit) {
  state.running = false;
  state.board = { width, height };
  state.cells = cells;
  state.seed = cells.slice();
  state.target = target;
  state.scratch = createGrid(width, height);
  state.wrap = wrap;
  state.stepper = createStepper(width, height, { wrap });
  state.generation = 0;
  state.settle = settle;
  state.live = countLive(cells);
  state.view = "board";
  fitBoard();
  syncControls();
  draw();
}

function advance() {
  if (!state.stepper || !state.cells || !state.scratch || state.view === "target") return;
  state.live = state.stepper(state.cells, state.scratch);
  [state.cells, state.scratch] = [state.scratch, state.cells];
  state.generation += 1;
  // The construction is finished and every cell of it is a still life, so
  // there is nothing left to watch: stop on the generation it settles. The
  // camera stays where it was, so the text is seen at the size it was built.
  if (state.settle && state.generation === state.settle) {
    setStatus("settled");
    setRunning(false);
  }
  syncStats();
  draw();
}

function setRunning(running: boolean) {
  // Stopping always works; there is nothing to start on an empty board, or
  // while the finished text is on screen instead of the live one.
  if (running && (!state.board.width || state.view === "target")) return;
  state.running = running;
  state.lastFrame = performance.now();
  ui.run.textContent = running ? "pause" : "run";
}

function reset() {
  if (!state.seed) return;
  state.cells = state.seed.slice();
  state.generation = 0;
  state.live = countLive(state.cells);
  state.view = "board";
  setRunning(false);
  syncControls();
  draw();
  setStatus("back to the seed");
}

function setView(next: View) {
  if (next === "target" && !state.target) return;
  setRunning(false);
  state.view = next;
  syncControls();
  draw();
}

function frame(timestamp: number) {
  requestAnimationFrame(frame);
  if (!state.running) return;
  if (timestamp - state.lastFrame < 1000 / Number(ui.speed.value)) return;
  state.lastFrame = timestamp;
  advance();
}

async function generate() {
  const engine = state.engine;
  if (!engine) return;

  const text = ui.textInput.value;
  if (!text.trim()) {
    setError("type at least one visible character");
    return;
  }
  const unsupported = [...text].find(
    (character) => character !== "\n" && !engine.supported.includes(character)
  );
  if (unsupported) {
    setError(`the 5x7 font has no glyph for ${JSON.stringify(unsupported)}. remove it to build.`);
    return;
  }

  setError();
  setRunning(false);
  state.planning = true;
  ui.generate.disabled = true;
  ui.generate.dataset.busy = "";
  ui.generate.textContent = "planning";
  setStatus("planning glider collisions");
  // The planner reports one step per block, and the first report only arrives
  // once it knows how many blocks there are.
  setProgress("plan", 0, 0);
  syncControls();

  try {
    const plan = await engine.generateText(text, (done, total) => {
      setProgress("plan", done, total);
    });
    // A plan can take seconds; the user may have moved on to another mode.
    if (state.mode !== "text") return;
    state.planning = false;
    state.text = text;
    setBoard({
      width: plan.width,
      height: plan.height,
      cells: pointsToGrid(plan.initial, plan.width, plan.height),
      target: pointsToGrid(plan.target, plan.width, plan.height),
      settle: plan.generations
    });
    setStatus(`${plan.generations} generations to settle`);
  } catch (error: unknown) {
    setError(String(error instanceof Error ? error.message : error).toLowerCase());
    setStatus("board unchanged");
  } finally {
    state.planning = false;
    ui.generate.disabled = false;
    delete ui.generate.dataset.busy;
    ui.generate.textContent = "build";
    syncControls();
  }
}

function buildBoard() {
  const width = clampInt(ui.width.value, MIN_DIMENSION, MAX_DIMENSION);
  const height = clampInt(ui.height.value, MIN_DIMENSION, MAX_DIMENSION);
  ui.width.value = String(width);
  ui.height.value = String(height);

  const random = state.mode === "random";
  const cells = random
    ? randomGrid(width, height, Number(ui.density.value) / 100, Number(ui.seed.value) || 0)
    : createGrid(width, height);

  setError();
  setBoard({ width, height, cells, wrap: ui.wrap.checked });
  setStatus(
    random ? `random fill, seed ${ui.seed.value}` : "empty board, switch on draw to fill it"
  );
}

function setMode(mode: Mode) {
  state.mode = mode;
  state.target = null;
  state.settle = 0;
  setError();
  syncControls();
  if (mode === "text") void generate();
  else buildBoard();
}

function setDrawMode(drawMode: boolean) {
  state.drawMode = drawMode;
  ui.draw.setAttribute("aria-pressed", String(drawMode));
  ui.canvasWrap.classList.toggle("drawing", drawMode);
}

function pointerPoint(event: PointerEvent | WheelEvent): Vector {
  const rect = ui.canvasWrap.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function paintAt(event: PointerEvent) {
  const pointer = state.pointer;
  if (!pointer || !state.cells || !state.seed) return;
  const cell = cellAt(state.camera, pointerPoint(event), state.board);
  if (!cell || cell.index === pointer.lastCell) return;

  pointer.lastCell = cell.index;
  const before = state.cells[cell.index];
  const after = state.paintAlive ? 1 : 0;
  if (before === after) return;

  state.cells[cell.index] = after;
  // Drawn cells belong to the seed too, so reset keeps them.
  state.seed[cell.index] = after;
  state.live += after - before;
  syncStats();
  draw();
}

function exportPattern() {
  const grid = visibleGrid();
  if (!grid) return;
  // Name what is actually on screen: the planned text, the untouched seed, or
  // the generation the board has reached.
  const label = state.mode === "text" && state.text ? slug(state.text) : state.mode;
  const stage =
    state.view === "target" ? "target" : state.generation === 0 ? "seed" : `gen${state.generation}`;
  const name = `${label || "board"}-${stage}`;
  const rle = toRle(gridToPoints(grid, state.board.width), {
    name,
    comment: state.text
      ? `game-of-life-text ${stage} for "${state.text}"`
      : "game-of-life-text board"
  });

  const url = URL.createObjectURL(new Blob([rle], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${name}.rle`;
  link.click();
  URL.revokeObjectURL(url);
  setStatus(`exported ${name}.rle`);
}

function zoom(factor: number, anchor?: Vector) {
  if (!state.board.width) return;
  const size = viewport();
  state.camera = zoomAt(
    state.camera,
    factor,
    anchor ?? {
      x: size.width / 2,
      y: size.height / 2
    }
  );
  draw();
}

/** Grow the field with the text, so a multi-line word is visible as typed. */
function fitTextInput() {
  ui.textInput.style.height = "auto";
  ui.textInput.style.height = `${ui.textInput.scrollHeight + 2}px`;
}

const TYPING = new Set(["INPUT", "TEXTAREA"]);

function onKeyDown(event: KeyboardEvent) {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (TYPING.has((event.target as HTMLElement | null)?.tagName ?? "")) return;

  const actions: Record<string, () => void> = {
    " ": () => setRunning(!state.running),
    s: () => {
      setRunning(false);
      advance();
    },
    r: reset,
    f: () => {
      fitBoard();
      draw();
    },
    d: () => {
      if (!ui.draw.disabled) setDrawMode(!state.drawMode);
    },
    g: () => {
      if (!ui.ghostToggle.hidden) ui.ghost.click();
    },
    t: () => setView(state.view === "target" ? "board" : "target"),
    "+": () => zoom(ZOOM_STEP),
    "=": () => zoom(ZOOM_STEP),
    "-": () => zoom(1 / ZOOM_STEP)
  };

  const action = actions[event.key.length === 1 ? event.key.toLowerCase() : event.key];
  if (!action) return;
  event.preventDefault();
  action();
}

function bind() {
  for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="mode"]')) {
    radio.addEventListener("change", () => setMode(radio.value as Mode));
  }

  ui.theme.addEventListener("click", () => {
    setTheme(currentTheme() === "dark" ? "light" : "dark");
  });
  // Follow the system until the user picks a side, then stop.
  matchMedia("(prefers-color-scheme: light)").addEventListener("change", (event) => {
    let chosen: string | null = null;
    try {
      chosen = localStorage.getItem("theme");
    } catch {
      // No storage means no stored choice to respect.
    }
    if (!chosen) setTheme(event.matches ? "light" : "dark", false);
  });

  ui.textInput.addEventListener("input", () => {
    ui.textCount.textContent = `${ui.textInput.value.length}/${TEXT_LIMIT}`;
    fitTextInput();
  });
  ui.textInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) void generate();
  });
  ui.generate.addEventListener("click", () => void generate());
  ui.build.addEventListener("click", buildBoard);
  ui.density.addEventListener("input", () => {
    ui.densityOut.textContent = `${ui.density.value}%`;
  });
  ui.wrap.addEventListener("change", () => {
    state.wrap = ui.wrap.checked;
    state.stepper = createStepper(state.board.width, state.board.height, { wrap: state.wrap });
  });

  ui.run.addEventListener("click", () => setRunning(!state.running));
  ui.step.addEventListener("click", () => {
    setRunning(false);
    advance();
  });
  ui.reset.addEventListener("click", reset);
  ui.speed.addEventListener("input", () => {
    ui.speedOut.textContent = `${ui.speed.value}/s`;
  });

  ui.viewBoard.addEventListener("click", () => setView("board"));
  ui.viewTarget.addEventListener("click", () => setView("target"));
  ui.ghost.addEventListener("change", () => {
    state.ghost = ui.ghost.checked;
    draw();
  });
  ui.draw.addEventListener("click", () => setDrawMode(!state.drawMode));
  ui.zoomIn.addEventListener("click", () => zoom(ZOOM_STEP));
  ui.zoomOut.addEventListener("click", () => zoom(1 / ZOOM_STEP));
  ui.fit.addEventListener("click", () => {
    fitBoard();
    draw();
  });
  ui.export.addEventListener("click", exportPattern);

  const wrapper = ui.canvasWrap;
  wrapper.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoom(event.deltaY < 0 ? WHEEL_STEP : 1 / WHEEL_STEP, pointerPoint(event));
  });
  wrapper.addEventListener("pointerdown", (event) => {
    if (!state.cells || state.view === "target") return;
    wrapper.setPointerCapture(event.pointerId);
    state.pointer = { ...pointerPoint(event), camera: state.camera, lastCell: null };
    if (state.drawMode) {
      const cell = cellAt(state.camera, pointerPoint(event), state.board);
      state.paintAlive = cell ? state.cells[cell.index] === 0 : true;
      paintAt(event);
    } else {
      wrapper.classList.add("panning");
    }
  });
  wrapper.addEventListener("pointermove", (event) => {
    const pointer = state.pointer;
    if (!pointer) return;
    if (state.drawMode) {
      paintAt(event);
      return;
    }
    const point = pointerPoint(event);
    state.camera = pan(pointer.camera, point.x - pointer.x, point.y - pointer.y);
    draw();
  });
  for (const type of ["pointerup", "pointercancel"] as const) {
    wrapper.addEventListener(type, (event) => {
      state.pointer = null;
      wrapper.classList.remove("panning");
      if (wrapper.hasPointerCapture(event.pointerId)) {
        wrapper.releasePointerCapture(event.pointerId);
      }
    });
  }

  document.addEventListener("keydown", onKeyDown);

  new ResizeObserver(() => {
    const size = renderer.resize(viewport());
    if (state.board.width) state.camera = center(state.camera, state.board, size);
    draw();
  }).observe(wrapper);
}

async function start() {
  bind();
  // The head script picked the theme; this labels the switch to match it.
  setTheme(currentTheme(), false);
  renderer.resize(viewport());
  requestAnimationFrame(frame);
  fitTextInput();
  syncControls();

  try {
    state.engine = await createEngine({
      onStatus: (label) => {
        ui.bootLabel.textContent = label;
      }
    });
    ui.boot.hidden = true;
    // Land on something to look at, unless the wait was spent elsewhere. The
    // first construction plays itself: the point of the page is the assembly.
    if (state.mode === "text") {
      await generate();
      if (state.target && !matchMedia("(prefers-reduced-motion: reduce)").matches) setRunning(true);
    }
  } catch (error: unknown) {
    console.error(error);
    ui.bootLabel.textContent = "python failed to load";
    setError("the python planner could not load. check your connection and reload the page.");
    setStatus("no engine");
  }
}

void start();
