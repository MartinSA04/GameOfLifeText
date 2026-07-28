// The message protocol between the page and the Pyodide worker. This file is a
// declaration file on purpose: both sides of the boundary compile in separate
// tsconfigs (DOM versus WebWorker lib), and a .d.ts can be shared by both
// without either emitting a copy of it.

/** A single live cell, as the planner reports it. */
export type Point = [x: number, y: number];

/** A planned construction: the seed, what it settles into, and how long that takes. */
export interface Plan {
  width: number;
  height: number;
  generations: number;
  initial: Point[];
  target: Point[];
}

/** How far a synchronous Python call has got. */
export type Progress = (done: number, total: number) => void;

/** Page to worker. */
export interface WorkerRequest {
  id: number;
  method: string;
  args: unknown[];
}

/** Worker to page. */
export type WorkerMessage =
  | { type: "status"; label: string }
  | { type: "ready"; supported: string }
  | { type: "fatal"; error: string }
  | { type: "progress"; id: number; done: number; total: number }
  | { type: "result"; id: number; result: unknown }
  | { type: "error"; id: number; error: string };
