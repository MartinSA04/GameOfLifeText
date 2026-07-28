/**
 * Main-thread client for the Pyodide worker. The planner is Python, so it runs
 * off the main thread and is reached through ordinary async calls.
 */

import type { Plan, Progress, WorkerMessage } from "../types/protocol.js";

export interface Engine {
  /** Every character the bundled bitmap font can render. */
  supported: string;
  /** Plan a construction, reporting how many blocks are placed as it goes. */
  generateText(text: string, onProgress?: Progress): Promise<Plan>;
}

export interface EngineOptions {
  onStatus?: (label: string) => void;
}

interface Pending {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  onProgress?: Progress;
}

const WORKER_URL = new URL("../engine-worker.js", import.meta.url);

export function createEngine({ onStatus }: EngineOptions = {}): Promise<Engine> {
  const worker = new Worker(WORKER_URL, { type: "module" });
  const pending = new Map<number, Pending>();
  let nextId = 1;

  // Assigned by the promise executor below, which runs synchronously.
  let resolveReady!: (engine: Engine) => void;
  let rejectReady!: (error: Error) => void;
  const ready = new Promise<Engine>((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });

  const call = (method: string, args: unknown[], onProgress?: Progress) =>
    new Promise<unknown>((resolve, reject) => {
      const id = nextId;
      nextId += 1;
      pending.set(id, onProgress ? { resolve, reject, onProgress } : { resolve, reject });
      worker.postMessage({ id, method, args });
    });

  worker.addEventListener("message", ({ data }: MessageEvent<WorkerMessage>) => {
    if (data.type === "status") {
      onStatus?.(data.label);
      return;
    }
    if (data.type === "ready") {
      resolveReady({
        supported: data.supported,
        generateText: async (text, onProgress) =>
          JSON.parse(String(await call("generate_text", [text], onProgress))) as Plan
      });
      return;
    }
    if (data.type === "fatal") {
      rejectReady(new Error(data.error));
      return;
    }
    const request = pending.get(data.id);
    if (!request) return;
    if (data.type === "progress") {
      request.onProgress?.(data.done, data.total);
      return;
    }
    pending.delete(data.id);
    if (data.type === "error") request.reject(new Error(data.error));
    else request.resolve(data.result);
  });

  worker.addEventListener("error", (event) => {
    rejectReady(new Error(event.message || "the python worker could not start"));
  });

  return ready;
}
