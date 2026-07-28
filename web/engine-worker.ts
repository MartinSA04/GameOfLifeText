import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.2/full/pyodide.mjs";

import type { Progress, WorkerMessage, WorkerRequest } from "./types/protocol.js";

/** Where the planner package is unpacked inside the Pyodide filesystem. */
const MOUNT = "/app";

/**
 * Everything the page is allowed to call, so a stray message cannot reach into
 * arbitrary Python globals. Each takes a progress reporter as its last
 * argument.
 */
const METHODS = new Set(["generate_text"]);

const PROGRESS_INTERVAL_MS = 40;

const scope = self as unknown as DedicatedWorkerGlobalScope;
const send = (message: WorkerMessage) => scope.postMessage(message);
const status = (label: string) => send({ type: "status", label });

interface Manifest {
  package: string;
  modules: string[];
}

async function fetchFile(url: URL): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`could not load ${url.pathname}: ${response.status}`);
  }
  return response.text();
}

async function boot() {
  status("loading pyodide");
  const pyodide = await loadPyodide();

  status("loading numpy");
  await pyodide.loadPackage("numpy");

  status("loading planner");
  // build_web.py writes this manifest, so the list of modules that ship to the
  // browser lives in one place instead of being repeated here.
  const manifest = JSON.parse(
    await fetchFile(new URL("./pysrc/manifest.json", import.meta.url))
  ) as Manifest;

  const packageDir = `${MOUNT}/${manifest.package}`;
  pyodide.FS.mkdirTree(packageDir);
  const sources = await Promise.all(
    manifest.modules.map((name) =>
      fetchFile(new URL(`./pysrc/${manifest.package}/${name}`, import.meta.url))
    )
  );
  manifest.modules.forEach((name, index) => {
    pyodide.FS.writeFile(`${packageDir}/${name}`, sources[index]);
  });

  await pyodide.runPythonAsync(`
import sys
sys.path.insert(0, ${JSON.stringify(MOUNT)})
from ${manifest.package}.browser import generate_text, supported_characters
`);

  return pyodide;
}

const ready = boot();

ready
  .then((pyodide) => {
    const characters = pyodide.globals.get("supported_characters");
    if (!characters) throw new Error("the planner did not expose its font");
    try {
      send({ type: "ready", supported: String(characters()) });
    } finally {
      characters.destroy();
    }
  })
  .catch((error: unknown) => {
    send({ type: "fatal", error: error instanceof Error ? error.message : String(error) });
  });

/**
 * A reporter to hand to Python, which calls it once per placed block. Those
 * land in the hundreds, so everything but the last one is rate limited: the
 * page cannot paint faster than this anyway.
 */
function reporter(id: number): Progress {
  let last = 0;
  return (done, total) => {
    const now = performance.now();
    if (done < total && now - last < PROGRESS_INTERVAL_MS) return;
    last = now;
    send({ type: "progress", id, done, total });
  };
}

async function handle({ id, method, args }: WorkerRequest) {
  try {
    if (!METHODS.has(method)) {
      throw new Error(`unknown python method: ${method}`);
    }
    const pyodide = await ready;
    const call = pyodide.globals.get(method);
    if (!call) throw new Error(`the planner does not define ${method}`);
    try {
      // Pyodide wraps the reporter so Python can call it straight through.
      send({ type: "result", id, result: await call(...args, reporter(id)) });
    } finally {
      call.destroy();
    }
  } catch (error: unknown) {
    send({ type: "error", id, error: error instanceof Error ? error.message : String(error) });
  }
}

scope.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  void handle(event.data);
});
