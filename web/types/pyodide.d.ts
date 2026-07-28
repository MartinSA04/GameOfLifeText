// The slice of the Pyodide runtime the worker actually uses, declared for the
// exact CDN module it imports. Pyodide publishes no types for its ESM build,
// and pulling the npm package in for types alone would add a dependency the
// app never loads.
declare module "https://cdn.jsdelivr.net/pyodide/v314.0.2/full/pyodide.mjs" {
  interface PyProxyCallable {
    (...args: unknown[]): unknown;
    destroy(): void;
  }

  interface PyodideFS {
    mkdirTree(path: string): void;
    writeFile(path: string, contents: string): void;
  }

  interface Pyodide {
    FS: PyodideFS;
    globals: { get(name: string): PyProxyCallable | undefined };
    loadPackage(names: string | string[]): Promise<unknown>;
    runPythonAsync(code: string): Promise<unknown>;
  }

  export function loadPyodide(): Promise<Pyodide>;
}
