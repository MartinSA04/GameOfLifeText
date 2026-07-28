import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

// main.ts throws on a missing element rather than limping along with nulls, so
// a renamed id in the markup takes the whole page down on load. Nothing else in
// the suite touches the DOM, which makes this the check that catches it.

const read = (path: string) => readFileSync(new URL(`../../${path}`, import.meta.url), "utf8");

const html = read("web/index.html");
const main = read("web/src/main.ts");
const styles = read("web/styles.css");

const matches = (source: string, pattern: RegExp) => [...source.matchAll(pattern)].map((m) => m[1]);

const declaredIds = new Set(matches(html, /\sid="([^"]+)"/g));
const lookedUpIds = matches(main, /element(?:<[^>]+>)?\("([^"]+)"\)/g);

describe("index.html", () => {
  it("declares every id the app looks up", () => {
    expect(lookedUpIds.length).toBeGreaterThan(20);
    for (const id of lookedUpIds) {
      expect(declaredIds, `#${id} is looked up by main.ts`).toContain(id);
    }
  });

  it("has no duplicate ids", () => {
    const all = matches(html, /\sid="([^"]+)"/g);
    expect(all.length).toBe(declaredIds.size);
  });

  it("points every label and output at an element that exists", () => {
    for (const target of matches(html, /\sfor="([^"]+)"/g)) {
      expect(declaredIds, `for="${target}"`).toContain(target);
    }
  });

  it("offers exactly the modes the app handles", () => {
    const values = matches(html, /name="mode"[^>]*value="([^"]+)"/g);
    expect(values).toEqual(["text", "random", "blank"]);
  });

  it("loads the compiled entry point and the local assets", () => {
    expect(html).toContain('src="./src/main.js"');
    expect(html).toContain('href="./styles.css"');
    expect(html).toContain('href="./favicon.svg"');
  });
});

describe("styles.css", () => {
  it("defines every custom property the renderer reads", () => {
    const render = read("web/src/render.ts");
    for (const property of matches(render, /read\("(--[\w-]+)"\)/g)) {
      expect(styles, property).toContain(`${property}:`);
    }
  });
});
