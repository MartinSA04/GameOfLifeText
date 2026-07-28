import { describe, expect, it } from "vitest";

import type { Point } from "../../web/types/protocol.js";
import { toRle } from "../../web/src/rle.js";

const header = (rle: string) => rle.split("\n")[0];
const body = (rle: string) => rle.trim().split("\n").slice(1).join("\n");

describe("toRle", () => {
  it("writes a header with the bounding box and the Life rule", () => {
    const rle = toRle([
      [0, 0],
      [1, 0],
      [2, 0]
    ]);

    expect(header(rle)).toBe("x = 3, y = 1, rule = B3/S23");
    expect(body(rle)).toBe("3o!");
  });

  it("encodes a glider", () => {
    const glider: Point[] = [
      [1, 0],
      [2, 1],
      [0, 2],
      [1, 2],
      [2, 2]
    ];

    expect(body(toRle(glider))).toBe("bo$2bo$3o!");
  });

  it("normalizes the pattern to its own origin", () => {
    const shifted = toRle([
      [41, 12],
      [42, 12],
      [43, 12]
    ]);

    expect(header(shifted)).toBe("x = 3, y = 1, rule = B3/S23");
    expect(body(shifted)).toBe("3o!");
  });

  it("collapses empty rows into one row token", () => {
    const gap = toRle([
      [0, 0],
      [0, 3]
    ]);

    expect(body(gap)).toBe("o3$o!");
  });

  it("drops dead cells at the end of a row", () => {
    const ragged = toRle([
      [0, 0],
      [3, 0],
      [0, 1]
    ]);

    expect(body(ragged)).toBe("o2bo$o!");
  });

  it("keeps lines within the 70 character limit", () => {
    const comb: Point[] = Array.from({ length: 300 }, (_, index) => [index * 2, 0]);
    const lines = toRle(comb).trim().split("\n").slice(1);

    expect(lines.length).toBeGreaterThan(1);
    for (const line of lines) expect(line.length).toBeLessThanOrEqual(70);
  });

  it("emits optional name and comment lines", () => {
    const rle = toRle([[0, 0]], { name: "life-seed", comment: "hello" });

    expect(rle.split("\n").slice(0, 3)).toEqual([
      "#N life-seed",
      "#C hello",
      "x = 1, y = 1, rule = B3/S23"
    ]);
  });

  it("handles an empty pattern", () => {
    expect(toRle([])).toBe("x = 0, y = 0, rule = B3/S23\n!\n");
  });
});
