/**
 * Run Length Encoded output — the format Golly, LifeViewer and every other
 * Life tool reads, so an exported pattern can be opened somewhere else.
 */

import type { Point } from "../types/protocol.js";

export interface RleOptions {
  name?: string;
  comment?: string;
  rule?: string;
}

const LINE_LIMIT = 70;
const DEFAULT_RULE = "B3/S23";

const token = (count: number, tag: string) => (count === 1 ? tag : `${count}${tag}`);

function wrap(tokens: readonly string[]): string[] {
  const lines: string[] = [];
  let line = "";
  for (const piece of tokens) {
    if (line.length + piece.length > LINE_LIMIT) {
      lines.push(line);
      line = "";
    }
    line += piece;
  }
  if (line) lines.push(line);
  return lines;
}

function encodeRow(alive: ReadonlySet<number>, width: number): string[] {
  const runs: { tag: string; length: number }[] = [];
  let x = 0;
  while (x < width) {
    const tag = alive.has(x) ? "o" : "b";
    let length = 1;
    while (x + length < width && alive.has(x + length) === (tag === "o")) length += 1;
    runs.push({ tag, length });
    x += length;
  }
  // Dead cells at the end of a row are implied by the row terminator.
  if (runs.length && runs[runs.length - 1].tag === "b") runs.pop();
  return runs.map((run) => token(run.length, run.tag));
}

/** Encode points as an RLE pattern file, normalized to its own origin. */
export function toRle(points: readonly Point[], options: RleOptions = {}): string {
  const { name = "", comment = "", rule = DEFAULT_RULE } = options;
  const header: string[] = [];
  if (name) header.push(`#N ${name}`);
  if (comment) header.push(`#C ${comment}`);

  if (points.length === 0) {
    header.push(`x = 0, y = 0, rule = ${rule}`, "!");
    return `${header.join("\n")}\n`;
  }

  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const width = Math.max(...xs) - minX + 1;
  const height = Math.max(...ys) - minY + 1;

  const rows = Array.from({ length: height }, () => new Set<number>());
  for (const [x, y] of points) rows[y - minY].add(x - minX);

  const tokens: string[] = [];
  let pending = 0;
  for (const alive of rows) {
    if (alive.size === 0) {
      pending += 1;
      continue;
    }
    if (pending) tokens.push(token(pending, "$"));
    tokens.push(...encodeRow(alive, width));
    pending = 1;
  }
  tokens.push("!");

  header.push(`x = ${width}, y = ${height}, rule = ${rule}`);
  return `${[...header, ...wrap(tokens)].join("\n")}\n`;
}
