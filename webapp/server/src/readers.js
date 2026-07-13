/**
 * Tolerant filesystem readers for persisted NetSentinel artefacts.
 *
 * Mirrors src/dashboard/loader.py semantics: every reader tolerates absent
 * files/folders and returns null / [] instead of throwing, so a missing
 * artefact set degrades to an "unavailable" panel rather than a 500.
 */

import fs from "node:fs";
import path from "node:path";

/** Read and parse a JSON file; null on any problem (never throws). */
export function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch {
    return null;
  }
}

/** Read a JSONL file into an array of parsed lines (bad lines skipped). */
export function readJsonl(filePath) {
  let text;
  try {
    text = fs.readFileSync(filePath, "utf-8");
  } catch {
    return [];
  }
  const rows = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      rows.push(JSON.parse(trimmed));
    } catch {
      /* skip malformed line */
    }
  }
  return rows;
}

/** Read a UTF-8 text file; null when absent/unreadable. */
export function readText(filePath) {
  try {
    return fs.readFileSync(filePath, "utf-8");
  } catch {
    return null;
  }
}

/** List immediate subdirectory names of a directory, sorted (empty when absent). */
export function listDirs(dirPath) {
  try {
    return fs
      .readdirSync(dirPath, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
      .sort();
  } catch {
    return [];
  }
}

/** True when the path exists and is a regular file. */
export function isFile(filePath) {
  try {
    return fs.statSync(filePath).isFile();
  } catch {
    return false;
  }
}

/** True when the path exists and is a directory. */
export function isDir(dirPath) {
  try {
    return fs.statSync(dirPath).isDirectory();
  } catch {
    return false;
  }
}

/** Join path segments (re-export so loaders avoid importing node:path). */
export function join(...segments) {
  return path.join(...segments);
}
