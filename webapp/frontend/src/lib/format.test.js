import { describe, expect, it } from "vitest";
import { pct, metric, int, ts, titleCase, sortedEntries } from "./format.js";

describe("pct", () => {
  it("formats ratios as percentages", () => {
    expect(pct(0.1234)).toBe("12.3%");
    expect(pct(1, 0)).toBe("100%");
  });
  it("returns a dash for absent values", () => {
    expect(pct(null)).toBe("—");
    expect(pct(undefined)).toBe("—");
    expect(pct(NaN)).toBe("—");
  });
});

describe("metric", () => {
  it("formats to fixed digits", () => {
    expect(metric(0.98765)).toBe("0.9877");
    expect(metric(0.5, 2)).toBe("0.50");
  });
  it("returns a dash for non-numbers", () => {
    expect(metric("x")).toBe("—");
  });
});

describe("int", () => {
  it("adds thousands separators", () => {
    expect(int(1234567)).toBe("1,234,567");
    expect(int(0)).toBe("0");
  });
  it("returns a dash for absent values", () => {
    expect(int(null)).toBe("—");
  });
});

describe("ts", () => {
  it("formats ISO timestamps in UTC", () => {
    expect(ts("2026-07-07T20:16:45.738765+00:00")).toBe("2026-07-07 20:16 UTC");
  });
  it("returns a dash for junk", () => {
    expect(ts("nope")).toBe("—");
    expect(ts(null)).toBe("—");
  });
});

describe("titleCase", () => {
  it("converts snake_case identifiers", () => {
    expect(titleCase("engine_a")).toBe("Engine A");
    expect(titleCase("config_finding")).toBe("Config Finding");
  });
});

describe("sortedEntries", () => {
  it("sorts descending by count", () => {
    expect(sortedEntries({ a: 1, b: 3, c: 2 })).toEqual([
      ["b", 3],
      ["c", 2],
      ["a", 1],
    ]);
  });
  it("tolerates null", () => {
    expect(sortedEntries(null)).toEqual([]);
  });
});
