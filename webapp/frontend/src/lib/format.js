/**
 * Formatting helpers shared by every section.
 *
 * Pure functions only — no React, no IO — so they are unit-testable.
 */

/** Format a 0–1 ratio as a percentage string, "—" when absent. */
export function pct(value, digits = 1) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}%`
    : "—";
}

/** Format a metric in [0, 1] to fixed digits, "—" when absent. */
export function metric(value, digits = 4) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}

/** Format an integer with thousands separators, "—" when absent. */
export function int(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.round(value).toLocaleString("en-US")
    : "—";
}

/** Format an ISO timestamp as "YYYY-MM-DD HH:MM UTC", "—" when absent. */
export function ts(value) {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "—";
  const pad = (n) => String(n).padStart(2, "0");
  return (
    `${parsed.getUTCFullYear()}-${pad(parsed.getUTCMonth() + 1)}-` +
    `${pad(parsed.getUTCDate())} ${pad(parsed.getUTCHours())}:` +
    `${pad(parsed.getUTCMinutes())} UTC`
  );
}

/** Title-case a snake_case identifier ("engine_a" -> "Engine A"). */
export function titleCase(value) {
  return String(value ?? "")
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

/** Sort {name: count} record into [[name, count], …] descending by count. */
export function sortedEntries(record) {
  return Object.entries(record ?? {}).sort((a, b) => b[1] - a[1]);
}
