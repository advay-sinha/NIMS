/**
 * Free-text row filtering for the monitoring sections.
 *
 * Sections search their incident/device/finding tables by incident id or
 * device (plus a few adjacent identifier fields). Matching is a
 * case-insensitive substring test across the named keys; array values (e.g.
 * affected_devices) are flattened before comparison.
 */

/** True when any of `keys` on `row` contains the query substring. */
export function rowMatches(row, query, keys) {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return keys.some((key) => {
    const value = row?.[key];
    if (value == null) return false;
    const text = Array.isArray(value) ? value.join(" ") : String(value);
    return text.toLowerCase().includes(q);
  });
}

/** Filter `rows` to those matching `query` on any of `keys`. */
export function filterByQuery(rows, query, keys) {
  if (!query.trim()) return rows ?? [];
  return (rows ?? []).filter((row) => rowMatches(row, query, keys));
}
