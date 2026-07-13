/**
 * API access — a single fetch hook with loading / error / data states.
 */

import { useEffect, useState } from "react";

/** Fetch a /api path; refetches when the path changes. */
export function useApi(path) {
  const [state, setState] = useState({ loading: true, error: null, data: null });

  useEffect(() => {
    let cancelled = false;
    setState({ loading: true, error: null, data: null });
    fetch(path)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setState({ loading: false, error: null, data });
      })
      .catch((err) => {
        if (!cancelled)
          setState({ loading: false, error: String(err.message ?? err), data: null });
      });
    return () => {
      cancelled = true;
    };
  }, [path]);

  return state;
}
