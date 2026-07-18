// Client for the Trends Arc backend (see `backend/`).
//
// `PUBLIC_*` vars are statically inlined into the bundle at build time, not read
// at runtime — the deployed site carries whatever value was set when it was
// built, so changing the API URL means rebuilding, not just restarting.
const API_URL = import.meta.env.PUBLIC_API_URL ?? 'http://localhost:8000';

export interface HealthResult {
  ok: boolean;
  /** Present only when the request failed — surfaced to the user verbatim. */
  error?: string;
}

/**
 * Pings `GET /health`. Resolves either way; a rejected fetch (backend down,
 * CORS refused, DNS) is reported as `ok: false` rather than thrown, because the
 * page treats "unreachable" as a state to display, not an exception.
 */
export async function checkHealth(): Promise<HealthResult> {
  try {
    const response = await fetch(`${API_URL}/health`, {
      headers: { Accept: 'application/json' },
    });

    if (!response.ok) {
      return { ok: false, error: `Backend responded ${response.status}` };
    }

    const body = (await response.json()) as { status?: string };
    return body.status === 'ok'
      ? { ok: true }
      : { ok: false, error: 'Unexpected response from /health' };
  } catch (cause) {
    return {
      ok: false,
      error: cause instanceof Error ? cause.message : 'Request failed',
    };
  }
}

// `submitForecast()` is deliberately absent: `POST /forecast` has not been
// built (design-doc Phases 2-4), and a client for a nonexistent endpoint is how
// placeholder UI gets written.

export { API_URL };
