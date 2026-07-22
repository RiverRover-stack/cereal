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

export interface DailyForecastPoint {
  date: string;
  predicted_revenue: number;
}

export interface ForecastSuccess {
  ok: true;
  storeLevel: {
    dailyForecast: DailyForecastPoint[];
    total30Day: number;
  };
  rows: number;
  rawRows: number;
  aggregated: boolean;
  dateRange: { start: string; end: string };
  /** Plain-English SHAP attributions, ranked, 0-5 sentences (ADR-0006). */
  explanation: string[];
}

export interface ForecastFailure {
  ok: false;
  /** The backend's own message (422/413 `detail`, or a network-failure summary). */
  error: string;
}

export type ForecastResult = ForecastSuccess | ForecastFailure;

/**
 * Response shape from `backend/app/main.py`'s `POST /forecast`. `sku_level` is
 * always null in V1 (design-doc.md:184) and not surfaced here since nothing
 * reads it yet.
 */
interface ForecastResponseBody {
  store_level: {
    daily_forecast: DailyForecastPoint[];
    total_30_day: number;
  };
  rows: number;
  raw_rows: number;
  aggregated: boolean;
  date_range: { start: string; end: string };
  explanation: string[];
}

/**
 * Posts a CSV to `POST /forecast` and normalizes both HTTP failure (422
 * validation, 413 too-large) and network failure into the same `ForecastFailure`
 * shape, carrying the backend's own message — this page renders the backend's
 * text verbatim rather than a generic failure state (design-doc.md:237-238).
 */
export async function submitForecast(file: File): Promise<ForecastResult> {
  const body = new FormData();
  // Field name must be "file" — it binds to `UploadFile = File(...)` on the
  // FastAPI side by parameter name.
  body.append('file', file);

  let response: Response;
  try {
    response = await fetch(`${API_URL}/forecast`, { method: 'POST', body });
  } catch (cause) {
    return {
      ok: false,
      error:
        cause instanceof Error
          ? `Could not reach the backend: ${cause.message}`
          : 'Could not reach the backend.',
    };
  }

  if (!response.ok) {
    let detail = `Backend responded ${response.status}.`;
    try {
      const errorBody = (await response.json()) as { detail?: string };
      if (errorBody.detail) detail = errorBody.detail;
    } catch {
      // Non-JSON error body (e.g. a proxy's plain-text 502) — fall back to the
      // status-only message set above rather than throwing on a bad parse.
    }
    return { ok: false, error: detail };
  }

  const parsed = (await response.json()) as ForecastResponseBody;
  return {
    ok: true,
    storeLevel: {
      dailyForecast: parsed.store_level.daily_forecast,
      total30Day: parsed.store_level.total_30_day,
    },
    rows: parsed.rows,
    rawRows: parsed.raw_rows,
    aggregated: parsed.aggregated,
    dateRange: parsed.date_range,
    explanation: parsed.explanation,
  };
}

export { API_URL };
