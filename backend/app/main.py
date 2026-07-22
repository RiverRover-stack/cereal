"""Trends Arc API.

`POST /forecast` validates the upload (Phase 2), returns a 30-day revenue
forecast (Phase 3), and attaches a SHAP-based plain-English explanation
(Phase 4, ADR-0006).
"""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.explain import explain
from app.forecasting import InsufficientHistoryError, forecast_revenue
from app.validation import (
    CsvValidationError,
    FileTooLargeError,
    read_upload,
    validate_csv,
)

app = FastAPI(
    title="Trends Arc API",
    description="Shopify sales forecasting backend.",
    version="0.1.0",
)

# The site is a static Astro build served from a different origin than this API,
# so every browser call is cross-origin — without this the fetch fails at the
# preflight and the page can only report "unreachable".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast")
async def forecast(file: UploadFile = File(...)) -> dict[str, object]:
    """Validate an uploaded sales CSV and return a 30-day revenue forecast.

    The model is fitted on this upload, inside this request (design-doc.md:107)
    — there is no global pre-trained model, so training latency is request
    latency. See `tests/backtest.py` for the measured fit time.

    `sku_level` is null and stays null in V1 (design-doc.md:184). It is present
    rather than omitted so per-SKU forecasting (design-doc.md:73) can fill it
    without a breaking schema change.

    The Phase 2 validation metadata (`rows`, `raw_rows`, `aggregated`,
    `date_range`) is kept alongside the forecast: `rows` is the count *after*
    aggregating to one row per date, which is the series the model actually
    trained on, and `raw_rows` keeps the original line count visible so a large
    collapse (a per-line-item export) reads as aggregation rather than data loss.
    """
    try:
        raw = await read_upload(file)
    except FileTooLargeError as cause:
        raise HTTPException(status_code=413, detail=str(cause)) from cause

    try:
        daily, raw_rows = validate_csv(raw)
    except CsvValidationError as cause:
        raise HTTPException(status_code=422, detail=str(cause)) from cause

    # design-doc.md:187 — refuse rather than return a forecast we know is
    # unreliable. Same 422 channel as validation, because from the user's side
    # it is the same class of problem: the file cannot produce a forecast, and
    # the message says what to do about it.
    try:
        result = forecast_revenue(daily)
    except InsufficientHistoryError as cause:
        raise HTTPException(status_code=422, detail=str(cause)) from cause

    # SHAP is additive to the forecast, not load-bearing for it (ADR-0006) — a
    # failure here shouldn't turn an otherwise-valid forecast into a 500.
    try:
        explanation = explain(result)
    except Exception:
        explanation = []

    daily_forecast = [
        {"date": date.date().isoformat(), "predicted_revenue": round(float(value), 2)}
        for date, value in zip(result.dates, result.predictions)
    ]

    return {
        "store_level": {
            "daily_forecast": daily_forecast,
            # Summed before rounding so the total matches the series to the cent
            # rather than drifting by the accumulated rounding of 30 days.
            "total_30_day": round(result.total, 2),
        },
        "sku_level": None,
        "explanation": explanation,
        "status": "valid",
        "rows": len(daily),
        "raw_rows": raw_rows,
        "aggregated": len(daily) < raw_rows,
        "date_range": {
            "start": daily["date"].iloc[0].date().isoformat(),
            "end": daily["date"].iloc[-1].date().isoformat(),
        },
    }
