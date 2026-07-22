"""Integration test for `POST /forecast`'s explanation wiring (ADR-0006).

Exercises the real endpoint end to end (validation -> forecast -> explain), not
a mocked model, so it catches wiring mistakes the unit tests in
`test_explain.py` can't see (e.g. `main.py` not calling `explain`, or a shape
mismatch between what `forecast_revenue` produces and what `explain` expects).
"""

from __future__ import annotations

import io

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _synthetic_csv(days: int = 60) -> bytes:
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    revenue = [1000 + 20 * i + (200 if d.weekday() >= 5 else 0) for i, d in enumerate(dates)]
    units = [10 + i for i in range(days)]
    frame = pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "revenue": revenue, "units_sold": units}
    )
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def test_forecast_response_includes_explanation():
    response = client.post(
        "/forecast",
        files={"file": ("sales.csv", _synthetic_csv(), "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()

    assert "explanation" in body
    assert isinstance(body["explanation"], list)
    assert 1 <= len(body["explanation"]) <= 5
    assert all(isinstance(sentence, str) and sentence for sentence in body["explanation"])


def test_insufficient_history_still_422s_without_touching_explain():
    short_csv = _synthetic_csv(days=10)

    response = client.post(
        "/forecast",
        files={"file": ("sales.csv", short_csv, "text/csv")},
    )

    assert response.status_code == 422
