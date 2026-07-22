"""Tests for `app.explain` (design-doc Phase 4, ADR-0006).

design-doc.md:216-218 specifies: "Write a test that mocks a trained model and
confirms the explanation list is generated with the right sign/direction for a
synthetic case where one feature clearly dominates." That's `test_dominant_feature_
gets_correct_sign_and_rank` below — a real (tiny) XGBRegressor fit on data
engineered so only one feature has any signal, so SHAP has nothing else to
attribute to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from app.explain import explain
from app.forecasting import Forecast

_FEATURE_NAMES = ("day_of_week", "day_of_month", "time_index", "rolling_7")


def _dominant_rolling_7_model() -> XGBRegressor:
    """A model where `rolling_7` is the only feature that varies.

    Every other feature is held constant across the training rows, so it has
    zero variance and therefore zero information — SHAP has nothing to
    attribute to it. The target increases strictly with `rolling_7`, so the
    sign is unambiguous too.
    """
    rng = np.random.default_rng(0)
    n = 80
    rolling_7 = rng.uniform(500.0, 5000.0, size=n)
    frame = pd.DataFrame(
        {
            "day_of_week": 2,
            "day_of_month": 15,
            "time_index": 100,
            "rolling_7": rolling_7,
        }
    )
    target = rolling_7 * 1.1
    model = XGBRegressor(n_estimators=50, max_depth=3, random_state=0)
    model.fit(frame[list(_FEATURE_NAMES)], target)
    return model


def _forecast(model: XGBRegressor, feature_frame: pd.DataFrame, n: int) -> Forecast:
    return Forecast(
        dates=pd.date_range("2026-01-01", periods=n),
        predictions=np.full(n, 5000.0),
        model=model,
        feature_frame=feature_frame,
        feature_names=_FEATURE_NAMES,
        strategy="recursive",
        target="level",
    )


def test_dominant_feature_gets_correct_sign_and_rank():
    model = _dominant_rolling_7_model()
    forecast_rows = pd.DataFrame(
        {
            "day_of_week": [1, 2, 3],
            "day_of_month": [10, 11, 12],
            "time_index": [100, 101, 102],
            "rolling_7": [4800.0, 4900.0, 5000.0],  # high and rising
        }
    )

    sentences = explain(_forecast(model, forecast_rows, 3))

    assert sentences, "expected at least one explanation sentence"
    top = sentences[0].lower()
    assert "recent 7-day sales trend" in top
    assert "dominant" in top
    assert "pushing this forecast up" in top


def test_low_dominant_feature_pulls_forecast_down():
    model = _dominant_rolling_7_model()
    forecast_rows = pd.DataFrame(
        {
            "day_of_week": [1, 2, 3],
            "day_of_month": [10, 11, 12],
            "time_index": [100, 101, 102],
            "rolling_7": [600.0, 550.0, 500.0],  # low
        }
    )

    sentences = explain(_forecast(model, forecast_rows, 3))

    assert sentences
    assert "pulling this forecast down" in sentences[0].lower()


def test_returns_at_most_five_sentences():
    model = _dominant_rolling_7_model()
    forecast_rows = pd.DataFrame(
        {
            "day_of_week": [0, 1],
            "day_of_month": [1, 2],
            "time_index": [1, 2],
            "rolling_7": [1000.0, 1000.0],
        }
    )

    sentences = explain(_forecast(model, forecast_rows, 2))

    assert len(sentences) <= 5
    assert all(isinstance(sentence, str) and sentence for sentence in sentences)
