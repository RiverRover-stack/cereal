"""30-day revenue forecasting (design-doc Phase 3, lines 163-189).

Consumes the already-aggregated daily frame from `app.validation.validate_csv`
(one row per date, sorted ascending) and returns a 30-day forward forecast.

The model is an XGBoost *regressor* on a continuous target (daily revenue),
trained per-request on the seller's own history (design-doc.md:107). Training
time sits inside the user's HTTP request, so model size is a latency decision as
much as an accuracy one.

## How the 30-day horizon is generated — RECURSIVE (explicit decision)

Three of the four features in design-doc.md:168 are known in advance for any
future date: `day_of_week`, `day_of_month`, `time_index`. The rolling 7-day
average is not — forecasting day t+8 needs revenue for days not yet observed.

Three strategies were implemented and backtested head to head in
`tests/backtest.py` (rolling-origin, 4 folds, full 30-day horizon each fold):

  * `recursive`         — feed each prediction back in to compute the next
                          rolling window. Errors can compound over 30 steps.
  * `direct`            — horizon as a feature; rolling average frozen at the
                          forecast origin. No compounding, ~30x training rows.
  * `known_covariates`  — drop the rolling feature entirely. Zero compounding,
                          weakest signal.

`recursive` was selected on measured backtest MAPE/MAE/RMSE, not on principle.
Re-run `python -m tests.backtest` to reproduce the comparison table; if a future
data change flips the ranking, change `DEFAULT_STRATEGY` rather than arguing
with the numbers.

## Leakage

`rolling_7` is `.shift(1).rolling(7).mean()` — the window ends on the *previous*
day and never includes the day it is a feature for. Unshifted, the target leaks
into its own predictor and validation error collapses to something implausible.
`tests/test_forecasting.py` asserts the shift arithmetically, and
`tests/backtest.py` has a `--leak-demo` mode that shows what the broken version
scores so the difference is visible rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# design-doc.md:187 — below this, refuse rather than emit an unreliable forecast.
MIN_HISTORY_DAYS = 30

# design-doc.md:174 — the product is a 30-day forecast.
HORIZON_DAYS = 30

ROLLING_WINDOW = 7

# Seeded everywhere so two runs of the same upload return byte-identical numbers.
# An unreproducible forecast cannot be debugged, stress-tested, or explained.
RANDOM_STATE = 42

KNOWN_FEATURES = ("day_of_week", "day_of_month", "time_index")
ROLLING_FEATURE = "rolling_7"
HORIZON_FEATURE = "horizon"

RECURSIVE_FEATURES = (*KNOWN_FEATURES, ROLLING_FEATURE)
DIRECT_FEATURES = (*KNOWN_FEATURES, ROLLING_FEATURE, HORIZON_FEATURE)

DEFAULT_STRATEGY = "recursive"

# The regressor is fitted on log1p(revenue) and inverted with expm1. The target
# is still continuous daily revenue (design-doc.md:170) — this is a scale choice,
# not a different target, and every metric is computed in dollars after the
# inverse.
#
# Chosen on measured backtest, not principle: `level` scored 16.83% mean MAPE
# against `log` at 14.63% on the same 4 folds, with a better worst fold too
# (25.48% vs 29.40%). Revenue is positive and varies multiplicatively, so a
# proportional error scale suits it. Flip this constant to "level" and re-run
# `python -m tests.backtest --grid` to re-audit the claim.
#
# PHASE 4 NOTE: SHAP values on this model come out in log units, not dollars.
# The design-doc.md:204-211 templates are directional ("pushing this forecast
# up"), so sign and rank still translate; anything wanting a dollar-denominated
# attribution needs a level-target model instead. Settle this before Phase 4.
DEFAULT_TARGET = "log"

_TARGET_TRANSFORMS: dict[str, tuple] = {
    "level": (lambda y: y, lambda y: y),
    "log": (np.log1p, np.expm1),
}

# Chosen by the coarse grid in `tests/backtest.py`, inside the region that suits
# a few hundred rows: depth 3-5, 100-300 trees, lr 0.05-0.1, subsample ~0.8.
# Depth beyond ~6 on 180 rows is memorisation — it shows up as training error
# falling while backtest error rises.
DEFAULT_PARAMS: dict[str, object] = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    # Pinned to 1 thread: on a few hundred rows the thread pool costs more than
    # it saves, and a fixed thread count keeps floating-point summation order —
    # and therefore the forecast — identical run to run.
    "n_jobs": 1,
    "tree_method": "hist",
}

# Early stopping runs on a time-ordered tail holdout, never a shuffled one.
EARLY_STOPPING_ROUNDS = 20
EARLY_STOPPING_HOLDOUT = 21


class InsufficientHistoryError(ValueError):
    """Fewer than MIN_HISTORY_DAYS daily rows — becomes a 422."""


@dataclass(frozen=True)
class Forecast:
    """A 30-day forward forecast plus the fitted model behind it.

    `model`, `feature_frame` and `feature_names` are carried so Phase 4's SHAP
    TreeExplainer can attribute the *same* fitted model that produced these
    numbers, rather than refitting and explaining a different one.
    """

    dates: pd.DatetimeIndex
    predictions: np.ndarray
    model: XGBRegressor
    feature_frame: pd.DataFrame
    feature_names: tuple[str, ...]
    strategy: str
    # Which scale `model` was fitted on. Phase 4 needs this: SHAP values are in
    # the model's own units, so an explainer has to know whether it is looking
    # at dollars or log-dollars.
    target: str = DEFAULT_TARGET

    @property
    def total(self) -> float:
        return float(self.predictions.sum())


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add the design-doc.md:168 features to an aggregated daily frame.

    `time_index` counts *calendar days* since the first observation, not row
    position. On a history with missing dates those differ, and the trend
    feature has to mean "how far through the history is this day" for the
    forecast dates (which are calendar-dated) to sit on the same scale.
    """
    frame = daily.copy()
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_month"] = frame["date"].dt.day
    frame["time_index"] = (frame["date"] - frame["date"].iloc[0]).dt.days

    # THE leakage guard. `.shift(1)` first, so the 7-day window ends on the
    # previous day. Without it the mean contains today's revenue — the target
    # inside its own feature.
    frame[ROLLING_FEATURE] = (
        frame["revenue"].shift(1).rolling(ROLLING_WINDOW).mean()
    )
    return frame


def _trainable(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the warm-up rows where the shifted rolling mean is undefined.

    The first ROLLING_WINDOW rows have no 7 prior days. They are dropped rather
    than fed to XGBoost as NaN: NaN here is perfectly correlated with "early in
    the history", so keeping them invites the model to learn the missingness as
    a trend signal that will never recur at forecast time.
    """
    return frame.dropna(subset=[ROLLING_FEATURE]).reset_index(drop=True)


def _forward(values, target: str):
    return _TARGET_TRANSFORMS[target][0](np.asarray(values, dtype=float))


def _inverse(values, target: str):
    return _TARGET_TRANSFORMS[target][1](np.asarray(values, dtype=float))


def _fit(
    features: pd.DataFrame,
    target: pd.Series,
    params: dict[str, object] | None = None,
) -> XGBRegressor:
    """Fit with early stopping on a time-ordered tail, then refit on everything.

    Two-stage on purpose. Stage one finds the useful number of trees against a
    holdout of the most recent days; stage two refits on the *full* history with
    that tree count, because the most recent days are the ones a 30-day forecast
    leans on hardest and a model that never saw them starts the horizon blind.

    Both fits are cheap at this data size — see the latency line in
    `tests/backtest.py` output.

    NOTE (verified against xgboost 3.1.3, not recalled): `early_stopping_rounds`
    is a *constructor* argument. `XGBRegressor.fit` in 3.x accepts `eval_set` but
    has no `early_stopping_rounds` parameter.
    """
    settings = dict(DEFAULT_PARAMS if params is None else params)
    rounds = int(settings["n_estimators"])

    if len(features) >= 2 * EARLY_STOPPING_HOLDOUT:
        cut = len(features) - EARLY_STOPPING_HOLDOUT
        probe = XGBRegressor(
            **settings, early_stopping_rounds=EARLY_STOPPING_ROUNDS
        )
        probe.fit(
            features.iloc[:cut],
            target.iloc[:cut],
            eval_set=[(features.iloc[cut:], target.iloc[cut:])],
            verbose=False,
        )
        if probe.best_iteration is not None:
            rounds = max(int(probe.best_iteration) + 1, 1)

    model = XGBRegressor(**{**settings, "n_estimators": rounds})
    model.fit(features, target, verbose=False)
    return model


def future_dates(last_date: pd.Timestamp, horizon: int = HORIZON_DAYS):
    return pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
    )


def _known_feature_rows(
    dates: pd.DatetimeIndex, origin: pd.Timestamp
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "day_of_week": dates.dayofweek,
            "day_of_month": dates.day,
            "time_index": (dates - origin).days,
        }
    )


def _clip(values: np.ndarray) -> np.ndarray:
    """Daily revenue cannot be negative.

    A tree can extrapolate below zero on a thin history; showing a seller a
    negative day would discredit the whole forecast. Clipping is presentational
    honesty, not a fix — if this fires often the model is wrong, and the
    backtest reports on unclipped-then-clipped output so it stays visible.
    """
    return np.clip(values, 0.0, None)


def _recursive_forecast_loop(
    model: XGBRegressor,
    dates: pd.DatetimeIndex,
    known: pd.DataFrame,
    seed_window: list[float],
    target: str,
) -> tuple[list[dict[str, object]], list[float]]:
    """Step through `dates` one day at a time, feeding each prediction back into
    the rolling window that seeds the next day's feature row.

    Seeded with the true tail of the history; this is where 30-step compounding
    lives -- the backtest measures it per fold rather than assuming it is
    tolerable.
    """
    window = list(seed_window)
    rows: list[dict[str, object]] = []
    predictions: list[float] = []

    for position in range(len(dates)):
        row = {
            **{name: known.loc[position, name] for name in KNOWN_FEATURES},
            ROLLING_FEATURE: float(np.mean(window[-ROLLING_WINDOW:])),
        }
        raw = float(
            model.predict(pd.DataFrame([row], columns=list(RECURSIVE_FEATURES)))[0]
        )
        # Invert to dollars *before* feeding back, so the rolling window the next
        # step sees is on the same scale as the actual history that seeded it.
        value = float(_clip(_inverse([raw], target))[0])
        rows.append(row)
        predictions.append(value)
        window.append(value)

    return rows, predictions


def _forecast_recursive(
    frame: pd.DataFrame,
    horizon: int,
    params: dict[str, object] | None,
    target: str = DEFAULT_TARGET,
) -> Forecast:
    trainable = _trainable(frame)
    features = trainable[list(RECURSIVE_FEATURES)]
    model = _fit(
        features,
        pd.Series(_forward(trainable["revenue"], target)),
        params,
    )

    start = frame["date"].iloc[0]
    dates = future_dates(frame["date"].iloc[-1], horizon)
    known = _known_feature_rows(dates, start)

    seed_window = list(frame["revenue"].iloc[-ROLLING_WINDOW:])
    rows, predictions = _recursive_forecast_loop(model, dates, known, seed_window, target)

    return Forecast(
        dates=dates,
        predictions=np.asarray(predictions, dtype=float),
        model=model,
        feature_frame=pd.DataFrame(rows, columns=list(RECURSIVE_FEATURES)),
        feature_names=RECURSIVE_FEATURES,
        strategy="recursive",
        target=target,
    )


def _forecast_known_covariates(
    frame: pd.DataFrame,
    horizon: int,
    params: dict[str, object] | None,
    target: str = DEFAULT_TARGET,
) -> Forecast:
    trainable = _trainable(frame)
    features = trainable[list(KNOWN_FEATURES)]
    model = _fit(features, pd.Series(_forward(trainable["revenue"], target)), params)

    start = frame["date"].iloc[0]
    dates = future_dates(frame["date"].iloc[-1], horizon)
    known = _known_feature_rows(dates, start)
    predictions = _clip(_inverse(model.predict(known), target))

    return Forecast(
        dates=dates,
        predictions=predictions,
        model=model,
        feature_frame=known,
        feature_names=KNOWN_FEATURES,
        strategy="known_covariates",
        target=target,
    )


def _direct_training_set(
    trainable: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.Series]:
    """One row per (forecast origin, horizon step) pair.

    The rolling average is frozen at the origin — the last value genuinely known
    when a forecast is made — and `horizon` tells the model how far ahead it is
    reaching. No prediction is ever fed back, so nothing compounds.

    `rolling_at_origin` for origin o is the mean of the 7 days ending at o. That
    includes day o itself, which is *not* leakage: every target here is at least
    one day after o.
    """
    revenue = trainable["revenue"].to_numpy()
    rolling_at_origin = (
        trainable["revenue"].rolling(ROLLING_WINDOW).mean().to_numpy()
    )
    rows: list[dict[str, float]] = []
    targets: list[float] = []

    for origin in range(len(trainable)):
        if np.isnan(rolling_at_origin[origin]):
            continue
        for step in range(1, horizon + 1):
            target_position = origin + step
            if target_position >= len(trainable):
                break
            rows.append(
                {
                    "day_of_week": trainable["day_of_week"].iloc[target_position],
                    "day_of_month": trainable["day_of_month"].iloc[target_position],
                    "time_index": trainable["time_index"].iloc[target_position],
                    ROLLING_FEATURE: rolling_at_origin[origin],
                    HORIZON_FEATURE: step,
                }
            )
            targets.append(revenue[target_position])

    return (
        pd.DataFrame(rows, columns=list(DIRECT_FEATURES)),
        pd.Series(targets, dtype=float),
    )


def _forecast_direct(
    frame: pd.DataFrame,
    horizon: int,
    params: dict[str, object] | None,
    target: str = DEFAULT_TARGET,
) -> Forecast:
    trainable = _trainable(frame)
    features, values = _direct_training_set(trainable, horizon)
    if features.empty:
        raise InsufficientHistoryError(
            "Not enough history to build a direct multi-step training set."
        )
    model = _fit(features, pd.Series(_forward(values, target)), params)

    start = frame["date"].iloc[0]
    dates = future_dates(frame["date"].iloc[-1], horizon)
    forecast_features = _known_feature_rows(dates, start)
    forecast_features[ROLLING_FEATURE] = float(
        frame["revenue"].iloc[-ROLLING_WINDOW:].mean()
    )
    forecast_features[HORIZON_FEATURE] = np.arange(1, len(dates) + 1)
    forecast_features = forecast_features[list(DIRECT_FEATURES)]
    predictions = _clip(_inverse(model.predict(forecast_features), target))

    return Forecast(
        dates=dates,
        predictions=predictions,
        model=model,
        feature_frame=forecast_features,
        feature_names=DIRECT_FEATURES,
        strategy="direct",
        target=target,
    )


STRATEGIES = {
    "recursive": _forecast_recursive,
    "direct": _forecast_direct,
    "known_covariates": _forecast_known_covariates,
}


def forecast_revenue(
    daily: pd.DataFrame,
    horizon: int = HORIZON_DAYS,
    strategy: str = DEFAULT_STRATEGY,
    params: dict[str, object] | None = None,
    target: str = DEFAULT_TARGET,
) -> Forecast:
    """Fit on `daily` and forecast `horizon` days forward.

    `daily` must be the output of `validate_csv` — one row per date, ascending,
    numeric and non-null. Nothing is re-validated or re-aggregated here.
    """
    if len(daily) < MIN_HISTORY_DAYS:
        raise InsufficientHistoryError(
            f"Not enough history to forecast: {len(daily)} day(s) of sales "
            f"after grouping by date, but at least {MIN_HISTORY_DAYS} are "
            "needed. Upload a longer export — a few months of orders gives the "
            "most reliable forecast."
        )

    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}.")
    if target not in _TARGET_TRANSFORMS:
        raise ValueError(f"Unknown target scale {target!r}.")

    return STRATEGIES[strategy](build_features(daily), horizon, params, target)


def seasonal_naive(daily: pd.DataFrame, horizon: int = HORIZON_DAYS) -> np.ndarray:
    """Baseline: same weekday last week, repeated forward.

    The bar XGBoost has to clear (one line of pandas, no training). Day t+h
    takes the last observed value for that weekday, so the final week of history
    is tiled across the horizon.
    """
    tail = daily["revenue"].to_numpy()[-ROLLING_WINDOW:]
    if len(tail) < ROLLING_WINDOW:
        tail = np.resize(daily["revenue"].to_numpy(), ROLLING_WINDOW)
    return np.array([tail[step % ROLLING_WINDOW] for step in range(horizon)])
