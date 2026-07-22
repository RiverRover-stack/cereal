"""Rolling-origin backtest for the live `POST /forecast` model.

Run from `backend/`:

    .venv\\Scripts\\python -m tests.backtest

This exercises the *actual* code path a real upload goes through
(`app.validation.validate_csv` + `app.forecasting.forecast_revenue`) against real
history in `data/sales_daily.csv` — not a separate offline model. The point is
measuring what a user would actually receive.

Why rolling-origin, not a random train/test split: a random split lets the model
train on days *after* the ones it's scored on, which a real forecast can never do.
Every fold here trains only on data strictly before its 30-day forecast window,
exactly like a real upload's history — this is design-doc.md's own definition of
the backtest methodology (see `app/forecasting.py`'s module docstring).

Every fold is scored against `seasonal_naive()` (design-doc.md's implicit bar:
"repeat last week"). A trained model that doesn't clearly beat that baseline
isn't earning its complexity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.forecasting import (
    HORIZON_DAYS,
    MIN_HISTORY_DAYS,
    ROLLING_WINDOW,
    forecast_revenue,
    seasonal_naive,
)
from app.validation import validate_csv
from tests.metrics import mae, mape, rmse

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sales_daily.csv"

# Fractions of the series length used as fold origins (expanding window: later
# folds train on more history). Spread across the series rather than adjacent
# days, so folds land in different stretches/seasons instead of all scoring the
# same few weeks.
FOLD_FRACTIONS = (0.40, 0.55, 0.70, 0.85, 0.97)


def load_daily(path: Path = DATA_PATH) -> pd.DataFrame:
    raw = path.read_bytes()
    daily, _ = validate_csv(raw)
    return daily


def run_fold(daily: pd.DataFrame, cut: int, horizon: int) -> dict[str, float] | None:
    train = daily.iloc[:cut].reset_index(drop=True)
    actual_frame = daily.iloc[cut : cut + horizon]
    if len(actual_frame) < horizon or len(train) < MIN_HISTORY_DAYS:
        return None

    actual = actual_frame["revenue"].to_numpy()
    forecast = forecast_revenue(train, horizon=horizon)
    naive = seasonal_naive(train, horizon=horizon)

    return {
        "origin_date": str(train["date"].iloc[-1].date()),
        "train_days": len(train),
        "model_mae": mae(actual, forecast.predictions),
        "model_rmse": rmse(actual, forecast.predictions),
        "model_mape": mape(actual, forecast.predictions),
        "naive_mae": mae(actual, naive),
        "naive_rmse": rmse(actual, naive),
        "naive_mape": mape(actual, naive),
    }


def leak_demo(daily: pd.DataFrame) -> None:
    """Show what the `.shift(1)` guard in `build_features()` is actually preventing.

    One-step-ahead diagnostic: "predict today's revenue with a 7-day rolling
    mean." The correct version uses only the *prior* 7 days; the leaky version's
    window includes today, so roughly 1/7 of the value it's "predicting" is
    baked into the prediction itself. If the leaky version scores meaningfully
    better, that gap is what target leakage looks like in this metric — not a
    better model, an easier (and fake) problem.
    """
    revenue = daily["revenue"]
    shifted = revenue.shift(1).rolling(ROLLING_WINDOW).mean()
    leaky = revenue.rolling(ROLLING_WINDOW).mean()

    mask = shifted.notna()
    actual = revenue[mask].to_numpy()
    correct_pred = shifted[mask].to_numpy()
    leaky_pred = leaky[mask].to_numpy()

    print("\nLeakage check - one-step-ahead 'predict today with a 7-day rolling mean':")
    print(
        f"  Correct (prior 7 days only):  MAE {mae(actual, correct_pred):8.2f}  "
        f"MAPE {mape(actual, correct_pred):6.2f}%"
    )
    print(
        f"  Leaky (window includes today): MAE {mae(actual, leaky_pred):8.2f}  "
        f"MAPE {mape(actual, leaky_pred):6.2f}%"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=HORIZON_DAYS)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument(
        "--leak-demo",
        action="store_true",
        help="Also print the one-step-ahead shifted-vs-unshifted rolling-mean comparison.",
    )
    return parser


def _run_folds(daily: pd.DataFrame, horizon: int) -> list[dict[str, float]]:
    n = len(daily)
    results = []
    for frac in FOLD_FRACTIONS:
        cut = int(n * frac)
        result = run_fold(daily, cut, horizon)
        if result is not None:
            results.append(result)

    if not results:
        raise SystemExit(
            "No folds produced — series too short for the configured fold fractions "
            f"(n={n}, horizon={horizon}, min_history={MIN_HISTORY_DAYS})."
        )
    return results


def _print_fold_table(results: list[dict[str, float]]) -> str:
    """Print the per-fold comparison table and return the header (its width sets
    the divider length the caller prints around the summary)."""
    header = (
        f"{'origin':<12}{'train_days':>11}"
        f"{'model_MAE':>12}{'model_RMSE':>12}{'model_MAPE':>11}"
        f"{'naive_MAE':>12}{'naive_RMSE':>12}{'naive_MAPE':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['origin_date']:<12}{r['train_days']:>11}"
            f"{r['model_mae']:>12.2f}{r['model_rmse']:>12.2f}{r['model_mape']:>10.2f}%"
            f"{r['naive_mae']:>12.2f}{r['naive_rmse']:>12.2f}{r['naive_mape']:>10.2f}%"
        )
    return header


def _print_mape_summary(results: list[dict[str, float]], header: str) -> None:
    mean_model_mape = float(np.mean([r["model_mape"] for r in results]))
    mean_naive_mape = float(np.mean([r["naive_mape"] for r in results]))
    print("-" * len(header))
    print(f"Mean model MAPE: {mean_model_mape:.2f}%   Mean naive MAPE: {mean_naive_mape:.2f}%")

    lift = mean_naive_mape - mean_model_mape
    if lift > 0:
        print(f"Model beats the seasonal-naive baseline by {lift:.2f} points of MAPE.")
    else:
        print(f"Model is WORSE than the seasonal-naive baseline by {-lift:.2f} points of MAPE.")


def main() -> None:
    args = _build_arg_parser().parse_args()

    daily = load_daily(args.data)
    results = _run_folds(daily, args.horizon)
    header = _print_fold_table(results)
    _print_mape_summary(results, header)

    if args.leak_demo:
        leak_demo(daily)


if __name__ == "__main__":
    main()
