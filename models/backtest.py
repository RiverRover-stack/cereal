"""Rolling-origin backtesting, metrics, baselines and the 30-day horizon simulator.

-------------------------------------------------------------------------------
WHY NOT KFold / train_test_split
-------------------------------------------------------------------------------
`date` is not unique in this dataset -- it is one row per SKU per day, 40 SKUs over
1,088 days. A random row split puts 2025-06-14 in both train and test, so the model
sees the same day's demand shock for SKU A while predicting SKU B, and it sees the
future while predicting the past. Every split here is a CALENDAR DATE CUT: every row
on or after the cut is test, every row before is train.

-------------------------------------------------------------------------------
THE 30-DAY HORIZON PROBLEM -- explicit decision
-------------------------------------------------------------------------------
Of the features we use, `is_promo_active`, `base_price`, `month`, `day_of_month`,
`time_idx`, `sku`, `category` and `day_of_week` are all KNOWN in advance for any
future date. `units_lag_7` and `units_rolling_7d_avg` are NOT: forecasting day t+8 needs
demand from days you have not observed.

We implement and measure all three standard answers rather than asserting one:

  * ORACLE   -- feed the true lag features for every test day. This is NOT a
                deployable forecaster (it needs tomorrow's data to predict the day
                after). It is reported as a diagnostic upper bound only, and to make
                the hyperparameter search cheap.
  * RECURSIVE-- feed each prediction back in to build the next lag window. This is
                what production would do. Errors compound across the 30 steps and we
                measure that compounding by horizon-day bucket.
  * KNOWN-ONLY- drop both lag features entirely and fit on covariates that are known
                in advance. Weakest signal, zero compounding.

The headline numbers in the report are the RECURSIVE ones, because that is the
regime production runs in.

Note on what recursive mode still gets for free: the dataset has no rows for days a
SKU sold nothing, so the set of test rows tells us WHICH days a SKU transacted. We do
not predict occurrence, only volume on days where a sale happened. That is a real
caveat and it is in the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features import FEATURES, inverse_cbrt


@dataclass(frozen=True)
class Fold:
    idx: int
    train_end: pd.Timestamp  # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # inclusive

    def label(self, df: pd.DataFrame) -> tuple[str, str]:
        tr = df[df["date"] < self.train_end]["date"]
        return (
            f"{tr.min():%Y-%m-%d}..{tr.max():%Y-%m-%d}",
            f"{self.test_start:%Y-%m-%d}..{self.test_end:%Y-%m-%d}",
        )


def make_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    horizon_days: int = 30,
    last_test_end: pd.Timestamp | None = None,
) -> list[Fold]:
    """Expanding-window rolling origin: fit on everything up to day n, forecast the
    next `horizon_days` calendar days, roll the origin forward, repeat.

    Train windows GROW (expanding), test windows are fixed length and disjoint.

    `last_test_end` lets the caller build an EARLIER block of folds that is disjoint
    from the reported ones. Model selection runs on that earlier block so that no
    reported fold ever influences the configuration it is scoring -- selecting on a
    fold you then report is a soft form of testing on train.
    """
    last = last_test_end if last_test_end is not None else df["date"].max()
    folds = []
    for k in range(n_folds):
        test_end = last - pd.Timedelta(days=horizon_days * (n_folds - 1 - k))
        test_start = test_end - pd.Timedelta(days=horizon_days - 1)
        folds.append(Fold(k + 1, test_start, test_start, test_end))
    return folds


def split(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] < fold.train_end]
    test = df[(df["date"] >= fold.test_start) & (df["date"] <= fold.test_end)]
    return train, test


def time_ordered_valid(train: pd.DataFrame, valid_days: int = 30):
    """Carve an early-stopping validation set off the END of the training window.

    Time-ordered, never shuffled: the validation set is strictly later than the
    fitting set, which mirrors how the model is used.
    """
    cut = train["date"].max() - pd.Timedelta(days=valid_days - 1)
    return train[train["date"] < cut], train[train["date"] >= cut]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    # MAPE guard: only score rows whose actual is at least 1 unit. In this dataset
    # min(total_units) == 1 so nothing is actually dropped, but the guard has to be
    # here for data where zero-demand days are present as rows.
    m = np.abs(y_true) >= 1.0
    mape = float(np.mean(np.abs(err[m] / y_true[m])) * 100) if m.any() else float("nan")
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": mape,
        "n": int(len(y_true)),
    }


def decile_mae(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """MAE and mean bias per decile of the ACTUAL target.

    This is how we test the skewness report's prediction that a raw-target model
    over-predicts the low-volume SKUs. Positive bias = over-prediction.
    """
    d = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
    d["decile"] = pd.qcut(d["y"].rank(method="first"), n_bins, labels=False) + 1
    out = d.groupby("decile").apply(
        lambda g: pd.Series(
            {
                "actual_mean": g["y"].mean(),
                "pred_mean": g["p"].mean(),
                "MAE": (g["p"] - g["y"]).abs().mean(),
                "bias": (g["p"] - g["y"]).mean(),
                "n": len(g),
            }
        ),
        include_groups=False,
    )
    return out


def directional(test: pd.DataFrame, y_pred) -> dict:
    """Post-hoc directional accuracy and F1, derived from the regressor's output.

    Not a second model. For each test row we compare the forecast to that SKU's most
    recent OBSERVED value before the forecast window, and score whether the model
    called the sign of the change correctly. "Up" (pred >= prior) is the positive
    class. Ties in the actual are counted as "not up".
    """
    from sklearn.metrics import f1_score

    d = test[["sku", "date", "total_units", "units_rolling_7d_avg"]].copy()
    d["pred"] = np.asarray(y_pred, float)
    # prior reference = previous observation for that SKU inside the evaluation frame,
    # falling back to the (strictly past) 7-day mean for the first row of each SKU.
    d = d.sort_values(["sku", "date"], kind="mergesort")
    prior = d.groupby("sku", observed=True)["total_units"].shift(1)
    d["prior"] = prior.fillna(d["units_rolling_7d_avg"])
    d = d.dropna(subset=["prior"])

    actual_up = (d["total_units"] > d["prior"]).astype(int)
    pred_up = (d["pred"] > d["prior"]).astype(int)
    return {
        "accuracy": float((actual_up == pred_up).mean()),
        "f1_up": float(f1_score(actual_up, pred_up, pos_label=1, zero_division=0)),
        "f1_down": float(f1_score(1 - actual_up, 1 - pred_up, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(actual_up, pred_up, average="macro", zero_division=0)),
        "actual_up_share": float(actual_up.mean()),
        "pred_up_share": float(pred_up.mean()),
        "n": int(len(d)),
    }


# --------------------------------------------------------------------------- #
# Baselines -- "does the model beat doing nothing"
# --------------------------------------------------------------------------- #
def baseline_lag7(test: pd.DataFrame) -> np.ndarray:
    """Predict the value 7 observations ago. Strictly past, no fitting."""
    return test["units_lag_7"].to_numpy(dtype=float)


def baseline_rolling(test: pd.DataFrame) -> np.ndarray:
    """Predict the trailing 7-observation mean -- the baseline the brief asked for.

    Since the source fix this is `units_rolling_7d_avg` used directly: its window is
    t-7..t-1, so the baseline is not scoring itself against part of its own answer.
    Before the fix this function had to substitute a locally-shifted column.
    """
    return test["units_rolling_7d_avg"].to_numpy(dtype=float)


def baseline_rolling_leaky(test: pd.DataFrame) -> np.ndarray:
    """The OLD leaky window (t-6..t) used as a predictor. NOT a real baseline.

    Reported only to keep the size of the discarded free lunch on the record: a
    predictor that can see 1/7 of the answer beats every honest model. If a future
    model ever scores near this line, suspect a leak before celebrating.
    """
    return test["units_rolling_7d_avg_LEAKY_DIAGNOSTIC"].to_numpy(dtype=float)


def baseline_flat_last7(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The only baseline here that is genuinely deployable over a 30-day horizon.

    IMPORTANT: `baseline_lag7` and `baseline_rolling7_shifted` above are ONE-STEP-AHEAD
    baselines. On day 20 of a 30-day forecast window they read actual demand from days
    13-19 of that same window -- data production would not have yet. They are a fair
    yardstick for a next-day forecast and an unfair one for a 30-day forecast.

    This baseline takes each SKU's mean of its last 7 observations at the moment the
    forecast is made and holds it flat for all 30 days. That is "do nothing", properly.
    Falls back to the global train mean for a SKU with no training history.
    """
    last7 = (
        train.sort_values("date")
        .groupby("sku", observed=True)["total_units"]
        .apply(lambda s: s.tail(7).mean())
    )
    fallback = float(train["total_units"].mean())
    return test["sku"].map(last7).fillna(fallback).to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# Horizon strategies
# --------------------------------------------------------------------------- #
def predict_oracle(model, test: pd.DataFrame, features: list[str], cbrt: bool) -> np.ndarray:
    p = model.predict(test[features])
    return inverse_cbrt(p) if cbrt else np.asarray(p, float)


def predict_recursive(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    cbrt: bool,
) -> np.ndarray:
    """Walk the test window forward one calendar day at a time, rebuilding the lag
    features from the model's OWN previous predictions.

    Per-SKU history buffers reproduce the exact feature definitions used in training:
      units_lag_7      = history[-7]        (7 observations back, matching shift(7))
      units_rolling_7d_avg = mean(history[-7:]) (window ending the previous observation)
    """
    hist: dict[str, list[float]] = {
        str(s): g["total_units"].astype(float).tolist()
        for s, g in train.groupby("sku", observed=True)
    }
    test = test.sort_values(["date", "sku"], kind="mergesort")
    out = pd.Series(index=test.index, dtype=float)

    for day, chunk in test.groupby("date", sort=True):
        rows = chunk.copy()
        lag, roll = [], []
        for s in rows["sku"].astype(str):
            h = hist.get(s, [])
            lag.append(h[-7] if len(h) >= 7 else np.nan)
            roll.append(float(np.mean(h[-7:])) if len(h) >= 1 else np.nan)
        rows["units_lag_7"] = lag
        rows["units_rolling_7d_avg"] = roll

        raw = model.predict(rows[features])
        pred = inverse_cbrt(raw) if cbrt else np.asarray(raw, float)
        # Demand cannot be negative. Clipping happens AFTER the inverse transform so
        # the cube-root model's own sign handling is preserved.
        pred = np.clip(pred, 0.0, None)
        out.loc[rows.index] = pred

        for s, p in zip(rows["sku"].astype(str), pred):
            hist.setdefault(s, []).append(float(p))

    return out.loc[test.sort_index().index].sort_index().to_numpy()


def horizon_buckets(test: pd.DataFrame, y_pred, fold: Fold) -> pd.DataFrame:
    """Error by how far into the 30-day window we are -- the compounding measurement."""
    d = pd.DataFrame(
        {
            "y": test["total_units"].to_numpy(float),
            "p": np.asarray(y_pred, float),
            "h": (test["date"] - fold.test_start).dt.days.to_numpy() + 1,
        }
    )
    d["bucket"] = pd.cut(
        d["h"], bins=[0, 7, 14, 21, 30], labels=["d1-7", "d8-14", "d15-21", "d22-30"]
    )
    return d.groupby("bucket", observed=True).apply(
        lambda g: pd.Series(
            {
                "MAE": (g["p"] - g["y"]).abs().mean(),
                "RMSE": float(np.sqrt(((g["p"] - g["y"]) ** 2).mean())),
                "MAPE": float((np.abs((g["p"] - g["y"]) / g["y"])).mean() * 100),
                "n": len(g),
            }
        ),
        include_groups=False,
    )
