"""Feature assembly for the Trends Arc demand forecaster.

Input is the FROZEN dataset `data/processed/master_df_cbrt.csv` only. Nothing here
re-derives features from `data/sales_line_items.csv`; the only transformation we apply
(adding calendar/time-index columns) is a pure function of the `date` column that
already exists in the frozen file.

-------------------------------------------------------------------------------
LEAKAGE DECISIONS -- all four are enforced in code, not just documented.
-------------------------------------------------------------------------------

1. `total_revenue` and `price_per_unit` are EXCLUDED as features.
   `scripts/build_dataset.py` computes `price_per_unit = total_revenue / total_units`
   and `total_revenue` is summed from the same line items as `total_units` on the same
   row. Both are functions of the target for the row being predicted. Using either is
   direct target leakage. Their `_cbrt` variants are excluded for the same reason.

2. `units_rolling_7d_avg` IS NOW CLEAN AT SOURCE (contract changed 2026-07-21).
   HISTORY: the original build script used `x.rolling(window=7).mean()` with no
   `.shift()`, so the window was rows t-6..t INCLUSIVE -- the target contributed 1/7
   of its own predictor. This module compensated by applying its own `.shift(1)` at
   load time.
   NOW: `scripts/build_dataset.py` builds it as
   `x.shift(1).rolling(window=7).mean()`, i.e. rows t-7..t-1, strictly past. The
   compensating shift here has been REMOVED -- keeping it would double-shift and
   silently throw away a day of signal.
   The assertions in `assert_no_target_leakage()` were inverted to match the new
   contract and are the runtime proof. They are deliberately two-sided: the column
   must equal the excluding-today window AND must NOT equal the including-today one,
   so a regression in either direction fails the run rather than degrading it. That
   is exactly how the contract change was caught: the old assertions raised
   "expected the frozen rolling column to include the current day" against the
   rebuilt file, and training refused to start.

3. `units_lag_7` is safe but MISNAMED. It is `groupby(sku).shift(7)`: seven ROWS back,
   not seven days back. SKUs only have rows on days they sold something, so seven rows
   back is exactly seven calendar days only 38.6% of the time (measured). It is still
   strictly in the past -- no leakage -- but it is a "lag-7-observations" feature and
   any weekday-seasonality story told about it would be wrong. Out of scope to rename.

4. Splits are by calendar date, never by row. See backtest.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "master_df_cbrt.csv"

TARGET_RAW = "total_units"
TARGET_CBRT = "total_units_cbrt"

# Categorical features are handed to XGBoost as pandas `category` dtype with
# `enable_categorical=True` (native categorical splits). Chosen over one-hot because
# `sku` has 40 levels and one-hot would give the tree 40 near-empty binary columns to
# split on, and over ordinal-integer encoding because that would impose a meaningless
# order on SKU ids. Categories are fixed from the FULL dataset vocabulary so that a
# level present only in a test fold does not silently become NaN.
CATEGORICAL = ["sku", "category", "day_of_week"]

NUMERIC = [
    "is_promo_active",
    "base_price",
    "month",
    "day_of_month",
    "time_idx",
    "units_lag_7",
    "units_rolling_7d_avg",  # clean at source since 2026-07-21; window is t-7..t-1
]

FEATURES = CATEGORICAL + NUMERIC

# Explicitly banned. Kept as a named list so the guard in assert_no_target_leakage()
# can fail loudly if anyone adds one back.
#
# `units_rolling_7d_avg` / `units_rolling_7d_avg_cbrt` were on this list until the
# source fix. They are no longer leaky and the raw one is now a live feature.
BANNED_FEATURES = [
    "total_revenue",
    "total_revenue_cbrt",
    "price_per_unit",
    TARGET_RAW,
    TARGET_CBRT,
    # Reconstructed by load_frame() purely so the backtest can still quantify how much
    # the old leak was flattering us. Never a feature -- the guard below enforces it.
    "units_rolling_7d_avg_LEAKY_DIAGNOSTIC",
]

# Not leaky, just not used: monotone transforms of features we already have. Trees
# split on rank order, so `x_cbrt` and `x` give a tree identical splits. Kept out of
# FEATURES to avoid handing XGBoost two perfectly-correlated columns to dilute
# `colsample_bytree` and SHAP attribution across.
UNUSED_REDUNDANT = ["units_lag_7_cbrt", "units_rolling_7d_avg_cbrt"]

DOW_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def signed_cbrt(x):
    """Cube root defined on the whole real line: cbrt(-8) == -2.

    Matches `scripts/skewness_analysis.py:37`. Accepts negatives so a future refund
    or correction row cannot NaN the pipeline.
    """
    return np.cbrt(np.asarray(x, dtype=float))


def inverse_cbrt(x):
    """Inverse of signed_cbrt: plain cubing. Also total on the real line."""
    return np.asarray(x, dtype=float) ** 3


def load_frame() -> pd.DataFrame:
    """Load the frozen dataset and attach leakage-safe derived columns."""
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df.sort_values(["sku", "date"], kind="mergesort").reset_index(drop=True)

    # NOTE: there is deliberately NO `.shift(1)` on units_rolling_7d_avg here.
    # The build script now shifts at source. Re-applying it would double-shift the
    # window to t-8..t-2 and throw away a day of the most recent signal.

    # Calendar/time features. All three are knowable for any future date without
    # observing any future demand -- this is what makes multi-step forecasting
    # possible at all (see backtest.py, horizon strategies).
    df["day_of_month"] = df["date"].dt.day
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days

    # Reconstruction of the OLD, leaky window (t-6..t inclusive). Not a feature -- it
    # is banned. It exists only so the backtest can keep reporting how well a leaky
    # predictor "scores", which is the reference point for spotting the next leak.
    df["units_rolling_7d_avg_LEAKY_DIAGNOSTIC"] = df.groupby("sku")[
        "total_units"
    ].transform(lambda s: s.rolling(7).mean())

    # With the shift at source, the extra NaN row per SKU is already consumed by the
    # `units_lag_7` shift(7) dropna in the build script, so nothing should drop here.
    # Asserted rather than assumed: if a future rebuild reintroduces NaNs we want to
    # know, not to silently lose rows.
    before = len(df)
    df = df.dropna(subset=list(NUMERIC)).reset_index(drop=True)
    df.attrs["rows_dropped_for_nan_features"] = before - len(df)
    assert df.attrs["rows_dropped_for_nan_features"] == 0, (
        f"expected 0 rows dropped, dropped {before - len(df)} -- the frozen file "
        "has NaNs in a feature column"
    )

    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=DOW_ORDER)
    df["sku"] = pd.Categorical(df["sku"], categories=sorted(df["sku"].unique()))
    df["category"] = pd.Categorical(
        df["category"], categories=sorted(df["category"].unique())
    )
    return df


def X_of(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    return df[features or FEATURES].copy()


def assert_no_target_leakage(df: pd.DataFrame) -> dict:
    """Runtime proof of the two leakage claims. Raises if either fails.

    Returns the measured evidence so the training script can print it.
    """
    evidence = {}

    # (a) No banned column may appear in the feature list.
    overlap = sorted(set(FEATURES) & set(BANNED_FEATURES))
    assert not overlap, f"banned column used as a feature: {overlap}"
    evidence["banned_features_used"] = overlap

    # (b/c) TWO-SIDED contract check on the frozen rolling column. Reconstruct both
    #     candidate windows from the target and require an exact match on the
    #     strictly-past one and an explicit NON-match on the leaky one. Checking only
    #     the positive side would let a future rebuild that reverts to the unshifted
    #     window slip through if the two happened to agree on the compared subset.
    g = df.groupby("sku", observed=True)["total_units"]
    incl_today = g.transform(lambda s: s.rolling(7).mean())
    excl_today = g.transform(lambda s: s.shift(1).rolling(7).mean())
    frozen = df["units_rolling_7d_avg"]

    m_exc = excl_today.notna()
    evidence["frozen_rolling_equals_window_EXCLUDING_today"] = bool(
        np.allclose(excl_today[m_exc], frozen[m_exc])
    )
    assert evidence["frozen_rolling_equals_window_EXCLUDING_today"], (
        "units_rolling_7d_avg is not the strictly-past (t-7..t-1) window -- the "
        "build script contract has changed; do NOT train on this"
    )
    evidence["rows_compared_excluding_window"] = int(m_exc.sum())

    m_inc = incl_today.notna()
    evidence["frozen_rolling_equals_window_INCLUDING_today"] = bool(
        np.allclose(incl_today[m_inc], frozen[m_inc])
    )
    assert not evidence["frozen_rolling_equals_window_INCLUDING_today"], (
        "units_rolling_7d_avg matches the window that INCLUDES the target row -- "
        "the target is leaking into its own predictor"
    )

    # (d) Correlation with the target. The leaky window is inflated because 1/7 of it
    #     IS the target; the clean column sits measurably lower. Both are computed so
    #     the gap stays visible in the run log rather than becoming folklore.
    evidence["corr_target_vs_clean_rolling(used)"] = float(frozen.corr(df["total_units"]))
    evidence["corr_target_vs_leaky_rolling(reconstructed, NOT used)"] = float(
        incl_today.corr(df["total_units"])
    )

    # (e) Every feature we use must be computable without the current row's target.
    #     units_lag_7 / units_rolling_7d_avg are shifts; the rest are calendar or
    #     catalogue attributes. Assert the shift columns are never equal to the
    #     target on more rows than chance would give.
    recomputed_lag = df.groupby("sku", observed=True)["total_units"].shift(7)
    m3 = recomputed_lag.notna()  # first 7 rows per SKU can't be re-derived post-trim
    evidence["units_lag_7_is_strictly_past"] = bool(
        np.allclose(recomputed_lag[m3], df["units_lag_7"][m3])
    )
    evidence["units_lag_7_rows_verified"] = int(m3.sum())
    assert evidence["units_lag_7_is_strictly_past"]

    # (f) How often `units_lag_7` is genuinely 7 CALENDAR days back. Not a leak,
    #     but it means the feature is not a weekday-seasonality signal.
    evidence["units_lag_7_pct_exactly_7_calendar_days"] = float(
        (df.groupby("sku", observed=True)["date"].diff(7).dt.days == 7).mean()
    )

    return evidence
