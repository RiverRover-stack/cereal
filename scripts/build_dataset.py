"""Reproduce the feature engineering from `Data Analysis.ipynb` and export the
resulting dataframe as the canonical training dataset.

Run:  python scripts/build_dataset.py
Out:  data/processed/master_df.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "processed" / "master_df.csv"


def build() -> pd.DataFrame:
    df_line_items = pd.read_csv(DATA / "sales_line_items.csv")
    df_events = pd.read_csv(DATA / "events.csv")
    df_catalogue = pd.read_csv(DATA / "catalogue.csv")

    df_line_items["date"] = pd.to_datetime(df_line_items["date"])
    df_events["date"] = pd.to_datetime(df_events["date"])

    daily_sku = (
        df_line_items.groupby(["date", "sku"])
        .agg(total_units=("units_sold", "sum"), total_revenue=("revenue", "sum"))
        .reset_index()
    )

    master_df = pd.merge(daily_sku, df_events[["date", "event"]], on="date", how="left")
    master_df["is_promo_active"] = master_df["event"].notna().astype(int)
    master_df = pd.merge(
        master_df,
        df_catalogue[["sku", "category", "base_price"]],
        on="sku",
        how="left",
    )

    master_df.sort_values(["sku", "date"], inplace=True)

    master_df["units_lag_7"] = master_df.groupby("sku")["total_units"].shift(7)

    # .shift(1) BEFORE .rolling(7): without it the window is rows t-6..t inclusive,
    # so today's total_units supplies 1/7 of its own predictor and the model leaks.
    # With the shift the window is t-7..t-1 -- strictly past.
    master_df["units_rolling_7d_avg"] = master_df.groupby("sku")["total_units"].transform(
        lambda x: x.shift(1).rolling(window=7).mean()
    )

    master_df.drop(columns=["event"], inplace=True)
    master_df.dropna(inplace=True)
    master_df.reset_index(drop=True, inplace=True)

    master_df["price_per_unit"] = master_df["total_revenue"] / master_df["total_units"]
    master_df["month"] = master_df["date"].dt.month
    master_df["day_of_week"] = master_df["date"].dt.day_name()

    return master_df


if __name__ == "__main__":
    master_df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  shape={master_df.shape}")
    print(master_df.dtypes)
