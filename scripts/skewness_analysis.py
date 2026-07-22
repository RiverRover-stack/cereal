"""Skewness check on the exported training dataset, plus the cube-root transform.

Run:  python scripts/skewness_analysis.py
Out:  reports/skewness/*.png
      data/processed/master_df_cbrt.csv
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "master_df.csv"
OUT_CSV = ROOT / "data" / "processed" / "master_df_cbrt.csv"
FIG = ROOT / "reports" / "skewness"

# Heavy-tailed count/money columns. is_promo_active (binary), month, base_price
# (a per-SKU constant) and price_per_unit are excluded deliberately.
SKEW_CANDIDATES = [
    "total_units",
    "total_revenue",
    "units_lag_7",
    "units_rolling_7d_avg",
]
TARGET = "total_units"


def signed_cbrt(s: pd.Series) -> pd.Series:
    """Cube root that is defined for negative inputs: cbrt(-8) == -2.

    Unlike log1p this never produces NaN/-inf, so a future negative value
    (a return, a correction, a differenced feature) cannot break the pipeline.
    """
    return np.cbrt(s.astype(float))


def describe_skew(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        raw = df[c].astype(float)
        rows.append(
            {
                "column": c,
                "skew_raw": stats.skew(raw),
                "kurtosis_raw": stats.kurtosis(raw),
                "skew_cbrt": stats.skew(signed_cbrt(raw)),
                "kurtosis_cbrt": stats.kurtosis(signed_cbrt(raw)),
                "skew_log1p": stats.skew(np.log1p(raw)),
                "max_over_median": raw.max() / raw.median(),
                "p99": raw.quantile(0.99),
                "max": raw.max(),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def plot_distributions(df: pd.DataFrame, cols: list[str]) -> None:
    fig, axes = plt.subplots(len(cols), 3, figsize=(16, 4 * len(cols)))
    for i, c in enumerate(cols):
        raw = df[c].astype(float)
        panels = [
            (raw, f"{c} — raw", "#c0392b"),
            (signed_cbrt(raw), f"{c} — cube root", "#27ae60"),
            (np.log1p(raw), f"{c} — log1p", "#2980b9"),
        ]
        for j, (vals, title, colour) in enumerate(panels):
            ax = axes[i, j]
            sns.histplot(vals, bins=60, ax=ax, color=colour, edgecolor=None)
            ax.set_title(f"{title}\nskew = {stats.skew(vals):.2f}", fontsize=10)
            ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIG / "distributions_raw_vs_transformed.png", dpi=120)
    plt.close(fig)


def plot_qq(df: pd.DataFrame) -> None:
    """Q-Q plots of the target: how far from normal before vs after."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    raw = df[TARGET].astype(float)
    for ax, vals, title in [
        (axes[0], raw, "total_units — raw"),
        (axes[1], signed_cbrt(raw), "total_units — cube root"),
    ]:
        stats.probplot(vals, dist="norm", plot=ax)
        ax.set_title(f"{title}\nskew = {stats.skew(vals):.2f}")
    fig.tight_layout()
    fig.savefig(FIG / "qq_target.png", dpi=120)
    plt.close(fig)


def plot_tail_weight(df: pd.DataFrame) -> None:
    """How much of total volume sits in the top 1% of rows — the bias risk."""
    raw = df[TARGET].astype(float).sort_values()
    cum = raw.cumsum() / raw.sum()
    pct_rows = np.arange(1, len(raw) + 1) / len(raw) * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(pct_rows, cum * 100, color="#8e44ad")
    ax.axvline(99, ls="--", color="grey")
    top1 = raw.tail(int(len(raw) * 0.01)).sum() / raw.sum() * 100
    ax.set_title(f"Cumulative share of units by row\ntop 1% of rows = {top1:.1f}% of all units")
    ax.set_xlabel("percentile of rows (sorted by units)")
    ax.set_ylabel("cumulative % of total units")
    fig.tight_layout()
    fig.savefig(FIG / "target_tail_concentration.png", dpi=120)
    plt.close(fig)


def plot_residual_leverage(df: pd.DataFrame) -> None:
    """Squared-error leverage: contribution to MSE if the model predicted the mean.

    This is the mechanism by which skew biases XGBoost's default objective.
    """
    raw = df[TARGET].astype(float)
    cb = signed_cbrt(raw)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, vals, title in [(axes[0], raw, "raw"), (axes[1], cb, "cube root")]:
        err2 = (vals - vals.mean()) ** 2
        share = np.sort(err2)[::-1].cumsum() / err2.sum()
        ax.plot(np.arange(1, len(share) + 1) / len(share) * 100, share * 100, color="#d35400")
        top1 = share[int(len(share) * 0.01)] * 100
        ax.set_title(f"{title}: top 1% of rows drive {top1:.1f}% of squared error")
        ax.set_xlabel("% of rows (worst first)")
        ax.set_ylabel("cumulative % of squared error")
        ax.set_xlim(0, 20)
    fig.tight_layout()
    fig.savefig(FIG / "squared_error_leverage.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC, parse_dates=["date"])

    summary = describe_skew(df, SKEW_CANDIDATES)
    print("\n=== Skewness summary ===")
    print(summary.round(3).to_string())

    plot_distributions(df, SKEW_CANDIDATES)
    plot_qq(df)
    plot_tail_weight(df)
    plot_residual_leverage(df)

    # Apply the transform. Original columns are kept so nothing downstream
    # loses information and predictions can be cubed back to units.
    out = df.copy()
    for c in SKEW_CANDIDATES:
        out[f"{c}_cbrt"] = signed_cbrt(out[c])

    # Sanity check: the transform round-trips and survives negatives.
    back = out[f"{TARGET}_cbrt"] ** 3
    assert np.allclose(back, df[TARGET]), "cube root does not round-trip"
    assert np.isfinite(signed_cbrt(pd.Series([-8.0, -1.0, 0.0]))).all()
    print(f"\nround-trip max abs error: {np.abs(back - df[TARGET]).max():.2e}")
    print("signed_cbrt([-8,-1,0]) =", signed_cbrt(pd.Series([-8.0, -1.0, 0.0])).tolist())

    out.to_csv(OUT_CSV, index=False)
    summary.round(4).to_csv(FIG / "skewness_summary.csv")
    print(f"\nwrote {OUT_CSV}  shape={out.shape}")
    print(f"wrote plots to {FIG}")


if __name__ == "__main__":
    main()
