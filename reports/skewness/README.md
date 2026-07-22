# Skewness check on the Trends Arc training data

**Date:** 2026-07-21
**Dataset:** `data/processed/master_df.csv` — 31,736 rows, exported from the
feature-engineering steps in `Data Analysis.ipynb`.

---

## The short version

The demand data is badly right-skewed, and it skews in the one place that
actually matters: the thing we are trying to predict.

`total_units` has a **skewness of 4.69**. For reference, a perfectly symmetric
bell curve has skewness 0; anything past about 1 is usually treated as "skewed
enough to do something about." The median row sells 4 units; the biggest sells
149. Half the dataset sits in a narrow band near zero while a thin tail stretches
out 37× further.

**Why this biases XGBoost specifically.** XGBoost's default regression objective
is squared error, so every row's influence on the trees is proportional to the
*square* of how wrong the prediction is. A row that is off by 100 units pulls on
the model 10,000× harder than a row that is off by 1. With this tail, that is not
a hypothetical:

> **The top 1% of rows account for 44.4% of the total squared error.**
> After cube-rooting, the same 1% account for **15.1%**.

That is the whole argument in one number. Untransformed, roughly 317 blockbuster
rows out of 31,736 get to steer nearly half the model's learning, and the model
pays for it by systematically over-predicting the ordinary low-volume SKUs that
make up the bulk of the catalogue.

**Action taken:** applied a **signed cube-root** transform. New columns written to
`data/processed/master_df_cbrt.csv`.

---

## Measured numbers

Reproduce with: `python scripts/skewness_analysis.py`

| column | skew (raw) | kurtosis (raw) | skew (cube root) | skew (log1p) | max ÷ median |
|---|---|---|---|---|---|
| `total_units` | 4.694 | 42.365 | **1.097** | 0.621 | 37.3 |
| `total_revenue` | 3.540 | 20.455 | **0.884** | 0.036 | 27.1 |
| `units_lag_7` | 4.738 | 43.891 | **1.093** | 0.622 | 37.3 |
| `units_rolling_7d_avg` | 4.148 | 27.429 | **1.508** | 1.022 | 20.8 |

Kurtosis measures how fat the tails are; 0 is normal. A kurtosis of 42 means
extreme values occur far more often than a bell curve would ever produce — these
are not a handful of data-entry outliers, they are a real, recurring December
demand spike.

Full CSV: `reports/skewness/skewness_summary.csv`

## Plots

| file | what it shows |
|---|---|
| `distributions_raw_vs_transformed.png` | Histograms of all four columns: raw vs cube root vs log1p. The raw column is a spike against the left wall; the cube-root column is a recognisable hump. |
| `qq_target.png` | Q-Q plot of `total_units`. Points on the diagonal = normally distributed. Raw curves sharply away at the top end; cube root tracks the line far longer. |
| `target_tail_concentration.png` | What share of all units sold sits in the top slice of rows. |
| `squared_error_leverage.png` | **The key plot.** Cumulative share of squared error by row, raw vs cube root — the 44.4% → 15.1% result above. |

---

## Why cube root and not log

You asked for cube root specifically for robustness against negatives, and that
reasoning holds up:

| | negative input | zero input |
|---|---|---|
| `log(x)` | NaN — crashes | −inf — crashes |
| `log1p(x)` | NaN below −1 | fine |
| `np.cbrt(x)` | **fine** — `cbrt(−8) = −2` | fine |

Cube root is defined and smooth across the entire real line, so a future refund,
returns row, stock correction, or differenced feature that comes through negative
will pass through the pipeline instead of poisoning it with NaN. `log1p` corrects
the skew slightly better on this data (0.62 vs 1.10 on the target) but breaks the
moment any value drops below −1. Given this is going into a live forecasting
service, that trade is worth making.

The transform is also exactly invertible — cube the prediction to get units back.
Verified in the script: round-trip error is **4.26e-14**, i.e. floating-point
noise only. Unlike `expm1(log1p(y))`, cubing introduces no retransformation bias
beyond the usual Jensen gap.

---

## What was and wasn't transformed

Transformed (added as `*_cbrt`, originals kept):
`total_units`, `total_revenue`, `units_lag_7`, `units_rolling_7d_avg`

Left alone, deliberately:
- `is_promo_active` — binary, skew is meaningless
- `month` — cyclic integer, not a magnitude
- `base_price` — constant per SKU, a catalogue attribute not a measurement
- `price_per_unit` — already tightly clustered around `base_price`

## An honest caveat

Gradient-boosted trees split on *rank order*, so monotone transforms of the
**input features** change essentially nothing for XGBoost — the `units_lag_7_cbrt`
and `units_rolling_7d_avg_cbrt` columns are there for consistency and for any
linear baseline, not because they will move the tree metrics. The transform that
genuinely matters is the one on the **target**, because that is what reshapes the
loss surface.

So the cube-root target should be treated as a strong hypothesis backed by the
leverage plot, not as a settled result. The training script is instructed to
**A/B it against the raw target on a real time-based backtest** and keep whichever
actually wins on MAE in original units.

## Not tested

- Whether the extreme rows are genuine demand or data errors. They cluster in
  November/December across many SKUs, which reads as real seasonality, but this
  was not verified against the raw line items.
- Per-category skew. The aggregate number may hide categories that are far worse
  or already well behaved.
- Any actual model fit. No XGBoost was trained as part of this analysis — that is
  the next step.

---

## Correction: a target leak was found, then fixed at source (2026-07-21)

**Found.** After the first model was trained, the frozen dataset turned out to
**leak the target**. The notebook built the rolling feature without a shift —

```python
master_df['units_rolling_7d_avg'] = master_df.groupby('sku')['total_units'].transform(
    lambda x: x.rolling(window=7).mean()
)
```

`rolling(7)` at row *t* averages rows *t-6 … t* **inclusive**, so today's target
contributes 1/7 of its own predictor. The first model worked around this at load
time in `models/features.py`.

**Fixed.** The leak is now corrected at source — in notebook cell `dcfe85bd` and
in `scripts/build_dataset.py` — by shifting before rolling:

```python
lambda x: x.shift(1).rolling(window=7).mean()
```

Verified after rebuild (`python scripts/build_dataset.py`):

```
matches EXCLUDING-today window: True
matches INCLUDING-today window: False
corr with target: 0.7201
```

Correlation with the target fell **0.794 → 0.720**, exactly the drop the training
code predicted. Row count is unchanged at 31,736 — the extra NaN row per SKU was
already being dropped by the `units_lag_7` shift, so nothing was lost.

**What changed in the numbers above.** Only `units_rolling_7d_avg`, and only
slightly: skew 4.148 → 4.155, kurtosis 27.43 → 27.58, p99 31.379 → 31.143. The
table has been left at the original values with this note rather than silently
overwritten. Every conclusion about the target holds — `total_units` was never
touched by this leak.

**Still outstanding:** `units_lag_7` is misnamed. It is 7 *rows* back, which is
genuinely 7 calendar days only 38.7% of the time, because SKUs have no rows on
zero-sale days. Not a leak (it is strictly in the past), but any weekday-seasonality
story told about that feature would be wrong. Not fixed.

## Files produced

| path | what |
|---|---|
| `scripts/build_dataset.py` | Reproduces the notebook's feature engineering, exports the canonical dataset |
| `scripts/skewness_analysis.py` | This analysis: stats, plots, transform |
| `data/processed/master_df.csv` | Exported notebook dataframe — 31,736 × 12 |
| `data/processed/master_df_cbrt.csv` | Same + 4 `*_cbrt` columns — 31,736 × 16 |
| `reports/skewness/*.png` | Plots listed above |
