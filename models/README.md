# Trends Arc demand forecaster — training, backtest and honest limits

**Date:** 2026-07-21 (revised same day after the rolling-window leak was fixed at source)
**Data:** `data/processed/master_df_cbrt.csv` (frozen; 31,736 rows, 40 SKUs,
2023-01-08 … 2025-12-30)
**Reproduce everything in this file with one command:**

```
backend/.venv/Scripts/python.exe models/train.py
```

Runtime ~28s. Every number below is pasted from that run's stdout. Nothing is
estimated or recalled.

---

## Revision note — what changed and why

The first version of this report was produced against a dataset where
`units_rolling_7d_avg` leaked the target (details in section 1a). That has since been
fixed in `scripts/build_dataset.py` and both processed CSVs were regenerated. This
report has been re-run end to end against the fixed data.

**Numbers that moved because of the fix are shown as `before → after` throughout, and
two conclusions reversed.** They are called out where they occur:

1. The **feature-set recommendation reversed** (section 3). Before the fix the
   no-lag model was competitive; now the lag features carry real signal and the
   full feature set is clearly better.
2. The **cube-root-vs-raw sign flip did NOT go away** (section 4) but its magnitude
   collapsed, and further testing showed the remaining difference is hyperparameter
   noise rather than a property of the transform.

A third, unrelated methodology change was made in the same pass: **model selection now
runs on its own block of four earlier folds** that never appear in the reported
backtest (section 3). The previous version selected hyperparameters and feature set on
reported folds 1-3, which biased those folds and, once the data was fixed, caused the
ablation to pick a model that lost across all five reported folds.

Two numbers that did **not** move are a useful cross-check that the fix and the old
workaround were doing the same thing: `baseline_rolling7_shifted` mean MAE is
**4.1079 both before and after**, and `baseline_flat_last7` is **4.6630 both before
and after**. The old local `.shift(1)` and the new source-level shift produce an
identical column.

---

## Headline

**The model is a real but modest improvement over doing nothing, and only if you
compare it against something that is actually deployable.**

Averaged over five 30-day rolling-origin backtest windows, mean absolute error in
units per SKU per day:

| model | MAE (before → after fix) | RMSE | MAPE | fair 30-day comparison? |
|---|---|---|---|---|
| **`xgb_full_recursive`** | 4.4099 → **4.3395** | 7.4600 | **63.89%** | yes |
| `xgb_known_covariates_only` | 4.4379 → 4.5856 | 7.4052 | 72.72% | yes |
| `baseline_flat_last7` (do-nothing) | 4.6630 → 4.6630 | 7.6086 | 90.65% | yes |
| `baseline_rolling7_shifted` | 4.1079 → 4.1079 | 6.6359 | 78.82% | **no — one-step-ahead** |
| `baseline_lag7` | 5.3321 → 5.3321 | 8.5134 | 94.04% | no — one-step-ahead |
| `xgb_full_oracle_lags` | 4.0083 → 3.9330 | 6.5883 | 64.92% | no — not deployable |
| `baseline_rolling7_LEAKY` | 3.5816 → 3.5816 | 5.7911 | 68.20% | no — reads the answer |

Against the only baseline that forecasts 30 days the way production would, the full
model wins by **-6.9% MAE** (was -5.4% before the fix) and by a wide margin on
percentage error (63.9% vs 90.7% MAPE). Against a one-step-ahead baseline it *loses*
by +5.6% MAE — but that baseline reads actual sales from inside the forecast window,
so it is not a comparison of two forecasters.

**Recommended configuration: the full feature set (with lags), cube-root target.**
This is *not* what the automated selection rule picks — see section 3, where that
disagreement is quantified rather than hidden.

**Three caveats that matter more than the headline:**

1. A **MAPE of 64%** is not a good forecast in absolute terms. The median SKU-day is
   4 units; being off by ~4.3 units on a 4-unit day is the normal case. This model is
   useful for direction and rough level, not for reorder quantities.
2. The advantage is concentrated in **days 1-7** of the horizon and gone by day 14
   (table below). The 30-day figure is an average of "clearly better early" and "no
   better late".
3. Fold 5 (December) is the worst fold for every model tested. Nothing here handles
   the holiday spike.

---

## Jargon, once

- **MAE** — mean absolute error. Average size of the miss, in units. Interpretable.
- **RMSE** — root mean squared error. Same idea but squares the errors first, so one
  huge miss hurts far more than several small ones. Used here to catch blown promo
  spikes that MAE smooths over.
- **MAPE** — mean absolute percentage error. The miss as a % of actual demand.
- **Rolling-origin backtest** — fit on everything up to a date, forecast the next 30
  days, move the date forward, repeat. The time-series replacement for cross-validation.
- **Leakage** — a feature that secretly contains the answer. Makes validation scores
  look great and the live model fail.
- **SHAP** — a method that splits a single prediction into per-feature contributions
  that sum to the prediction. Exact and fast on tree models.
- **Recursive forecasting** — to predict day 8 you need day 7's demand, which you do
  not have, so you feed in your own day-7 prediction. Errors accumulate.

---

## 1. The leakage findings — read this part first

Three problems were found in the frozen dataset. All three are handled in code and
verified at runtime by `models/features.py::assert_no_target_leakage`, which raises
if any check fails, so the training run cannot silently proceed with a leak.

### 1a. `units_rolling_7d_avg` leaked the target — NOW FIXED AT SOURCE

**The original problem.** `scripts/build_dataset.py` computed:

```python
lambda x: x.rolling(window=7).mean()           # OLD -- leaky
```

No `.shift()`. `rolling(7)` at row *t* averages rows *t-6 … t* **inclusive**, so the
value being predicted supplied one seventh of its own predictor.

**Now.** The build script computes:

```python
lambda x: x.shift(1).rolling(window=7).mean()  # NEW -- window is t-7..t-1
```

The compensating `.shift(1)` this module used to apply at load time has been
**removed**. Keeping it would have double-shifted the window to *t-8 … t-2* and
silently thrown away a day of the most recent signal.

**The guard caught the contract change, which is the whole point of having it.** Run
against the rebuilt file before any code was touched, training refused to start:

```
File "models/features.py", line 160, in assert_no_target_leakage
    assert evidence["frozen_rolling_equals_window_including_today"], (
AssertionError: expected the frozen rolling column to include the current day
```

The assertions are now inverted to match the new contract and are deliberately
**two-sided**: the column must equal the excluding-today window *and* must not equal
the including-today one, so a regression in either direction fails the run instead of
quietly degrading it. Current output:

```
frozen_rolling_equals_window_EXCLUDING_today: True
rows_compared_excluding_window: 31456
frozen_rolling_equals_window_INCLUDING_today: False
corr_target_vs_clean_rolling(used): 0.7200998403202493
corr_target_vs_leaky_rolling(reconstructed, NOT used): 0.7945789970442051
```

Correlation with the target is 0.720 for the clean column against 0.794 for the leaky
window — the drop predicted in the previous revision, now confirmed on fixed data.

**Row count, confirmed.** The 40 rows previously dropped for the local shift are no
longer dropped: `rows dropped for NaN features: 0`, and the dataset is back to the
full **31,736 rows (was 31,696)**. The extra NaN row per SKU was already being
consumed by the `units_lag_7` shift in the build script. This is asserted rather than
assumed — if a future rebuild reintroduces NaNs, `load_frame()` raises.

**How large was the flattery?** `baseline_rolling7_LEAKY` in the backtest is the old
leaky window — reconstructed on purpose as a banned diagnostic column — used as a
prediction: mean MAE **3.5816**, better than every honest model in the table,
including the non-deployable oracle-lag model. That number is kept on the record
deliberately. If a future model ever scores near it, suspect a leak before
celebrating.

### 1b. `total_revenue` and `price_per_unit` are excluded as direct target leakage

`total_revenue` is summed from the same line items as `total_units` on the same row,
and `scripts/build_dataset.py:51` defines `price_per_unit = total_revenue / total_units`.
Both are functions of the target for the row being predicted; either would give near-
perfect scores and a useless model. Their `_cbrt` variants go with them.

They are listed in `BANNED_FEATURES` and an assertion fails if any appears in the
feature list. Run output:

```
EXCLUDED as target leakage / target itself: ['total_revenue', 'total_revenue_cbrt',
'price_per_unit', 'total_units', 'total_units_cbrt',
'units_rolling_7d_avg_LEAKY_DIAGNOSTIC']
EXCLUDED as redundant (monotone transforms): ['units_lag_7_cbrt',
'units_rolling_7d_avg_cbrt']
banned_features_used: []
```

`units_rolling_7d_avg` and `units_rolling_7d_avg_cbrt` were on the banned list in the
previous revision. After the source fix they are no longer leaky, so the raw one is
now a live feature and the `_cbrt` one moved to a separate "redundant" list — trees
split on rank order, so a monotone transform of a feature gives identical splits, and
including both would only dilute `colsample_bytree` and split SHAP attribution across
two identical columns.

### 1c. `units_lag_7` is safe but misnamed (kept, documented)

It is `groupby("sku").shift(7)` — seven **rows** back, not seven **days** back. A SKU
only has a row on days it sold something, so:

```
units_lag_7_pct_exactly_7_calendar_days: 0.38654719838465423
```

Only **38.7%** of the time is it genuinely a week ago. It is still strictly in the
past, so there is no leakage, but it is not a weekday-seasonality signal and should
not be described as one in any customer-facing explanation. Verified against a
re-derivation on 31,456 rows (was 31,416 — the 40 rows recovered by the fix).
Left as-is; renaming it is out of scope.

### 1d. No random splitting anywhere

`date` is not unique — this is one row per SKU per day, 40 SKUs across 1,087 dates.
A random row split would put the same calendar day in both train and test. Every
split in `models/backtest.py` is a **calendar date cut**: `train = df[df.date < cut]`,
`test = df[df.date >= cut]`. `KFold`, `train_test_split` and `shuffle` do not appear
in the codebase. The early-stopping holdout is also time-ordered — the last 30 days
of the training window, never a random sample. Model selection now runs on a block of
folds strictly earlier than every reported fold, asserted at runtime:

```
assert sel_folds[-1].test_end < folds[0].test_start, "selection/report overlap"
```

---

## 2. The 30-day horizon decision

Of the ten candidate features, eight (`sku`, `category`, `day_of_week`,
`is_promo_active`, `base_price`, `month`, `day_of_month`, `time_idx`) are known in
advance for any future date. Two (`units_lag_7`, `units_rolling_7d_avg`) are not —
forecasting day *t+8* needs demand you have not observed.

All three standard answers were implemented and measured rather than one being
asserted:

- **Recursive** — feed each prediction back to build the next lag window. Deployable.
- **Known-covariates only** — drop both lag features. Deployable, cannot compound.
- **Oracle lags** — feed true lags. **Not deployable**; reported only as a diagnostic
  upper bound.

The backtest forecasts a full 30 days exactly the way production would, per fold.

### Measured cost of recursion

Oracle lags mean MAE **3.9330** (was 4.0083); the same model forecasting recursively
**4.3395** (was 4.4099). So the lag features are worth about **0.41 units of MAE**
when you have them, and recursion gives back roughly all of that. Both figures
improved slightly with the fix and the gap between them is essentially unchanged —
the compounding cost was never an artefact of the leak.

### Where the advantage actually lives

MAE by position in the 30-day window, pooled over folds (after the fix):

| bucket | xgb_full_recursive | xgb_known_only | baseline_flat_last7 | baseline_rolling7 (1-step) |
|---|---|---|---|---|
| d1-7 | **3.437** | 3.912 | 3.754 | 3.947 |
| d8-14 | 3.792 | 4.145 | 4.054 | **3.377** |
| d15-21 | 4.088 | 4.555 | 3.939 | **3.443** |
| d22-30 | 5.542 | 5.379 | 6.321 | **5.238** |

The recursive model is the best of the deployable options in week 1 and loses that
lead from day 8. On MAPE it stays ahead of both baselines in every bucket
(66.9/67.2/64.2/59.3 vs 90.7/96.2/89.1/88.6 for flat).

The fix widened the recursive model's week-1 lead over the no-lag model from
3.431 vs 3.784 to **3.437 vs 3.912** — as expected, since the lag features now carry
a full extra day of clean signal.

**Recommendation:** ship this as a 7-14 day forecast with a widening uncertainty band
beyond that, or accept that days 15-30 are a level estimate rather than a forecast.

---

## 3. Hyperparameters — searched, not asserted

Grid: `max_depth ∈ {3,4,5,6,8}` × `learning_rate ∈ {0.05,0.1}`, with
`subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `reg_lambda=1.0`,
`n_estimators=600` capped by early stopping (50 rounds) on a time-ordered holdout.

### Methodology change: selection folds are now disjoint from reported folds

The previous revision scored the sweep and the ablation on **reported folds 1-3**.
That is a soft form of testing on train — it biases those three folds, and once the
data was fixed it caused a concrete failure (below). Selection now runs on its own
block of four earlier 30-day windows that never appear in the reported backtest:

```
SELECTION folds (tuning only, never reported):
  fold 1: train 2023-01-08..2025-04-04   test 2025-04-05..2025-05-04 (n=865)
  fold 2: train 2023-01-08..2025-05-04   test 2025-05-05..2025-06-03 (n=787)
  fold 3: train 2023-01-08..2025-06-03   test 2025-06-04..2025-07-03 (n=772)
  fold 4: train 2023-01-08..2025-07-03   test 2025-07-04..2025-08-02 (n=939)

REPORTED folds (never used for selection):
  fold 1: train 2023-01-08..2025-08-02   test 2025-08-03..2025-09-01 (n=981)
  ... through ...
  fold 5: train 2023-01-08..2025-11-30   test 2025-12-01..2025-12-30 (n=1152)
```

Non-overlap is asserted at runtime, not eyeballed.

### The sweep

```
 max_depth  learning_rate  train_MAE  backtest_MAE  backtest_RMSE  backtest_MAPE  gap(bt-train)  best_iter
         5           0.05     2.4129        2.7351         4.6728        63.8175         0.3223     128.75
         6           0.05     2.3317        2.7378         4.6180        63.9376         0.4061     124.50
         4           0.05     2.4810        2.7639         4.7490        63.9844         0.2830     153.50
         6           0.10     2.4144        2.7752         4.7805        65.5772         0.3608      41.75
         3           0.05     2.5594        2.7797         4.7920        65.2539         0.2203     177.25
         5           0.10     2.4626        2.7959         4.7613        62.4718         0.3333      53.25
         8           0.10     2.2669        2.8197         4.8360        65.3939         0.5529      30.25
         8           0.05     2.2081        2.8213         4.8400        65.3094         0.6132      74.75
         4           0.10     2.4282        2.8244         4.8803        62.8486         0.3962     114.50
         3           0.10     2.5533        2.8727         4.9737        63.5961         0.3194      97.25
```

Selected: **`max_depth=5, learning_rate=0.05`** (was `max_depth=6, learning_rate=0.1`
before the fix). Note these MAE values are lower than the previous revision's sweep
table purely because the selection folds are now spring/summer windows, which are
easier than the autumn ones used before — they are not comparable to the old numbers
and should not be read as an improvement.

The memorisation check still works. At `learning_rate=0.05`, going from depth 5 to
depth 8 lowers training MAE (2.4129 → 2.2081 — the model fits the training data
better) while *raising* backtest MAE (2.7351 → 2.8213). The train-to-backtest gap
nearly doubles, 0.3223 → 0.6132. That is memorising rather than learning, and it is
why depth 8 was rejected despite the better training fit. Depth 8 is the worst gap in
the table at both learning rates.

Honest note on the spread: best and worst configurations differ by 0.138 MAE
(2.7351 vs 2.8727), about 5%. Depth matters less here than the horizon strategy does.

### Feature-set ablation — and a conclusion that reversed

```
          feature_set  n_features    MAE   RMSE    MAPE
known_covariates_only           8 2.6824 4.4597 69.1169
                 full          10 2.7351 4.6728 63.8175
          no_time_idx           9 2.7892 4.8087 58.0780
```

`known_covariates_only` wins on the selection folds by 2.0% MAE, so that is what the
script selects and saves.

**On the reported folds it loses, and the fix is what changed this.** The lag
features now carry a full extra day of clean signal, so the model that uses them
improved while the model that ignores them got relatively worse:

| feature set | mean MAE before fix | mean MAE after fix | vs `baseline_flat_last7` |
|---|---|---|---|
| full (with lags), recursive | 4.4099 | **4.3395** | **-6.9%** (was -5.4%) |
| known_covariates_only | 4.4379 | 4.5856 | -1.7% (was -4.8%) |

Before the fix the two were within 0.6% of each other and the no-lag model was the
better bet. After the fix the full feature set is **5.7% better** on reported folds,
and it is better on MAPE by a wide margin (63.9% vs 72.7%) and on worst-fold MAE
(6.685 vs 7.821). **Recommendation reversed: use the full feature set.**

**The automated rule still picks the wrong one, and that is reported rather than
patched.** Selecting on MAE over four clean, disjoint folds is a defensible rule and
it still misfires here, because the selection folds are spring/summer windows where
lag features matter less. Options considered and rejected:

- Re-select on the reported folds — that is tuning on the test set.
- Change the rule to MAE-with-a-MAPE-tiebreak after seeing that MAE-only misfires —
  a rule chosen because it gives the answer you already wanted is not validation.

So the script keeps the honest rule and saves its choice; this report recommends the
full feature set and states plainly that **the recommendation is informed by the
reported folds and therefore needs re-validating on genuinely fresh data.** Both
configurations are measured in section 5 so nothing is hidden behind the choice.

### Final artefact fitted without early stopping — and why

With early stopping the holdout is the last 30 days of the *full* dataset, which is
December — the demand spike. Validation error bottomed out after 5 trees and produced
a badly underfit artefact. The final model therefore uses a fixed tree count equal to
the median stopping point the backtest folds actually chose:

```
per-fold early-stopping iterations: [369, 77, 36, 264, 49] -> median+1 = 78
```

(was `[83, 28, 13, 99, 40] -> 41`; the lower learning rate accounts for most of the
increase.) Note the spread — 36 to 369, wider than before. Early stopping on this data
is unstable, which is a further reason not to rely on a single holdout.

---

## 4. A/B: raw target vs cube-root target

The skewness report predicted that a raw target would over-predict low-volume SKUs.
**It does, and the measurement confirms the mechanism** — but the MAE difference is
small, and the answer to the question you asked is: **the sign flip survived the fix,
though it shrank a lot, and follow-up testing shows what remains is noise.**

Pooled over all five reported folds, both scored in **original units** (cube-root
predictions cubed back first):

| feature set | target | MAE before fix | MAE after fix | RMSE | MAPE |
|---|---|---|---|---|---|
| full | raw | **4.3150** | **4.3696** | 7.5014 | 82.67 |
| full | cbrt | 4.4952 | 4.4202 | 8.1038 | **63.79** |
| known_covariates_only | raw | 4.6444 | 4.8924 | 8.1712 | 90.95 |
| known_covariates_only | cbrt | **4.5348** | **4.6933** | 8.2372 | **72.68** |

The decision rule was fixed in advance — pooled MAE in original units on the selected
feature set — which gives **cube-root, by -0.1991 MAE (-4.1%)**, a slightly wider
margin than the -2.4% before the fix.

**On the sign flip specifically.** Raw still wins on the full feature set, but the
margin collapsed from **+4.2% to +1.2%**. More usefully, an intermediate run during
this work — same fixed data, different hyperparameters (`depth=6, lr=0.1` instead of
`depth=5, lr=0.05`) — gave `full/raw 4.4885` vs `full/cbrt 4.4796`, i.e. **cube-root
winning by 0.2%**. The sign on the full feature set therefore flips with the
hyperparameters alone. That is the definition of noise, and it settles the question:
**on MAE the two targets are indistinguishable on the full feature set, and cube-root
is genuinely better on the no-lag set.**

On **MAPE the cube-root target wins decisively and consistently in every condition
tested, before and after the fix** (-18.9 points on the selected set, -18.9 on the
full set). If percentage error is what the product surfaces to a seller, cube-root is
the clear choice and the MAE comparison is a wash. That conclusion strengthened with
the fix.

### Per-decile MAE — the skewness report's prediction, tested

Rows bucketed into deciles by *actual* units. `bias > 0` means over-prediction.

Feature set `known_covariates_only`, after the fix:

```
        actual_mean  MAE_raw  bias_raw  MAE_cbrt  bias_cbrt      n
decile
1             1.088    3.625     3.622     2.747      2.747  521.0
2             2.000    3.136     3.134     2.120      2.109  520.0
3             2.981    2.268     2.127     1.534      1.209  520.0
4             3.956    2.400     1.875     1.743      0.806  520.0
5             5.063    2.176     0.957     1.827     -0.083  521.0
6             6.390    2.583     0.460     2.442     -0.794  520.0
7             8.192    2.897    -0.359     3.022     -1.854  520.0
8            11.121    4.737    -1.861     4.999     -3.591  520.0
9            16.000    7.988    -3.341     7.697     -5.976  520.0
10           34.246   17.100   -14.095    18.784    -17.939  521.0
```

**Confirmed, and the pattern is unchanged by the fix.** The raw-target model
over-predicts the bottom decile by **+3.62 units** (was +3.26) on an actual mean of
1.09 — it predicts over four times the true demand for the quietest SKU-days. The
cube-root target cuts that to **+2.75** (was +2.60) and cuts bottom-decile MAE by 24%
(3.625 → 2.747; was a 20% cut). Deciles 1-5 all improve, as before. The raw target's
low-volume over-prediction got slightly *worse* after the fix, so this is the more
robust of the two arguments for cube-root.

**The trade is visible too, and also unchanged:** the cube-root model is *worse* on
the top decile (MAE 18.784 vs 17.100; bias -17.939 vs -14.095). Shrinking the tail's
influence means under-calling blockbuster days harder. Both models under-predict the
top decile severely — see limitations.

---

## 5. Rolling-origin backtest — every fold

Expanding train window, five disjoint 30-day test windows:

```
fold 1: train 2023-01-08..2025-08-02 (n=26533)   test 2025-08-03..2025-09-01 (n=981)
fold 2: train 2023-01-08..2025-09-01 (n=27514)   test 2025-09-02..2025-10-01 (n=995)
fold 3: train 2023-01-08..2025-10-01 (n=28509)   test 2025-10-02..2025-10-31 (n=1020)
fold 4: train 2023-01-08..2025-10-31 (n=29529)   test 2025-11-01..2025-11-30 (n=1055)
fold 5: train 2023-01-08..2025-11-30 (n=30584)   test 2025-12-01..2025-12-30 (n=1152)
```

(Train counts are each 40 rows larger than before the fix — the recovered rows.
Test windows are identical, so the fold-level comparisons above are like-for-like.)

MAE per fold, after the fix — the mean alone would hide folds 4 and 5:

| model | f1 | f2 | f3 | f4 | f5 | mean | worst |
|---|---|---|---|---|---|---|---|
| `xgb_full_recursive` | 3.348 | 2.873 | 3.444 | 5.347 | 6.685 | **4.340** | 6.685 |
| `xgb_known_covariates_only` | 3.287 | 2.780 | 3.384 | 5.655 | 7.821 | 4.586 | 7.821 |
| `baseline_flat_last7` | 3.335 | 3.660 | 3.182 | 6.119 | 7.018 | 4.663 | 7.018 |
| `baseline_rolling7_shifted` (1-step) | 3.172 | 2.876 | 3.084 | 5.394 | 6.014 | 4.108 | 6.014 |
| `baseline_lag7` (1-step) | 3.982 | 3.844 | 4.004 | 6.899 | 7.932 | 5.332 | 7.932 |
| `xgb_full_oracle_lags` (not deployable) | — | — | — | — | — | 3.933 | 5.940 |
| `baseline_rolling7_LEAKY` (reads answer) | — | — | — | — | — | 3.582 | 5.342 |

**Folds 4 and 5 (November and December) are roughly twice as hard as folds 1-3 for
every model** — unchanged by the fix. Fold 4 RMSE for the recursive model is 11.32
against a fold-2 RMSE of 4.46. That gap between MAE (5.35) and RMSE (11.32) in fold 4
says the damage is a small number of very large misses — the Black Friday / holiday
spikes — not a general degradation. A model averaging 4.34 MAE that has an
11.32-RMSE month in it is not a model you should quote a single accuracy number for.

Where the fix helped most is fold 5 (December): the full recursive model went
**6.894 → 6.685** while the no-lag model went **7.385 → 7.821**. Clean recent-demand
signal matters most exactly when the level is moving fastest.

### On the baselines

`baseline_lag7` and `baseline_rolling7_shifted` were requested in the brief. Both are
**one-step-ahead** baselines: on day 20 of a 30-day window they read actual sales from
days 13-19 of that same window. Production would not have that data. They are
reported because they were asked for and because they are a fair yardstick for a
next-day forecast, but they are labelled in the output and they are not the
comparison the headline uses. `baseline_flat_last7` — each SKU's mean of its last 7
observations at forecast time, held flat for 30 days — is the like-for-like one.

---

## 6. Directional accuracy and F1

Derived post-hoc from the regressor's continuous output, not a second model. For each
forecast row, "up" means the forecast exceeds that SKU's previous observed value.

Scored on the model the script selects (`known_covariates_only`, cube-root):

```
 fold  accuracy  f1_up  f1_down  f1_macro  actual_up_share  pred_up_share    n
    1    0.6901 0.6337   0.7314    0.6826           0.4608         0.3853  981
    2    0.7327 0.6928   0.7633    0.7281           0.4402         0.4302  995
    3    0.6843 0.6632   0.7030    0.6831           0.4510         0.4863 1020
    4    0.6199 0.6678   0.5559    0.6118           0.4436         0.7005 1055
    5    0.6762 0.5046   0.7595    0.6321           0.4444         0.2092 1152
```

Mean accuracy **0.6806** (was 0.6890 before the fix), mean macro-F1 **0.6675** (was
0.6772), with a class balance of **44.8% up / 55.2% down** — unchanged, and close
enough to even that accuracy is not being inflated by a lopsided series. Calling ~68%
of daily moves correctly is meaningfully better than a coin flip and is probably a
more useful trust signal for a seller than dollar precision, given the ~64-73% MAPE.

The small drop is within the fold-to-fold spread and should not be read as the fix
making directionality worse. Note it is measured on the no-lag model the rule picks,
not the full-feature model this report recommends.

The instability is in *when* it says up, and it did not improve. Predicted-up share
swings from 20.9% (fold 5) to 70.1% (fold 4) against an actual that barely moves
(44-46%) — a slightly wider swing than before the fix (21.2% to 61.3%). In fold 5 the
model became strongly bearish and `f1_up` collapsed to 0.505 while `f1_down` rose to
0.760 — it called the downs and missed the ups. Macro-F1 is reported precisely so this
shows up instead of being averaged away by accuracy.

---

## 7. Explainability

`shap.TreeExplainer` (shap 0.50.0) on the fitted model, applied to the last fold's
1,152 forecast rows, aggregated per feature:

```
        feature  mean_abs_shap  mean_signed_shap
       time_idx         0.3365            0.3365
          month         0.2718            0.2718
            sku         0.2549           -0.0428
    day_of_week         0.0522           -0.0048
       category         0.0375           -0.0021
     base_price         0.0191           -0.0029
is_promo_active         0.0160            0.0090
   day_of_month         0.0123           -0.0018
```

Templated sentences, generated by `feature name + sign + magnitude bucket`, in SHAP's
own order with no re-ranking:

```
- the long-run trend is a major driver (34% of total attribution), on average pushing the forecast up across this window.
- the time of year is a major driver (27% of total attribution), on average pushing the forecast up across this window.
- which product it is is a major driver (25% of total attribution), on average pulling the forecast down across this window.
- the day of the week is a minor driver (5% of total attribution), on average pulling the forecast down across this window.
- the product category is a minor driver (4% of total attribution), on average pulling the forecast down across this window.
```

Ranking is essentially unchanged by the fix (`time_idx` 36% → 34%; `month` and `sku`
swapped 2nd/3rd place, but they are within 2 points of each other and that ordering
should not be treated as meaningful).

**Read this attribution sceptically.** `time_idx` is a monotone counter and every
future date exceeds every training value, so a tree cannot extrapolate on it — every
forecast row lands in the same right-most leaf. Its `mean_signed_shap` exactly equals
its `mean_abs_shap` (0.3365 both), meaning it pushes *every single row* in the same
direction by a similar amount. That is a constant offset presenting itself as "the
trend", not a trend the model is tracking. `month` shows the same signature
(0.2718 both). The `no_time_idx` ablation was run for this reason; it was worse on
MAE (2.7892 vs 2.7351) but best on MAPE (58.08 vs 63.82), so the column is doing
something real for large SKUs. Saying "the long-run trend is a major driver" to a
seller is defensible but thin.

**This attribution is for the model the script saves** — the no-lag one. It therefore
cannot attribute anything to recent demand, which is a further reason the
full-feature model is the better product choice: its explanations can actually
reference the seller's recent sales.

Note also that `is_promo_active` remains near the bottom (0.0160, up from 0.0122).
Promotions are where the largest errors are, and the model is barely using the promo
flag.

---

## 8. Reproducibility

```
bit-identical predictions across two fits: True
max abs diff: 0.000e+00   seed=20260721
```

Two independent fits with the same seed produce byte-identical predictions.
`random_state=20260721` is set on every estimator; `tree_method="hist"`,
`n_jobs=4`. Versions recorded in `models/artifacts/feature_metadata.json`:
xgboost 3.1.3, pandas 2.3.1, shap 0.50.0, scikit-learn 1.9.0 — all read from the
running interpreter, not from `requirements.txt`.

---

## 9. Encoding choice

`sku`, `category` and `day_of_week` are passed as pandas `category` dtype with
`enable_categorical=True` (XGBoost's native categorical splits). Chosen over one-hot
because `sku` has 40 levels and one-hot would hand the tree 40 near-empty binary
columns; chosen over ordinal integers because that imposes a meaningless order on SKU
ids. Category vocabularies are fixed from the full dataset and stored in
`feature_metadata.json`, so a level appearing only in a test fold does not silently
become NaN.

---

## 10. Where this breaks — limitations

None of these were fixed by the leakage fix. All are still live.

1. **Promo and holiday spikes are where it fails, and it fails big.** Fold 4
   (November) has MAE 5.35 against RMSE 11.32. The top decile of actual demand is
   under-predicted by 17.9 units on an actual mean of 34.2 — roughly half the volume
   missed on the biggest days. The `is_promo_active` flag carries almost no SHAP
   weight. A promo-aware model (event type, days-to-event, promo depth) is the
   highest-value next step and none of that is in the frozen dataset.
2. **The 30-day horizon is not really a 30-day forecast.** The advantage over the
   flat baseline lives in days 1-7 and is gone by day 8. Days 15-30 are a level
   estimate.
3. **~64% MAPE.** Half the rows are under 4 units and a 2-3 unit miss on a 3-unit day
   is a 100% error. Percentage error on low-count data is harsh, but sellers will read
   it that way, so it should not be presented as accuracy.
4. **`time_idx` cannot extrapolate.** See section 7. Any explanation attributing the
   forecast to "the long-run trend" is describing a constant offset. Retraining
   frequently is the mitigation; removing the feature costs MAE but improves MAPE.
5. **`units_lag_7` is 7 observations, not 7 days,** 61.3% of the time (section 1c).
   The dataset has no rows for days a SKU sold nothing, so gaps are silently skipped.
   Left unfixed by request — but note this now applies to `units_rolling_7d_avg` too,
   whose window is the last 7 *observations*, not the last 7 days.
6. **Zero-demand days are invisible.** `min(total_units) == 1`. The model only ever
   predicts volume *given a sale occurred*; it never predicts whether a sale occurs.
   The recursive backtest is told which days each SKU transacted on, which is a small
   piece of information from the future. Real forecasting needs an occurrence model or
   a dataset with explicit zeros.
7. **The feature-set selection rule picks the wrong model** (section 3). Even after
   moving selection onto four clean disjoint folds, it selects the no-lag model, which
   loses by 5.7% on the reported folds. The saved artefact is that model. Use the full
   feature set; re-validate on fresh data.
8. **Early stopping is unstable** — 36 to 369 rounds across folds, a wider spread than
   before.
9. **Five reported folds, all from the last 5 months of a 3-year dataset.** The
   reported backtest never evaluates a spring or summer window, and two of the five
   folds are the holiday period. The mean is weighted toward the hardest part of the
   year. Conversely the four selection folds are *all* spring/summer, which is the
   likely reason the selection rule undervalues the lag features.
10. **Directional calls are erratic in level.** Predicted-up share swings 20.9% to
    70.1% across folds against an actual that stays near 45%.

---

## 11. What was NOT tested, and why

- **Direct multi-step forecasting** (a model per horizon, or horizon as a feature).
  It is the standard fix for the compounding measured in section 2 and would likely
  help days 8-30. Not done: it multiplies training cost, and the design doc puts
  training inside the user's HTTP request. Recommended as the next experiment.
- **Per-SKU or per-category models.** One global model across 40 SKUs. Not tested
  against 40 small models.
- **Quantile / Tweedie / Poisson objectives.** `reg:squarederror` throughout, since
  the brief framed the tail problem as a target-transform question. A count-native
  objective (`count:poisson`, `reg:tweedie`) is arguably the better answer to skewed
  count data than cube-rooting and was not compared.
- **Prediction intervals.** Point forecasts only. Given the fold-4/5 variance, a band
  matters more than the point estimate; nothing here produces one.
- **Whether the extreme rows are genuine demand or data errors.** Inherited unresolved
  from `reports/skewness/README.md`. They cluster in Nov/Dec, which reads as real.
- **The `sku`/`category` encoding alternatives.** Native categorical was chosen on
  reasoning (section 9), not benchmarked against one-hot or target encoding.
- **Hyperparameters beyond depth and learning rate.** `subsample`, `colsample_bytree`,
  `min_child_weight` and `reg_lambda` were held fixed at conventional values to keep
  the sweep inside a sensible runtime. They were not searched.
- **Anything on the `backend/tests/fixtures/*.csv` single-seller path.** This work is
  entirely on the 40-SKU panel dataset the brief specified.
- **Re-running the pre-fix configuration on the fixed data.** The before/after pairs
  in this report compare the previous revision's full run against this one. Because
  the selection-fold change landed in the same pass, the chosen hyperparameters differ
  (`depth 6 / lr 0.1` → `depth 5 / lr 0.05`), so a small part of each delta is
  attributable to that rather than to the leak fix. The one place this mattered — the
  cube-root sign flip — was isolated by an explicit intermediate run and is reported
  in section 4. Elsewhere the deltas were not decomposed.
- **Whether `units_lag_7` and `units_rolling_7d_avg` are now redundant.** Both
  summarise the same recent-demand history and they were not tested separately.

---

## Files

| path | what |
|---|---|
| `models/features.py` | Loading, feature list, two-sided runtime leakage assertions |
| `models/backtest.py` | Fold generation, metrics, baselines, the recursive 30-day simulator |
| `models/train.py` | Entry point: sweep → ablation → A/B → backtest → SHAP → artefacts |
| `models/artifacts/model.json` | Trained XGBoost booster |
| `models/artifacts/feature_metadata.json` | Features, category vocabularies, params, versions, target transform |
| `models/artifacts/hyperparameter_sweep.csv` | Section 3 table |
| `models/artifacts/feature_set_ablation.csv` | Section 3 ablation |
| `models/artifacts/ab_target_transform.csv` | Section 4 table |
| `models/artifacts/decile_mae.csv` | Section 4 decile table |
| `models/artifacts/backtest_folds.csv` | Section 5, every model × every fold |
| `models/artifacts/horizon_degradation.csv` | Section 2 bucket table |
| `models/artifacts/directional.csv` | Section 6 |
| `models/artifacts/shap_attribution.csv` | Section 7 |
