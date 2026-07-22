# Validating the Trends Arc forecast — methodology and first real numbers

**Question this answers:** the backend runs and returns a number without erroring —
but is that number any *good*? Those are two different checks. This is the second one.

## The method: rolling-origin backtest, not a train/test split

A forecast can't be validated by holding out a random 20% of rows the way you would
for, say, a churn classifier. A random split would let the model train on days
*after* the ones it's being scored on — something a real forecast can never do,
since on the day a seller uploads their file, tomorrow hasn't happened yet.

The correct method is a **rolling-origin backtest**: pick a cutoff date, train only
on data before it, forecast the next 30 days, and compare to what actually
happened. Repeat at several cutoff dates so the model is tested across different
stretches of the history, not just one lucky (or unlucky) window.

Two more things a bare error number doesn't tell you:

- **Is it better than doing nothing clever?** A model that's merely okay in
  isolation might still be worse than "repeat last week's numbers" — the
  simplest thing a seller could do by eye. `app/forecasting.py` already ships a
  `seasonal_naive()` baseline for exactly this comparison.
- **Is the model cheating?** Time-series features are the easiest place to
  accidentally leak the answer into the input — e.g. a "7-day rolling average"
  that, if built wrong, includes the day it's supposed to be predicting.

## What I built

`backend/tests/backtest.py` (+ `backend/tests/metrics.py` for the MAE/RMSE/MAPE
helpers). These didn't exist in the checkout — `app/forecasting.py`'s own
docstring and `backend/pytest.ini` both reference a `tests/` suite that was
missing (flagged in the earlier MVP gap review, `docs/mvp-testing-gaps.md`). This
rebuilds the backtest piece of it, run directly against the live code path
(`app.validation.validate_csv` + `app.forecasting.forecast_revenue` — the exact
functions `POST /forecast` calls), not a separate offline model.

**To reproduce, from `backend/`:**
```
.venv\Scripts\python -m tests.backtest --leak-demo
```

## Real measured results (on `data/sales_daily.csv`, 1095 days)

5 folds, expanding window, spread across the series so each one lands in a
different stretch of history:

```
origin       train_days   model_MAE  model_RMSE model_MAPE   naive_MAE  naive_RMSE naive_MAPE
---------------------------------------------------------------------------------------------
2024-03-13          438     1054.36     1413.03     22.86%     1452.08     1864.13     27.16%
2024-08-24          602     1552.86     1917.31     20.62%     1578.40     1908.15     20.87%
2025-02-04          766     2478.63     3048.92     25.17%     3453.31     4201.13     36.26%
2025-07-18          930     1588.77     1900.50     18.71%     3346.59     3815.47     41.97%
2025-11-27         1062    12455.29    14858.89     36.75%    16475.45    18647.89     52.81%
---------------------------------------------------------------------------------------------
Mean model MAPE: 24.82%   Mean naive MAPE: 35.81%
Model beats the seasonal-naive baseline by 10.99 points of MAPE.
```

**Reading this:**
- The model beats the "repeat last week" baseline in every single fold, not just
  on average — that's the more important claim than the average, since a model
  that wins on average but loses badly on one fold is a model you can't trust
  fold-to-fold.
- The last fold (train through 2025-11-27) has much larger absolute error than
  the others — MAE of ~$12,455 vs. ~$1,000-2,500 elsewhere. That's very likely a
  late-year demand spike (holiday season) in the synthetic data that's harder to
  extrapolate into — both the model and the naive baseline get worse here, and
  the model still wins by a wide margin (36.75% vs. 52.81% MAPE), but I'm
  flagging the spike rather than averaging it away, since it's exactly the kind
  of period where trusting the number matters most.
- 20-25% MAPE in the calmer folds means a $10,000 day could plausibly forecast
  anywhere from ~$7,500-$12,500. Whether that's "good enough" isn't a number I
  can answer for you — it depends on what a seller does with the forecast — but
  now there's a real figure to make that call against, instead of no figure.

## Leakage check — is the model being handed the answer?

One-step-ahead diagnostic: predict a day's revenue with a plain 7-day rolling
mean, once correctly (mean of the *prior* 7 days) and once with the bug the
`.shift(1)` in `build_features()` exists to prevent (mean *including* the day
itself):

```
Correct (prior 7 days only):  MAE  2083.98  MAPE  23.40%
Leaky (window includes today): MAE  1859.37  MAPE  20.97%
```

The leaky version does score better — about 2.4 points of MAPE — which is the
leak made visible: roughly 1/7 of the value it's "predicting" is baked into its
own input. The effect is real but modest here, which makes sense given a 7-day
average dilutes any single day's influence to ~14%. This confirms the `.shift(1)`
guard in `app/forecasting.py` is doing real, load-bearing work, not defensive
boilerplate — removing it wouldn't crash anything, it would just quietly make
every number look better while becoming useless as a real forecast.

## What this does and doesn't establish

**Established:** on this dataset, the live forecasting code beats a naive
baseline by a meaningful margin, consistently across time, and the leakage
guard is doing what it claims to do.

**Not established:**
- **Generalization to real seller data.** `data/sales_daily.csv` is a synthetic
  fixture (see `data/generate_sales_data.py`) with clean daily rows and no
  gaps. A real Shopify export will have messier patterns — actual promo timing,
  irregular gaps, genuinely noisier demand. This backtest can't speak to that
  until it's run against real or more adversarially-generated data.
- **Sensitivity to short histories.** The shortest fold here trains on 438 days.
  The product's stated minimum is 30 days (`MIN_HISTORY_DAYS`) — nothing here
  tests accuracy near that floor, which is exactly the regime design-doc.md
  Section 8 flags as a risk ("forecast quality on thin/messy data"). Worth a
  follow-up run with `FOLD_FRACTIONS` adjusted to include an early, thin-history
  fold.
- **The `direct` and `known_covariates` strategies.** `app/forecasting.py`
  implements three strategies and picked `recursive` based on backtest numbers
  cited in its own comments — but those numbers aren't reproducible from this
  checkout (same missing-tests gap). This backtest only re-confirms `recursive`
  in isolation; it doesn't re-run the three-way comparison that originally
  justified the choice.
- **Multiple real sellers' data.** One fixture file, one business's shape of
  demand. Real validation needs more than one series.
