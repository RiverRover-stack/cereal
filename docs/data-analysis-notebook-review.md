# Review of `Data Analysis.ipynb`

**Date:** 2026-07-21 (second pass)
**Scope:** fix errors + refactor.
**Backup of the version I started from:** `docs/Data Analysis.ORIGINAL.ipynb`

> **Note on this file.** An earlier version of this review covered a different
> version of the notebook — one with a daily-revenue EDA section reading a
> `valid_daily.csv` that did not exist. That section is gone from the notebook now,
> so this document has been rewritten against what is actually there: the
> feature-engineering pipeline and the new "preparing the data for training"
> section. The two findings that carried over (leaky rolling window, row-based
> lag) are still findings, because the same code was still present.

---

## The short version

Seven problems. One stopped the notebook running at all; three would have
corrupted a model trained on this table; three were making the summary statistics
wrong or misleading.

| # | Problem | Severity |
|---|---------|----------|
| 1 | Cell 29 is not valid Python — `y='')']` | **Crashes** |
| 2 | `total_revenue` / `price_per_unit` fed to the model — same-day columns, r≈0.98 with the target | **Leakage** |
| 3 | `units_rolling_7d_avg` included today's sales | **Leakage** |
| 4 | `units_lag_7` was "7 rows back", not "7 days back" | **Wrong feature** |
| 5 | `revenue_7d_avg` rolled across SKU boundaries | Wrong for 240 rows |
| 6 | `df.resample('ME').sum()` summed text and non-additive columns | Wrong output |
| 7 | Weekday averages labelled as business totals were per-SKU-day averages | Misleading |

Plus: triplicated imports, five empty cells, and `set_index(inplace=True)` calls
that made the notebook non-re-runnable.

---

## 1. The notebook did not parse

```python
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='total_revenue', hue='sku', marker='o')

sns.lineplot(data=df, x=df.index, y='')']      # <- SyntaxError
```

An unbalanced quote and bracket, left mid-edit. The cell had never been executed
(`execution_count: null`), and everything after it in the notebook was empty or
unrun.

**Fix.** The intent — revenue over time — needed rethinking as well as repairing.
`hue='sku'` on this data draws **40 overlapping lines with 1,088 markers each**;
it is unreadable and slow. Section 5 now plots total revenue across the catalogue
with its 7-day average, and uses **category** rather than SKU for the breakdown,
which is the level the trend is actually visible at.

---

## 2. Same-day revenue was going into the model

This is the one that matters most, because the new training section was about to
consume it:

```python
master_df['price_per_unit'] = master_df['total_revenue'] / master_df['total_units']
...
train_df = master_df[master_df.index <= split_date]   # every column comes along
```

`train_df` carried `total_revenue` and `price_per_unit`, both of which are
**same-day** quantities. Revenue is units × average price, so within any one SKU it
tracks the target almost perfectly:

```
within-SKU corr(total_revenue, total_units): min 0.950  median 0.977  max 0.992
```

Since `sku` is itself a model feature, the model can condition on it and read the
target straight out of the revenue column. `price_per_unit` is worse — it is the
target used as a denominator, and it is undefined on the zero-demand days that make
up a quarter of a properly filled panel.

**Fix.** Features are now named explicitly rather than "whatever is in the frame",
with an assertion that no same-day column sneaks in:

```python
FEATURES = [
    "sku", "category", "base_price",
    "month", "day_of_week_num", "is_weekend", "is_promo_active",
    "units_lag_7", "units_rolling_7d_avg", "revenue_rolling_7d_avg",
]
LEAKY = ["total_revenue", "price_per_unit"]

assert TARGET not in FEATURES, "target leaked into the feature list"
assert not set(FEATURES) & set(LEAKY), "same-day column leaked into the feature list"
```

Revenue is not discarded — it comes back as `revenue_rolling_7d_avg`, the lagged
form, which is legitimately known on forecast morning.

`date` is also excluded as a raw column. A tree model splits on the raw timestamp
and learns "later = higher", which cannot extrapolate past the last training date;
the calendar features carry the part of it that generalises.

---

## 3. Target leakage in the rolling average

```python
master_df['units_rolling_7d_avg'] = master_df.groupby('sku')['total_units'].transform(
    lambda x: x.rolling(window=7).mean()
)
```

Pandas' `rolling(7)` window at row *t* spans rows *t-6 … t* — **including t**. So
the "average of the last 7 days" contains one seventh of the value being predicted.

Measured (section 4 of the rebuilt notebook):

```
corr(rolling avg, today's units)
  leak-free (shifted) : 0.738
  leaky   (unshifted) : 0.807   <- inflated
```

That 0.07 is borrowed against validation scores and repaid in production, where
today's sales are not yet known.

**Fix:**

```python
panel["units_rolling_7d_avg"] = by_sku[TARGET].transform(
    lambda s: s.shift(1).rolling(ROLLING_WINDOW).mean()
)
```

The window now ends *yesterday*. Section 4 asserts the alignment against a
hand-computed mean rather than trusting it.

---

## 4. `units_lag_7` was not a 7-day lag

```python
master_df['units_lag_7'] = master_df.groupby('sku')['total_units'].shift(7)
```

`shift(7)` moves seven **rows**. That equals seven days only if every SKU has a row
for every calendar day — and it does not, because `groupby(['date','sku'])` on line
items only emits a row when that SKU sold something.

Measured on the raw data:

```
rows for ACC-001: 664    date span: 1094 days
gaps between consecutive observations:
  1 day → 426    2 days → 141    3 days → 50
  4 days →  26   5 days →   8    6 days →  7
```

"Seven rows back" therefore averaged about **eleven calendar days**, varying by SKU
and by row. Whatever weekly seasonality the feature was meant to capture was smeared
across a moving, data-dependent window.

**Fix.** Reindex onto a dense (date × SKU) grid, filling absent days with zero —
a day with no order is a genuine observation of zero demand — and only then shift:

```python
panel = (
    daily_sku
    .set_index(["date", "sku"])
    .reindex(pd.MultiIndex.from_product([all_dates, all_skus], names=["date", "sku"]),
             fill_value=0)
    .reset_index()
    .sort_values(["sku", "date"], ignore_index=True)
)
```

```
43,800 rows after filling (11,784 zero-demand days added — 26.9% of the panel)
```

Just over a quarter of the panel had been missing. Section 4 now asserts every
SKU's dates are consecutive, so this cannot silently come back.

---

## 5. `revenue_7d_avg` rolled across SKU boundaries

```python
df['revenue_7d_avg'] = df['total_revenue'].rolling(window=7).mean()
```

No `groupby`. The frame is sorted by `sku` then `date`, so at each product boundary
the window mixes the last days of one SKU with the first days of the next — blending
a $20 accessory with a $200 item.

Measured: **240 of 32,016 rows** differ from the correct per-SKU calculation (six
corrupted rows at each of the 40 boundaries). Small, but wrong, and invisible unless
you go looking at exactly those rows.

**Fix.** Two separate things were conflated here, so they are now two things: the
per-SKU model feature is `by_sku["total_revenue"].transform(...)`, and the business
chart in section 5 aggregates across SKUs *first* and then rolls the daily totals.

---

## 6. `resample('ME').sum()` on a mixed-type frame

```python
monthly_summary = df.resample('ME').sum()
print(monthly_summary[['total_revenue', 'total_units']])
```

By this point `df` also held `sku`, `category` and `day_of_week` (strings) plus
`month`, `base_price` and `price_per_unit` (numbers that must not be summed).
`.sum()` computed all of them before the two useful columns were selected: it
**concatenated 40 SKU codes per month into one enormous string**, and summed a
per-unit price, which is not an additive quantity. The printed slice looked fine, so
none of it was visible.

**Fix** — select the additive columns first:

```python
monthly_summary = (
    daily_totals.set_index("date")[["total_revenue", "total_units"]].resample("ME").sum()
)
```

---

## 7. Weekday averages answered a different question

```python
revenue_by_day = df.groupby('day_of_week')['total_revenue'].mean()
print("Average Revenue by Day of Week:\n", revenue_by_day)
```

`df` is one row per SKU per day, so this is the mean revenue of **a single
SKU-day** — around $280 — printed under a heading that reads as business-wide
revenue. It is a real number, just not the number the label promises.

**Fix.** Section 5 aggregates to daily catalogue totals first, then averages:

```
Monday        7864.99
Tuesday       7497.48
Wednesday     7663.18
Thursday      8166.99
Friday        9989.05
Saturday     11588.30
Sunday        9551.06
```

The weekend lift is the same shape either way — but it is now the actual weekend
lift, not a per-SKU proxy for it.

---

## Smaller things

- **`import pandas as pd` appeared three times**, matplotlib/seaborn twice, because
  the notebook had accumulated several beginnings. All imports are in section 0,
  along with the constants (`LAG_DAYS`, `ROLLING_WINDOW`, `TEST_DAYS`, paths), so
  changing the lag no longer means hunting for a magic `7`.
- **`master_df.set_index('date', inplace=True)` was not re-runnable** — running the
  cell twice raises `KeyError: 'date'`, because the column has become the index.
  The same pattern appeared twice. The notebook now keeps `date` as a column
  throughout; every cell can be re-run.
- **`master_df.dropna(inplace=True)` dropped rows on *any* null column.** It happened
  to only hit warm-up rows, but a null appearing anywhere else later would silently
  delete data. Now `dropna(subset=LAG_FEATURES)`.
- **Five empty cells** removed.
- **`master_df` → `panel`.** "Master dataframe" describes nothing; the object is a
  panel — a per-entity time series — which is what it is.
- **Joins were unverified.** Both merges now assert unique keys, an unchanged row
  count, and no SKU missing from the catalogue.
- **Execution counts were scrambled** (25–58, with several cells never run). The
  notebook has been executed top-to-bottom in one pass, so saved outputs match saved
  code.

---

## New structure

```
0. Setup                      imports, paths, constants, load_csv()
1. Load the raw tables
2. Build the (date × SKU) panel
   2.1 Aggregate line items
   2.2 Fill the calendar       <- fixes the lag
   2.3 Join promos + catalogue with assertions
3. Feature engineering
   3.1 Calendar features
   3.2 Lag and rolling         <- leak-free
   3.3 Drop warm-up rows
4. Verification                4 assertions + the leakage measurement
5. Exploratory plots           aggregate-then-plot
6. Prepare for training
   6.1 Explicit FEATURES list  <- fixes the revenue leak
   6.2 Time-based split
```

Final state: **43,520 rows**, split at 2025-11-30 into 42,320 train / 1,200 test.

---

## Reproducing this

Run the whole notebook — it exits non-zero if any cell raises, so the assertions in
section 4 and 6 act as a regression test:

```bash
cd "C:/Users/ASUS/OneDrive/Desktop/cereal"
python -m jupyter nbconvert --to notebook --execute --inplace "Data Analysis.ipynb" \
  --ExecutePreprocessor.timeout=900
```

Confirm the revenue leak (finding 2):

```bash
python -c "
import pandas as pd
li = pd.read_csv('data/sales_line_items.csv', parse_dates=['date'])
d = li.groupby(['date','sku'], as_index=False).agg(u=('units_sold','sum'), r=('revenue','sum'))
w = d.groupby('sku').apply(lambda g: g.r.corr(g.u), include_groups=False)
print('within-SKU corr: min %.4f median %.4f max %.4f' % (w.min(), w.median(), w.max()))
"
```

Confirm the date gaps that broke the lag (finding 4):

```bash
python -c "
import pandas as pd
li = pd.read_csv('data/sales_line_items.csv', parse_dates=['date'])
d = li.groupby(['date','sku'])['units_sold'].sum().reset_index().sort_values(['sku','date'])
g = d[d.sku=='ACC-001']
print('rows:', len(g), 'span days:', (g.date.max()-g.date.min()).days+1)
print(g.date.diff().dt.days.value_counts().head(6))
"
```

Confirm the cross-SKU rolling contamination (finding 5):

```bash
python -c "
import pandas as pd, numpy as np
li = pd.read_csv('data/sales_line_items.csv', parse_dates=['date'])
d = li.groupby(['date','sku'], as_index=False).agg(r=('revenue','sum')).sort_values(['sku','date']).reset_index(drop=True)
wrong = d['r'].rolling(7).mean()
right = d.groupby('sku')['r'].transform(lambda s: s.rolling(7).mean())
print('rows differing:', int((~np.isclose(wrong, right)).sum()), 'of', len(d))
"
```

---

## What I did *not* check

- **No model was trained.** The leakage figures are correlations, not a measured
  drop in backtest error. The real cost of a leak shows up as the gap between
  validation and production accuracy, and I did not fit anything to quantify that.
- **Whether zero-fill is right for your model.** It is right for *demand*
  forecasting (no order = zero demand). If a SKU was out of stock rather than
  unwanted, a zero is misleading — you would want a stock-availability flag.
  Nothing in `catalogue.csv` records stock, so the two cases cannot be told apart.
- **`is_promo_active` still flags only the exact event date.** `events.csv` carries
  a `sigma_days` column (2.0–4.0), meaning the generator spreads each promotion over
  several days around its peak, so the binary flag under-counts promo influence —
  3.3% of rows are marked. I left the original behaviour because widening it is a
  modelling decision, not a bug fix. A decaying promo-proximity feature would
  probably beat the flag; worth testing before training.
- **The lag features straddle the train/test boundary.** The first week of the test
  set uses lag values from inside the training window. This is correct for one-step
  forecasting but means the sets are not fully independent — noted in the notebook.
  For multi-step forecasts the lags must be built from predictions, not actuals.
- **`data/components.csv` is unused** by the notebook and I did not look at it.
- **Whether the generated data is realistic** — I verified the notebook handles the
  data correctly, not that `backend/scripts/generate_sales_data.py` simulates
  anything sensible.
