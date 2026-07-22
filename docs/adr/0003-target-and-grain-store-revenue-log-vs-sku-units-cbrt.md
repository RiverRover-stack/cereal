# 0003 — Target and grain: store revenue (log) vs. SKU units (cube-root)

## Status
Accepted (documents an existing, previously-implicit split — not a new choice)

## Context
The live serving path and the offline research pipeline predict genuinely
different things, and this was not written down anywhere before this review,
which made the pretrained-model proposal (ADR-0002) look like a simple
serving-mode swap when it is not.

- **Live path** (`backend/app/forecasting.py:62-90`): predicts store-wide
  **daily revenue in dollars**, one row per calendar date. Target is
  transformed with `log1p` and inverted with `expm1` — chosen because it beat
  the untransformed target on measured backtest (14.63% vs. 16.83% mean MAPE,
  per the code comment at `forecasting.py:76-80`), consistent with revenue
  being positive and varying multiplicatively.
- **Offline path** (`models/features.py`, `models/artifacts/feature_metadata.json:11-12`):
  predicts **units sold per SKU per day**, one row per (SKU, date). Target is
  `total_units_cbrt` (signed cube root of units), inverted by cubing. Chosen
  because raw `total_units` is heavily right-skewed (skew 4.69, reduced to
  1.10 after the transform — `reports/skewness/`), and the top 1% of rows
  drove 44.4% of squared error before the transform, 15.1% after.

Both transform choices are individually well-justified by measurement. The
problem is not the choices themselves — it's that they were made
independently, on different targets, at different grains, without either side
recording that the other exists.

## Decision
Record, explicitly, that these are two different prediction problems and keep
them distinguishable in code and docs going forward:
- Live serving predicts **store revenue ($, log-transformed)**.
- Offline research predicts **per-SKU units (cube-root transformed)**.

If a future pretrained model is built to serve production traffic (per the
ADR-0002 gate), it must predict the same thing the API contract promises —
store-level daily revenue — or the API contract and frontend must change
first. Do not silently convert a units-per-SKU model's output into a revenue
number by multiplying by price; that reintroduces `price_per_unit`, which is
explicitly banned as a leakage-adjacent feature (`feature_metadata.json:103-109`)
and would need its own validation.

## Consequences
- **Easier**: anyone evaluating "should we deploy the pretrained model" can
  check this ADR first and immediately see it's not a drop-in swap.
- **Harder**: a future global model (ADR-0002 Phase 2) has to be trained on the
  right target from the start — reusing the existing offline pipeline's target
  choice without adapting it to store-revenue grain would repeat this problem.
- **Debt**: none created by this ADR itself; it documents debt that already
  existed (two silently-diverged targets) so it's now visible instead of
  latent.

## Alternatives considered
- **Standardize on one target now** (e.g. make the live path also predict
  per-SKU units). Rejected for this ADR's scope — the live path's revenue
  target matches the product's actual promise (a 30-day revenue forecast,
  `design-doc.md`) and store-level sellers often don't even have clean SKU
  data in their CSV (SKU column is optional per `design-doc.md:150`). Revisit
  if per-SKU forecasting (`design-doc.md` §5 item 4, the reserved `sku_level`
  field) is prioritized.

## Links
- [`docs/architecture-review.md`](../architecture-review.md) §2.1, §3 (Q1)
- ADR-0002 (the serving-strategy decision this feeds into)
- `reports/skewness/README.md` (cube-root justification detail)
