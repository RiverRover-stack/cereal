# 0007 — Rolling-origin backtest as the evaluation method and deploy gate

## Status
Accepted (method, already in use) / Proposed (as an automated deploy gate)

## Context
A forecast cannot be honestly validated with a random train/test split — that
would let the model train on days *after* the ones it's scored on, which is
impossible in production (on the day a seller uploads their file, tomorrow
hasn't happened yet). `docs/forecast-validation.md:6-16` establishes this and
implements the correct alternative: a **rolling-origin backtest** — pick a
cutoff date, train only on data before it, forecast 30 days forward, compare
to what actually happened, and repeat at several cutoffs.

This method is already used in two places, independently:
- Live path: `backend/tests/backtest.py` + `backend/tests/metrics.py`, run
  directly against the real `validate_csv` + `forecast_revenue` functions
  (not a separate offline copy). Measured result: model beats a
  seasonal-naive baseline in every one of 5 folds, mean MAPE 24.82% vs.
  35.81% naive (`docs/forecast-validation.md:43-59`).
- Offline path: `models/backtest.py`, expanding-window folds by calendar
  date, plus hyperparameter selection done on a *separate, earlier* block of
  folds than the ones reported, to avoid tuning on the test set
  (`models/train.py:56-60, 107-137`, non-overlap asserted at runtime).

Neither is currently wired into anything that blocks a deploy. A regression
(a code change that quietly makes forecasts worse) would ship silently today.
Separately, `backend/tests/` has the harness code (`backtest.py`,
`metrics.py`) but no actual `test_*.py` files, despite `forecasting.py`'s own
docstring and `pytest.ini` referencing a test suite that doesn't exist yet —
`pytest` currently collects nothing.

## Decision
1. **Standardize on rolling-origin backtesting** as the evaluation method for
   any forecasting model in this repo, live or offline — never a random
   split, never tuning and reporting on the same fold.
2. **Propose wiring the existing backtest into a deploy gate**: run
   `python -m tests.backtest` (or equivalent) before deploy, compare MAPE
   against the last known-good run, and fail the deploy on a meaningful
   regression. Threshold and exact CI mechanism are left open — this ADR
   records the requirement, not the implementation.
3. **Backfill `backend/tests/test_*.py`** so `pytest` has real unit tests
   (validation edge cases, forecast response shape, and the explanation
   sign/direction test that `design-doc.md:216-218` specified for Phase 4 and
   was never written) — separate from, and in addition to, the backtest
   harness.

## Consequences
- **Easier**: future model or feature changes get an objective pass/fail
  signal instead of relying on someone remembering to re-run the backtest
  manually.
- **Harder**: deploys take longer (a 5-fold backtest isn't instant), and
  someone has to decide and maintain a sensible regression threshold — too
  strict blocks legitimate improvements that happen to shift the metric
  slightly, too loose doesn't catch real regressions.
- **Debt this closes**: currently, `docs/forecast-validation.md:116-121`
  notes that the backtest numbers justifying the `recursive` vs `direct` vs
  `known_covariates` strategy choice (cited in `forecasting.py`'s own
  comments) aren't reproducible from this checkout — a gate that runs the
  backtest on every change would prevent this kind of unreproducible-claim
  drift going forward.

## Alternatives considered
- **Manual backtest runs before major changes, no automated gate.** This is
  the status quo; rejected as the long-term answer because it already failed
  once (the strategy-comparison numbers in `forecasting.py`'s comments are
  no longer reproducible) and there's no reason to expect manual discipline
  to hold better going forward.
- **Gate on a stricter metric than MAPE** (e.g. worst-fold MAPE, to catch the
  kind of large single-fold error seen in the holiday-spike fold,
  `docs/forecast-validation.md:66-72`). Not rejected — worth considering
  alongside mean MAPE when the gate is actually implemented, since
  `docs/forecast-validation.md` itself flags that a model winning on average
  but losing badly on one fold is a model you can't trust fold-to-fold.

## Links
- [`docs/architecture-review.md`](../architecture-review.md) §8, §10
- `docs/forecast-validation.md` (the existing measured results this codifies)
