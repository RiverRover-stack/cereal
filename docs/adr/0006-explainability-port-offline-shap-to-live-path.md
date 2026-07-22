# 0006 — Port offline SHAP explainability to the live request path

## Status
Accepted — implemented. `backend/app/explain.py` runs `shap.TreeExplainer`
against the per-request fitted model and returns sign/rank-only sentences
(no dollar magnitude, per the Decision below); wired into `POST /forecast`'s
`explanation` field and rendered in `src/pages/app.astro`'s "why" panel.
Covered by `backend/tests/test_explain.py` (the dominant-feature sign/rank
test specified in `design-doc.md:216-218`) and
`backend/tests/test_forecast_endpoint.py` (end-to-end wiring). Dollar-
denominated attribution remains deferred, as decided below.

## Context
TrendsArc's stated differentiator is explainable forecasting: a 30-day
forecast, SHAP-based attributions, and plain-English reasoning
(`design-doc.md` Phase 4, product context given for this review). SHAP is a
standard technique for explaining what a trained model's prediction was
driven by, feature by feature.

This is fully implemented and working — just in the wrong place:
- `models/train.py:416-466` runs `shap.TreeExplainer` against the offline
  model, aggregates values per feature, and templates them into plain-English
  sentences by feature name + sign + magnitude bucket
  (`models/train.py:434-460`).
- The **live** `POST /forecast` endpoint has never had this step added.
  `backend/app/main.py:3-5` states it directly: "The explanation step (Phase
  4: SHAP) is not built yet, so no `explanation` key is returned." The
  frontend's "why" panel (`src/pages/app.astro:121-128`) is a hardcoded
  placeholder telling the user this breakdown "is planned but not built."

Two things make this cheaper to close than it looks:
- `forecasting.py`'s `Forecast` dataclass already carries `model`,
  `feature_frame`, `feature_names`, and `target` specifically so a SHAP
  explainer can attribute the *exact fitted model* that produced the forecast
  (`forecasting.py:122-140` docstring: "so Phase 4's SHAP TreeExplainer can
  attribute the same fitted model that produced these numbers, rather than
  refitting and explaining a different one").
- The templating pattern from `models/train.py` is directly reusable — it's
  feature name + sign + magnitude → sentence, and the live path's 4 features
  (`day_of_week`, `day_of_month`, `time_index`, `rolling_7`) are a subset of
  what the offline templates already handle conceptually.

One real technical constraint: the live model is fit on `log1p(revenue)`
(`forecasting.py:71-90`), so SHAP values computed against it are in log units,
not dollars. `forecasting.py:82-85` already flags this and says to settle it
before Phase 4 — sign and relative rank survive the log transform fine;
dollar-denominated attribution does not, without further work.

## Decision
Build `backend/app/explain.py`: run `shap.TreeExplainer` against the
per-request fitted model (using the `Forecast` object's already-carried
`model`/`feature_frame`), aggregate per feature over the 30-day forecast
window, and template into 3-5 plain-English sentences using **sign and
relative rank only** (not dollar magnitude) for V1, given the log-target
constraint above. Wire the result into `POST /forecast`'s response as
`explanation: string[]`, and update `src/lib/api.ts` and
`src/pages/app.astro` to carry and render it, replacing the current
placeholder.

Defer dollar-denominated attribution ("$X of this forecast came from Y") to a
follow-up decision once it's decided whether the live model should move to a
level (non-log) target, or whether a log-aware SHAP adjustment is worth the
complexity.

## Consequences
- **Easier**: ships the product's stated differentiator using code that
  already exists and works, adapted rather than rewritten.
- **Easier**: explanations are honestly personal — they describe the seller's
  own fitted model, not a population-average global model — which is a
  stronger claim than Option B (ADR-0002) could make even if it existed today.
- **Harder**: adds SHAP computation to the request path, which has its own
  cost (proportional to tree count × feature count × forecast rows) — should
  be measured once implemented, not assumed negligible.
- **Debt**: sign/rank-only explanations are a deliberate scope cut for V1;
  if product requirements later demand dollar-denominated "why" statements,
  that reopens the log-vs-level target question this ADR defers.

## Alternatives considered
- **Move the live model to a level (non-log) target now, to get dollar SHAP
  immediately.** Rejected for this ADR — the log target was chosen because it
  measurably outperforms the level target on backtest (14.63% vs 16.83% mean
  MAPE, `forecasting.py:76-80`); trading away that accuracy win for
  dollar-denominated explanations wasn't asked for and shouldn't happen as a
  side effect of an explainability change.
- **Wait for Phase 2/a global model before adding SHAP**, on the theory that
  explainability should be built once against whatever the "final"
  architecture turns out to be. Rejected — ADR-0002 keeps per-request fitting
  as the accepted V1 path, and the SHAP logic needed here is largely reusable
  regardless of which model produces the forecast, so there's no reason to
  block the differentiator on an architecture decision that's already been
  made.

## Links
- [`docs/architecture-review.md`](../architecture-review.md) §7, §10
- ADR-0002 (confirms per-request model as what this attaches to)
- ADR-0003 (log-target constraint this defers around)
