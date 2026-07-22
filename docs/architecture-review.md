# TrendsArc architecture review: pretrained-model pivot

**What this is:** a critical review of the proposed shift from "train a model on
each seller's upload" to "deploy one pretrained model and only run inference."
Written for the founder, not for an ML audience — jargon is explained on first
use. Findings are backed by file/line citations so you can check them yourself.

**Companion doc:** [`docs/adr/`](adr/README.md) — the individual decisions below,
recorded as short, revisitable Architecture Decision Records (ADRs) so future-you
doesn't have to remember *why*, just look it up.

---

## 1. TL;DR — what to fix first

1. **The pivot as described can't be deployed today.** The "pretrained model"
   (`models/artifacts/model.json`) and the live API (`backend/app/forecasting.py`)
   solve two different problems — different target, different grain, different
   math. You cannot point the API at `model.json` without a rewrite of both sides.
   See §2.
2. **The training data is one store, not many.** `master_df_cbrt.csv` is 40 SKUs
   × ~3 years for a *single* synthetic seller. "Merged many synthetic stores"
   didn't happen — there's no store identifier in the data at all. A model can't
   learn to generalize across sellers from one seller's history. See §2.3.
3. **The pretrained model is structurally unable to serve a new seller.** It was
   trained with `sku` as a categorical feature, and the 40 SKU values it knows
   about are frozen into the saved model. A real seller's product codes won't be
   any of `ACC-001`, `APP-001`, … — the model has never seen them and has nothing
   to fall back on. See §2.2.
4. **The premise "pretrained is cheaper/faster" doesn't hold at this scale.**
   Fitting XGBoost on a few hundred rows takes well under a second (§2.4). The
   latency and cost you'd actually be fighting on Cloud Run is *cold start*
   (container spin-up), which a pretrained model doesn't fix — you still need
   inference code, feature engineering, and a warm container either way.
5. **The thing actually worth building next is the differentiator, not a
   pretrained model.** SHAP explanations and plain-English reasoning are fully
   built and working in the *offline* research pipeline (`models/train.py`) but
   never reached the live API — `POST /forecast` still returns no `explanation`
   field, and the frontend's "why" panel is a hardcoded placeholder
   (`src/pages/app.astro:121-128`). Porting that is higher-value than switching
   serving strategy. See §7.

Recommendation in one line: **keep per-request fitting as the V1 architecture,
port the already-built SHAP explainability to it, and treat "pretrained global
model" as a Phase-2+ upgrade you earn once you have data from more than one
store and a backtest that proves it's actually better.** Full reasoning in §5.

---

## 2. Critique of the proposed architecture

### 2.1 Two disconnected models, not one pipeline with a serving-mode toggle

| | **Live serving path** (what actually answers `POST /forecast`) | **Offline research path** (what you're proposing to deploy) |
|---|---|---|
| Code | `backend/app/forecasting.py`, `backend/app/main.py` | `models/train.py`, `models/features.py` |
| Predicts | Store-wide **daily revenue ($)** | **Units sold, per SKU, per day** |
| Target transform | `log1p` / `expm1` (`forecasting.py:86-90`) | Signed cube root, `x**3` to invert (`feature_metadata.json:11-12`) |
| Grain | One row per calendar date | One row per (SKU, date) — 40 rows/day |
| Features | `day_of_week`, `day_of_month`, `time_index`, `rolling_7` (`forecasting.py:62-67`) | `sku`, `category`, `day_of_week`, `is_promo_active`, `base_price`, `month`, `day_of_month`, `time_idx` (`feature_metadata.json:13-22`) |
| Trained on | The seller's own uploaded CSV, at request time | `data/processed/master_df_cbrt.csv`, 31,736 rows, one synthetic store |
| SHAP | Not implemented (`main.py:3-5`) | Fully implemented (`models/train.py:416-466`) |
| Saved to disk | Never — nothing survives the request | `models/artifacts/model.json` + `feature_metadata.json` |
| Wired to an endpoint | Yes — this is production | **No.** Nothing in `backend/` loads `model.json`, imports `shap`, or references `joblib`/`pickle`. It's a standalone evaluation artifact. |

These aren't two versions of the same model — they answer different questions
(revenue vs. units, store vs. SKU) with different math. "Deploy the pretrained
model" isn't a config change; it's building a second product surface (per-SKU
unit forecasts) and then figuring out how to turn that into the revenue number
your API contract promises the frontend (`src/lib/api.ts:71-80`).

### 2.2 The frozen `sku` vocabulary blocks generalization to any real seller

`feature_metadata.json:36-101` shows the model was trained with `sku` as a
`category`-dtype feature (`enable_categorical=True`) and the exact 40 SKU codes
it learned are baked into the saved model:

```
ACC-001 … ACC-008, APP-001 … APP-010, FOO-001 … FOO-006,
HOM-001 … HOM-006, OUT-001 … OUT-005, SUM-001 … SUM-005
```

XGBoost's native categorical support has no "unknown category" fallback the way
some encoders do — a SKU outside this list has no learned split to route
through. Every real Shopify seller has their own product codes, so this model,
as trained, cannot score a single row of real-world data. This isn't a tuning
problem; it's a feature-choice problem baked into training.

The selected feature set is called `known_covariates_only` in the metadata
(`feature_metadata.json:23`) — a slightly misleading name, since `sku` and
`base_price` are "known" only in the sense of "known in advance for this one
store," not "known and meaningful for any store."

### 2.3 "Many synthetic stores merged" — verified against the data, this is one store

The plan described training data as many synthetic stores generated and merged
into a master dataframe. Checked against the actual files:

- `data/processed/master_df_cbrt.csv` — 31,736 rows = 40 SKUs × ~793 usable
  days after feature warm-up (source range 2023-01-08 to 2025-12-30,
  `feature_metadata.json:7-10`).
- **No store or seller ID column exists anywhere** in `master_df.csv`,
  `master_df_cbrt.csv`, `catalogue.csv`, or `events.csv`.
- `catalogue.csv` carries generation parameters (`annual_phase`,
  `annual_amplitude`, `popularity`, `promo_sensitivity`) for 40 *products*, and
  `components.csv` is the literal trend/weekly/annual/event/promo/noise
  decomposition the series was built from — strong, direct evidence this is one
  procedurally generated store's catalogue and sales history, not a merge of
  independent stores.

This matters because "trained on many synthetic stores" and "trained on one
store's 40 products" imply very different generalization properties. A model
that has only ever seen one store's demand pattern, price points, and category
mix has no basis to claim it will transfer to a different store's shape of
demand — it's had exactly one example to learn "what a store looks like" from.

### 2.4 The cost/latency premise doesn't hold at this scale

Motivations given for the pivot were faster inference, lower Cloud Run cost, and
better scalability. Checking each against what's actually happening today:

- **Fit time.** `DEFAULT_PARAMS` in `forecasting.py:97-111` is 200 trees, max
  depth 3, on typically a few hundred rows (a seller's daily history). That's a
  sub-second fit on Cloud Run's CPU tier — this is not a workload where "skip
  training" meaningfully moves latency. (Exact warm-run timing wasn't
  re-measured in this review — see §13 — but the order of magnitude is not in
  question at this data size and tree depth.)
- **What actually drives Cloud Run cost/latency for this app is cold start** —
  container spin-up when there's no warm instance — which is orthogonal to
  whether the model trains or just runs inference. A pretrained model still
  needs the container to boot, the CSV to be validated, features to be built,
  and (per the differentiator) SHAP to run. None of that goes away.
- **Scalability**: per-request fitting is embarrassingly parallel and fully
  stateless (`design-doc.md:106`) — each request is independent, so this
  already scales horizontally the same way inference-only would. The place
  "many pretrained models" would actually help is if you had *so many* uploads
  per second that CPU-bound fitting became the bottleneck — worth solving when
  you have that traffic, not before.

None of this means pretrained-and-cached is a bad idea forever — see §5 and §9
— just that the cost/speed argument, as currently framed, is weaker than it
sounds and shouldn't be the deciding factor.

---

## 3. Answers to your specific questions

**1. Is deploying a pretrained model trained only on synthetic data a good
production architecture?** Not as currently built — see §2.1–2.3. The
*concept* (offline training, cached inference) is sound engineering practice in
general; the specific artifact you have today (single-store, SKU-identity
features, wrong target) isn't ready to serve real traffic behind your current
API contract.

**2. What risks does synthetic-only training introduce?** Two distinct risks,
often conflated:
- *Synthetic vs. real* — synthetic data is clean and gap-free by construction
  (`docs/forecast-validation.md:105-109` already flags this for the live path's
  own training fixture). Real Shopify exports have irregular gaps, messy promo
  timing, refunds, currency quirks, multi-channel double-counting — none of
  which the synthetic generator models.
- *One store vs. many* — even setting aside "synthetic vs. real," a model that
  has only ever seen one store's price points, category mix, and seasonality
  shape has no evidence it generalizes to a *different* store, synthetic or
  real. This is the bigger, more fixable risk (§9 Phase 2).

**3. Will the model generalize to unseen real sellers?** The per-SKU model:
almost certainly not, for the concrete reason in §2.2 (frozen SKU vocabulary
plus single-store price/seasonality patterns). The per-request store-level
model: better positioned, because it only uses features that are meaningful for
*any* store's own history (see §4) — but it has never been backtested against a
real seller either (§13).

**4. Which parts of the current feature engineering are likely to generalize?**
The live path's features are all self-referential to each store's own time
series and transfer cleanly to any seller: `day_of_week`, `day_of_month`,
`time_index` (linear trend), `rolling_7` (recent-momentum signal, properly
shifted — see below). None of them encode this-store-specific identity.

**5. Which features may cause leakage or poor generalization?**
- **Poor generalization, not leakage**: `sku` and `base_price` in the offline
  model (§2.2) — these encode *this store's* identity and price points, not a
  transferable pattern.
- **Leakage, already found and fixed**: the offline pipeline's
  `units_rolling_7d_avg` originally included the day it was predicting;
  `.shift(1)` was added before `.rolling(7)` to fix it, and a diagnostic column
  (`units_rolling_7d_avg_LEAKY_DIAGNOSTIC`) is kept specifically to show the
  gap (`models/features.py`, per `docs/data-analysis-notebook-review.md`).
  `total_revenue`, `total_revenue_cbrt`, and `price_per_unit` (=
  revenue/units) are explicitly banned as leakage
  (`feature_metadata.json:103-109`) since they're derived from or highly
  correlated with the target.
- **Leakage, quantified on the live path**: `forecasting.py`'s `rolling_7` uses
  the same `.shift(1)` guard, and `docs/forecast-validation.md:78-96` measured
  the effect directly — removing the shift improves MAPE by ~2.4 points
  (20.97% vs. 23.40%), i.e. the guard is real, load-bearing protection, not
  defensive boilerplate.
- **Not leakage, but a real caveat**: `units_lag_7` in the offline pipeline is
  `.shift(7)` by *row*, not by *calendar day*. Because the panel originally had
  gaps, this lands on exactly 7 calendar days back only ~38.6% of the time
  (documented in `models/features.py:35-39`). Worth knowing if you ever revive
  that feature — it currently isn't used, since the selected feature set
  (`known_covariates_only`) drops both lag features anyway.

**6. What architecture would you recommend instead?** See §5 and §6 — keep
per-request store-level fitting for V1, add the SHAP/explanation layer to it,
and treat a pretrained *store-level* (not SKU-level) global model as an earned
Phase 2 upgrade once you have more than one store's worth of data to prove it
generalizes.

---

## 4. What each feature actually is, in plain terms

| Feature | Meaning | Transfers to any store? |
|---|---|---|
| `day_of_week` | Mon–Sun, captures "sales are higher on weekends" type patterns | Yes |
| `day_of_month` | 1–31 | Yes |
| `time_index` | Days since the seller's first record — the model's handle on "trending up/down over time" | Yes |
| `rolling_7` | Average revenue over the last 7 days, *not counting today* | Yes |
| `sku` | This store's specific product codes | **No** — identity, not pattern |
| `base_price` | This store's price per SKU | **No** — store-specific |
| `category` | 6 product categories used by this one store's catalogue | Partially — category *names* might not even match a real seller's taxonomy |
| `is_promo_active` | Whether a promo event is running that day | Conceptually yes, but needs a real promo calendar per seller — not free |

---

## 5. Option A vs. Option B — the actual decision to make

### Option A: keep per-request fitting (today's live architecture)

**How it works today:** seller uploads CSV → validate → fit a small XGBoost
model on their own history inside the request → recursive 30-day forecast →
(missing today) SHAP → response.

| | |
|---|---|
| Inference latency | Fast in absolute terms for this data size (§2.4); dominated by Cloud Run cold start, not fit time |
| Cloud Run cost | Same cost driver as any stateless request — cold start and container size, not training |
| Generalization to unseen sellers | Best available — the model only ever sees the uploading seller's own data, so there's no cross-seller transfer problem to solve |
| Explainability fit | Best available — SHAP explains *this seller's own fitted model*, so "why" statements are about their actual sales pattern, not a population average |
| Data required today | None beyond what the seller uploads — works with zero additional data collection |
| Statelessness | Fully stateless already (`design-doc.md:106`) |
| Maintenance | One code path to test and backtest (`backend/tests/backtest.py`) |
| Weakness | No opportunity to learn from patterns *across* sellers (e.g. "stores like this one tend to spike around X") — every forecast starts from zero prior knowledge each time |

### Option B: pretrained global model, inference only

**How it would work:** offline training on many stores' historical data → one
saved model → seller uploads CSV → feature-engineer their data → run inference
against the saved model → SHAP → response.

| | |
|---|---|
| Inference latency | Marginally faster than a from-scratch fit, but the gap is small at this data size and swamped by cold start (§2.4) |
| Cloud Run cost | Marginally lower per-request CPU, but same cold-start floor; also now carries a model-artifact load cost |
| Generalization to unseen sellers | **Currently unproven and structurally blocked** (§2.2, §2.3) — would need training data from many real or varied synthetic stores, and features that don't encode any one store's identity |
| Explainability fit | Works, but explanations describe "what the global model learned across stores," which is a weaker, less personal "why" than Option A's — a risk to the stated differentiator |
| Data required today | Substantially more than exists now — needs many stores' worth of history, ideally with real seller variety |
| Statelessness | Still stateless per request, but now depends on a versioned model artifact that must be built, validated, and redeployed out-of-band |
| Maintenance | Two pipelines to maintain: the offline training/eval pipeline and the online serving path, kept in sync |
| Strength | Once real cross-seller data exists, a global model could pick up patterns a single upload can't see (e.g. category-wide seasonality), and would open the door to lower inference cost at very high request volume |

### My recommendation

**Ship Option A for V1, with SHAP added.** It's the only one that's real today,
it's already the more generalizable design (no seller-identity features to
leak), and it directly serves the stated differentiator — an explanation that's
actually about *this* seller's data. Option B is a legitimate future
architecture, but its precondition (real multi-store training data or a much
larger, more varied synthetic corpus, plus proof via backtest that a global
model beats per-seller fitting) doesn't exist yet. Building toward it now would
mean optimizing a cost/latency problem you don't currently have (§2.4) while
leaving the actual differentiator unbuilt.

This isn't a rejection of Option B — it's a sequencing call. See the
option-neutral roadmap in §9: Option B stays reachable, gated on evidence
rather than committed to on schedule.

---

## 6. Proposed architecture (diagram)

```
                         ┌─────────────────────────────────────┐
                         │             Astro frontend            │
                         │   CSV upload → chart → "why" panel    │
                         └───────────────────┬───────────────────┘
                                             │ HTTP/JSON (stateless)
                                             ▼
                         ┌─────────────────────────────────────┐
                         │           FastAPI (Cloud Run)         │
                         │                                       │
   1. validate CSV  ─────┼──▶ app/validation.py                  │
                         │                                       │
   2. get a forecasting  │                                       │
      model  ────────────┼──▶  Option A (V1): fit fresh on this  │
                         │      upload — app/forecasting.py      │
                         │      Option B (earned, Phase 2+):      │
                         │      load a versioned pretrained       │
                         │      model artifact, run inference     │
                         │      only — SAME interface, so the     │
                         │      caller doesn't care which one     │
                         │      is behind it                      │
                         │                                       │
   3. explain   ─────────┼──▶ app/explain.py (NEW — port of       │
                         │      models/train.py's SHAP +          │
                         │      templating logic, §7)             │
                         │                                       │
   4. respond  ──────────┼──▶ {store_level, sku_level, explanation}│
                         └─────────────────────────────────────┘

  Offline / eval track (unchanged in spirit, kept separate from serving):
   data/ ──▶ scripts/build_dataset.py ──▶ scripts/skewness_analysis.py
         ──▶ models/train.py (backtest, ablation, SHAP eval)
         ──▶ models/artifacts/ (model.json, feature_metadata.json, eval CSVs)
   This track's job is to PROVE whether a pretrained model beats Option A
   before it ever gets wired to the live API (§9 Phase 2).
```

The key design point: both options should sit behind the same internal
interface (something like `get_forecast(daily_frame) -> Forecast`), so choosing
A or B later is a config/deploy decision, not a rewrite of the request flow or
the frontend contract.

---

## 7. Updated request flow — the one change that matters most right now

Current flow, annotated (`backend/app/main.py:42-100`):

```
POST /forecast (multipart CSV)
  → read_upload()            (validation.py — 5MB cap, 413 on overflow)
  → validate_csv()           (validation.py — schema/date/numeric checks, 422 on failure)
  → forecast_revenue()       (forecasting.py — fits XGBoost, recursive 30-day forecast, 422 if <30 days history)
  → [NOTHING HERE TODAY]     ← Phase 4 (SHAP) was never wired in; main.py:3-5 says so explicitly
  → JSON response: {store_level, sku_level: null, status, rows, raw_rows, aggregated, date_range}
```

**What to add:** a SHAP step between forecasting and response. This is lower
effort than it sounds, because the pieces already exist:

- `forecasting.py`'s `Forecast` dataclass (`:122-140`) *already* carries
  `model`, `feature_frame`, `feature_names`, and `target` specifically so "Phase
  4's SHAP TreeExplainer can attribute the same fitted model that produced
  these numbers" — this was designed for, just never finished.
- The templating logic — mapping SHAP sign + magnitude to a plain-English
  sentence — already exists and works in `models/train.py:434-460`. It needs to
  be ported to the live per-request model's feature set (`day_of_week`,
  `day_of_month`, `time_index`, `rolling_7`), not reinvented.
- **One real caveat to resolve first**: the live model is fit on
  `log1p(revenue)` (`forecasting.py:71-85`), so raw SHAP values come out in log
  units, not dollars. The code comment at `forecasting.py:82-85` already flags
  this and says to settle it before Phase 4. The templates only need *sign and
  relative rank* ("pushing this forecast up" / "pulling it down"), which
  survive the log transform fine — but if you ever want dollar-denominated
  attributions, that needs either a level-target model or a SHAP
  interaction/link-function adjustment. Recommend: ship sign/rank-based
  templates first (no model change needed), revisit dollar attribution later.

---

## 8. Updated training / evaluation pipeline

**What's already solid and should be kept as-is:**
- Rolling-origin backtesting (not random split) — correctly reflects that a
  forecast can never see its own future (`docs/forecast-validation.md:6-16`).
- Leakage guards with the shift quantified, not just asserted
  (`docs/forecast-validation.md:78-96`).
- Reproducibility: fixed seed, `n_jobs=1`, `tree_method="hist"` for
  byte-identical reruns (`forecasting.py:58-60`).
- Offline pipeline's feature-set ablation and target-transform A/B
  (`models/train.py`) — good practice, and directly answers "should we use
  cube-root or raw" and "which features actually help" with measured numbers
  rather than assumption.
- `models/artifacts/feature_metadata.json` as a versioning record (seed,
  library versions, feature list, params) — this is the right shape for a model
  card; just needs to exist for whatever model *does* get deployed.

**What's missing before either path is production-grade:**
- **No real test suite.** `backend/tests/` has `backtest.py` and `metrics.py`
  (harness code) but no actual `test_*.py` files, despite `forecasting.py`'s own
  docstring and `pytest.ini` referencing one. `pytest` currently has nothing to
  collect. This was flagged before (`docs/mvp-testing-gaps.md`) and is still
  true.
- **No deploy gate.** The backtest exists and produces real numbers
  (`docs/forecast-validation.md`) but nothing runs it automatically before a
  deploy — a regression could ship silently.
- **No model/version registry for the live path.** Option A doesn't need one
  (nothing is persisted), but the moment any pretrained artifact goes live
  (Option B), you need to know which model version served which request, for
  debugging and rollback.
- **No monitoring.** Neither path logs prediction quality in production — you'd
  only find out about a bad forecast if a user complained. Even simple logging
  (predicted vs. eventual actual, when a returning user re-uploads) would give
  you a live accuracy signal.
- **Thin-history accuracy is untested.** The product's floor is 30 days
  (`MIN_HISTORY_DAYS`), but the backtest's shortest fold trains on 438 days
  (`docs/forecast-validation.md:110-115`) — the regime right above the minimum,
  which is exactly where new sellers will land, has never been measured.
- **The `direct`/`known_covariates` strategy comparison isn't reproducible.**
  `forecasting.py`'s comments cite backtest numbers that justified picking
  `recursive`, but those numbers can't be regenerated from this checkout
  (`docs/forecast-validation.md:116-121`) — another symptom of the missing test
  suite.

---

## 9. Migration roadmap (kept option-neutral — both A and B stay reachable)

**Phase 1 — Ship the differentiator on today's architecture**
- Keep per-request fitting (Option A) as the production path.
- Port SHAP + plain-English templates into the live request flow (§7).
- Backfill the missing `backend/tests/test_*.py` suite; wire
  `backend/tests/backtest.py` into a deploy gate so a regression can't ship
  silently.
- Add thin-history backtest folds near `MIN_HISTORY_DAYS=30` to know accuracy
  at the floor, not just on hundreds of days of history.

**Phase 2 — Earn the case for a pretrained model, don't assume it**
- Generate data from *many* distinct synthetic stores (varied price points,
  categories, seasonality shapes) — not more days of the one existing store.
- Train a store-level (not SKU-level) global model using only
  transferable features — the ones in §4's "Yes" column — explicitly excluding
  any store-identity feature like `sku`.
- Backtest that global model against Option A's per-request fit, on held-out
  synthetic stores it never trained on. **Only proceed to Phase 3 if it wins**
  on this comparison; if it doesn't, Option A stays as the architecture and
  this phase's output is a documented "not yet" in the ADR (§10, ADR-0002).

**Phase 3 — Real data, opt-in**
- Collect anonymized, opt-in real seller data (this requires storage and a
  privacy/consent flow that don't exist yet in this no-database V1 —
  `design-doc.md:106` explicitly deferred this).
- Validate the Phase 2 model (or Option A) against real data before trusting
  either on real traffic.

**Phase 4 — Periodic offline retraining, informed by real data**
- Retrain the chosen model on a schedule as real data accumulates.
- Add production monitoring (predicted vs. actual, when available) so accuracy
  drift is caught rather than assumed.
- Revisit Option A vs. B one more time at this point — with real data, a global
  model's cross-seller advantage becomes a real, testable claim instead of a
  hypothesis.

---

## 10. Concrete codebase changes required (described here; not applied by this review)

- **New** `backend/app/explain.py` — SHAP `TreeExplainer` against the
  per-request fitted model (using the `Forecast.model`/`feature_frame` already
  carried for this purpose), ported templating logic from
  `models/train.py:434-460`, adapted to the live feature set.
- **`backend/app/main.py`** — call the new explain step after
  `forecast_revenue()`, add `"explanation": [...]` to the response dict
  (`main.py:84-100`).
- **`src/lib/api.ts`** — extend `ForecastResponseBody` (`:71-80`) and
  `ForecastSuccess` (`:46-56`) with an `explanation: string[]` field; thread it
  through `submitForecast()`'s return (`:119-130`).
- **`src/pages/app.astro`** — replace the hardcoded placeholder text at
  `:121-128` ("that breakdown is planned but not built") with real rendering of
  the `explanation` array.
- **`backend/tests/`** — add real `test_*.py` files (validation edge cases,
  forecast shape, explanation sign/direction on a synthetic case — this last
  one was literally specified as a Phase 4 test requirement in
  `design-doc.md:216-218` and never written).
- **CI/deploy** — wire `python -m tests.backtest` as a gate; fail the deploy if
  MAPE regresses past a threshold vs. the last known-good run.
- **Decide and document** (as an ADR) whether to keep log-target SHAP
  (sign/rank only) or move to a level-target model for dollar-denominated
  attributions — don't leave this as an open code comment
  (`forecasting.py:82-85`) indefinitely.

---

## 11. Risks and technical debt

| Risk | Why it matters | Mitigation |
|---|---|---|
| Two-paths confusion | Anyone reading this repo cold (including future-you) could reasonably assume `model.json` is what's live — it isn't. Easy to ship a bug by wiring the wrong one in. | ADR-0002 makes the split explicit; consider renaming `models/` → `models/research/` or adding a top-level README note. |
| Synthetic-only, single-store generalization gap | Current backtest numbers (24.82% mean MAPE) are real but only proven on one synthetic store's shape of demand — no evidence they hold for a different store, synthetic or real. | Phase 2/3 roadmap (§9); don't market accuracy claims beyond what's been measured. |
| `sku`-identity feature blocks reuse | Any future attempt to "just deploy `model.json`" will silently fail or produce garbage on unknown SKUs, with no error signal unless one is added. | If Phase 2 proceeds, explicitly exclude identity features and add an out-of-vocabulary guard/test. |
| No monitoring in production | A model that quietly gets worse (data drift, a bad seasonal period) has no signal to be caught by. | Cheap to add: log prediction vs. later-observed actual when available. |
| Log-space SHAP | Sign/rank explanations are safe today; a future ask for "$X of this forecast came from Y" would be wrong if built naively on the log-target model. | Resolve as an explicit ADR before dollar-denominated SHAP is requested. |
| Docs partly stale | `docs/mvp-testing-gaps.md` still says the frontend isn't wired to the backend — it is now (Phase 5 shipped). Anyone reading gap docs without checking dates could waste time re-litigating a solved problem. | Add a "last verified" date to living docs; this review supersedes that section. |

---

## 12. Final recommendation and justification

**Keep per-request fitting as the V1 production architecture. Do not deploy
`models/artifacts/model.json` behind the live API in its current form — it
solves a different problem (per-SKU units, not store revenue), was trained on
one store, and cannot score a real seller's SKUs by construction.**

Justification, in order of weight:
1. It's the only architecture that's actually correct and generalizable *today*
   — Option A's features (§4) are the only ones proven not to encode
   store-specific identity that would break on a new seller.
2. It directly serves the stated differentiator. Explainability that describes
   *this seller's own fitted model* is a stronger, more honest "why" than
   explaining a population-average global model — and it's cheaper to ship,
   since the SHAP logic already exists offline and just needs porting (§7).
3. The cost/latency argument for going pretrained doesn't hold at current data
   scale (§2.4) — you'd be optimizing a problem you don't have yet, at the cost
   of the problem (missing explainability) you do have.
4. This doesn't foreclose Option B. §9 keeps it reachable, gated on real
   evidence (a global model beating per-seller fitting on a proper backtest)
   rather than committed to on a schedule you can't yet justify.

---

## 13. What was NOT tested, and why

- **No real Shopify seller data was available for this review** — every claim
  about accuracy or generalization is scoped to the synthetic fixtures in the
  repo (`data/sales_daily.csv`, `data/processed/master_df_cbrt.csv`). Treat all
  MAPE/MAE numbers cited here as "measured on synthetic data," not as
  production accuracy guarantees.
- **No new backtests were run for this review** — all quantitative results
  (§2.4 fit-time claim excepted, which is qualitative/order-of-magnitude) are
  cited from `docs/forecast-validation.md`, which was itself produced in a
  prior session and is reproducible via
  `backend/.venv\Scripts\python -m tests.backtest --leak-demo`.
- **`TrendsArc_PRD.docx` was not read** — it's a binary `.docx` file and wasn't
  opened as part of this review; if it contains requirements not reflected in
  `design-doc.md`, they aren't accounted for here.
- **Cloud Run cold-start timing was not measured** — §2.4's claim that cold
  start dominates fit time is based on general Cloud Run behavior and the small
  size of the fitting workload, not a fresh production measurement. Worth an
  actual cold-start benchmark before treating it as settled.
- **No code was changed by this review** — per the agreed scope, this is
  documentation only; §10's changes are described, not implemented.
