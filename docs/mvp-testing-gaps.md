# Trends Arc — what's left to test before MVP is done

**Read against:** `design-doc.md` (Sections 4, 6, 7, 8)
**Method:** read every file the design doc's 7 build phases touch, then actually ran the
backend (`uvicorn app.main:app --port 8123`) and fired real requests at it with `curl`,
using the real `data/sales_daily.csv` fixture and some hand-built bad CSVs. Not a code
read-through guess — the numbers below are what the server actually returned.

## The one thing to fix first: the frontend doesn't talk to the backend

This is the biggest gap, and it's not a testing gap — it's a missing feature. The backend's
`/forecast` endpoint is fully built and working (see "What I confirmed works" below), but
`src/pages/app.astro` never calls it. Open the file and the "Get my forecast" button's click
handler is literally:

```js
submitReadout.textContent =
  'Forecasting is not implemented yet — there is no endpoint to submit to, so nothing was sent...';
```

And `src/lib/api.ts` has a comment explaining why `submitForecast()` doesn't exist yet:

> `submitForecast()` is deliberately absent: `POST /forecast` has not been built
> (design-doc Phases 2-4), and a client for a nonexistent endpoint is how placeholder
> UI gets written.

That comment is now stale — Phase 2 and 3 (validation + forecasting) **are** built on the
backend. The frontend just hasn't caught up. This is design-doc **Phase 5** (Frontend
Dashboard, lines 223-242) and it hasn't been started: no upload wiring, no chart, no "why"
panel, no loading state tied to a real request. Until this is built, there is nothing an
end user can actually do on the site — the "Definition of done for V1" (design-doc.md:65,
"a seller can land on the marketing page, upload a real Shopify CSV export, and... see a
30-day revenue forecast") cannot happen no matter how well the backend works.

## What I confirmed works (backend, tested live)

Started the real server and hit it for real — not simulated:

```
cd backend && .venv/Scripts/python -m uvicorn app.main:app --port 8123
```

| Test | Command | Result |
|---|---|---|
| Health check | `curl http://localhost:8123/health` | `{"status":"ok"}` |
| Real forecast, 1095-row CSV | `curl -X POST .../forecast -F "file=@data/sales_daily.csv"` | **HTTP 200** in **0.33s** — 30-day forecast returned, total `$673,144.57`, `rows: 1095`, `aggregated: false` |
| Missing column | CSV with no `revenue` column | **HTTP 422** — `"Missing required column(s): revenue..."` |
| Empty file | 0-byte file | **HTTP 422** — `"The uploaded file is empty."` |
| Insufficient history | 10-day CSV | **HTTP 422** — `"Not enough history to forecast: 10 day(s)... at least 30 are needed."` |
| Malformed date | `13/45/2024` | **HTTP 422** — `"Unparseable date(s) on line(s) 2: '13/45/2024'..."` |
| Oversized file | 7.3MB CSV | **HTTP 413** — `"File is larger than the 5MB limit..."` |

Every Phase 2 test case the design doc calls for (design-doc.md:157-158: valid CSV, missing
column, empty file, malformed date, oversized file) passes when exercised directly against
the running server. The `InsufficientHistoryError` guard from Phase 3 (design-doc.md:187)
also fires correctly.

One measured finding worth flagging positively: design-doc.md Section 8 lists **per-request
training latency** as an open risk ("could make the free-tool experience feel slow").
On a realistic 3-year, 1095-row history, the full validate → feature-engineer → train →
recursive-forecast pipeline took **0.33 seconds**. That risk looks resolved for
realistically-sized CSVs — but this was tested locally with a warm process, not through a
Cloud Run cold start, so the cold-start half of that risk (Section 8, first bullet) is
still untested.

## What's built but not wired into the request path

`backend/app/forecasting.py` is more sophisticated than the design doc's Phase 3 prompt
asked for — it implements and documents three forecasting strategies (`recursive`,
`direct`, `known_covariates`) and two target scales (`log`, `level`), with comments citing
backtest numbers that justify the defaults (`recursive` + `log`, chosen over `level` because
it scored 14.63% MAPE vs 16.83%). That's good, evidence-based work.

But **the tests that produced those numbers don't exist in this checkout.** The code
repeatedly cites files that aren't there:

- `backend/app/forecasting.py` references `tests/backtest.py`, `tests/test_forecasting.py`,
  and a `--leak-demo` mode
- `backend/pytest.ini` sets `testpaths = tests`
- `backend/requirements-dev.txt` references `tests/backtest.py` and `tests/metrics.py`
- `backend/.gitignore`-adjacent rule mentions `backend/tests/fixtures/generate_fixtures.py`

None of `backend/tests/`, `backend/tests/test_forecasting.py`, or `backend/tests/backtest.py`
exist on disk right now:

```
$ find backend -iname "test_*.py"
(no output)
```

`backend/.pytest_cache/v/cache/nodeids` is empty too, meaning the last time pytest ran here
it collected zero tests. So the specific numbers baked into `DEFAULT_STRATEGY` and
`DEFAULT_TARGET` (14.63% vs 16.83% MAPE) are not independently reproducible in this repo as
it stands — you'd have to take the code comments' word for it, which is exactly the kind of
thing the design doc's own "standing safeguards" (design-doc.md:116) warn against ("Do not
state specific numbers... unless they come directly from a provided source").

Separately, `models/` contains a **different**, offline-trained model (`models/artifacts/model.json`,
78 rounds, `max_depth=5`) with its own backtest suite (`models/backtest.py`,
`models/train.py`, `models/artifacts/backtest_folds.csv`). That pipeline is not used by
`POST /forecast` at all — the live endpoint always trains fresh per-request per
design-doc.md:107. Worth confirming with whoever built `models/` whether that was meant to
feed back into `backend/app/forecasting.py`'s defaults, or is a separate research track.

## Phase 4 (SHAP explainability) — confirmed not started

`backend/app/main.py`'s own docstring says so directly: *"The explanation step (Phase 4:
SHAP) is not built yet, so no `explanation` key is returned."* I confirmed this against the
live response — the JSON has `store_level`, `sku_level`, `status`, `rows`, `raw_rows`,
`aggregated`, `date_range`, and no `explanation` key. `shap` is in `requirements.txt` but
never imported in `backend/app/`.

This matters more than a normal missing feature: per design-doc.md Section 1-2, the
"why panel" explaining the forecast **is the stated differentiator**, not a nice-to-have.
The landing page (`src/pages/index.astro`) already promises it in its own copy — the first
benefit bullet says *"Every prediction comes with a short 'why' panel"* — so right now the
marketing copy is ahead of the product.

Also worth noting: `forecasting.py`'s own comment flags an unresolved decision that Phase 4
will need to make before it can start: the model fits on `log1p(revenue)`, so raw SHAP
values come out in log units, not dollars. The design doc's example templates
(design-doc.md:204-211) are directional ("pushing this forecast up"), which still works,
but anyone wanting a dollar-denominated attribution needs to settle that first.

## Phase 6 (Deployment prep) — partially done

- `backend/Dockerfile` looks genuinely deploy-ready: reads `PORT` from env, has a
  `.dockerignore`, pinned requirements. Not tested here (didn't build/run the container).
- **No Privacy Policy or Terms page exists.** `find src/pages -iname "*privacy*" -o -iname "*terms*"`
  returns nothing — only `404.astro`, `app.astro`, `index.astro`. This is worse than "not
  built yet": the landing page's own FAQ answer says *"See our Privacy Policy for the
  specifics"* (`src/pages/index.astro:53`) — that's a promise to a real link that doesn't
  exist. Design-doc.md:259-261 explicitly calls for at least a placeholder page here.
- **No rate limiting.** Grepped `backend/app/*.py` and `requirements.txt` for
  `rate.?limit|slowapi` — nothing. Design-doc.md Section 8 calls this out explicitly as
  *"a launch blocker, not a nice-to-have"* given the endpoint trains a model per request
  (compute cost + abuse surface). Not present at all right now.
- No `wrangler.toml` or Cloudflare Pages config found anywhere in the repo root.

## Not tested, and why

- **The actual browser experience** — I didn't drive this through Chrome. Given the
  frontend doesn't call the backend yet (see above), there's nothing end-to-end to click
  through; testing the UI now would only confirm the placeholder message shows up.
- **Cloud Run cold start behavior** — only tested against a warm local `uvicorn` process.
  The design doc's cold-start + training-latency stacking risk (Section 8) is only half
  addressed by the 0.33s number above.
- **Docker build** — didn't run `docker build`/`docker run` against `backend/Dockerfile`,
  so I can't confirm the container actually starts cleanly, only that the file reads as
  correct.
- **Explanation quality** — can't test what doesn't exist (Phase 4).
- **Mobile responsiveness** — design-doc.md:65 calls for "on both desktop and mobile";
  not checked visually in this pass.

## Priority order to close before calling this an MVP

1. **Wire the frontend to the backend** (Phase 5) — nothing works for a real user until
   this exists; this is the only item that blocks the "Definition of done" in
   design-doc.md:65 outright.
2. **Add rate limiting** — design doc calls this a launch blocker, and it's currently fully
   absent.
3. **Add a real (even placeholder) Privacy Policy page** — the landing page already links
   to one that doesn't exist.
4. **Write the missing backend test suite** (`backend/tests/`) — `pytest.ini` and
   `requirements-dev.txt` are already configured for it, and the forecasting module's
   design decisions (recursive vs. direct, log vs. level target) currently rest on numbers
   that can't be reproduced in this checkout.
5. **Phase 4 (SHAP explanation)** — the product's actual differentiator; landing page copy
   already promises it.
6. Cold-start / Docker build verification, mobile check — lower risk, easy to defer to
   just before actual deploy.
