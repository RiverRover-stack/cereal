# Trends Arc — Design Doc
**Author:** Kaustubh | **Status:** Draft | **Target MVP timeline:** 2 weeks

## 1. Problem
E-commerce sellers make inventory and revenue decisions largely by looking backward. Shopify's native analytics (and most lightweight analytics apps) show *what happened* — last month's revenue, last week's units sold — but not *what's likely to happen next*. This leaves sellers guessing on decisions that directly cost money: how much stock to reorder, whether to ramp ad spend, whether a slow week is noise or the start of a real decline.

Forecasting tools that do exist tend to produce a number with no explanation — "next month: $42,300" — and stop there. For a solo seller or small team, an unexplained prediction is hard to trust and even harder to act on. If the forecast doesn't say *why* it's up or down, there's no way to sanity-check it against what the seller actually knows about their business (a promotion they're running, a product that's trending, a seasonal pattern), so it gets ignored rather than used.

The core problem, then, isn't "sellers lack a forecast" — it's "sellers lack a forecast they can trust enough to act on." That trust gap, more than raw predictive accuracy, is what's stopping small e-commerce sellers from planning inventory and revenue with any confidence.

## 2. Solution Overview
A web tool that turns a seller's raw sales history into a trustworthy, explained forecast — not just a number, but the reasoning behind it.

**How it works, end to end:**
1. Seller uploads a CSV export of their order history (date, revenue, units, SKU).
2. The tool cleans and validates the data, then runs it through a trained forecasting model to produce a 30-day revenue forecast.
3. Alongside the forecast, the tool surfaces a plain-English explanation of *why* the number looks the way it does — driven by SHAP feature attribution on the underlying model, translated from raw feature weights ("feature_3: +0.42") into readable statements ("your last 3 Fridays show a recurring spike; this is pulling the forecast up").
4. The seller sees a forecast chart plus a short "why" panel next to it, and can use both to decide on reordering, ad spend, or promotions with actual reasoning to check against what they already know about their business.

The differentiator isn't forecasting accuracy alone — plenty of tools forecast. It's that this tool is the only one in reach of a solo seller that explains its own reasoning, closing the trust gap described in Section 1. This directly leverages a forecasting + explainability pipeline already built and proven in a prior project (a knowledge-distillation pipeline with SHAP-based decision analysis), redeployed here for a domain where people will pay for it.

## 3. Tech Stack

**Frontend**
- **AstroJS** — marketing/SEO pages (free-tool landing page, ranks on Google) plus the lightweight app dashboard (upload form, forecast chart, "why" panel)
- **Cloudflare Pages** — hosting, free tier, ties into the original SEO/Cloudflare deployment plan

**Backend**
- **FastAPI (Python)** — serves the forecast endpoint, handles CSV upload/validation
- **Google Cloud Run** — hosting, chosen over Railway/Fly.io because: scale-to-zero keeps cost near-$0 pre-revenue, and it builds directly on Docker/FastAPI/GCP experience already established from a prior project (the energy consumption predictor). Cold-start risk (Python + ML libs) is mitigated by keeping the trained model loaded in a global variable so only the first request after idle pays the cold-start cost.
- **Docker** — containerizes the backend for Cloud Run deployment

**Model layer**
- **XGBoost** — core forecasting model for the 30-day revenue prediction. Chosen over an LSTM/ensemble approach for V1 specifically because: (1) target users likely have thin historical data (6-18 months), where XGBoost outperforms a data-hungry LSTM; (2) SHAP's `TreeExplainer` is fast and exact on tree models, versus slow/approximate `DeepExplainer` on LSTMs — and explainability is the core differentiator, so the model choice needs to make that easy; (3) faster to build correctly within the 2-week window. LSTM/ensemble forecasting is deferred to the Future Integrations roadmap (Section 5), once real usage data justifies longer-horizon or multi-series forecasts.
- **SHAP** — generates feature attributions for each forecast; attributions are translated into plain-English "why" statements for the explanation panel (templated for V1, not full LLM generation — keeps cost and output predictable)

**Data**
- CSV upload only for V1 (no Shopify OAuth) — sidesteps the Shopify app review process, which can take weeks, and lets the MVP ship in the 2-week window
- No persistent database required for V1 — forecasts run on the uploaded file per session; no user accounts or saved history yet (this is itself a V1 boundary, not an oversight — see Section 4)

**Why this stack overall:** every piece was chosen to hit two constraints simultaneously — ship a real, defensible MVP inside 2 weeks, and make the "why" explanation (the actual differentiator) as reliable and fast to build as possible. Nothing here is a placeholder; each piece is also the foundation the Future Integrations roadmap builds on directly (e.g., XGBoost + SHAP extends naturally to per-SKU forecasting before LSTM is ever needed).

## 4. MVP Scope (V1)

**In scope:**
- CSV upload of order history (date, revenue, units sold, SKU columns)
- Data validation on upload: reject/flag missing required columns, empty files, malformed dates, with a clear user-facing error (not a raw 500)
- 30-day revenue forecast (store-level, not per-SKU)
- XGBoost model trained on uploaded historical data, run server-side per upload
- SHAP-based feature attribution on the forecast, translated into 3-5 templated plain-English "why" bullets
- Single dashboard view: forecast chart + "why" panel, side by side
- No login required to try it (free, one-off upload → forecast → explanation) — this is the SEO/lead-gen hook
- Astro marketing/landing page targeting "shopify sales forecast" — style search intent

**Explicitly out of scope for V1 (not oversights — deferred deliberately):**
- User accounts, login, saved forecast history
- Live Shopify OAuth connection (CSV only)
- Per-SKU forecasting
- LSTM/ensemble modeling
- Payment/subscription billing
- Multi-store support
- Forecast accuracy tracking over time ("we predicted X, actual was Y")
- Inventory reorder recommendations, ad-spend suggestions

**Definition of done for V1:** A seller can land on the marketing page, upload a real Shopify CSV export, and within seconds see a 30-day revenue forecast with a readable explanation of what's driving it — end to end, no errors, on both desktop and mobile.

## 5. Future Integrations (Locked Roadmap, Not V1)
Sequenced roughly by value delivered vs. dependency on earlier items — not a strict order, but a sane build sequence once V1 has real users.

1. **Paid tier / billing (Stripe)** — needed early, right after V1 validates demand, to start converting free-tool users into paying customers. Unlocks everything below.
2. **Shopify OAuth live connection** — replaces manual CSV upload with an auto-refreshing live data feed. Highest-value convenience upgrade; likely the anchor feature of the first paid tier.
3. **Forecast accuracy tracking** — store each forecast and compare it against actual results a month later ("we predicted $42K, actual was $44K"). Requires persistence (a database, not needed in V1). This is a major trust-builder and retention driver — directly reinforces the core "trustworthy forecast" positioning.
4. **Per-SKU forecasting** — natural extension of the existing XGBoost + SHAP pipeline, just re-run per product instead of store-wide. High value for sellers with larger catalogs.
5. **LSTM / ensemble modeling for longer horizons (60-90 day)** — revisited once usage data shows users have enough history and are asking for longer-range forecasts (see Section 3 for why this is deferred, not dropped).
6. **Inventory reorder recommendations** — builds on per-SKU forecasting + accuracy tracking; turns the forecast into a concrete action ("reorder 120 units of SKU X by [date]").
7. **Multi-store support** — for sellers or agencies managing more than one storefront.
8. **Ad-spend suggestions** — requires integrating an ads platform API (Meta/Google Ads); furthest out, biggest scope increase.

Each of these should get its own short design note when it's actually prioritized — this list exists so V1 build decisions don't accidentally foreclose them (e.g., API response schemas should be written with per-SKU and multi-store in mind even though V1 doesn't use them yet).

## 6. Architecture

**Request flow (V1, stateless):**
```
User browser
   │
   ▼
Astro app (Cloudflare Pages)
   │  1. Landing/marketing page (SEO)
   │  2. Upload form + dashboard (app UI)
   │
   ▼  POST /forecast (multipart CSV upload)
FastAPI backend (Docker container on Cloud Run)
   │  1. Validate CSV (columns, types, size limit)
   │  2. Feature engineering (day-of-week, trend, recent aggregates)
   │  3. Train/run XGBoost model on the uploaded data
   │  4. Run SHAP TreeExplainer on the forecast
   │  5. Translate SHAP values → 3-5 templated "why" bullets
   │  6. Return JSON: { forecast: [...], explanation: [...] }
   │
   ▼
Astro frontend renders forecast chart + "why" panel
```

**Key architectural decisions:**
- **No database in V1** — each forecast is computed fresh per upload, nothing persisted. Keeps the backend stateless and simple, and sidesteps user-data storage/privacy questions until they need to be dealt with properly (see Section 8).
- **Model trained per-request, not pre-trained globally** — because each seller's data is different, the model fits to their uploaded history at request time rather than serving one global model. This keeps forecasts personalized but means request latency includes training time — worth watching in testing (see Section 8 risk on cold start + training latency stacking).
- **Backend and frontend are fully decoupled** — Astro talks to FastAPI purely over HTTP/JSON. This keeps the door open to add a mobile client or public API later without touching the frontend.
- **Response schema should anticipate Section 5 items** — e.g., structure the JSON response as `{ store_level: {...}, sku_level: null }` from the start, even though `sku_level` is unused in V1, so per-SKU forecasting (Section 5, item 4) doesn't require a breaking schema change later.

## 7. Claude Code Build Prompts (Phase-Wise)
Run these as separate Claude Code sessions/prompts, in order. Each phase is scoped to stay small so Claude Code has full context of just that piece — don't paste all phases into one session at once.

**Standing safeguards — apply to every phase prompt below, not just one:**
- Before writing anything, confirm which referenced files were actually read successfully (and roughly how long they are). If any referenced file/path wasn't found, stop and say so instead of proceeding on assumptions.
- Do not state specific numbers, accuracy percentages, benchmarks, or performance claims unless they come directly from a provided source. If a real number doesn't exist yet, use qualitative language instead of inventing one.
- Do not invent library/API details from memory when precision matters (e.g., XGBoost/SHAP method signatures, FastAPI patterns) — check installed package versions/docs if uncertain rather than guessing.
- If a source document doesn't cover something needed for the task, ask rather than filling the gap with an assumption.
- If a task depends on live web access (e.g., fetching a competitor site) and that access isn't actually available, say so explicitly rather than reconstructing plausible-sounding content from training data.

---

**Phase 1 — Project Scaffold**
```
Set up a monorepo for a web tool called Trends Arc with two parts:

1. `/frontend` — an AstroJS project. Include a placeholder landing page
   (title, one-paragraph description, a "Try it free" CTA button) and a
   placeholder `/app` route for the dashboard (empty for now).
2. `/backend` — a FastAPI project with a basic health-check endpoint
   (`GET /health` returning {"status": "ok"}), structured for Docker
   deployment to Google Cloud Run. Include a Dockerfile that installs
   dependencies from requirements.txt and runs the app with uvicorn.
   Include a requirements.txt with fastapi, uvicorn, pandas, xgboost, shap.

Add a root README.md explaining the two folders and how to run each
locally. Add .gitignore appropriate for a Python + Node project.

Do not implement any forecasting logic yet — this phase is scaffolding only.
```

---

**Phase 2 — CSV Upload + Validation**
```
In the /backend FastAPI project, implement a POST /forecast endpoint that:

1. Accepts a multipart CSV file upload.
2. Validates the CSV using pandas: must contain columns `date`, `revenue`,
   `units_sold` (SKU column optional for V1). Reject the file with a clear
   422 error message if columns are missing, dates are unparseable, the
   file is empty, or revenue/units_sold contain non-numeric values.
3. Enforce a max file size of 5MB, reject larger files with a 413 error.
4. For now, if validation passes, just return the parsed row count as
   {"status": "valid", "rows": N} — no forecasting yet.

Write a few pytest tests covering: valid CSV, missing column, empty file,
malformed date, oversized file.
```

---

**Phase 3 — Forecasting Model**
```
Extend the /backend POST /forecast endpoint. After CSV validation passes,
add a forecasting step:

1. Feature-engineer the validated data: day-of-week, rolling 7-day average
   revenue, day-of-month, and a linear time index (for trend).
2. Train an XGBoost regressor on this data to predict daily revenue,
   using the engineered features. Keep training fast — this runs
   synchronously per request, so favor sensible defaults over extensive
   hyperparameter tuning for V1.
3. Use the trained model to produce a 30-day forward daily revenue
   forecast, and also return a 30-day total.
4. Return JSON in this shape:
   {
     "store_level": {
       "daily_forecast": [{"date": "...", "predicted_revenue": ...}, ...],
       "total_30_day": ...
     },
     "sku_level": null
   }
   (sku_level stays null for V1 — reserved for a future per-SKU feature,
   do not implement it now.)

Handle the edge case of very little historical data (e.g., under 30 rows)
by returning a clear error asking the user to upload more history, rather
than producing an unreliable forecast.
```

---

**Phase 4 — SHAP Explainability Layer**
```
Extend the /backend forecasting logic to add explainability:

1. After the XGBoost model produces the forecast, run SHAP's
   TreeExplainer against the model to get feature attributions for the
   forecast period.
2. Aggregate the SHAP values per feature (day-of-week, rolling average,
   trend, day-of-month) across the forecast window.
3. Translate the top 3-5 features by absolute SHAP contribution into
   plain-English template strings. Examples:
   - rolling 7-day average has high positive contribution →
     "Your recent sales trend is strong, which is pushing this forecast up."
   - a specific day-of-week has high positive contribution →
     "Your sales tend to spike on [day], which factors into this forecast."
   - trend feature has negative contribution →
     "Your overall sales trend has been declining, which is pulling this
     forecast down."
   Write this as a small templating function that maps feature name +
   sign + magnitude bucket (high/medium/low) to a sentence.
4. Add "explanation": [list of 3-5 strings] to the JSON response.

Write a test that mocks a trained model and confirms the explanation
list is generated with the right sign/direction for a synthetic case
where one feature clearly dominates.
```

---

**Phase 5 — Frontend Dashboard**
```
In the /frontend AstroJS project, build the /app dashboard page:

1. A file upload component for the CSV (drag-and-drop or click-to-browse).
2. On upload, POST the file to the backend's /forecast endpoint (backend
   URL from an environment variable, not hardcoded).
3. Show a loading state while waiting for the response.
4. On success, render:
   - A line chart of the 30-day daily forecast (use a lightweight charting
     approach suitable for Astro — a small client-side island component).
   - A "Why this forecast" panel listing the explanation bullets returned
     by the backend.
   - The 30-day total revenue forecast as a prominent summary number.
5. On error (validation failure, insufficient data, etc.), show the
   backend's error message clearly, not a generic failure state.

Keep the design clean and minimal — this is a single-purpose tool page,
not a full app shell. Mobile-responsive.
```

---

**Phase 6 — Deployment Prep**
```
Prepare both projects for deployment:

1. /backend: finalize the Dockerfile for Cloud Run — ensure the app reads
   PORT from the environment (Cloud Run requirement), the trained-model
   dependencies are all in requirements.txt with pinned versions, and add
   a .dockerignore. Add a note in the README on deploying via
   `gcloud run deploy`.
2. /frontend: add a wrangler/Cloudflare Pages config if needed, ensure the
   backend API URL is read from a build-time environment variable so it
   can point to the deployed Cloud Run URL in production vs localhost in
   dev.
3. Add a basic Privacy Policy and Terms page (placeholder content is fine
   for now, flagged clearly as [PLACEHOLDER — replace before real launch])
   since the tool accepts user file uploads.

Do not actually deploy — just get both projects deploy-ready.
```

---

**Phase 7 — SEO Landing Page Content**
```
Rewrite the /frontend landing page content (not the /app dashboard) to be
SEO-optimized for the search intent "shopify sales forecast" and related
terms ("ecommerce revenue forecast tool", "sales prediction shopify").

Include:
- A clear H1 stating what the tool does in plain language
- A short explanation of the "why" differentiator (explained forecasts,
  not just numbers)
- A prominent free-tool CTA
- A brief FAQ section (3-5 Q&As) addressing likely searcher questions
  (e.g., "Do I need to connect my Shopify store?", "Is this free?",
  "How accurate is the forecast?")
- Proper semantic HTML (single H1, meaningful H2s, descriptive title tag
  and meta description)

Keep copy concise and avoid generic filler — every sentence should either
explain the tool or address a real objection a seller would have.
```

## 8. Open Questions / Risks

- **Per-request training latency** — training XGBoost synchronously per upload, stacked with Cloud Run cold starts, could make the free-tool experience feel slow. Needs real testing with realistic CSV sizes before assuming this is fine; may need a "processing..." UX pattern or a background job if latency is bad.
- **Forecast quality on thin/messy data** — many small sellers will have short or gappy histories. The tool needs a graceful minimum-data threshold (Phase 3 handles this) rather than silently producing a bad forecast that damages trust on the very first use.
- **Explanation template quality** — SHAP-to-plain-English templating (Phase 4) is inherently approximate; worth manually sanity-checking explanations against a handful of real CSVs before launch to make sure they don't sound generic or, worse, wrong.
- **Data privacy** — even without accounts, users are uploading real sales data. Needs at minimum a clear "we don't store your data" statement (if true) or a real privacy policy (Phase 6 flags this as a placeholder to replace before real launch) — not optional once this is live and collecting uploads publicly.
- **Abuse/cost risk** — a public, free, compute-heavy (model training per request) endpoint is a natural target for scraping or abuse, which directly costs money on Cloud Run. Rate limiting is listed in the deploy checklist but should be treated as a launch blocker, not a nice-to-have.
- **AdSense/SEO timeline dependency** — the original strategy assumes organic ranking within a few months; if that's slower than expected, the free-tool funnel won't produce paid conversions on the hoped-for timeline. Worth having a distribution plan (Product Hunt, Reddit, build-in-public) that doesn't solely depend on SEO.
- **Competitive risk** — established players (Triple Whale, Lifetimely, Inventory Planner) could add explainability features themselves; the moat here is being first, narrow, and cheap for small sellers, not unreplicable technology.

## 9. Cost Estimate (Build → Deploy → Maintenance)

**Build phase (0 → MVP launch, ~2 weeks):**
| Item | Cost |
|---|---|
| Domain (.com) | ~$10-15/year |
| Development (your own time via Claude Code) | $0 out-of-pocket |
| Cloudflare Pages hosting | $0 (free tier) |
| Google Cloud Run | $0 (free tier covers MVP-stage traffic — 2M requests/month, 360K GB-seconds) |
| Google Cloud account/billing setup | $0 (requires a card on file, but free tier should cover early usage) |
| **Total to launch** | **~$10-15** |

**Ongoing monthly costs (post-launch, low-to-moderate traffic):**
| Item | Cost |
|---|---|
| Cloud Run (assuming free tier is exceeded modestly) | ~$0-20/month depending on traffic and whether min-instances is set to avoid cold starts |
| Cloudflare Pages | $0 (free tier is generous for a site this size) |
| Domain renewal | ~$1/month amortized |
| Monitoring/error tracking (optional, e.g. a free-tier Sentry) | $0 on free tier |
| **Total ongoing (early stage)** | **~$0-25/month** |

**Costs that appear later, once paid tiers/integrations are added (Section 5):**
| Item | Cost |
|---|---|
| Stripe (billing) | 2.9% + $0.30 per transaction — no fixed cost, scales with revenue |
| Database (e.g., Cloud SQL or a managed Postgres) once persistence is needed | ~$10-30/month depending on provider/tier |
| Shopify OAuth app review/hosting | $0 in fees, but engineering time |
| Increased Cloud Run usage as traffic grows | Scales with usage — budget to monitor, not a fixed number |

**Bottom line:** V1 is close to free to build and run (~$10-15 one-time, plus low single-digit dollars/month at low traffic) — the main cost is your time, not infrastructure. Costs only become meaningful once you add persistence (a database) and billing, which is exactly why Section 5 sequences those as deliberate later steps rather than V1 requirements.
