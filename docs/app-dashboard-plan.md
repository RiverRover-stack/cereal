# Trends Arc — Stop the 404s + FastAPI Backend Scaffold

## Context

Every CTA on the landing page points at `/app` (`src/pages/index.astro:10`, rendered at
lines 97, 130, 183), but `src/pages/` contains **only `index.astro`**. So every button in
the site dead-ends on Astro's 404 page. That is the bug driving this pass.

The previous version of this plan proposed building a full forecast dashboard on top of a
local **mock**. That is now explicitly rejected: the frontend and backend are not being
developed independently, and nothing should be invented. So this pass builds a real
backend foundation and a real (honest) `/app` page — and stops before any forecasting
logic exists, so no fabricated numbers ever reach the screen.

### Verified before planning (not assumed)
- `src/pages/` = `index.astro` only. `src/components/` **exists but is empty**
  (the earlier plan wrongly said the folder didn't exist).
- `astro.config.mjs` has **no adapter and no `output`** → pure static. A same-origin
  `/api/*` route is therefore impossible today; the backend must be a separate origin.
- No backend code anywhere in the repo — only `.md` docs.
- Python **3.13.1**, pip **26.0.1**, Node **v22.13.0**.
- `fastapi 0.128.0`, `xgboost 3.1.3`, `shap 0.50.0` are **already installed** globally,
  so there is no Python-3.13 ML wheel risk to design around.
- Astro docs confirm `src/pages/404.astro` builds to `404.html`
  (docs.astro.build/en/basics/astro-pages/#custom-404-error-page).

### Decisions (from user)
- Build the `/app` route **and** a FastAPI backend — not a mock.
- Backend scope this pass: **scaffold + health check only** (design-doc Phase 1).
  Phases 2–4 (validation, XGBoost, SHAP) are explicitly deferred.
- Layout: **keep Astro at the repo root**, add a sibling `backend/`. Do not restructure
  into `/frontend` per design-doc Phase 1 — the architecture matches, the folder name doesn't.
- No backend exists elsewhere; nothing to point at yet.

---

## Work items

### 1. `backend/` — FastAPI scaffold (design-doc Phase 1)
New sibling folder at the repo root. **No forecasting logic** — scaffolding only.
- `backend/app/main.py` — FastAPI app, `GET /health` → `{"status": "ok"}`. CORS middleware
  allowing the Astro dev origin (`http://localhost:4321`) so the browser can actually reach it.
- `backend/requirements.txt` — `fastapi`, `uvicorn[standard]`, `pandas`, `xgboost`, `shap`,
  pinned to the versions verified installed above (xgboost 3.1.3, shap 0.50.0, fastapi 0.128.0);
  pandas/uvicorn pins to be read off the environment at implementation time, not guessed.
- `backend/Dockerfile` — installs from `requirements.txt`, runs uvicorn, reads `PORT` from
  the environment (Cloud Run requirement, design-doc Phase 6). `backend/.dockerignore`.
- `backend/README.md` — how to create a venv, install, and run locally.
- `.gitignore` — add Python entries (`__pycache__/`, `.venv/`, `*.pyc`); verify what the
  existing file already covers before appending.

`POST /forecast` is **not** created this pass. The endpoint does not exist, so the UI must
not pretend it does.

### 2. `src/pages/404.astro` — no more dead ends
A real 404 in the site's own design language (`Layout.astro` + the extracted header/footer),
with a short message and a link home. Builds to `404.html`.

### 3. `src/components/` — fill the empty folder
Extract the chrome currently inlined and duplicated in `index.astro` so `/app` and `404`
reuse one source, appearance unchanged:
- `Logo.astro` — the ink rounded-square SVG glyph + wordmark, duplicated verbatim at
  `index.astro:74-95` (header, 8×8/18px) and `199-220` (footer, 6×6/14px). Props for size
  and wrapper element.
- `SiteHeader.astro` — `index.astro:69-103`, with props for the CTA label/href.
- `SiteFooter.astro` — `index.astro:195-223`.
- `CtaButton.astro` — the ink pill repeated at `index.astro:96-101, 129-134, 182-187`;
  `sm`/`lg` sizes.

Then refactor `index.astro` to consume them. Visual output must be unchanged.

### 4. `src/pages/app.astro` — the real, honest tool page
Static page using `Layout` + `SiteHeader`/`SiteFooter`, in the existing stark design system.
- **Upload UI**: drag-and-drop + click-to-browse zone (`border-hairline`, `rounded-lg`,
  `bg-canvas-soft`), mono eyebrow, and the CSV requirements stated from design-doc §Phase 2
  (`date`, `revenue`, `units_sold` required; SKU optional; 5 MB max).
- **Genuinely wired**: a small `<script>` island calls `GET ${PUBLIC_API_URL}/health` on load
  and shows the live backend status. This proves the connection is real rather than asserted.
- **Submit is honest**: because `/forecast` does not exist yet, the submit path states plainly
  that forecasting isn't implemented yet and links to the backend status. **No mock forecast,
  no placeholder chart, no invented numbers.** The chart, KPI, and "why" panel arrive in a
  later pass once Phases 2–4 are built.
- `src/lib/api.ts` — reads `import.meta.env.PUBLIC_API_URL` (falling back to
  `http://localhost:8000`) and exposes `checkHealth()`. `submitForecast()` is deliberately
  **not** written yet.
- `.env.example` documenting `PUBLIC_API_URL`.

### 5. Deferred (explicitly not this pass)
Semantic error/warning tokens in `global.css`, the SVG forecast chart, the "why" panel,
scroll-reveal motion, and the `site.webmanifest` theme-color fix. All belong with the
Phase 2–4 backend work; adding them now would mean styling states that cannot occur.

---

## Files
**New:** `backend/app/main.py` · `backend/requirements.txt` · `backend/Dockerfile` ·
`backend/.dockerignore` · `backend/README.md` · `src/pages/404.astro` · `src/pages/app.astro` ·
`src/components/{Logo,SiteHeader,SiteFooter,CtaButton}.astro` · `src/lib/api.ts` · `.env.example`
**Modified:** `src/pages/index.astro` (consume components) · `.gitignore` (Python)

## Reuse (don't reinvent)
- `src/layouts/Layout.astro` as the shell for both new pages.
- Existing tokens/utilities in `src/styles/global.css` (`bg-canvas-soft`, `border-hairline`,
  `rounded-pill`, `shadow-soft`, `text-display-*`, `font-mono`) — no new tokens needed.
- Component specs in `DESIGN.md` (`form-input`, `card-marketing`, `ex-empty-state-card`).

## Verification (every claim checked, nothing asserted)
1. `python -m venv`, install `backend/requirements.txt`, run `uvicorn` — then `curl`
   `GET /health` and paste the actual response. If install fails, report it rather than
   claiming success.
2. `astro dev --background`, then `astro dev status`/`logs` (per `CLAUDE.md`) to confirm it's up.
3. With **claude-in-chrome**: load `/`, click **every** CTA and confirm each now reaches
   `/app` instead of the 404 page — this is the specific bug reported, so it gets clicked,
   not assumed.
4. On `/app`, confirm the health indicator shows the backend live; then stop the backend
   and confirm it reports unreachable. Check `read_console_messages` for errors and
   `read_network_requests` to confirm the `/health` call actually fired.
5. Visit a bogus URL (e.g. `/nope`) and confirm the styled 404 renders.
6. Compare `/` against the current page after the component extraction — confirm no visual regression.
7. `astro build` — confirm `index.html`, `app/index.html`, and `404.html` are all emitted, clean.
