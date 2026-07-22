# Trends Arc

*An explainable sales forecast for e-commerce sellers — not just a number, but why.*

## What is this?

If you sell things online, you can easily see what your sales *were* — last
week, last month. What's harder is knowing what they're *about to be*.

Trends Arc answers that. Upload a CSV of your order history and get back:

- A **30-day revenue forecast**
- A **plain-English explanation** of why — e.g. *"your last 3 Fridays show a
  recurring spike, which is pulling the forecast up"*

Most forecasting tools stop at the number. An unexplained prediction is hard
to trust and easy to ignore. Trends Arc shows its reasoning, so you can check
it against what you already know about your own business.

No sign-up. Under the hood, it trains a small model (XGBoost) on your data
and uses SHAP to turn its reasoning into readable sentences instead of raw
statistics.

## How it works

1. **Upload** a CSV — date, revenue, units, SKU
2. **Validate** — bad or missing data is flagged in plain language
3. **Forecast** — a model trains on your history and projects 30 days out
4. **Explain** — a "why" panel breaks down what's driving the forecast

## Current limitations (V1)

Deliberate scope cuts for this first version, not oversights:

- CSV upload only — no live Shopify connection
- No accounts or saved forecast history
- Store-level only — no per-product breakdown
- 30-day horizon only

For the full reasoning behind these calls, see [`design-doc.md`](design-doc.md)
and [`docs/product/`](docs/product/).

## Setup

Requires **Node ≥22.12** and **Python 3.13**.

```bash
npm install

# Backend: create the venv once
cd backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
cd ..

# Run frontend + backend together
npm run app
```

`npm run app` starts both dev servers (Astro on `:4321`, FastAPI on `:8000`)
and tears down cleanly on Ctrl+C. Run them separately with `npm run app:web`
and `npm run app:api`.

Copy `.env.example` to `.env` if the frontend needs to point at a non-default
API URL.

There's no persistent database and no pre-trained global model — the API
fits a fresh model inside each `/forecast` request, on the data just
uploaded.

## Architecture

| Layer | Location | Stack |
|---|---|---|
| Frontend | `src/` | Astro + Tailwind CSS |
| API | `backend/` | FastAPI, trains an XGBoost model per request |
| Forecasting model | `backend/app/forecasting.py` | Recursive 30-day XGBoost regressor |
| Explanation | `backend/app/explain.py` | SHAP attribution → templated sentences |
| Offline research/training | `models/` | Backtest + hyperparameter search over synthetic panel data |
| Synthetic data generation | `backend/scripts/generate_sales_data.py` | Trend/seasonality/event/promo/noise model with known ground truth |
