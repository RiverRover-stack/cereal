# Trends Arc

An explainable sales forecasting tool for e-commerce sellers. A seller uploads a CSV
of their order history; the tool returns a 30-day revenue forecast plus a plain-English
explanation of *why* the forecast looks the way it does (SHAP attribution, translated
into readable sentences). See `design-doc.md` for the full product rationale.

## Architecture

| Layer | Location | Stack |
|---|---|---|
| Frontend | `src/` | Astro + Tailwind CSS |
| API | `backend/` | FastAPI, trains an XGBoost model per request on the upload |
| Forecasting model | `backend/app/forecasting.py` | Recursive 30-day XGBoost regressor |
| Explanation | `backend/app/explain.py` | SHAP attribution → templated sentences |
| Offline research/training | `models/` | Standalone backtest + hyperparameter search over synthetic panel data |
| Synthetic data generation | `backend/scripts/generate_sales_data.py` | Multiplicative trend/seasonality/event/promo/noise model with known ground truth |

There is no persistent database and no pre-trained global model: the API fits a model
inside each `/forecast` request, on the data the seller just uploaded.

## Quickstart

Requires Node ≥22.12 and Python 3.13.

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

`npm run app` starts both dev servers (Astro on :4321, FastAPI on :8000) and tears
down cleanly on Ctrl+C. To run them separately: `npm run app:web` / `npm run app:api`.

Copy `.env.example` to `.env` if you need to point the frontend at a non-default API URL.

## Project structure

```
src/                    Astro frontend (pages, components, API client)
backend/                FastAPI service
  app/                    validation, forecasting, explanation, main API
  scripts/                synthetic sales data generator
  tests/                  pytest suite + live-model backtest
models/                 Offline training/backtest pipeline for the research model
  train.py                entry point: sweep -> ablation -> A/B -> backtest -> SHAP -> artifacts
  train_pipeline.py       the numbered pipeline stages train.py calls
  artifacts/              saved model + metrics from the last training run
data/                   Generated/processed CSVs (synthetic sales panel)
scripts/                Repo-root data-prep scripts (build_dataset.py, skewness_analysis.py)
docs/                   Design docs, ADRs, architecture/validation reports
  adr/                    Architecture Decision Records
  product/                Product planning docs (PRD, design language, landing copy, competitor research)
reports/                Generated analysis reports (e.g. skewness)
```

## Testing

```bash
cd backend
.venv\Scripts\python -m pytest -q          # unit + smoke tests
.venv\Scripts\python -m tests.backtest      # rolling-origin backtest against the live model
```

The offline research model has its own reproducible pipeline:

```bash
backend\.venv\Scripts\python.exe models\train.py
```

## Documentation

- `design-doc.md` — problem, solution, MVP scope. Cited by line number throughout
  the backend code (e.g. `design-doc.md:107`).
- `docs/adr/` — Architecture Decision Records.
- `docs/architecture-review.md`, `docs/forecast-validation.md`, `docs/mvp-testing-gaps.md` —
  point-in-time engineering reviews.
- `models/README.md` — training results, leakage findings, and honest limitations of
  the research forecasting model.
- `backend/README.md` — API endpoints and local/container run instructions.
- `docs/product/` — PRD, visual design language, landing page copy, competitor research.
