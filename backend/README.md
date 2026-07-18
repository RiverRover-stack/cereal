# Trends Arc — backend

FastAPI service for the Trends Arc forecast tool. This is the Phase 1 scaffold:
a health check and nothing else.

## What exists

| Method | Path      | Response            |
| ------ | --------- | ------------------- |
| GET    | `/health` | `{"status": "ok"}`  |

`POST /forecast` does **not** exist yet. CSV validation, the XGBoost model, and
the SHAP explanation are design-doc Phases 2–4 and have not been built, so the
`/app` page says so rather than showing a placeholder result.

## Run it locally

From this directory:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then:

```bash
curl http://localhost:8000/health
```

Port 8000 is what the frontend falls back to when `PUBLIC_API_URL` is unset
(see `../.env.example`). Installing the full requirements pulls xgboost and
shap — several hundred MB and slow on first run — even though the scaffold
imports neither yet.

## CORS

`app/main.py` allows `http://localhost:4321`, the Astro dev origin. Deploying
the frontend anywhere else means adding that origin there too, otherwise the
browser blocks the call before the request is made.

## Container

```bash
docker build -t trends-arc-api .
docker run -p 8080:8080 trends-arc-api
```

The image reads `PORT` from the environment (Cloud Run sets it), defaulting to
8080.
