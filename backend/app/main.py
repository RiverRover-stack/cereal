"""Trends Arc API — scaffold only (design-doc Phase 1).

No forecasting logic lives here yet. `POST /forecast` (Phases 2-4: CSV
validation, XGBoost, SHAP) is deliberately absent rather than stubbed, so the
frontend has nothing to pretend against.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Trends Arc API",
    description="Shopify sales forecasting backend.",
    version="0.1.0",
)

# The site is a static Astro build served from a different origin than this API,
# so every browser call is cross-origin — without this the fetch fails at the
# preflight and the page can only report "unreachable".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
