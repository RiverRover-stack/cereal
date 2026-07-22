# 0002 — Per-request fitting vs. pretrained global serving

## Status
Accepted (per-request fitting, for V1) / Proposed (pretrained global model,
deferred to a gated Phase 2 — see Consequences)

## Context
The original design trained a fresh XGBoost model on each seller's uploaded
CSV, inside the request (`design-doc.md:107`, `backend/app/forecasting.py`).
A later proposal was to switch to a pretrained model, deployed once, with the
live path doing inference only — motivated by faster inference, lower Cloud
Run cost, and better scalability.

Investigating the actual repo state surfaced three facts that bear directly on
this decision:

1. **A pretrained artifact already exists but solves a different problem.**
   `models/artifacts/model.json` predicts per-SKU daily *units* with a
   cube-root target, trained on `data/processed/master_df_cbrt.csv`
   (`feature_metadata.json:11-22`). The live API predicts store-wide daily
   *revenue* with a log target (`forecasting.py:62-90`). These are not two
   modes of the same model.
2. **The pretrained model's `sku` feature vocabulary is frozen to 40 specific
   codes** (`feature_metadata.json:43-84`, e.g. `ACC-001`…`SUM-005`) from the
   one synthetic store it was trained on. XGBoost's native categorical
   handling has no fallback for an unseen category — a real seller's SKU codes
   would not match any of these, so the model cannot score their data.
3. **The training data is one synthetic store** (40 SKUs, one catalogue, no
   store/seller ID column anywhere in the data — see ADR-0005), not multiple
   merged stores. There is no evidence yet that a model trained this way
   generalizes to a *different* store's demand pattern.

Separately, the cost/latency case for pretraining was checked against the
actual workload: `forecasting.py`'s default XGBoost config is 200 trees at
max depth 3 on a few hundred rows (`forecasting.py:97-111`) — a sub-second fit.
The latency/cost driver for this app on Cloud Run is much more likely to be
cold start (container spin-up) than model-fit time, and cold start is
unaffected by whether the model trains or just runs inference.

## Decision
Keep per-request fitting as the production serving path for V1. Do not deploy
`models/artifacts/model.json` (or any model trained the same way) behind the
live API. Treat "pretrained global model" as a legitimate future architecture,
reachable via the Phase 2 gate defined below — not adopted now, and not
rejected outright.

**The gate for revisiting this ADR:** a pretrained global model is only
adopted if (a) it is trained on transferable-only features with no store or
SKU identity (ADR-0004), (b) it is trained on data from multiple distinct
stores (ADR-0005 roadmap), and (c) it beats per-request fitting on a rolling-
origin backtest against stores/data it did not train on. Until all three hold,
this ADR's status stands.

## Consequences
- **Easier**: V1 ships on an architecture that is already correct for the
  generalization problem (no seller-identity features to fail on) and already
  stateless (ADR-0008).
- **Easier**: the SHAP explainability differentiator (ADR-0006) attaches
  naturally to a per-request model — the explanation is genuinely about the
  uploading seller's own fitted model, not a population average.
- **Harder**: the app cannot yet benefit from patterns learned across many
  sellers (e.g. category-wide seasonality) — each forecast starts from zero
  prior knowledge beyond the uploaded history.
- **Debt**: two disconnected model-training code paths now exist in the repo
  (`backend/app/forecasting.py` live, `models/train.py` research) with no
  runtime relationship. Anyone unfamiliar with this ADR could reasonably
  assume `model.json` is what's live — it isn't. Mitigation: this ADR, plus
  consider a top-level note in `models/README.md` pointing here.
- **Not a permanent rejection**: the gate above keeps Option B reachable and
  defines exactly what evidence would flip this decision.

## Alternatives considered
- **Deploy `models/artifacts/model.json` as-is behind `POST /forecast`.**
  Rejected outright — target/grain mismatch with the API contract (ADR-0003)
  and the frozen SKU vocabulary means it cannot score real seller data
  (Context, point 2).
- **Retrain a pretrained model now, on the existing single-store data, without
  waiting for multi-store data.** Rejected — would still fail the
  generalization test in the gate above; a model trained on one store's shape
  of demand has no demonstrated ability to transfer to a different store, and
  shipping it would risk quietly bad forecasts for every seller whose store
  doesn't resemble the training store.
- **Hybrid: pretrained model as a fallback when a seller has too little
  history to fit their own model.** Not rejected, genuinely interesting, but
  out of scope until the Phase 2 gate is met — flagging here so it isn't lost.

## Links
- [`docs/architecture-review.md`](../architecture-review.md) §2, §5, §6, §9
- ADR-0003 (target/grain mismatch detail)
- ADR-0004 (transferable-only feature requirement)
- ADR-0005 (single-store data status, multi-store roadmap)
- ADR-0006 (SHAP attaches to the per-request model)
