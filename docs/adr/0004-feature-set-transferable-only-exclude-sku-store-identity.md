# 0004 — Feature set: transferable-only, exclude store/SKU identity

## Status
Accepted (for the live path, already true) / Proposed (as a hard requirement
for any future pretrained/global model)

## Context
The live path's features — `day_of_week`, `day_of_month`, `time_index`,
`rolling_7` (`backend/app/forecasting.py:62-67`) — are all derived from a
store's own time series and carry no store-specific identity. They mean the
same thing for any seller: "what day of the week is it," "how many days into
this store's history are we," "what's this store's own recent momentum."

The offline research model's selected feature set (`known_covariates_only`,
`feature_metadata.json:23`) includes `sku`, `category`, and `base_price`
(`feature_metadata.json:13-22`). `sku` in particular is a 40-value categorical
feature whose exact vocabulary (`ACC-001`…`SUM-005`,
`feature_metadata.json:43-84`) is frozen into the trained model. This directly
causes the generalization failure described in ADR-0002: any seller whose
product codes aren't in that list is unscoreable by construction, not just
poorly predicted.

This is a specific instance of a general rule worth stating once rather than
rediscovering per model: a feature that encodes *which specific entity this
row belongs to* (a SKU code, a store ID, a customer ID) lets a tree model
learn an entity-specific intercept/adjustment that has no meaning for an
entity it never saw in training. That's different from — and a more severe
failure mode than — ordinary poor generalization from a small or skewed
training set.

## Decision
For any model intended to serve traffic from sellers not present in its
training data (i.e. any production or pretrained model, as opposed to a
model fit fresh on one seller's own upload), the feature set must exclude
identity features for the entity being generalized across — no `sku`, no
store/seller ID, no anything whose value space is fixed at training time and
meaningless outside it. `base_price` is similarly store-specific and should be
excluded or replaced with a relative/normalized version (e.g. price percentile
within category) if reintroduced.

This does not apply to the live per-request model (ADR-0002) — it fits fresh
on each seller's own data, so there's no "unseen entity" problem: `sku` would
be safe to use there *if* the CSV contained it, precisely because the model
is refit per seller rather than reused across sellers. The constraint is
specific to models meant to be reused across sellers without refitting.

## Consequences
- **Easier**: this ADR gives Phase 2 (ADR-0002, ADR-0005) a concrete,
  checkable requirement — "list the features, confirm none are identity
  features" — instead of a vague generalization concern.
- **Harder**: some genuinely useful signal (this SKU's specific price point,
  its specific historical performance) becomes unavailable to a global model
  unless re-expressed in a transferable form (e.g. price relative to category
  median, rather than the raw price).
- **Debt**: none new; this formalizes a constraint the live path already
  satisfies by construction and the offline path currently violates.

## Alternatives considered
- **Allow `sku`/store identity but add an out-of-vocabulary fallback** (e.g.
  route unknown categories to a learned "average" leaf). Not rejected outright
  — technically possible with some categorical-encoding schemes — but adds
  complexity and still means an unseen seller gets a degraded, "average
  store" prediction rather than one grounded in their own data, which is a
  worse experience than what per-request fitting already provides today.
  Revisit only if Phase 2 data shows transferable-only features are
  insufficient on their own.

## Links
- [`docs/architecture-review.md`](../architecture-review.md) §2.2, §3 (Q4, Q5), §4
- ADR-0002 (why this matters for the pretrained-model decision)
- ADR-0005 (multi-store data needed to even evaluate a transferable-only model)
