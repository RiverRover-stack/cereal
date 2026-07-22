# 0008 — Deployment: stateless Cloud Run, no DB in V1, cold start over pretraining as the real cost driver

## Status
Accepted

## Context
`design-doc.md:106` records the original decision: no database in V1 — each
forecast is computed fresh per upload, nothing persisted, keeping the backend
stateless and sidestepping user-data storage/privacy questions until they need
to be dealt with properly. This is still true and still a reasonable choice
for V1 — there's no user data at rest to secure, no consent flow needed yet,
and every request is independent, which is the easiest shape to run on Cloud
Run (each request can land on any instance, instances can scale to zero).

The pretrained-model proposal that prompted this review cited "lower Cloud Run
costs" as a motivation. Checked against the actual workload:
`forecasting.py:97-111`'s default config is 200 trees, max depth 3, fit on
typically a few hundred rows (one seller's daily history) — a sub-second CPU
operation. For a request-scoped, stateless FastAPI service on Cloud Run, the
latency and cost that actually matters most is usually **cold start** —
spinning up a fresh container instance when there's no warm one available —
which is a function of container size and startup work, not of whether the
model trains or only runs inference. A pretrained model still needs the
container to boot, dependencies to import, the CSV to validate, and (per
ADR-0006) SHAP to run.

## Decision
1. Keep the no-database, fully stateless architecture for V1 — reaffirms
   `design-doc.md:106`, now with an explicit ADR home.
2. Do not treat "avoid per-request model fitting" as a meaningful cost or
   latency lever at current data scale — the fit itself is not the dominant
   cost. If Cloud Run cost/latency becomes a real problem, first measure
   whether it's cold start, container size, or fit time before choosing a
   fix; don't assume it's the model.
3. If cold start turns out to be the actual bottleneck (once measured — see
   Consequences), the standard levers are a smaller container image, fewer
   heavy imports at startup, or Cloud Run minimum-instance configuration —
   independent of the per-request-fit-vs-pretrained decision in ADR-0002.

## Consequences
- **Easier**: no user-data storage, consent flow, or DB migrations to build
  or secure for V1.
- **Easier**: this ADR gives a concrete answer the next time "let's deploy a
  pretrained model to save cost" comes up — check cold start first.
- **Harder**: without persistence, there's genuinely no way to track a
  seller's forecast over time, compare it to what actually happened, or build
  the monitoring called out as missing in `docs/architecture-review.md` §8 —
  that's a real limitation of staying stateless, not a free lunch.
- **Open item**: this ADR's cold-start claim is reasoned from the workload's
  size and Cloud Run's known behavior, not from a fresh production
  measurement (see `docs/architecture-review.md` §13). If cost/latency
  becomes a live concern, measure cold start explicitly before spending
  effort on a pretrained-model rewrite to "fix" it.

## Alternatives considered
- **Add a database now to support caching a pretrained model artifact.**
  Rejected for V1 — no other feature currently needs persistence, and adding
  it solely to support a serving strategy (ADR-0002) that isn't yet justified
  by data (ADR-0005) or feature design (ADR-0004) would be building
  infrastructure ahead of a decision that hasn't been earned.
- **Measure cold start now, before writing this ADR.** Considered; not done
  in this review's scope (documentation only, no infrastructure access
  exercised) — recorded as an open item instead of guessed at.

## Links
- [`docs/architecture-review.md`](../architecture-review.md) §2.4, §6
- ADR-0002 (the decision this cost analysis directly informs)
- `design-doc.md` §4 (original no-DB decision, reaffirmed here)
