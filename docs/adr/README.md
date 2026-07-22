# Architecture Decision Records (ADRs)

This folder is the record of *why* TrendsArc's architecture looks the way it
does — not just what it does (the code already shows that). Before this folder
existed, decisions lived only in `design-doc.md` §4 and scattered code
comments; this makes them a first-class, searchable, revisitable artifact.

## When to write one

Write an ADR when a decision is genuinely hard to reverse, affects more than
one part of the system, or was chosen over a real alternative someone might
reasonably re-propose later (as happened with the pretrained-model question —
see ADR-0002). Don't write one for reversible implementation details that
belong in code comments instead.

## Format

Each ADR is a single Markdown file, numbered sequentially, named
`NNNN-short-kebab-title.md`. Sections:

- **Status** — `proposed`, `accepted`, `deferred`, `superseded by ADR-XXXX`, or
  `rejected`.
- **Context** — what problem or question forced this decision; cite the actual
  file/line evidence, not recollection.
- **Decision** — the choice made, stated plainly.
- **Consequences** — what this makes easier, what it makes harder, what debt it
  creates.
- **Alternatives considered** — options that were weighed and why they lost
  (or are deferred, not rejected outright).
- **Links** — related ADRs, the full review doc, relevant source files.

## Updating a decision

Never edit history — if a decision changes, write a new ADR and mark the old
one `superseded by ADR-XXXX`. The old file stays; it's still true that the
earlier decision was correct given what was known then.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-adopt-adr-process.md) | Adopt an ADR process | Accepted |
| [0002](0002-per-request-fit-vs-pretrained-global-serving.md) | Per-request fitting vs. pretrained global serving | Accepted (per-request) / Proposed (global, deferred) |
| [0003](0003-target-and-grain-store-revenue-log-vs-sku-units-cbrt.md) | Target and grain: store revenue (log) vs. SKU units (cube-root) | Accepted |
| [0004](0004-feature-set-transferable-only-exclude-sku-store-identity.md) | Feature set: transferable-only, exclude store/SKU identity | Accepted |
| [0005](0005-synthetic-only-training-data-status-and-multistore-roadmap.md) | Synthetic-only training data: current status and multi-store roadmap | Accepted (current), roadmap proposed |
| [0006](0006-explainability-port-offline-shap-to-live-path.md) | Port offline SHAP explainability to the live request path | Accepted — implemented |
| [0007](0007-evaluation-gate-rolling-origin-backtest.md) | Rolling-origin backtest as the evaluation method and deploy gate | Accepted (method) / Proposed (gate) |
| [0008](0008-deployment-stateless-cloud-run-cold-start-no-db-v1.md) | Deployment: stateless Cloud Run, no DB in V1, cold start over pretraining | Accepted |

See also: [`docs/architecture-review.md`](../architecture-review.md) — the full
narrative review these ADRs are extracted from.
