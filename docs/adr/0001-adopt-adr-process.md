# 0001 — Adopt an ADR process

## Status
Accepted

## Context
TrendsArc's architecture decisions were previously recorded in one place —
`design-doc.md` §4 ("Key architectural decisions") — as a short bullet list
with no per-decision status, alternatives, or update mechanism. That worked
while there were four decisions and one author. It stopped working the moment
a real architectural question came up for re-litigation: "should we deploy a
pretrained model instead of fitting per request?" (see ADR-0002) — there was
no structured place to record the answer, the reasoning, or that the question
had even been seriously considered, so the next time it comes up it would have
to be re-derived from scratch or from memory.

The founder asked explicitly for "a clear record of trade-offs" so decisions
don't rely on memory as the product grows toward serving many stores.

## Decision
Adopt a lightweight, MADR-inspired ADR process, stored under `docs/adr/`, one
file per decision, numbered sequentially. See `docs/adr/README.md` for the
format and index. `design-doc.md` §4 remains as historical context but new
architectural decisions are recorded here going forward.

## Consequences
- **Easier**: revisiting a past decision means reading one short file instead
  of reconstructing reasoning from code comments and chat history.
- **Easier**: proposing a change to something already decided (e.g. "let's
  deploy the pretrained model") can be answered by pointing at the relevant
  ADR's Context and Alternatives sections instead of re-debating from zero.
- **Harder**: someone has to remember to write one when a real architectural
  decision gets made — this is a process discipline, not a technical
  enforcement (no CI check requires an ADR for a given change).
- **Debt**: `design-doc.md` §4 and this folder now both describe some of the
  same ground (e.g. "no database in V1" appears in both). ADR-0008 supersedes
  design-doc.md §4's version of that specific point going forward; the rest of
  §4 is left as-is rather than migrated wholesale, to avoid rewriting history
  that was accurate when written.

## Alternatives considered
- **Keep using `design-doc.md` §4 only.** Rejected — it has no per-decision
  status field, no way to mark something superseded without editing history,
  and mixes decisions with a 7-phase build plan, making it hard to scan for
  "what was decided" versus "what was built."
- **A single `DECISIONS.md` log.** Considered but rejected in favor of one
  file per decision — a single growing file gets harder to link to precisely
  and harder to mark individual entries as superseded.

## Links
- [`docs/architecture-review.md`](../architecture-review.md)
- `design-doc.md` §4 (predecessor, not superseded — coexists)
