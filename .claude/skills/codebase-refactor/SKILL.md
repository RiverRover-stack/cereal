---
name: codebase-refactor
description: Refactor an existing codebase toward Big Tech engineering standards (Google/Microsoft-style structure, function/file size limits, complexity reduction, naming) using a token-efficient, script-first, batch-by-batch workflow on a dedicated git branch with test verification before and after every batch. Use this skill whenever the user asks to "refactor," "clean up," "restructure," "modernize," or "bring up to standard" a codebase or repo, wants files split up, functions extracted, complexity reduced, or code reorganized to match how professional engineering teams do it — even if they don't use the word "refactor" explicitly (e.g. "this file is a mess," "can you make this more maintainable," "split this up"). Always consult this skill before touching multiple files for a structural cleanup — do not freehand a multi-file refactor without it.
---

# Codebase Refactor

Refactors a codebase the way a senior engineer at a well-run company would: scripted detection of problems (never manual full-repo reading), small reviewable batches, a dedicated git branch, and tests as the safety net — never a single giant AI-authored diff.

## Core principles (non-negotiable)

1. **Never read the whole repo into context.** Use scripts to find hotspots. Only open files that scripts flag or that you're actively editing.
2. **Scripts do the mechanical work; Claude's reasoning is reserved for judgment calls** — how to split a file, what a function should be named, where a boundary belongs. Formatting, import sorting, lint auto-fixes are done by running tools, not by hand-editing.
3. **Everything happens on a new git branch.** Main/master is never touched directly.
4. **Tests gate every batch.** If there's a test suite, it must pass before the batch is committed. If it breaks, fix or revert before moving on.
5. **Work in batches, not one pass.** Batch by folder/module. Pause between batches for a status update (and user check-in if they want one) rather than streaming edits non-stop.
6. **State survives across pauses.** Progress is tracked in `.refactor/state.json` in the target repo so work can resume without re-scanning everything.

## Workflow

### Step 0 — Clarify scope (if not already known)
If the user hasn't specified, ask briefly: which repo/path, any folders to exclude (e.g. `vendor/`, `node_modules/`, generated code), and whether they want to review each batch or let it run through with periodic updates. Don't over-ask if they've already told you in this conversation — check first.

### Step 1 — Preflight
Run:
```bash
bash scripts/preflight.sh <repo_path>
```
This verifies the path is a git repo, checks for a clean working tree (uncommitted changes are not allowed to start — ask the user to commit/stash first if dirty), creates a branch named `refactor/<yyyymmdd-hhmm>`, detects the test command for the stack, and runs the baseline test suite, saving results to `.refactor/baseline_tests.txt`.

If baseline tests already fail, stop and tell the user — refactoring on top of a broken baseline makes it impossible to know if you caused a regression.

If there's no test suite at all, tell the user explicitly that changes can't be verified automatically and confirm they still want to proceed.

### Step 2 — Scan for hotspots
Run:
```bash
python3 scripts/scan_complexity.py <repo_path> --exclude <patterns>
```
This produces `.refactor/scan_report.json`: per-file line counts, per-function/method length, nesting depth, and a flagged list of "hotspots" (files over ~400 lines, functions over ~50 lines, nesting over 4 levels deep — see `references/thresholds.md` for exact defaults and how to tune them). It uses language-specific tools when available (radon for Python complexity, etc.) and a generic heuristic scanner otherwise — you don't need to read this report's full JSON, just the summary it prints.

Read only the printed summary, not every raw file, at this stage.

### Step 3 — Build the batch plan
Run:
```bash
python3 scripts/batch_plan.py <repo_path>
```
This groups hotspot files by top-level folder/module into `.refactor/batches.json`. Folders with too many hotspots (see `references/thresholds.md`) get split into sub-batches automatically. Present the batch plan to the user as a short numbered list (folder/module → file count → est. size) before starting, so they know what's coming.

### Step 4 — Process one batch at a time
For each batch, in order, consult `references/workflow-details.md` for the full per-batch procedure. In short:
1. Open only that batch's flagged files.
2. Run the mechanical formatter/linter pass first: `bash scripts/apply_formatter.sh <repo_path> <batch files>`.
3. Do the judgment-call refactoring yourself: split oversized files along cohesive responsibility lines, extract long functions, rename for clarity, per `references/style-guides.md`.
4. Run `bash scripts/run_tests.sh <repo_path>` — scoped to affected tests if the stack supports it, full suite otherwise.
5. If tests fail: fix within this batch, or revert the batch's changes (`git checkout -- <files>`) and report why rather than leaving a broken commit.
6. Commit the batch: `git commit -m "refactor(<module>): <summary>"`.
7. Update `.refactor/state.json` and give the user a short status: what changed, complexity before/after, test result.
8. **Pause.** Don't auto-continue into the next batch unless the user has said to run through without stopping. This is a hard requirement, not a suggestion — large uninterrupted multi-batch runs are exactly the failure mode this skill exists to avoid.

### Step 5 — Wrap-up
After the last batch, run the full test suite once more, print a before/after summary (total flagged hotspots resolved, lines changed, files touched), and remind the user the work is on the `refactor/...` branch, not merged — they review and merge it themselves.

## Token efficiency rules
- Never `cat` or `view` an entire large file when a script can report the specific lines/functions that matter.
- Never re-scan the whole repo between batches — `scan_complexity.py` supports `--only <folder>` to refresh just one area if needed.
- Prefer grep/ast-based targeted lookups (`git grep`, `scripts/scan_complexity.py --only`) over dumping files to find usages.
- Summarize completed batches in `.refactor/state.json` rather than keeping their full diffs in context going forward.

## Reference files
- `references/thresholds.md` — default size/complexity thresholds and how to override them
- `references/style-guides.md` — condensed Google/Microsoft-style engineering conventions per language (naming, function/file size, SRP, docstrings, nesting)
- `references/workflow-details.md` — the detailed per-batch procedure, commit message format, state file schema, and failure-handling rules
