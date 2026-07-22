# Default thresholds

These are the defaults baked into `scan_complexity.py` and `batch_plan.py`. They're reasonable
industry-standard-ish defaults, not laws — adjust per-project if the user asks (e.g. a codebase
of generated protobuf code shouldn't be flagged the same way as hand-written logic).

| Signal | Default | Rationale |
|---|---|---|
| File length | 400 lines | Google/Microsoft style guides generally treat files over ~300-500 lines as candidates for splitting; 400 is a reasonable middle ground before diminishing returns from arguing over exact number. |
| Function/method length | 50 lines | A function you can't see on one or two screens is a refactor candidate. Common convention (Clean Code, Google style) targets well under this. |
| Nesting depth | 4 levels | Beyond 4 levels of indentation, cyclomatic complexity and readability both suffer sharply; usually a sign of missing early-returns or extracted helpers. |
| Max hotspot files per batch | 6 | Keeps each batch small enough to reason about carefully and review as one diff, without so many batches that overhead dominates. |

## Overriding

- `scan_complexity.py` constants `LINE_THRESHOLD`, `FUNC_THRESHOLD`, `NEST_THRESHOLD` at the top of the file — edit directly if the user wants different numbers for a run.
- `batch_plan.py --max-per-batch N` — CLI flag, no code edit needed.
- Both scripts accept `--exclude PATTERN` for extra folders to skip (generated code, vendored deps, migrations, etc.) beyond the built-in defaults (`node_modules`, `vendor`, `.git`, `dist`, `build`, `__pycache__`, `.venv`, `venv`, `target`).

## When NOT to flag something

Some large files are legitimately fine: data tables, generated code, migrations, test fixtures with
lots of parametrized cases. If a "hotspot" the scanner flags is actually fine on inspection, skip it
in the batch and note why in the status update — don't force a split just because a script flagged it.
