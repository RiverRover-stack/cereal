# Per-batch procedure, in detail

## 1. Load only what this batch needs
Open the batch's flagged files (from `.refactor/batches.json`) — not the whole repo, not the whole
folder if only some files in it were flagged. If you need to know how a function is used elsewhere,
use `git grep -n "function_name" <repo>` instead of opening files speculatively.

## 2. Mechanical pass first
Run `scripts/apply_formatter.sh <repo_path> <batch files>`. This handles formatting, import order,
and auto-fixable lint issues via whatever tools are already installed in the environment (ruff/black/
isort for Python, prettier/eslint for JS/TS, gofmt for Go, rustfmt for Rust). Tools that aren't
installed are silently skipped — don't spend time installing new tooling into the user's project
unless they ask for it.

## 3. Judgment-call editing
This is the part that needs actual reasoning — apply `references/style-guides.md`:
- Split oversized files along responsibility boundaries.
- Extract long functions into smaller named pieces.
- Rename things that fail the "name for intent" test.
- Reduce nesting via early returns / guard clauses.

Keep edits scoped to what the scan flagged plus what's directly necessary to support that (e.g. if
splitting a file, you'll need to add imports in the new files) — don't drift into unrelated cleanup
in the same batch; that belongs in its own batch so diffs stay reviewable.

## 4. Test
Run `scripts/run_tests.sh <repo_path>`. On failure:
- First try to fix forward if the cause is clear (e.g. a broken import after a split).
- If not quickly fixable, revert this batch's files (`git checkout -- <files>`) and tell the user
  what broke and why, rather than leaving a red batch committed.
- Never commit a batch with failing tests.

## 5. Commit
Use Conventional Commits style: `refactor(<module>): <what changed>`, e.g.
`refactor(auth): split auth.py into auth/login.py and auth/tokens.py`. Keep the body brief — one or
two lines on what changed and why, not a full essay.

## 6. Update state and report
Append to `.refactor/state.json`:
```json
{
  "batches_completed": [
    {
      "module": "auth",
      "files": ["auth.py"],
      "commit": "<sha>",
      "summary": "split into login.py and tokens.py, extracted validate_token()",
      "tests": "pass"
    }
  ]
}
```
Then give the user a short status update — module name, what changed, test result — and stop.
Don't start the next batch until told to continue, unless the user explicitly asked for a
run-through-without-stopping mode at the start.

## Handling scope creep mid-batch
If while editing you discover an unrelated problem (e.g. a security issue, a bug, dead code in a
file you're already touching), don't silently fix it as part of this refactor batch. Note it in the
status update as a separate finding — refactor batches should stay focused so their diffs are easy
to review, and unrelated fixes deserve their own scrutiny/commit.

## Resuming after a pause
If picking this up in a new conversation/session, read `.refactor/state.json` and `.refactor/batches.json`
first — this tells you what's done and what's next without needing to re-scan the repo.
