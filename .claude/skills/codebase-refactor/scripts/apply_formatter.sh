#!/usr/bin/env bash
# Best-effort mechanical formatting/lint-fix pass, run BEFORE manual judgment edits
# so Claude never spends reasoning tokens on things a tool already fixes.
# Usage: apply_formatter.sh <repo_path> <file1> [file2 ...]
set -euo pipefail

REPO="${1:?Usage: apply_formatter.sh <repo_path> <files...>}"
shift
FILES=("$@")

cd "$REPO"

run_if_available() {
  local tool="$1"; shift
  if command -v "$tool" >/dev/null 2>&1; then
    echo "Running $tool..."
    "$tool" "$@" || echo "  ($tool reported issues; continuing)"
  fi
}

PY_FILES=()
JS_FILES=()
GO_FILES=()
RS_FILES=()

for f in "${FILES[@]}"; do
  case "$f" in
    *.py) PY_FILES+=("$f") ;;
    *.js|*.jsx|*.ts|*.tsx) JS_FILES+=("$f") ;;
    *.go) GO_FILES+=("$f") ;;
    *.rs) RS_FILES+=("$f") ;;
  esac
done

if [ ${#PY_FILES[@]} -gt 0 ]; then
  run_if_available ruff check --fix "${PY_FILES[@]}"
  run_if_available ruff format "${PY_FILES[@]}"
  run_if_available black "${PY_FILES[@]}"
  run_if_available isort "${PY_FILES[@]}"
fi

if [ ${#JS_FILES[@]} -gt 0 ]; then
  run_if_available prettier --write "${JS_FILES[@]}"
  run_if_available eslint --fix "${JS_FILES[@]}"
fi

if [ ${#GO_FILES[@]} -gt 0 ]; then
  run_if_available gofmt -w "${GO_FILES[@]}"
fi

if [ ${#RS_FILES[@]} -gt 0 ]; then
  run_if_available rustfmt "${RS_FILES[@]}"
fi

echo "Mechanical formatting pass done (tools not installed were silently skipped)."
