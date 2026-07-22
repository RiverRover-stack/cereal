#!/usr/bin/env bash
# Preflight checks before refactoring: clean git tree, new branch, baseline test run.
# Usage: preflight.sh <repo_path>
set -euo pipefail

REPO="${1:?Usage: preflight.sh <repo_path>}"
cd "$REPO"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: $REPO is not a git repository. Refactoring requires git for safety (branch + revert)." >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: working tree is not clean. Commit or stash your changes before refactoring." >&2
  git status --porcelain >&2
  exit 1
fi

mkdir -p .refactor

BRANCH="refactor/$(date +%Y%m%d-%H%M)"
git checkout -b "$BRANCH"
echo "Created and switched to branch: $BRANCH"
echo "$BRANCH" > .refactor/branch.txt

# Detect test command based on stack markers present in the repo root.
TEST_CMD=""
if [ -f "pyproject.toml" ] || [ -f "setup.py" ] || [ -f "pytest.ini" ]; then
  if command -v pytest >/dev/null 2>&1; then TEST_CMD="pytest -q"; fi
elif [ -f "package.json" ]; then
  if grep -q '"test"' package.json; then TEST_CMD="npm test --silent"; fi
elif [ -f "go.mod" ]; then
  TEST_CMD="go test ./..."
elif [ -f "Cargo.toml" ]; then
  TEST_CMD="cargo test"
elif [ -f "pom.xml" ]; then
  TEST_CMD="mvn -q test"
elif [ -f "build.gradle" ] || [ -f "build.gradle.kts" ]; then
  TEST_CMD="./gradlew test"
fi

echo "$TEST_CMD" > .refactor/test_cmd.txt

if [ -z "$TEST_CMD" ]; then
  echo "WARNING: no test command detected. Changes cannot be automatically verified."
  echo "no-test-suite" > .refactor/baseline_tests.txt
else
  echo "Detected test command: $TEST_CMD"
  echo "Running baseline test suite..."
  if eval "$TEST_CMD" > .refactor/baseline_tests.txt 2>&1; then
    echo "PASS" >> .refactor/baseline_tests.txt
    echo "Baseline tests PASSED."
  else
    echo "FAIL" >> .refactor/baseline_tests.txt
    echo "ERROR: baseline tests FAIL before any changes were made. Fix the baseline before refactoring." >&2
    tail -n 30 .refactor/baseline_tests.txt >&2
    exit 1
  fi
fi

# init state file
cat > .refactor/state.json <<EOF
{
  "branch": "$BRANCH",
  "test_cmd": "$TEST_CMD",
  "batches_completed": [],
  "batches_remaining": []
}
EOF

echo "Preflight complete."
