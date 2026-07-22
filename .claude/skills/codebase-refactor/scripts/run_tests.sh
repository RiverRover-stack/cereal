#!/usr/bin/env bash
# Run the repo's detected test suite (set during preflight) and report pass/fail.
# Usage: run_tests.sh <repo_path>
set -uo pipefail

REPO="${1:?Usage: run_tests.sh <repo_path>}"
cd "$REPO"

if [ ! -f ".refactor/test_cmd.txt" ]; then
  echo "ERROR: no .refactor/test_cmd.txt found. Run preflight.sh first." >&2
  exit 1
fi

TEST_CMD=$(cat .refactor/test_cmd.txt)

if [ -z "$TEST_CMD" ]; then
  echo "No test suite detected for this repo. Cannot verify changes automatically."
  exit 0
fi

echo "Running: $TEST_CMD"
if eval "$TEST_CMD" > .refactor/last_test_run.txt 2>&1; then
  echo "PASS"
  exit 0
else
  echo "FAIL"
  echo "--- last 40 lines of output ---"
  tail -n 40 .refactor/last_test_run.txt
  exit 1
fi
