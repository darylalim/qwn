#!/usr/bin/env bash
# Stop: if Python changed and isn't committed yet, require format/lint/types/fast tests to pass.
set -uo pipefail
input=$(cat)
[ "$(printf '%s' "$input" | jq -r '.stop_hook_active // false')" = "true" ] && exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
git status --porcelain -- '*.py' pyproject.toml | grep -q . || exit 0
fail() { printf '%s failed:\n%s\n' "$1" "$(printf '%s' "$2" | tail -40)" >&2; exit 2; }
out=$(uv run --frozen ruff format --check . 2>&1) || fail "ruff format --check" "$out"
out=$(uv run --frozen ruff check . 2>&1)          || fail "ruff check" "$out"
out=$(uv run --frozen ty check 2>&1)              || fail "ty check" "$out"
out=$(uv run --frozen pytest -m "not slow" -q -x 2>&1); rc=$?
[ "$rc" -eq 0 ] || [ "$rc" -eq 5 ] || fail "pytest -m 'not slow'" "$out"   # 5 = no tests collected
exit 0
