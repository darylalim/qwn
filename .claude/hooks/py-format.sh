#!/usr/bin/env bash
# PostToolUse(Write|Edit|MultiEdit): format + lint the edited Python file; report unfixable lint to Claude.
set -uo pipefail
f=$(jq -r '.tool_input.file_path // .tool_response.filePath // empty')
case "$f" in *.py|*.pyi) ;; *) exit 0 ;; esac
[ -f "$f" ] || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
ruff="$CLAUDE_PROJECT_DIR/.venv/bin/ruff"            # direct call: fast, no project build
[ -x "$ruff" ] || ruff="uv run --frozen --quiet ruff"
$ruff format --quiet "$f" 2>/dev/null                 # syntax errors surface in the check below
if ! out=$($ruff check --fix --quiet "$f" 2>&1); then
  printf 'ruff found issues it could not fix in %s:\n%s\n' "$f" "$out" >&2
  exit 2
fi
