#!/usr/bin/env bash
# PreToolUse(Write|Edit|MultiEdit): block edits to generated or user-owned paths.
set -uo pipefail
f=$(jq -r '.tool_input.file_path // empty')
[ -z "$f" ] && exit 0
rel=${f#"$CLAUDE_PROJECT_DIR"/}
case "$rel" in
  uv.lock)                 why="uv.lock is generated; change dependencies with 'uv add' / 'uv remove'." ;;
  .venv/*)                 why=".venv is managed by uv; run 'uv sync'." ;;
  data/*|index/*)          why="data/ and index/ hold the user's corpus and generated index; change them via 'qwn ingest'." ;;
  eval/*/baseline.json)    why="Update baselines only with 'qwn eval --update-baseline'." ;;
  .streamlit/secrets.toml) why="The secrets file is user-managed." ;;
  *) exit 0 ;;
esac
jq -n --arg r "$why" \
  '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $r}}'
