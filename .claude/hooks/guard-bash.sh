#!/usr/bin/env bash
# PreToolUse(Bash): enforce uv; confirm before model downloads/loads, servers, or deleting user data.
set -uo pipefail
cmd=$(jq -r '.tool_input.command // empty')
decide() {
  jq -n --arg d "$1" --arg r "$2" \
    '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: $d, permissionDecisionReason: $r}}'
  exit 0
}
has() { printf '%s' "$cmd" | grep -Eq "$1"; }

has '(^|[;&|[:space:]])(pip3?|python3? -m pip|uv pip) install' &&
  decide deny "Use 'uv add <pkg>' (or 'uv add --dev <pkg>') so pyproject.toml and uv.lock stay in sync."

if has 'pytest' && ! has 'not slow' && ! has 'tests/(unit|integration|ui)'; then
  decide ask "Full pytest includes @slow tests that download ~13 GB of models (~18 GB with voice) and use ~16+ GB of memory."
fi

has 'smoke_test\.py|hf download|huggingface-cli download|qwn (ingest|search|ask|eval|ui|models pull)|streamlit run' &&
  decide ask "This downloads or loads MLX models (~13 GB) or starts a long-running server."

has 'rm[[:space:]]+-[a-zA-Z]*r[a-zA-Z]*([[:space:]]+[^[:space:]]+)*[[:space:]]+([^[:space:]]*/)?(data|index)/?([[:space:];&|]|$)' &&
  decide ask "This deletes the user's corpus or index."

exit 0
