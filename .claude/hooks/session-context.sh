#!/usr/bin/env bash
# SessionStart(startup): orient a fresh implementation session. Stdout is added to Claude's context.
cd "$CLAUDE_PROJECT_DIR" || exit 0
echo "Branch: $(git branch --show-current)"
echo "Recent commits:"; git log --oneline -5
echo "Implement PLAN.md phase by phase. Check which phase's exit criteria are already met before starting."
