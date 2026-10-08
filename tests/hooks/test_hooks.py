"""Run each .claude/hooks/*.sh with JSON payloads and check its decision.

The cases are the pipe-test list in PLAN.md → Claude Code hooks → Installing and checking.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / ".claude" / "hooks"


def run_hook(name: str, payload: dict, project_dir: Path = ROOT) -> subprocess.CompletedProcess:
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(project_dir)}
    return subprocess.run(
        [str(HOOKS / name)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def decision(result: subprocess.CompletedProcess) -> str | None:
    """The PreToolUse permissionDecision, or None when the hook printed nothing (allowed)."""
    assert result.returncode == 0, result.stderr
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"]


def test_settings_json_points_at_existing_executable_scripts():
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    commands = [
        hook["command"]
        for groups in settings["hooks"].values()
        for group in groups
        for hook in group["hooks"]
    ]
    assert len(commands) == 5
    for command in commands:
        script = ROOT / command.replace('"$CLAUDE_PROJECT_DIR"/', "")
        assert script.is_file(), script
        assert os.access(script, os.X_OK), script


# H1 py-format.sh


def test_py_format_accepts_clean_file():
    result = run_hook(
        "py-format.sh", {"tool_input": {"file_path": str(ROOT / "src/qwn/__init__.py")}}
    )
    assert result.returncode == 0, result.stderr


def test_py_format_skips_non_python(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("import os\n")
    result = run_hook("py-format.sh", {"tool_input": {"file_path": str(f)}})
    assert result.returncode == 0
    assert f.read_text() == "import os\n"


def test_py_format_fixes_what_it_can_and_reports_the_rest(tmp_path):
    f = tmp_path / "bad.py"
    f.write_text("import os\nx=undefined_name\n")
    result = run_hook("py-format.sh", {"tool_input": {"file_path": str(f)}})
    assert result.returncode == 2
    assert "F821" in result.stderr
    assert f.read_text().strip() == "x = undefined_name"  # formatted, unused import removed


def test_py_format_reports_syntax_error(tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def f(:\n")
    result = run_hook("py-format.sh", {"tool_input": {"file_path": str(f)}})
    assert result.returncode == 2


# H2 protect-paths.sh


@pytest.mark.parametrize(
    "rel, expected",
    [
        ("uv.lock", "deny"),
        ("index/qwn.db", "deny"),
        ("data/report.pdf", "deny"),
        (".venv/bin/python", "deny"),
        ("eval/public/baseline.json", "deny"),
        (".streamlit/secrets.toml", "deny"),
        ("src/qwn/x.py", None),
        ("tests/data/x.txt", None),
    ],
)
def test_protect_paths(rel, expected):
    result = run_hook("protect-paths.sh", {"tool_input": {"file_path": str(ROOT / rel)}})
    assert decision(result) == expected


# H3 guard-bash.sh


@pytest.mark.parametrize(
    "command, expected",
    [
        ("pip install foo", "deny"),
        ("uv pip install foo", "deny"),
        ("python -m pip install foo", "deny"),
        ("uv run pytest", "ask"),
        ("uv run qwn ingest data/", "ask"),
        ("uv run qwn models pull", "ask"),
        ("streamlit run src/qwn/ui/app.py", "ask"),
        ("uv run python scripts/smoke_test.py", "ask"),
        ("rm -rf index", "ask"),
        ("rm -rf ./data", "ask"),
        ("rm -r -f /abs/path/index", "ask"),
        ("uv add foo", None),
        ('uv run pytest -m "not slow"', None),
        ("uv run pytest tests/unit", None),
        ("rm -rf .ruff_cache", None),
        ("rm -rf indexer", None),
        ("rm -rf data_old", None),
        ("ls data", None),
        ("uv run qwn models status", None),
    ],
)
def test_guard_bash(command, expected):
    result = run_hook("guard-bash.sh", {"tool_input": {"command": command}})
    assert decision(result) == expected


# H4 quality-gate.sh


def test_quality_gate_lets_claude_stop_on_second_attempt():
    result = run_hook("quality-gate.sh", {"stop_hook_active": True})
    assert result.returncode == 0


def test_quality_gate_skips_tree_without_python_changes(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "notes.md").write_text("hi\n")
    result = run_hook("quality-gate.sh", {}, project_dir=tmp_path)
    assert result.returncode == 0


# H5 session-context.sh


def test_session_context_prints_branch_and_reminder():
    result = run_hook("session-context.sh", {})
    assert result.returncode == 0
    assert result.stdout.startswith("Branch: ")
    assert "Recent commits:" in result.stdout
    assert "PLAN.md" in result.stdout
