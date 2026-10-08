import os

from typer.testing import CliRunner

import qwn.cli
from qwn.models_lock import MODELS, PinnedModel

runner = CliRunner()


def test_cli_forces_offline_mode():
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_without_home_commands_stop_with_a_hint(tmp_path, monkeypatch):
    monkeypatch.delenv("QWN_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(qwn.cli.app, ["models", "status"])
    assert result.exit_code == 1
    assert "set QWN_HOME" in result.output


def test_models_status_lists_every_pinned_model(home):
    result = runner.invoke(qwn.cli.app, ["models", "status"])
    assert result.exit_code == 0, result.output
    for m in MODELS.values():
        assert f"{m.repo}@{m.revision[:7]}" in result.output


def test_models_status_flags_unpinned_configured_model(home, monkeypatch):
    monkeypatch.setenv("QWN_GEN_MODEL", "someone/other-model")
    result = runner.invoke(qwn.cli.app, ["models", "status"])
    assert "not pinned: someone/other-model" in result.output


def test_repin_rewrites_only_changed_revisions(tmp_path, monkeypatch):
    lock = tmp_path / "models_lock.py"
    lock.write_text(qwn.cli.LOCK_FILE.read_text())
    monkeypatch.setattr(qwn.cli, "LOCK_FILE", lock)
    core = [m for m in MODELS.values() if m.group == "core"]
    new_sha = "f" * 40
    target = core[0]

    def latest(repo: str) -> str:
        return new_sha if repo == target.repo else MODELS[repo].revision

    out = qwn.cli._repin(core, latest)
    assert out[0] == PinnedModel(target.repo, new_sha, target.group, target.size_gb)
    assert out[1:] == core[1:]
    text = lock.read_text()
    assert new_sha in text and target.revision not in text
