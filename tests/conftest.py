import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An empty qwn home with an example config, used as QWN_HOME."""
    (tmp_path / "qwn.example.toml").write_text("")
    monkeypatch.setenv("QWN_HOME", str(tmp_path))
    monkeypatch.setenv("QWN_LOG_DIR", str(tmp_path / "logs"))
    for var in ("QWN_TOP_K", "QWN_INDEX_DIR", "QWN_PDF_EMBED", "QWN_CHUNK_CONTEXT", "QWN_HYBRID"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path
