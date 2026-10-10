"""AppTest fixtures: a tmp qwn home with a small corpus, the UI wired to fakes."""

import shutil

import pytest
import streamlit as st
from fake_app import APP, fake_registry
from streamlit.testing.v1 import AppTest

from qwn.config import Settings

TIMEOUT = 20


@pytest.fixture
def ui_home(home, monkeypatch):
    """The UI against an empty index in a tmp home, with fakes for every model."""
    import qwn.ui.services

    monkeypatch.setattr(qwn.ui.services, "make_registry", fake_registry)
    st.cache_resource.clear()  # services() is shared per process: start fresh
    yield home
    st.cache_resource.clear()


@pytest.fixture(scope="session")
def corpus_index(tmp_path_factory):
    """The integration corpus, ingested once per session: (corpus folder, index folder).

    The index stores absolute paths into the corpus folder, so tests copy only the index."""
    from sample_corpus import make_corpus

    from qwn.ingest import Ingester, open_index

    root = tmp_path_factory.mktemp("ui-corpus")
    (root / "qwn.example.toml").write_text("")
    docs = make_corpus(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("QWN_HOME", str(root))
        mp.delenv("QWN_INDEX_DIR", raising=False)
        settings = Settings()
        index = open_index(settings)
        Ingester(settings, index, fake_registry(settings).embedder).run([docs])
        index.close()
    return docs, settings.index_dir


@pytest.fixture
def corpus_home(ui_home, corpus_index):
    """As ui_home, with the integration corpus already indexed."""
    docs, index_dir = corpus_index
    shutil.copytree(index_dir, ui_home / "index")
    return ui_home, docs


def app(page: str | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    if page is not None:
        at.switch_page(f"app_pages/{page}.py").run()
    return at
