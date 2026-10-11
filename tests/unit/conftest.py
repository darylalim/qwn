import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def load_script(name: str, path: Path):
    """Import an eval corpus generator (a script, not a package module)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def public_corpus(tmp_path_factory):
    """The public eval corpus, built once: (generator module, out dir, layout). Read-only."""
    corpus = load_script("build_corpus", ROOT / "eval" / "public" / "build_corpus.py")
    out, layout = tmp_path_factory.mktemp("public"), {}
    corpus.build(out, layout)
    return corpus, out, layout
