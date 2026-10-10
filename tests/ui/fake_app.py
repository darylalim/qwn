"""The real app with fakes for every model, for `streamlit run` in test_responsive.py.

Streamlit reruns this script on every interaction: it re-points the UI's registry factory at the
fakes, then runs the app's own entry point unchanged.
"""

import runpy
import sys
from pathlib import Path

TESTS = Path(__file__).parents[1]
APP = TESTS.parent / "src" / "qwn" / "ui" / "app.py"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from fakes import (  # noqa: E402
    FakeAsr,
    FakeEmbedder,
    FakeGenerator,
    FakeGuard,
    FakeMemory,
    FakeReranker,
    FakeTts,
    FakeVad,
)

import qwn.ui.services  # noqa: E402
from qwn.config import Settings  # noqa: E402
from qwn.models import Registry  # noqa: E402


def fake_registry(settings: Settings, generator: FakeGenerator | None = None) -> Registry:
    return Registry(
        settings,
        overrides={
            "embedder": FakeEmbedder(settings.embed_dim),
            "reranker": FakeReranker(),
            "generator": generator or FakeGenerator(),
            "guard": FakeGuard(),
            "vad": FakeVad(),
            "asr": FakeAsr(),
            "tts": FakeTts(),
        },
        memory=FakeMemory(active_gb=12.4),
    )


if __name__ == "__main__":
    qwn.ui.services.make_registry = fake_registry
    runpy.run_path(str(APP), run_name="__main__")
