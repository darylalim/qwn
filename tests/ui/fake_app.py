"""The real app with fakes for every model, for `streamlit run` in test_responsive.py.

Streamlit reruns this script on every interaction: it re-points the UI's registry factory at the
fakes, then runs the app's own entry point unchanged.
"""

import os
import runpy
import sys
import threading
import time
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
from qwn.models import CORE_ROLES, Registry  # noqa: E402

# Seconds the warm load takes in all (default 0: instant). A question asked meanwhile waits for
# it, as with real models: the browser bug class that only shows during the warm load.
WARM_SECONDS = float(os.environ.get("QWN_FAKE_WARM_SECONDS", "0"))
ANSWER_SECONDS = float(os.environ.get("QWN_FAKE_ANSWER_SECONDS", "0"))


class SlowGenerator(FakeGenerator):
    def answer(self, *args, **kwargs):
        end = time.monotonic() + ANSWER_SECONDS
        while time.monotonic() < end:  # busy, as generation is: holds the GIL
            sum(range(10_000))
        return super().answer(*args, **kwargs)


class SlowWarmRegistry(Registry):
    def _get(self, role):
        with self._load_lock:  # any caller waits while the warm thread "loads"
            if threading.current_thread().name == "qwn-warm-load":
                time.sleep(WARM_SECONDS / len(CORE_ROLES))
        return super()._get(role)


def fake_registry(settings: Settings, generator: FakeGenerator | None = None) -> Registry:
    return (SlowWarmRegistry if WARM_SECONDS else Registry)(
        settings,
        overrides={
            "embedder": FakeEmbedder(settings.embed_dim),
            "reranker": FakeReranker(),
            "generator": generator or (SlowGenerator if ANSWER_SECONDS else FakeGenerator)(),
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
