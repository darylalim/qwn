import subprocess
import sys
import threading
import time

import pytest
from fakes import FakeEmbedder, FakeGenerator, FakeGuard, FakeReranker

from qwn.config import Settings
from qwn.interfaces import Item
from qwn.models import InsufficientMemory, ModelsInUse, ProcessLock, Registry

GB = 10**9


class FakeMemory:
    def __init__(self, active_gb=0.0, recommended_gb=26.8):
        self.active = int(active_gb * GB)
        self.recommended = int(recommended_gb * GB)
        self.cleared = 0

    def active_bytes(self):
        return self.active

    def peak_bytes(self):
        return self.active

    def recommended_bytes(self):
        return self.recommended

    def clear_cache(self):
        self.cleared += 1


def counting_loaders(memory: FakeMemory, calls: list[str]):
    sizes = {"guard": 1.2, "embedder": 2.7, "reranker": 2.7, "generator": 5.8}
    fakes = {
        "guard": FakeGuard,
        "embedder": FakeEmbedder,
        "reranker": FakeReranker,
        "generator": FakeGenerator,
    }

    def make(role):
        def load(settings):
            calls.append(role)
            memory.active += int(sizes[role] * GB)
            return fakes[role]()

        return load

    return {role: make(role) for role in sizes}


def test_overrides_are_used_without_loading(home):
    reg = Registry(Settings(), overrides={"embedder": FakeEmbedder(dim=8)}, memory=FakeMemory())
    out = reg.embedder().embed([Item(text="a")], is_query=True)
    assert out.shape == (1, 8)
    assert reg.loaded() == []
    assert not (home / "index" / "models.lock").exists()  # overrides never take the lock


def test_core_models_load_once_in_pipeline_order(home):
    memory, calls = FakeMemory(), []
    reg = Registry(Settings(), loaders=counting_loaders(memory, calls), memory=memory)
    reg.load_core()
    reg.load_core()
    reg.generator()
    assert calls == ["guard", "embedder", "reranker", "generator"]
    assert len(reg.loaded()) == 4
    assert reg.memory_gb()["active"] == pytest.approx(12.4)
    reg.process_lock.release()


def test_memory_check_refuses_to_load(home):
    memory, calls = FakeMemory(active_gb=20.0), []
    reg = Registry(Settings(), loaders=counting_loaders(memory, calls), memory=memory)
    with pytest.raises(InsufficientMemory, match=r"20\.0 GB active \+ 5\.8 GB"):
        reg.generator()
    assert calls == []
    assert memory.cleared == 1  # tried evicting voice models first
    reg.process_lock.release()


def test_headroom_counts(home):
    memory, calls = FakeMemory(active_gb=0.0, recommended_gb=7.0), []
    reg = Registry(
        Settings(memory_headroom_gb=1.5), loaders=counting_loaders(memory, calls), memory=memory
    )
    with pytest.raises(InsufficientMemory):
        reg.generator()  # 5.8 + 1.5 > 7.0
    reg.process_lock.release()


def test_unpinned_model_is_refused(home):
    memory, calls = FakeMemory(), []
    reg = Registry(
        Settings(gen_model="someone/else"), loaders=counting_loaders(memory, calls), memory=memory
    )
    with pytest.raises(KeyError, match="not pinned"):
        reg.generator()
    assert calls == []


def test_calls_are_serialised_by_the_mlx_lock(home):
    active, peak = 0, 0
    guard = threading.Lock()

    class Slow(FakeReranker):
        def score(self, query, docs):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            return super().score(query, docs)

    reg = Registry(Settings(), overrides={"reranker": Slow()}, memory=FakeMemory())
    threads = [
        threading.Thread(target=reg.reranker().score, args=(Item(text="q"), [Item(text="d")]))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1


HOLDER = """
import sys, time
from pathlib import Path
from qwn.models import ProcessLock
lock = ProcessLock(Path(sys.argv[1]))  # keep a reference: closing the file drops the lock
lock.acquire()
print("locked", flush=True)
time.sleep(60)
"""


def test_second_process_is_refused_until_first_exits(tmp_path):
    path = tmp_path / "models.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(ModelsInUse, match=rf"pid {holder.pid}"):
            ProcessLock(path).acquire()
    finally:
        holder.kill()
        holder.wait()
    lock = ProcessLock(path)
    lock.acquire()  # the OS released the dead holder's lock
    lock.release()


def test_fast_tests_do_not_import_mlx():
    code = (
        "import sys, qwn.cli, qwn.models, qwn.adapters.mlx_vlm_embed, "
        "qwn.adapters.mlx_vlm_gen, qwn.adapters.mlx_vlm_rerank, qwn.adapters.mlx_lm_guard; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('mlx', 'mlx_vlm', 'mlx_lm')]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
