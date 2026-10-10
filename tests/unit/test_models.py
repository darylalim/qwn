import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from fakes import (
    GB,
    FakeAsr,
    FakeEmbedder,
    FakeGenerator,
    FakeGuard,
    FakeMemory,
    FakeReranker,
    FakeTts,
    FakeVad,
)

from qwn.config import Settings
from qwn.interfaces import Item
from qwn.models import InsufficientMemory, ModelsInUse, ProcessLock, Registry, WarmLoad


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
        "qwn.adapters.mlx_vlm_gen, qwn.adapters.mlx_vlm_rerank, qwn.adapters.mlx_lm_guard, "
        "qwn.adapters.mlx_audio_vad, qwn.adapters.mlx_audio_asr, qwn.adapters.mlx_audio_tts, "
        "qwn.voice; "
        "bad = [m for m in sys.modules "
        "if m.split('.')[0] in ('mlx', 'mlx_vlm', 'mlx_lm', 'mlx_audio')]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# warm load (qwn ui)


def test_warm_load_uses_one_thread_however_often_it_is_started(home):
    memory, calls = FakeMemory(), []
    reg = Registry(Settings(), loaders=counting_loaders(memory, calls), memory=memory)
    warm = WarmLoad(reg)
    assert warm.state == "idle"
    for _ in range(5):  # every rerun / tab calls start()
        warm.start()
    warm.join(5)
    assert warm.state == "ready" and warm.done == warm.total == 4
    assert calls == ["guard", "embedder", "reranker", "generator"]  # once each, pipeline order


def test_warm_load_failure_falls_back_to_loading_on_first_use(home):
    memory, calls = FakeMemory(recommended_gb=8.0), []  # room for guard + embedder only
    reg = Registry(Settings(), loaders=counting_loaders(memory, calls), memory=memory)
    warm = WarmLoad(reg)
    warm.start()
    warm.join(5)
    assert warm.state == "failed" and "InsufficientMemory" in (warm.error or "")
    assert warm.done == 2 and calls == ["guard", "embedder"]
    memory.recommended = int(30 * GB)  # memory freed: the next use loads it lazily
    reg.reranker().score(Item(text="q"), [Item(text="q")])
    assert calls[-1] == "reranker"


# voice models: lazy and evictable


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def voice_loaders(memory: FakeMemory, calls: list[str]):
    sizes = {"vad": 0.002, "asr": 2.5, "tts": 2.9}
    fakes = {"vad": FakeVad, "asr": FakeAsr, "tts": FakeTts}

    def make(role):
        def load(settings):
            calls.append(role)
            memory.active += int(sizes[role] * GB)
            return fakes[role]()

        return load

    return {role: make(role) for role in sizes}


def test_voice_models_load_on_first_use_and_unload_when_idle(home):
    memory, calls, clock = FakeMemory(), [], Clock()
    reg = Registry(
        Settings(voice_idle_minutes=10),
        loaders=voice_loaders(memory, calls),
        memory=memory,
        clock=clock,
    )
    assert reg.loaded() == []
    asr = reg.asr()
    assert calls == ["asr"]
    clock.now = 9 * 60
    asr.transcribe(np.zeros(16_000, dtype=np.float32))  # use counts, not just loading
    clock.now = 18 * 60
    reg.evict_idle()
    assert reg.loaded() == [reg.repo("asr")]  # used 9 minutes ago: kept
    clock.now = 19 * 60
    reg.evict_idle()
    assert reg.loaded() == []
    assert memory.cleared == 1
    reg.asr()
    assert calls == ["asr", "asr"]  # loaded again on the next use
    reg.process_lock.release()


def test_any_model_request_unloads_idle_voice_models(home):
    memory, calls, clock = FakeMemory(), [], Clock()
    loaders = {**counting_loaders(memory, []), **voice_loaders(memory, calls)}
    reg = Registry(Settings(), loaders=loaders, memory=memory, clock=clock)
    reg.tts()
    clock.now = 11 * 60
    reg.guard()
    assert reg.loaded() == [reg.repo("guard")]
    reg.process_lock.release()


class FreeingMemory(FakeMemory):
    def clear_cache(self):  # dropping the voice models gives their memory back
        super().clear_cache()
        self.active = 0


def test_voice_models_are_evicted_to_make_room_for_core_ones(home):
    memory, calls = FreeingMemory(recommended_gb=12.0), []
    loaders = {**counting_loaders(memory, calls), **voice_loaders(memory, calls)}
    reg = Registry(Settings(memory_headroom_gb=1.0), loaders=loaders, memory=memory)
    reg.asr()
    reg.tts()
    assert memory.active == pytest.approx(5.4 * GB)
    reg.generator()  # 5.4 + 5.8 + 1.0 > 12: voice goes first
    assert reg.loaded() == [reg.repo("generator")]
    reg.process_lock.release()
