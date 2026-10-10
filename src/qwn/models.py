"""Model registry: lazy loading, memory policy, one MLX lock, one model-owning process.

See PLAN.md → Runtime robustness → Memory policy. Tests inject fakes through `overrides`.
"""

import fcntl
import functools
import gc
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Literal, Protocol, cast

from qwn.config import Settings
from qwn.interfaces import Embedder, Generator, Guard, Reranker
from qwn.models_lock import Group, pinned

logger = logging.getLogger(__name__)

Role = Literal["generator", "embedder", "reranker", "guard"]
CORE_ROLES: tuple[Role, ...] = ("guard", "embedder", "reranker", "generator")  # pipeline order
_SETTING: dict[Role, str] = {
    "generator": "gen_model",
    "embedder": "embed_model",
    "reranker": "rerank_model",
    "guard": "guard_model",
}

GB = 1e9


class InsufficientMemory(RuntimeError):
    pass


class ModelsInUse(RuntimeError):
    pass


class Memory(Protocol):
    def active_bytes(self) -> int: ...
    def peak_bytes(self) -> int: ...
    def recommended_bytes(self) -> int: ...
    def clear_cache(self) -> None: ...


class ProcessLock:
    """Exclusive, non-blocking flock on index_dir/models.lock, held until the process exits.

    The OS releases it if the process dies, so a stale lock can't block anyone.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: IO[str] | None = None

    def acquire(self) -> None:
        if self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = self.path.open("a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            pid = f.read().strip() or "?"
            f.close()
            raise ModelsInUse(
                f"Models are in use by another qwn process (pid {pid}, probably `qwn ui`). "
                "Close it, or use the UI."
            ) from None
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()))  # for the message above only; the lock decides
        f.flush()
        self._file = f

    def release(self) -> None:
        if self._file is not None:
            fcntl.flock(self._file, fcntl.LOCK_UN)
            self._file.close()
            self._file = None


class _Locked:
    """Routes every method call on a model through the registry's MLX lock."""

    def __init__(self, inner: object, lock: threading.Lock) -> None:
        self._inner = inner
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        @functools.wraps(attr)
        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attr(*args, **kwargs)

        return call


Loader = Callable[[Settings], object]


class Registry:
    def __init__(
        self,
        settings: Settings,
        *,
        overrides: dict[Role, object] | None = None,
        loaders: dict[Role, Loader] | None = None,
        memory: Memory | None = None,
    ) -> None:
        self.settings = settings
        self.lock = threading.Lock()  # MLX isn't thread-safe: every model call goes through it
        self._load_lock = threading.Lock()
        self._overrides: dict[Role, object] = dict(overrides or {})
        self._loaders: dict[Role, Loader] = {**_default_loaders(), **(loaders or {})}
        self._memory = memory
        self._loaded: dict[Role, object] = {}
        self._groups: dict[Role, Group] = {}
        self.process_lock = ProcessLock(settings.index_dir / "models.lock")

    # typed accessors

    def embedder(self) -> Embedder:
        return cast(Embedder, self._get("embedder"))

    def reranker(self) -> Reranker:
        return cast(Reranker, self._get("reranker"))

    def generator(self) -> Generator:
        return cast(Generator, self._get("generator"))

    def guard(self) -> Guard:
        return cast(Guard, self._get("guard"))

    def load_core(self) -> None:
        for role in CORE_ROLES:
            self._get(role)

    # state

    def repo(self, role: Role) -> str:
        return getattr(self.settings, _SETTING[role])

    def loaded(self) -> list[str]:
        return [self.repo(role) for role in self._loaded]

    def memory_gb(self) -> dict[str, float]:
        m = self.memory
        return {
            "active": m.active_bytes() / GB,
            "peak": m.peak_bytes() / GB,
            "recommended": m.recommended_bytes() / GB,
        }

    @property
    def memory(self) -> Memory:
        if self._memory is None:
            from qwn.adapters.mlx_memory import MlxMemory

            self._memory = MlxMemory()
        return self._memory

    # loading

    def _get(self, role: Role) -> object:
        if role in self._overrides:
            return _Locked(self._overrides[role], self.lock)
        with self._load_lock:
            if role not in self._loaded:
                self._load(role)
            return _Locked(self._loaded[role], self.lock)

    def _load(self, role: Role) -> None:
        repo = self.repo(role)
        model = pinned(repo)
        self.process_lock.acquire()  # before any load: one process owns the models
        self._check_memory(repo, model.size_gb)
        logger.info("loading %s (%s)", role, repo)
        self._loaded[role] = self._loaders[role](self.settings)
        self._groups[role] = model.group

    def _fits(self, size_gb: float) -> tuple[bool, float, float]:
        active = self.memory.active_bytes() / GB
        limit = self.memory.recommended_bytes() / GB
        return active + size_gb + self.settings.memory_headroom_gb <= limit, active, limit

    def _check_memory(self, repo: str, size_gb: float) -> None:
        ok, active, limit = self._fits(size_gb)
        if ok:
            return
        self.evict("voice")
        ok, active, limit = self._fits(size_gb)
        if not ok:
            raise InsufficientMemory(
                f"Not enough memory to load {repo}: {active:.1f} GB active + {size_gb:.1f} GB "
                f"model + {self.settings.memory_headroom_gb:.1f} GB headroom > {limit:.1f} GB "
                "recommended. Close other apps and try again."
            )

    def evict(self, group: Group) -> None:
        with self.lock:
            for role in [r for r, g in self._groups.items() if g == group]:
                logger.info("evicting %s", role)
                del self._loaded[role], self._groups[role]
            gc.collect()
            self.memory.clear_cache()


class WarmLoad:
    """Loads the core models in one background thread, in pipeline order (`qwn ui`, phase 5).

    `start` is idempotent, so reruns and extra tabs sharing this object never start a second
    thread. A failed load stops warming and is kept in `error`; the registry then loads each
    model on first use, as the CLI does.
    """

    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.total = len(CORE_ROLES)
        self.done = 0
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    @property
    def state(self) -> Literal["idle", "loading", "ready", "failed"]:
        if self.error is not None:
            return "failed"
        if self.done == self.total:
            return "ready"
        return "idle" if self._thread is None else "loading"

    def start(self) -> None:
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="qwn-warm-load", daemon=True)
                self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        for role in CORE_ROLES:
            try:
                self.registry._get(role)
            except Exception as e:  # e.g. InsufficientMemory, ModelsInUse, not downloaded
                logger.warning("warm load stopped at %s (%s)", role, type(e).__name__)
                self.error = f"{type(e).__name__}: {e}"
                return
            self.done += 1
        logger.info("warm load done (%d models)", self.done)


def _default_loaders() -> dict[Role, Loader]:
    # Adapters import MLX libraries lazily, so importing qwn.models stays light.
    def generator(s: Settings) -> object:
        from qwn.adapters.mlx_vlm_gen import MlxVlmGenerator

        return MlxVlmGenerator.load(s)

    def embedder(s: Settings) -> object:
        from qwn.adapters.mlx_vlm_embed import MlxVlmEmbedder

        return MlxVlmEmbedder.load(s)

    def reranker(s: Settings) -> object:
        from qwn.adapters.mlx_vlm_rerank import MlxVlmReranker

        return MlxVlmReranker.load(s)

    def guard(s: Settings) -> object:
        from qwn.adapters.mlx_lm_guard import MlxLmGuard

        return MlxLmGuard.load(s)

    return {"generator": generator, "embedder": embedder, "reranker": reranker, "guard": guard}
