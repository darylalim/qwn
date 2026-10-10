"""What the pages share: one set of cached resources per server process, plus per-session state.

The UI has no model or retrieval logic of its own (PLAN.md → Streamlit UI): it wires `qwn.*`
objects together and shows their results. Tests replace `make_registry` to inject fakes.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

from qwn import logging_setup
from qwn.answer import Answerer
from qwn.config import Settings
from qwn.index import Index, IndexMismatch
from qwn.ingest import Ingester, open_index
from qwn.jobs import JobRegistry
from qwn.models import ModelsInUse, Registry, WarmLoad
from qwn.retrieve import Retriever
from qwn.voice import Voice

# Sliders and toggles on the System page: applied per request, never persisted.
TUNABLE = ("top_k", "rerank_k", "max_images", "guard_enabled", "controversial")


def _registry(settings: Settings) -> Registry:
    return Registry(settings)


# The one place the UI gets models from; tests replace it to inject fakes.
make_registry: Callable[[Settings], Registry] = _registry


@dataclass
class Services:
    settings: Settings  # base settings (defaults → qwn.toml → env)
    registry: Registry
    warm: WarmLoad
    jobs: JobRegistry
    index: Index | None  # None when the index can't be opened (see `problems`)
    problems: list[str] = field(default_factory=list)  # shown on every page

    @property
    def ready(self) -> bool:
        return self.index is not None and not self.problems


@st.cache_resource
def services() -> Services:
    """Created once per server process, shared by every tab and rerun."""
    settings = Settings()
    logging_setup.configure(settings.log_dir)
    registry = make_registry(settings)
    problems: list[str] = []
    try:
        registry.process_lock.acquire()  # one process owns the models: the UI, while it runs
    except ModelsInUse as e:
        problems.append(str(e))
    index: Index | None = None
    try:
        index = open_index(settings)
    except IndexMismatch as e:
        problems.append(str(e))
    warm = WarmLoad(registry)
    if settings.warm_load and not problems:
        warm.start()
    return Services(settings, registry, warm, JobRegistry(), index, problems)


def init_session(svc: Services) -> None:
    """Session state, initialised in one place (the app entry point runs it before every page)."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("settings", {key: getattr(svc.settings, key) for key in TUNABLE})


def request_settings(svc: Services) -> Settings:
    """Base settings with this session's sliders applied (per request; not persisted)."""
    return svc.settings.model_copy(update=dict(st.session_state.get("settings", {})))


def require_index(svc: Services) -> Index:
    """The index, for pages that have already stopped on `show_problems`."""
    if svc.index is None:
        raise RuntimeError("the index isn't open")
    return svc.index


def answerer(svc: Services) -> Answerer:
    settings = request_settings(svc)
    retriever = Retriever(settings, require_index(svc), svc.registry)
    return Answerer(settings, retriever, svc.registry)


def voice(svc: Services) -> Voice:
    return Voice(request_settings(svc), svc.registry)


def ingester(svc: Services) -> Ingester:
    return Ingester(svc.settings, require_index(svc), svc.registry.embedder)


def show_problems(svc: Services) -> bool:
    """Show what stops the app working (models in use elsewhere, index mismatch). True if any."""
    for problem in svc.problems:
        st.error(problem, icon=":material/error:")
    return bool(svc.problems)


def display_path(path: str, data_dir: Path) -> str:
    """Paths inside the library folder are shown relative to it; others with ~ for home."""
    p = Path(path)
    if p.is_relative_to(data_dir):
        return str(p.relative_to(data_dir))
    home = Path.home()
    return f"~/{p.relative_to(home)}" if p.is_relative_to(home) else path


_IMAGE = re.compile(r"!\[")


def safe_markdown(text: str) -> str:
    """Model output for `st.markdown`. Answers can echo document text, so markdown images are
    neutralised: rendering one would make the browser fetch a URL a document chose. (HTML is
    already escaped by Streamlit.)"""
    return _IMAGE.sub(r"!\\[", text)


def memory_line(svc: Services) -> dict[str, Any] | None:
    try:
        return svc.registry.memory_gb()
    except Exception:  # MLX unavailable
        return None
