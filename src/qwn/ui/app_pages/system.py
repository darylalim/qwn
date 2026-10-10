"""System: models, memory, paths and per-session settings (PLAN.md → Streamlit UI)."""

import streamlit as st

from qwn.models import CORE_ROLES
from qwn.ui import services as ui

st.set_page_config(page_title="qwn · System", layout="wide")
svc = ui.services()
st.title("System")
ui.show_problems(svc)

st.subheader("Models")
warm = svc.warm
if warm.state == "loading":
    st.info(
        f"Loading models in the background ({warm.done}/{warm.total})…",
        icon=":material/hourglass_top:",
    )
elif warm.state == "failed":
    st.error(
        f"Warm-up failed: {warm.error}. Models will load on first use instead; closing other "
        "apps frees memory.",
        icon=":material/error:",
    )
loaded = svc.registry.loaded()
st.dataframe(
    [
        {
            "Role": role,
            "Model": svc.registry.repo(role),
            "Loaded": svc.registry.repo(role) in loaded,
        }
        for role in CORE_ROLES
    ],
    hide_index=True,
    width="stretch",
    alt="Core models and whether each is loaded",
)

mem = ui.memory_line(svc)
na = "n/a"
metrics = {
    "Active (GB)": na if mem is None else f"{mem['active']:.1f}",
    "Peak (GB)": na if mem is None else f"{mem['peak']:.1f}",
    "Limit (GB)": na if mem is None else f"{mem['recommended']:.1f}",
    "Models loaded": f"{len(loaded)}/{len(CORE_ROLES)}",
}
for col, (label, value) in zip(st.columns(4), metrics.items(), strict=True):
    col.metric(label, value, border=True)

st.subheader("Settings for this session")
st.caption("Applied to each question you ask from now on; they reset when the app restarts.")
current = st.session_state.settings


def apply() -> None:
    """Widget callback: copy the widgets into this session's settings before the rerun, so the
    sidebar's Guard line (drawn before this page) is already up to date."""
    state = st.session_state
    state.settings = {
        "top_k": state["set-top_k"],
        "rerank_k": state["set-rerank_k"],
        "max_images": state["set-max_images"],
        "guard_enabled": state["set-guard"],
        "controversial": "block" if state["set-block"] else "warn",
    }


left, mid, right = st.columns(3)
left.slider(
    "Candidates (top_k)", 5, 200, current["top_k"], step=5, key="set-top_k", on_change=apply
)
mid.slider(
    "Sources per answer (rerank_k)",
    1,
    10,
    current["rerank_k"],
    key="set-rerank_k",
    on_change=apply,
)
right.slider(
    "Page images sent (max_images)",
    0,
    8,
    current["max_images"],
    key="set-max_images",
    on_change=apply,
)
with st.container():
    st.toggle(
        "Guard checks questions and answers",
        current["guard_enabled"],
        key="set-guard",
        on_change=apply,
    )
    st.toggle(
        "Block controversial (instead of warning)",
        current["controversial"] == "block",
        key="set-block",
        on_change=apply,
        disabled=not current["guard_enabled"],
    )

st.subheader("Paths")
s = svc.settings
st.table(
    {
        "Home": str(s.home),
        "Library folder": str(s.data_dir),
        "Index": str(s.index_dir),
        "Logs": str(s.log_dir),
    },
    alt="Where qwn keeps its files",
)
