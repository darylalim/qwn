"""qwn Streamlit app: `qwn ui` runs `streamlit run` on this file (PLAN.md → Streamlit UI)."""

import os

# Offline at runtime: huggingface_hub reads this once, at import, so it's set before anything
# imports it (as in qwn.cli).
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path

import streamlit as st

from qwn.ui import services as ui

PAGES = Path(__file__).parent / "app_pages"

svc = ui.services()
ui.init_session(svc)

page = st.navigation(
    [
        st.Page(PAGES / "chat.py", title="Chat", icon=":material/chat:", default=True),
        st.Page(PAGES / "library.py", title="Library", icon=":material/library_books:"),
        st.Page(PAGES / "system.py", title="System", icon=":material/monitor_heart:"),
    ]
)


@st.fragment(run_every="1s" if svc.warm.state == "loading" else None)
def status_line() -> None:
    """The sidebar's one status line: model loading, then Guard on/off."""
    warm = svc.warm
    if warm.state == "loading":
        st.caption(f":material/hourglass_top: Loading models ({warm.done}/{warm.total})…")
    elif warm.state == "failed":
        st.caption(":material/error: Model warm-up failed: see System")
    elif st.session_state.settings["guard_enabled"]:
        st.caption(":material/shield: Guard on")
    else:
        st.caption(":material/remove_moderator: Guard off")


with st.sidebar:
    status_line()

page.run()
