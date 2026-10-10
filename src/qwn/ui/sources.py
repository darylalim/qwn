"""Source cards under an answer, and the "Open page" dialog with the supporting passage
outlined (PLAN.md → Responsive layout, → Answer highlighting)."""

import logging

import streamlit as st

from qwn.adapters.images import highlight
from qwn.answer import Response, cite_label
from qwn.interfaces import Box, Source
from qwn.retrieve import Hit
from qwn.ui import services as ui

logger = logging.getLogger(__name__)

THUMB_PX = 96
VISIBLE_ROWS = 2  # more cards than this many rows go in "More sources"
PER_ROW = 2


def _box(response: Response, source: Source, msg_id: str) -> Box | None:
    """The supporting passage on this page, found once per (message, source) and kept in the
    session. A model error isn't kept, so opening the page again retries."""
    boxes: dict[str, Box | None] = st.session_state.boxes
    key = f"{msg_id}:{source.label}"
    if key not in boxes:
        try:
            with st.spinner("Finding the supporting passage…"):
                boxes[key] = ui.answerer(ui.services()).locate(response, source.label)
        except Exception:  # the page is still worth showing without its highlight
            logger.exception("locate failed")
            return None
    return boxes[key]


@st.dialog("Source page", width="medium")
def open_page(label: str, source: Source, response: Response, msg_id: str) -> None:
    st.markdown(f"**{label}**")
    st.caption(source.path)
    if source.image_path is None or not source.image_path.is_file():
        st.text(source.text)  # document text: plain, never rendered as markdown
        return
    box = _box(response, source, msg_id)
    if box is None:
        st.image(str(source.image_path), width="stretch", alt=f"Page {label}")
        st.caption(":material/search_off: Couldn't pinpoint the passage")
    else:
        alt = f"Page {label}, with the supporting passage outlined"
        st.image(highlight(source.image_path, box), width="stretch", alt=alt)
        st.caption(":material/highlight: Supporting passage (approximate)")


def _card(source: Source, hit: Hit, response: Response, msg_id: str) -> None:
    label = cite_label(source, hit)
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        if source.image_path is not None and source.image_path.is_file():
            st.image(str(source.image_path), width=THUMB_PX, alt=f"Thumbnail of {label}")
        with st.container(gap="xsmall"):
            st.markdown(f"**{label}**")
            st.badge(f"score {source.score:.2f}", icon=":material/sort:", color="gray")
            key = f"open-{msg_id}-{source.label}"
            if st.button("Open page", key=key, type="tertiary", icon=":material/open_in_full:"):
                open_page(label, source, response, msg_id)


def cards(response: Response, msg_id: str) -> None:
    """The cited sources as cards, two per row; rows past the second go in an expander."""
    cited = response.cited_sources()
    if not cited:
        return
    rows = [cited[i : i + PER_ROW] for i in range(0, len(cited), PER_ROW)]

    def draw(row_list: list[list[tuple[Source, Hit]]]) -> None:
        for row in row_list:
            for col, (source, hit) in zip(st.columns(PER_ROW), row, strict=False):
                with col:
                    _card(source, hit, response, msg_id)

    draw(rows[:VISIBLE_ROWS])
    if len(rows) > VISIBLE_ROWS:
        with st.expander("More sources"):
            draw(rows[VISIBLE_ROWS:])
