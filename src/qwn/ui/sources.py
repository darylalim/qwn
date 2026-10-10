"""Source cards under an answer, and the "Open page" dialog (PLAN.md → Responsive layout)."""

from pathlib import Path

import streamlit as st

from qwn.answer import Response, cite_label
from qwn.interfaces import Source
from qwn.retrieve import Hit

THUMB_PX = 96
VISIBLE_ROWS = 2  # more cards than this many rows go in "More sources"
PER_ROW = 2


@st.dialog("Source page", width="medium")
def open_page(label: str, path: str, image_path: Path | None, text: str) -> None:
    st.markdown(f"**{label}**")
    st.caption(path)
    if image_path is not None and image_path.is_file():
        st.image(str(image_path), width="stretch", alt=f"Page {label}")
    else:
        st.text(text)  # document text: plain, never rendered as markdown


def _card(source: Source, hit: Hit, key: str) -> None:
    label = cite_label(source, hit)
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        if source.image_path is not None and source.image_path.is_file():
            st.image(str(source.image_path), width=THUMB_PX, alt=f"Thumbnail of {label}")
        with st.container(gap="xsmall"):
            st.markdown(f"**{label}**")
            st.badge(f"score {source.score:.2f}", icon=":material/sort:", color="gray")
            if st.button("Open page", key=key, type="tertiary", icon=":material/open_in_full:"):
                open_page(label, source.path, source.image_path, source.text)


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
                    _card(source, hit, key=f"open-{msg_id}-{source.label}")

    draw(rows[:VISIBLE_ROWS])
    if len(rows) > VISIBLE_ROWS:
        with st.expander("More sources"):
            draw(rows[VISIBLE_ROWS:])
