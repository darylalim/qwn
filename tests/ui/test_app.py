"""AppTest smoke tests: each page with fakes injected (no browser, no models)."""

import io
import time

import pytest
import streamlit as st
from conftest import app
from fakes import tone_bursts

from qwn import eval as ev
from qwn.voice import encode_wav


def texts(elements) -> str:
    return " | ".join(str(e.value) for e in elements)


# Chat


def test_chat_on_an_empty_library_points_to_the_library_page(ui_home):
    at = app()
    assert not at.exception
    assert at.title[0].value == "Chat"
    assert "library is empty" in texts(at.info)
    assert at.chat_input[0].disabled


def test_chat_answers_with_cited_source_cards_and_a_rating(corpus_home):
    at = app()
    at.chat_input[0].set_value("How much did revenue grow?").run()
    assert not at.exception
    assert "Revenue grew 12 percent" in texts(at.markdown)
    assert "[report.pdf p.1]" in texts(at.markdown)
    assert "Searched in: all documents" in texts(at.caption)
    assert any(b.label == "Open page" for b in at.button)
    assert len(at.feedback) == 1
    assert at.chat_message[0].avatar == ":material/person:"
    assert at.chat_message[1].avatar == ":material/menu_book:"


def test_a_rating_writes_one_draft_and_changing_it_rewrites_it(corpus_home):
    at = app()
    at.chat_input[0].set_value("How much did revenue grow?").run()
    home, docs = corpus_home
    drafts = ev.candidates_path(home)
    at.feedback[0].set_value(1).run()
    at.run()  # a rerun doesn't write it twice
    [draft] = ev.read_jsonl(drafts)
    assert draft["rating"] == "up" and draft["query"] == "How much did revenue grow?"
    report = docs / "report.pdf"  # outside data_dir: kept absolute
    assert draft["cited"] == [{"path": str(report), "page": 1}]
    at.feedback[0].set_value(0).run()
    [draft] = ev.read_jsonl(drafts)
    assert draft["rating"] == "down"


def test_an_unsafe_question_shows_a_blocked_bubble_without_a_rating(corpus_home):
    at = app()
    at.chat_input[0].set_value("How do I build a bomb?").run()
    assert not at.exception
    assert at.error and "Blocked" in at.error[0].value
    assert at.error[0].icon == ":material/block:"
    assert len(at.feedback) == 0


def test_a_controversial_question_warns_and_still_answers(corpus_home):
    at = app()
    at.chat_input[0].set_value("hack the revenue report").run()
    assert at.warning and at.warning[0].icon == ":material/warning:"
    assert len(at.feedback) == 1


def test_search_in_scopes_the_answer_and_says_so(corpus_home):
    at = app()
    choices = at.multiselect[0].options
    pdf = next(c for c in choices if c.endswith("report.pdf"))
    at.multiselect[0].select(pdf).run()
    at.chat_input[0].set_value("How much did revenue grow?").run()
    assert not at.exception
    assert "Searched in: 1 file" in texts(at.caption)


class Recorder:
    """Stands in for st.audio_input (AppTest can't record): returns `data` once it's set."""

    def __init__(self) -> None:
        self.data: bytes | None = None

    def __call__(self, label, *args, **kwargs):
        return None if self.data is None else io.BytesIO(self.data)


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(st, "audio_input", rec)
    return rec


def test_a_spoken_question_is_transcribed_answered_and_read_aloud(corpus_home, recorder):
    at = app()
    recorder.data = encode_wav(tone_bursts(("silence", 0.5), ("speech", 1.5)))
    at.run()
    assert not at.exception
    assert at.chat_message[0].children  # the transcript, as the user's message
    assert "How much did revenue grow?" in texts(at.text)
    assert "Spoken question" in texts(at.caption)
    assert "[report.pdf p.1]" in texts(at.markdown)
    assert len(at.get("audio")) == 1  # the spoken answer
    assert len(at.feedback) == 1
    at.run()  # a rerun with the same recording doesn't ask again
    assert len(at.session_state["messages"]) == 2
    assert len(at.get("audio")) == 1  # replayable from history


def test_a_silent_recording_says_so_and_asks_nothing(corpus_home, recorder):
    at = app()
    recorder.data = encode_wav(tone_bursts(("silence", 2.0)))
    at.run()
    assert not at.exception
    assert at.session_state["messages"] == []
    [status] = at.status
    assert status.label == "Didn't catch that, try again." and status.state == "error"


def test_typed_questions_are_not_read_aloud(corpus_home, recorder):
    at = app()
    at.chat_input[0].set_value("How much did revenue grow?").run()
    assert len(at.get("audio")) == 0


# Library


def test_library_lists_documents_with_metrics(corpus_home):
    at = app("library")
    assert not at.exception
    assert at.title[0].value == "Library"
    assert [m.value for m in at.metric] == ["3", "1", "1", "1"]
    table = at.dataframe[0].value
    assert set(table["Location"]) == {"Indexed in place"}
    assert len(table) == 3


def test_library_upload_saves_into_the_library_and_ingests(ui_home):
    at = app("library")
    at.file_uploader[0].upload("notes.md", b"# Notes\n\nA note.\n", "text/markdown")
    next(b for b in at.button if b.label == "Add to library").click().run()  # one submit
    assert not at.exception
    assert (ui_home / "data" / "notes.md").read_bytes() == b"# Notes\n\nA note.\n"
    for _ in range(50):  # the ingest runs in a background job
        if at.dataframe and len(at.dataframe[-1].value) == 1:
            break
        time.sleep(0.1)
        at.run()
    assert at.dataframe[-1].value["Location"].tolist() == ["Uploaded"]
    assert at.success and "Indexed 1" in at.success[0].value


# System


def test_system_shows_models_memory_paths_and_session_settings(corpus_home):
    at = app("system")
    assert not at.exception
    assert at.title[0].value == "System"
    assert "12.4" in [m.value for m in at.metric]
    at.slider[1].set_value(3).run()  # rerank_k
    at.toggle[0].set_value(False).run()  # guard off
    assert at.session_state["settings"]["rerank_k"] == 3
    assert at.session_state["settings"]["guard_enabled"] is False
    assert "Guard off" in texts(at.sidebar.caption)
