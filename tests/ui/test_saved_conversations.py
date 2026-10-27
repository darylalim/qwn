"""Saved conversations on the Chat page (PLAN.md → CLI → Saved conversations), with fakes."""

import pytest
import streamlit as st
from conftest import app
from fakes import FakeGenerator

from qwn.conversations import Exchange, Store, make

REVENUE = "How much did revenue grow?"


def texts(elements) -> str:
    return " | ".join(str(e.value) for e in elements)


def button(at, label):
    return next(b for b in at.button if b.label == label)


@pytest.fixture
def generator(monkeypatch):
    from fake_app import fake_registry

    import qwn.ui.services

    gen = FakeGenerator()
    monkeypatch.setattr(
        qwn.ui.services, "make_registry", lambda settings: fake_registry(settings, gen)
    )
    st.cache_resource.clear()
    return gen


@pytest.fixture
def store(corpus_home):
    return Store(corpus_home[0] / "index")


def files(store: Store) -> list[str]:
    return sorted(p.name for p in store.folder.glob("*.json"))


def open_saved(at, conversation_id: str):
    at.sidebar.selectbox[0].select(conversation_id).run()
    button(at, "Open").click().run()
    assert not at.exception
    return at


AUTO = "new answers are saved automatically"


def has_save(at) -> bool:
    return any(b.label == "Save" for b in at.button)


def test_save_writes_a_file_then_new_answers_are_saved_automatically(corpus_home, generator, store):
    at = app()
    assert "Saved conversations appear here." in texts(at.sidebar.caption)
    assert not has_save(at)
    at.chat_input[0].set_value(REVENUE).run()
    button(at, "Save").click().run()
    assert not at.exception
    [name] = files(store)
    assert AUTO in texts(at.caption) and not has_save(at)
    at.chat_input[0].set_value("And in 2025?").run()
    assert files(store) == [name]  # the same file, written without pressing Save
    assert len(store.load(name.removesuffix(".json")).exchanges) == 2
    assert AUTO in texts(at.caption) and "not saved" not in texts(at.caption)
    assert not has_save(at) and any(b.label == "Export" for b in at.download_button)
    [option] = at.sidebar.selectbox[0].options
    assert option.startswith(f"{REVENUE} · ") and option.endswith(" · 2 questions")


def test_an_error_exchange_is_not_saved(corpus_home, generator, store, monkeypatch):
    answer = generator.answer
    failures = iter([True])

    def flaky(question, sources, *, greedy=False):
        if next(failures, False):
            raise RuntimeError("boom")
        return answer(question, sources, greedy=greedy)

    monkeypatch.setattr(generator, "answer", flaky)
    at = app()
    at.chat_input[0].set_value(REVENUE).run()
    assert at.error  # the failed exchange
    at.chat_input[0].set_value("Which plan has single sign-on?").run()
    button(at, "Save").click().run()
    [summary] = store.list()
    [exchange] = store.load(summary.id).exchanges
    assert exchange.question == "Which plan has single sign-on?"


def test_an_error_after_saving_is_not_autosaved(corpus_home, generator, store, monkeypatch):
    at = app()
    at.chat_input[0].set_value(REVENUE).run()
    button(at, "Save").click().run()
    [summary] = store.list()

    def boom(question, sources, *, greedy=False):
        raise RuntimeError("boom")

    monkeypatch.setattr(generator, "answer", boom)
    at.chat_input[0].set_value("And in 2025?").run()
    assert at.error and not has_save(at) and AUTO in texts(at.caption)
    assert len(store.load(summary.id).exchanges) == 1


def test_a_conversation_never_saved_writes_no_file(corpus_home, generator, store):
    at = app()
    for question in (REVENUE, "And in 2025?", "Which plan has single sign-on?"):
        at.chat_input[0].set_value(question).run()
    assert not at.exception and files(store) == [] and has_save(at)
    assert "bookmark_added" not in texts(at.caption)  # no "Saved 14:02" caption


def test_open_restores_the_conversation_and_follow_ups_carry_on(corpus_home, generator, store):
    at = app()
    pdf = next(c for c in at.multiselect[0].options if c.endswith("report.pdf"))
    at.multiselect[0].select(pdf).run()
    at.chat_input[0].set_value(REVENUE).run()
    at.chat_input[0].set_value("And in 2025?").run()
    button(at, "Save").click().run()
    [summary] = store.list()

    at = open_saved(app(), summary.id)  # a new session, as after a restart
    messages = at.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"] * 2
    assert at.multiselect[0].value == [pdf]
    assert "Searched for: How much did revenue grow? And in 2025?" in texts(at.caption)
    assert texts(at.caption).count("Searched in: 1 file") == 2
    assert "Revenue grew 12 percent" in texts(at.markdown)
    assert any(b.label == "Open page" for b in at.button)
    assert f"Saved {summary.saved[11:16]}" in texts(at.caption)
    generator.rewrites.clear()
    at.chat_input[0].set_value("And in 2024?").run()
    [(history, message)] = generator.rewrites
    assert [t.question for t in history] == [REVENUE, "And in 2025?"]
    assert message == "And in 2024?"
    assert AUTO in texts(at.caption) and not has_save(at)
    assert [e.question for e in store.load(summary.id).exchanges][-1] == "And in 2024?"
    assert files(store) == [f"{summary.id}.json"]


def test_new_conversation_after_open_saves_a_second_file(corpus_home, generator, store):
    at = app()
    at.chat_input[0].set_value(REVENUE).run()
    button(at, "Save").click().run()
    [summary] = store.list()
    at = open_saved(app(), summary.id)
    button(at, "New conversation").click().run()
    assert at.session_state["saved"] is None
    at.chat_input[0].set_value("Which plan has single sign-on?").run()
    assert files(store) == [f"{summary.id}.json"]  # a new conversation isn't autosaved
    assert len(store.load(summary.id).exchanges) == 1
    button(at, "Save").click().run()
    assert len(files(store)) == 2


def test_a_scope_no_longer_indexed_is_dropped_with_a_message(corpus_home, generator, store):
    at = app()
    at.chat_input[0].set_value(REVENUE).run()
    button(at, "Save").click().run()
    [summary] = store.list()
    first = store.load(summary.id).exchanges[0]
    store.save(make([Exchange(first.id, first.question, first.response, ["/gone/old.pdf"])]))
    store.delete(summary.id)
    [moved] = store.list()
    at = open_saved(app(), moved.id)
    assert at.multiselect[0].value == []
    assert "no longer in your library; searching all documents." in texts(at.info)


def test_a_missing_page_shows_no_thumbnail_and_no_open_page(corpus_home, generator, store):
    at = app()
    at.chat_input[0].set_value(REVENUE).run()
    button(at, "Save").click().run()
    [summary] = store.list()
    for page in (store.index_root / "pages").rglob("*.webp"):
        page.unlink()
    at = open_saved(app(), summary.id)
    assert "No longer in your library" in texts(at.caption)
    assert not any(b.label == "Open page" for b in at.button)
    assert not at.image


def test_delete_asks_first_then_removes_the_file(corpus_home, generator, store):
    at = app()
    at.chat_input[0].set_value(REVENUE).run()
    button(at, "Save").click().run()
    [summary] = store.list()
    at.sidebar.selectbox[0].select(summary.id).run()
    button(at, "Delete").click().run()
    assert "This removes the saved file." in texts(at.markdown)
    assert files(store)  # not yet
    # AppTest doesn't rerun a dialog by itself (the browser reruns it as a fragment): press the
    # sidebar's Delete again so the dialog is drawn on the run that clicks its own Delete
    sidebar, confirm = (b for b in at.button if b.label == "Delete")
    sidebar.click()
    confirm.click().run()
    assert not at.exception
    assert files(store) == [] and at.session_state["saved"] is None
    assert "Saved conversations appear here." in texts(at.sidebar.caption)
    at.chat_input[0].set_value("And in 2025?").run()
    assert files(store) == [] and has_save(at)  # deleting stops its autosave


# Export and Rename (PLAN.md → CLI → Export and rename conversations)


@pytest.fixture
def downloads(monkeypatch):
    """The arguments of every `st.download_button` call, newest last (AppTest keeps the file
    only during a run), with a fixed clock for the export time."""
    import qwn.conversations

    calls: list[dict] = []
    real = st.download_button

    def record(label, data, **kwargs):
        calls.append({"label": label, "data": data, **kwargs})
        return real(label, data, **kwargs)

    monkeypatch.setattr(st, "download_button", record)
    monkeypatch.setattr(qwn.conversations, "now", lambda: "2026-10-26T14:05:31+08:00")
    return calls


def test_export_offers_the_conversation_on_screen_as_markdown(corpus_home, generator, downloads):
    at = app()
    assert not at.download_button and not downloads  # nothing to export before an answer
    at.chat_input[0].set_value(REVENUE).run()
    at.chat_input[0].set_value("And in 2025?").run()
    assert not at.exception
    [export] = at.download_button
    assert export.label == "Export"
    last = downloads[-1]
    assert last["file_name"] == "qwn-conversation-2026-10-26-1405.md"
    assert last["mime"] == "text/markdown" and last["on_click"] == "ignore"
    text = last["data"]
    assert text.startswith(f"# {REVENUE}\n\nExported from qwn, 2026-10-26 14:05 · 2 questions")
    assert "· not saved\n" in text  # unsaved exchanges are exported too
    assert "## 2. And in 2025?" in text and f"Searched for: {REVENUE} And in 2025?" in text
    assert "[report.pdf p.1]" in text and f"{corpus_home[1] / 'report.pdf'}, page 1" in text
    button(at, "Save").click().run()
    assert "· saved 2026-10-26 " in downloads[-1]["data"]


def rename_to(at, title: str):
    """Open the Rename dialog and press its Rename on the same run (AppTest doesn't rerun a
    dialog by itself; see the Delete test)."""
    button(at, "Rename").click().run()
    [field] = [t for t in at.text_input if t.label == "Title"]
    field.set_value(title)
    sidebar, confirm = (b for b in at.button if b.label == "Rename")
    sidebar.click()
    confirm.click().run()
    assert not at.exception
    return at


def test_rename_through_the_dialog(corpus_home, generator, store):
    at = app()
    at.chat_input[0].set_value(REVENUE).run()
    button(at, "Save").click().run()
    [summary] = store.list()
    at = app()  # a new session: the open conversation isn't this one
    assert button(at, "Rename").disabled
    at.sidebar.selectbox[0].select(summary.id).run()
    rename_to(at, "  ")
    assert "A title can't be empty." in texts(at.error)
    assert store.load(summary.id).title == REVENUE  # unchanged, the dialog still open
    rename_to(at, "Revenue by year")
    assert not at.error
    [option] = at.sidebar.selectbox[0].options
    assert option.startswith("Revenue by year · ")
    assert store.load(summary.id).saved == summary.saved


def test_renaming_the_open_conversation_then_saving_keeps_the_title(corpus_home, generator, store):
    at = app()
    at.chat_input[0].set_value(REVENUE).run()
    button(at, "Save").click().run()
    [summary] = store.list()
    at.sidebar.selectbox[0].select(summary.id).run()
    rename_to(at, "Mine")
    assert at.session_state["saved"].title == "Mine"
    at.chat_input[0].set_value("And in 2025?").run()
    saved = store.load(summary.id)
    assert saved.title == "Mine" and len(saved.exchanges) == 2


# Search (PLAN.md → CLI → Search saved conversations)

SSO = "Which plan has single sign-on?"


def two_saved(store: Store) -> tuple[str, str]:
    """The revenue conversation and the sign-on one, saved in that order: ids (revenue, sso)."""
    ids = []
    for question in (REVENUE, SSO):
        at = app()
        at.chat_input[0].set_value(question).run()
        button(at, "Save").click().run()
        ids.append(at.session_state["saved"].id)
    return ids[0], ids[1]


def search(at, text: str):
    at.text_input(key="saved_search").set_value(text).run()
    assert not at.exception
    return at


def test_search_narrows_the_saved_list(corpus_home, generator, store):
    revenue, sso = two_saved(store)
    at = app()
    assert "match" not in texts(at.sidebar.caption)  # nothing typed: no count
    assert at.sidebar.selectbox[0].options and len(at.sidebar.selectbox[0].options) == 2
    for text, expect in (("12 PERCENT", [revenue]), ("report.pdf", [revenue]), ("sign-on", [sso])):
        search(at, text)
        assert [o.split(" · ")[0] for o in at.sidebar.selectbox[0].options] == [
            store.load(i).title for i in expect
        ]
        assert "1 of 2 match" in texts(at.sidebar.caption)
    search(at, "zebra")
    assert "No saved conversation matches." in texts(at.sidebar.caption)
    assert not at.sidebar.selectbox and not any(b.label == "Open" for b in at.button)
    search(at, "")
    assert len(at.sidebar.selectbox[0].options) == 2 and "match" not in texts(at.sidebar.caption)


def test_a_pick_the_search_filters_out_is_cleared(corpus_home, generator, store):
    revenue, _ = two_saved(store)
    at = app()
    at.sidebar.selectbox[0].select(revenue).run()
    search(at, "percent")
    assert at.sidebar.selectbox[0].value == revenue  # still matches: kept
    search(at, "sign-on")
    assert at.session_state["saved_pick"] is None and at.sidebar.selectbox[0].value is None
    assert button(at, "Open").disabled


def test_renaming_the_pick_out_of_the_search_clears_it(corpus_home, generator, store):
    revenue, _ = two_saved(store)
    store.rename(revenue, "Zebra notes")
    at = search(app(), "zebra")  # only the title matches
    at.sidebar.selectbox[0].select(revenue).run()
    rename_to(at, "Mine")
    assert store.load(revenue).title == "Mine"
    assert at.session_state["saved_pick"] is None
    assert "No saved conversation matches." in texts(at.sidebar.caption)


def test_open_after_a_search_restores_the_conversation(corpus_home, generator, store):
    _, sso = two_saved(store)
    at = search(app(), "sign-on")
    at = open_saved(at, sso)
    assert [m["role"] for m in at.session_state["messages"]] == ["user", "assistant"]
    assert at.session_state["messages"][0]["content"] == SSO
    assert at.text_input(key="saved_search").value == "sign-on"  # the search stays


# Autosave (PLAN.md → CLI → Autosave saved conversations)

COPIED = "This conversation changed in another window, so it was saved as a copy."


def saved_once(store: Store):
    at = app()
    at.chat_input[0].set_value(REVENUE).run()
    button(at, "Save").click().run()
    [summary] = store.list()
    return summary.id


def test_two_windows_on_one_conversation_keep_both_answers(corpus_home, generator, store):
    conversation_id = saved_once(store)
    first, second = open_saved(app(), conversation_id), open_saved(app(), conversation_id)
    first.chat_input[0].set_value("And in 2025?").run()
    assert COPIED not in texts(first.info)
    second.chat_input[0].set_value(SSO).run()
    assert not second.exception and COPIED in texts(second.info)
    original = store.load(conversation_id)
    assert [e.question for e in original.exchanges] == [REVENUE, "And in 2025?"]
    [copy] = [s for s in store.list() if s.id != conversation_id]
    assert [e.question for e in store.load(copy.id).exchanges] == [REVENUE, SSO]
    assert second.session_state["saved"].id == copy.id
    assert len(second.sidebar.selectbox[0].options) == 2
    second.chat_input[0].set_value("And in 2024?").run()  # the copy carries on
    assert len(store.load(copy.id).exchanges) == 3 and COPIED not in texts(second.info)


def test_a_conversation_deleted_in_another_window_is_not_recreated(corpus_home, generator, store):
    conversation_id = saved_once(store)
    at = open_saved(app(), conversation_id)
    store.delete(conversation_id)  # as the other window's Delete does
    at.chat_input[0].set_value("And in 2025?").run()
    assert "deleted in another window. Save to keep it again." in texts(at.info)
    assert files(store) == [] and at.session_state["saved"] is None and has_save(at)


def test_an_unreadable_file_is_left_alone_and_save_writes_a_copy(corpus_home, generator, store):
    conversation_id = saved_once(store)
    at = open_saved(app(), conversation_id)
    store.path(conversation_id).write_text("{ hand edited")
    at.chat_input[0].set_value("And in 2025?").run()
    assert "Couldn't save to this conversation's file" in texts(at.error)
    assert "1 newer exchange not saved" in texts(at.caption) and has_save(at)
    assert store.path(conversation_id).read_text() == "{ hand edited"
    button(at, "Save").click().run()
    [copy] = store.list()  # the unreadable file isn't listed
    assert copy.id != conversation_id and copy.exchanges == 2
    assert store.path(conversation_id).read_text() == "{ hand edited"


def test_a_failed_write_shows_the_error_and_save(corpus_home, generator, store, monkeypatch):
    conversation_id = saved_once(store)
    at = open_saved(app(), conversation_id)

    def full(self, conversation):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Store, "autosave", full)
    at.chat_input[0].set_value("And in 2025?").run()
    assert "Couldn't save the conversation: [Errno 28] No space left" in texts(at.error)
    assert "1 newer exchange not saved" in texts(at.caption)
    monkeypatch.undo()
    button(at, "Save").click().run()
    assert len(store.load(conversation_id).exchanges) == 2 and AUTO in texts(at.caption)
