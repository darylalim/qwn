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


def test_save_writes_a_file_and_says_what_is_not_saved(corpus_home, generator, store):
    at = app()
    assert "Saved conversations appear here." in texts(at.sidebar.caption)
    assert not any(b.label == "Save" for b in at.button)
    at.chat_input[0].set_value(REVENUE).run()
    button(at, "Save").click().run()
    assert not at.exception
    [name] = files(store)
    assert "Saved " in texts(at.caption) and "not saved" not in texts(at.caption)
    at.chat_input[0].set_value("And in 2025?").run()
    assert "1 newer exchange not saved" in texts(at.caption)
    button(at, "Save").click().run()
    assert files(store) == [name]  # the same file, overwritten
    assert len(store.load(name.removesuffix(".json")).exchanges) == 2
    assert "not saved" not in texts(at.caption)
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
    assert "1 newer exchange not saved" in texts(at.caption)


def test_new_conversation_after_open_saves_a_second_file(corpus_home, generator, store):
    at = app()
    at.chat_input[0].set_value(REVENUE).run()
    button(at, "Save").click().run()
    [summary] = store.list()
    at = open_saved(app(), summary.id)
    button(at, "New conversation").click().run()
    assert at.session_state["saved"] is None
    at.chat_input[0].set_value("Which plan has single sign-on?").run()
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
