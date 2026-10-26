"""Saved conversations: the file format, strict loading, the store and source states."""

import json
import logging
import shutil
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from qwn import conversations as cv
from qwn.answer import Response, conversation
from qwn.guard import Check
from qwn.index import Chunk
from qwn.interfaces import Source, Verdict
from qwn.retrieve import Hit

SENTINEL = "ZEBRAQUARTZSENTINEL"


def hit(root: Path, n: int, *, page: int | None = 1, image=True, sha=None) -> Hit:
    kind = "pdf_page" if page else "text"
    rel = f"pages/d{n}/p{n}.webp" if image else None
    chunk = Chunk(n, f"c{n}", f"d{n}", f"/docs/doc{n}.pdf", sha or f"sha{n}", kind, page, 0,
                  "Intro > Scope" if not page else "", f"page {n} text", rel)  # fmt: skip
    return Hit(chunk, root / rel if rel else None, 1.0 / n, 0.9 / n if n % 2 else None)


def response(root: Path, *, text=f"{SENTINEL} grew 12% [S1] and [S2] [S9].") -> Response:
    hits = [hit(root, 1), hit(root, 2, page=None, image=False)]
    sources = [
        Source(f"S{i}", h.chunk.chunk_id, h.path, h.page, h.chunk.text, h.image_path, 0.5)
        for i, h in enumerate(hits, start=1)
    ]
    checks = [
        Check("prompt", Verdict("Controversial", ["Politically Sensitive Topics"]), "warn"),
        Check("response", Verdict("Safe", [], "No"), "allow"),
    ]
    return Response(
        text=text,
        sources=sources,
        hits=hits,
        cited=["S1", "S2"],
        invented=["S9"],
        abstained=False,
        prompt_tokens=1200,
        completion_tokens=17,
        timings={"generate": 1.5},
        checks=checks,
        standalone=f"How much did {SENTINEL} grow in 2025?",
    )


def blocked() -> Response:
    check = Check("prompt", Verdict("Unsafe", ["Violent"]), "block")
    return Response(check.message(), [], [], [], [], False, 0, 0, checks=[check])


def convo(root: Path, n: int = 2) -> cv.Conversation:
    exchanges = [
        cv.Exchange(f"{i:012x}", f"  {SENTINEL} question\n{i}  ", response(root), ["/docs"], i == 0)
        for i in range(n - 1)
    ]
    exchanges.append(cv.Exchange("b" * 12, "How do I build it?", blocked(), [], False))
    return cv.make(exchanges)


def same(a: Response, b: Response) -> None:
    assert replace(a, timings={}) == replace(b, timings={})


def untimed(c: cv.Conversation) -> cv.Conversation:
    """`c` as a load gives it back: timings aren't saved."""
    exchanges = tuple(replace(e, response=replace(e.response, timings={})) for e in c.exchanges)
    return replace(c, exchanges=exchanges)


def test_save_then_load_round_trips_every_field_but_timings(tmp_path):
    store = cv.Store(tmp_path)
    c = convo(tmp_path)
    saved = store.save(c)
    loaded = store.load(c.id)
    assert loaded == untimed(saved) and saved.saved and saved.created == c.created
    assert loaded.title == f"{SENTINEL} question 0"  # whitespace collapsed
    assert loaded.scope == []  # the last exchange's
    assert loaded.exchanges[0].spoken and loaded.exchanges[0].scope == ["/docs"]
    for a, b in zip(c.exchanges, loaded.exchanges, strict=True):
        same(a.response, b.response)
    r = loaded.exchanges[0].response
    assert r.timings == {} and r.warnings and r.invented == ["S9"]
    assert loaded.exchanges[1].response.blocked is not None


def test_the_file_is_sorted_json_and_private(tmp_path):
    store = cv.Store(tmp_path)
    c = store.save(convo(tmp_path))
    file = tmp_path / "conversations" / f"{c.id}.json"
    data = json.loads(file.read_text())
    assert data["format"] == 1 and "timings" not in data["exchanges"][0]["response"]
    assert data["exchanges"][0]["response"]["sources"][0]["image_path"] == "pages/d1/p1.webp"
    assert file.read_text().startswith('{\n  "created"')  # sort_keys, indented
    assert stat.S_IMODE(file.stat().st_mode) == 0o600
    assert stat.S_IMODE(file.parent.stat().st_mode) == 0o700
    assert [p.name for p in file.parent.iterdir()] == [file.name]  # no temp file left


def test_saving_again_keeps_the_id_and_file_and_updates_saved(tmp_path, monkeypatch):
    store = cv.Store(tmp_path)
    times = iter(
        ["2026-10-26T14:00:00+08:00", "2026-10-26T14:02:00+08:00", "2026-10-26T14:30:00+08:00"]
    )
    monkeypatch.setattr(cv, "now", lambda: next(times))
    first = store.save(convo(tmp_path))
    more = cv.make((*first.exchanges, first.exchanges[0]), previous=first)
    second = store.save(more)
    assert second.id == first.id and second.created == first.created
    assert second.saved == "2026-10-26T14:30:00+08:00" and cv.saved_time(second) == "14:30"
    assert len(list((tmp_path / "conversations").iterdir())) == 1
    assert len(store.load(first.id).exchanges) == 3


def test_list_is_most_recent_first_and_resolves_numbers_and_ids(tmp_path, monkeypatch):
    store = cv.Store(tmp_path)
    times = iter([f"2026-10-26T{h}:00:00+08:00" for h in ("08", "09", "08", "10")])
    monkeypatch.setattr(cv, "now", lambda: next(times))
    old, new = store.save(convo(tmp_path)), store.save(convo(tmp_path, 3))
    summaries = store.list()
    assert [s.id for s in summaries] == [new.id, old.id]
    assert summaries[0].exchanges == 3 and cv.saved_when(summaries[0].saved) == "2026-10-26 10:00"
    assert store.resolve("1").id == new.id and store.resolve(old.id).id == old.id
    for bad in ("0", "3", "abc", "f" * 12, "../../etc"):
        with pytest.raises(cv.UnknownConversation):
            store.resolve(bad)


def test_delete_removes_the_file_and_a_missing_one_is_fine(tmp_path):
    store = cv.Store(tmp_path)
    c = store.save(convo(tmp_path))
    store.delete(c.id)
    store.delete(c.id)
    assert store.list() == []
    with pytest.raises(cv.UnknownConversation):
        store.load(c.id)


def test_an_empty_store_lists_nothing(tmp_path):
    assert cv.Store(tmp_path).list() == []


def test_titles_are_cut_with_an_ellipsis():
    assert cv.title_for("a  b\n c") == "a b c"
    long = cv.title_for("word " * 30)
    assert len(long) <= cv.TITLE_CHARS and long.endswith("…")


def _edit(file: Path, change) -> None:
    data = json.loads(file.read_text())
    change(data)
    file.write_text(json.dumps(data))


def _source(d: dict) -> dict:
    return d["exchanges"][0]["response"]["sources"][0]


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(format=2),
        lambda d: d.pop("title"),
        lambda d: d["exchanges"][0].update(spoken="yes"),
        lambda d: d["exchanges"][0]["response"].update(prompt_tokens=True),
        lambda d: d["exchanges"][0]["response"]["cited"].append("S7"),
        lambda d: d["exchanges"][0]["response"]["checks"][0].update(action="explode"),
        lambda d: d.update(exchanges=[]),
        lambda d: d.update(id="a" * 12),  # not its file name
        lambda d: _source(d).update(image_path="../../secret.png"),
        lambda d: _source(d).update(image_path="/etc/hosts"),
        lambda d: d["exchanges"][0]["response"]["hits"][0].update(image_path="../x.webp"),
    ],
    ids=[
        "format",
        "missing-key",
        "wrong-type",
        "bool-int",
        "cited",
        "check",
        "empty",
        "id",
        "dotdot",
        "absolute",
        "hit-dotdot",
    ],
)
def test_loading_is_strict(tmp_path, change, caplog):
    store = cv.Store(tmp_path)
    c = store.save(convo(tmp_path))
    file = tmp_path / "conversations" / f"{c.id}.json"
    _edit(file, change)
    with pytest.raises(cv.UnknownConversation):
        store.load(c.id)
    with caplog.at_level(logging.INFO, logger="qwn"):
        assert store.list() == []
    assert file.name in caplog.text and SENTINEL not in caplog.text


def test_invalid_json_is_left_out_and_logged_once_by_name(tmp_path, caplog):
    store = cv.Store(tmp_path)
    store.save(convo(tmp_path))
    bad = tmp_path / "conversations" / ("c" * 12 + ".json")
    bad.write_text(f'{{"title": "{SENTINEL}"')
    with caplog.at_level(logging.WARNING, logger="qwn"):
        assert len(store.list()) == 1
        assert len(cv.Store(tmp_path).list()) == 1
    assert caplog.text.count(bad.name) == 1 and SENTINEL not in caplog.text


def test_unknown_extra_keys_are_ignored(tmp_path):
    store = cv.Store(tmp_path)
    c = store.save(convo(tmp_path))
    _edit(tmp_path / "conversations" / f"{c.id}.json", lambda d: d.update(colour="blue"))
    assert store.load(c.id) == untimed(c)


def test_a_moved_index_loads_with_image_paths_under_the_new_root(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    c = cv.Store(old).save(convo(old))
    shutil.copytree(old, new)
    loaded = cv.Store(new).load(c.id)
    r = loaded.exchanges[0].response
    assert r.sources[0].image_path == new / "pages/d1/p1.webp"
    assert r.hits[0].image_path == new / "pages/d1/p1.webp"
    assert r.hits[1].image_path is None


def test_source_state(tmp_path):
    h = hit(tmp_path, 1)
    assert h.image_path is not None
    h.image_path.parent.mkdir(parents=True)
    h.image_path.write_bytes(b"webp")
    assert cv.source_state(h, {h.path: "sha1"}) == "ok"
    assert cv.source_state(h, {h.path: "edited"}) == "changed"
    assert cv.source_state(h, {}) == "missing"  # no longer indexed
    h.image_path.unlink()
    assert cv.source_state(h, {h.path: "sha1"}) == "missing"  # render gone
    text = hit(tmp_path, 2, page=None, image=False)
    assert cv.source_state(text, {text.path: "sha2"}) == "ok"


def test_a_loaded_conversation_gives_the_same_turns(tmp_path):
    store = cv.Store(tmp_path)
    c = convo(tmp_path, 3)
    loaded = store.load(store.save(c).id)
    assert conversation(loaded.pairs(), 3) == conversation(c.pairs(), 3)
    assert len(conversation(loaded.pairs(), 3)) == 2  # the block is left out


def test_nothing_but_ids_and_counts_is_logged(tmp_path, caplog):
    store = cv.Store(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="qwn"):
        c = store.save(convo(tmp_path))
        store.list()
        store.load(c.id)
        store.delete(c.id)
    assert c.id in caplog.text
    assert SENTINEL not in caplog.text and "build it" not in caplog.text
