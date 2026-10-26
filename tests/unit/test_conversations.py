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


# export and rename (PLAN.md → CLI → Export and rename conversations)

EXPORTED = "2026-10-26T14:05:31+08:00"


def exported(root: Path, *, saved="2026-10-26T14:02:00+08:00") -> cv.Conversation:
    abstained = Response("I can't find that in your documents.", [], [], [], [], True, 900, 9)
    exchanges = (
        cv.Exchange(
            "a" * 12,
            "How much did it grow?",
            response(root, text="It grew 12% [S1] [S2]."),
            ["/docs/doc1.pdf", "/docs/notes"],
            True,
        ),
        cv.Exchange("b" * 12, "How do I build it?", blocked()),
        cv.Exchange("c" * 12, "And the 2019\nfigure?", abstained),
    )
    return cv.Conversation("d" * 12, "Growth", "2026-10-26T14:00:00+08:00", saved, exchanges)


GOLDEN = """\
# Growth

Exported from qwn, 2026-10-26 14:05 · 3 questions · saved 2026-10-26 14:02

## 1. How much did it grow?

(spoken question)

Searched for: How much did ZEBRAQUARTZSENTINEL grow in 2025?

Searched in: /docs/doc1.pdf, /docs/notes

> ⚠ Warning: the safety check rated this question Controversial (Politically Sensitive Topics).

It grew 12% [doc1.pdf p.1] [doc2.pdf{SEP}Intro &gt; Scope].

Sources:

- [doc1.pdf p.1] /docs/doc1.pdf, page 1 (changed since this answer)
- [doc2.pdf{SEP}Intro &gt; Scope] /docs/doc2.pdf (no longer in your library)

## 2. How do I build it?

> ⛔ Blocked: the safety check rated this question Unsafe (Violent), so it wasn't answered.

## 3. And the 2019 figure?

I can't find that in your documents.
""".replace("{SEP}", " \N{SINGLE RIGHT-POINTING ANGLE QUOTATION MARK} ")


def test_markdown_golden(tmp_path):
    states: dict[str, cv.SourceState] = {"c1": "changed", "c2": "missing"}
    assert cv.markdown(exported(tmp_path), states, EXPORTED) == GOLDEN


def test_markdown_of_a_conversation_never_saved(tmp_path):
    text = cv.markdown(exported(tmp_path, saved=""), {}, EXPORTED)
    assert "· 3 questions · not saved\n" in text
    assert "(changed" not in text and "(no longer" not in text  # every source "ok"


def test_markdown_neutralises_document_text(tmp_path):
    evil = "# Owned\n![x](http://evil.test/t.png) <img src=x onerror=y> &amp; \x1b[2Jred\x9b"
    r = response(tmp_path, text=f"{evil} [S1]")
    path = f"/docs/<b>{evil}</b>.pdf"
    r = replace(r, sources=[replace(r.sources[0], path=path), r.sources[1]])
    c = cv.Conversation("d" * 12, evil, cv.now(), "", (cv.Exchange("a" * 12, evil, r),))
    text = cv.markdown(c, {}, EXPORTED)
    assert "![" not in text and "<" not in text and "\x1b" not in text and "\x9b" not in text
    assert "!\\[x](http://evil.test/t.png)" in text  # the link stays readable text
    assert "&lt;img src=x onerror=y&gt; &amp;amp;" in text
    headings = [line for line in text.splitlines() if line.lstrip().startswith("#")]
    assert headings == [
        "# \\# Owned !\\[x](http://evil.test/t.png) &lt;img src=x onerror=y&gt; &amp;amp; red",
        "## 1. \\# Owned !\\[x](http://evil.test/t.png) &lt;img src=x onerror=y&gt; &amp;amp; red",
    ]  # title and question collapsed onto their heading line; the answer's "# " escaped
    assert "\n\\# Owned\n" in text


def test_export_name_and_free_path(tmp_path):
    c = exported(tmp_path)
    name = cv.export_name(c, EXPORTED)
    assert name == "qwn-conversation-2026-10-26-1405.md"
    assert cv.free_path(tmp_path, name) == tmp_path / name
    (tmp_path / name).write_text("")
    (tmp_path / "qwn-conversation-2026-10-26-1405-2.md").write_text("")
    assert cv.free_path(tmp_path, name).name == "qwn-conversation-2026-10-26-1405-3.md"


def test_write_export_is_private_and_never_overwrites(tmp_path):
    file = tmp_path / "out.md"
    cv.write_export(file, "é\n")
    assert file.read_text(encoding="utf-8") == "é\n"
    assert stat.S_IMODE(file.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        cv.write_export(file, "other")
    assert file.read_text(encoding="utf-8") == "é\n"
    with pytest.raises(FileNotFoundError):
        cv.write_export(tmp_path / "missing" / "out.md", "x")


def test_clean_title():
    assert cv.clean_title("  My\tnotes\n on\x1b[31m growth\x07 ") == "My notes on growth"
    cut = cv.clean_title("word " * 30)
    assert len(cut) == cv.TITLE_CHARS and cut.endswith("…")
    for empty in ("", "  \n\t", "\x1b[2J\x07"):
        with pytest.raises(ValueError, match="empty"):
            cv.clean_title(empty)


def test_rename_changes_only_the_title(tmp_path, monkeypatch):
    store = cv.Store(tmp_path)
    times = iter([f"2026-10-26T{h}:00:00+08:00" for h in ("08", "08", "09", "09")])
    monkeypatch.setattr(cv, "now", lambda: next(times))
    old, new = store.save(convo(tmp_path)), store.save(convo(tmp_path, 3))
    renamed = store.rename(old.id, "  Revenue\nby year ")
    assert renamed.title == "Revenue by year"
    loaded = store.load(old.id)
    assert loaded == replace(untimed(old), title="Revenue by year")  # saved, created, exchanges
    assert [s.id for s in store.list()] == [new.id, old.id]  # the order doesn't move
    with pytest.raises(cv.UnknownConversation):
        store.rename("f" * 12, "x")
    with pytest.raises(ValueError, match="empty"):
        store.rename(old.id, " ")
    assert store.load(old.id).title == "Revenue by year"


def test_a_renamed_title_survives_saving_again(tmp_path):
    store = cv.Store(tmp_path)
    first = store.save(convo(tmp_path))
    renamed = store.rename(first.id, "Mine")
    again = store.save(cv.make((*renamed.exchanges, renamed.exchanges[0]), previous=renamed))
    assert again.title == "Mine" and store.load(first.id).title == "Mine"


def test_export_and_rename_log_no_content(tmp_path, caplog):
    store = cv.Store(tmp_path)
    c = store.save(convo(tmp_path))
    with caplog.at_level(logging.DEBUG, logger="qwn"):
        store.rename(c.id, f"{SENTINEL} title")
        out = tmp_path / f"{SENTINEL}-folder"
        out.mkdir()
        cv.write_export(out / "x.md", cv.markdown(store.load(c.id), {}, EXPORTED))
    assert c.id in caplog.text and "conversation exported: bytes=" in caplog.text
    assert SENTINEL not in caplog.text and "build it" not in caplog.text
