"""Growing the private set: Chat ratings → drafts → `qwn eval review` → queries."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import qwn.cli
from qwn import eval as ev
from qwn.answer import Response
from qwn.index import Chunk
from qwn.ingest import HEADING_SEP as SEP
from qwn.interfaces import Source
from qwn.retrieve import Hit

runner = CliRunner()


def response(home: Path, *, cited: bool = True) -> Response:
    pdf = Chunk(1, "c1", "d1", str(home / "data/q3.pdf"), "s", "pdf_page", 4, 0, "", "t", None)
    md = Chunk(2, "c2", "d2", "/elsewhere/n.md", "s", "text", None, 0, f"Plans{SEP}SSO", "t", None)
    hits = [Hit(pdf, None, 1.0), Hit(md, None, 0.5)]
    sources = [
        Source(f"S{i}", h.chunk.chunk_id, h.path, h.page, "t", None, h.score)
        for i, h in enumerate(hits, 1)
    ]
    labels = ["S1", "S2"] if cited else []
    return Response("x", sources, hits, labels, [], not cited, 1, 1)


def draft(draft_id: str, rating: str = "up", **extra) -> dict:
    row = {
        "draft_id": draft_id,
        "created": "2026-10-09T12:00:00+00:00",
        "query": f"question {draft_id}?",
        "rating": rating,
        "scope": [],
        "cited": [{"path": "q3.pdf", "page": 4}],
        "abstained": False,
    }
    return row | extra


@pytest.fixture
def paths(home):
    return ev.candidates_path(home), home / "eval" / "private" / "queries.jsonl"


def review(stdin: str):
    return runner.invoke(qwn.cli.app, ["eval", "review"], input=stdin)


def lines(path: Path) -> list[dict]:
    return ev.read_jsonl(path)


# drafts


def test_make_draft_keeps_cited_pages_not_answer_text(home):
    d = ev.make_draft("m1", "Q?", "up", [str(home / "data/notes")], response(home), home / "data")
    assert d["cited"] == [
        {"path": "q3.pdf", "page": 4},
        {"path": "/elsewhere/n.md", "heading": "SSO"},
    ]
    assert d["scope"] == ["notes"] and d["abstained"] is False
    assert set(d) == {"draft_id", "created", "query", "rating", "scope", "cited", "abstained"}


def test_save_draft_rewrites_the_same_id_and_remove_drops_it(paths):
    drafts, _ = paths
    ev.save_draft(drafts, draft("a"))
    ev.save_draft(drafts, draft("b"))
    ev.save_draft(drafts, draft("a", rating="down"))  # changed their mind
    assert [(d["draft_id"], d["rating"]) for d in lines(drafts)] == [("a", "down"), ("b", "up")]
    ev.remove_draft(drafts, "a")
    assert [d["draft_id"] for d in lines(drafts)] == ["b"]


def test_parse_and_format_expected_round_trip():
    text = "docs/q3.pdf:4, notes/pricing.md#Enterprise, charts/a.png"
    items = ev.parse_expected(text)
    assert items == [
        {"path": "docs/q3.pdf", "page": 4},
        {"path": "notes/pricing.md", "heading": "Enterprise"},
        {"path": "charts/a.png"},
    ]
    assert ev.format_expected(items) == text
    assert ev.parse_expected(" , ") == []


# qwn eval review


def test_review_with_no_drafts(paths):
    result = review("")
    assert result.exit_code == 0 and "No drafts" in result.output


def test_accept_turns_a_thumbs_up_into_a_query(paths):
    drafts, queries = paths
    queries.parent.mkdir(parents=True)
    queries.write_text(json.dumps({"id": "priv-007", "query": "q", "expected": []}) + "\n")
    ev.save_draft(drafts, draft("a", scope=["notes"]))
    result = review("a\n")
    assert result.exit_code == 0, result.output
    assert lines(queries)[-1] == {
        "id": "priv-008",
        "query": "question a?",
        "expected": [{"path": "q3.pdf", "page": 4}],
        "tags": [],
        "in": ["notes"],
    }
    assert lines(drafts) == []
    assert "--update-baseline" in result.output


def test_accepting_an_abstained_thumbs_up_makes_it_unanswerable(paths):
    drafts, queries = paths
    ev.save_draft(drafts, draft("a", cited=[], abstained=True))
    review("a\n")
    [q] = lines(queries)
    assert q["id"] == "priv-001" and q["expected"] == [] and q["tags"] == ["unanswerable"]


def test_a_thumbs_down_cannot_be_accepted_and_is_edited_instead(paths):
    drafts, queries = paths
    ev.save_draft(drafts, draft("a", rating="down"))
    result = review("a\ne\nreport.pdf:2, notes.md#SSO\n18.4, APAC\nchart, exact\n")
    assert "use [e]dit" in result.output
    [q] = lines(queries)
    assert q["expected"] == [
        {"path": "report.pdf", "page": 2},
        {"path": "notes.md", "heading": "SSO"},
    ]
    assert q["answer_contains"] == ["18.4", "APAC"] and q["tags"] == ["chart", "exact"]


def test_unanswerable_skip_discard_and_quit(paths):
    drafts, queries = paths
    for name in "abcd":
        ev.save_draft(drafts, draft(name))
    review("u\ns\nd\nq\n")  # a: unanswerable, b: skip, c: discard, d: quit before it
    [q] = lines(queries)
    assert q["query"] == "question a?" and q["expected"] == [] and q["tags"] == ["unanswerable"]
    assert [d["draft_id"] for d in lines(drafts)] == ["b", "d"]  # skipped and unreviewed stay
