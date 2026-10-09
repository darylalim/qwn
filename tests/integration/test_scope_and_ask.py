"""Scoped search (real sqlite-vec) and `qwn ask` end to end with fakes."""

import json

import numpy as np
import pytest
from fakes import FakeEmbedder, FakeGenerator, FakeMemory, FakeReranker
from sample_corpus import make_corpus
from typer.testing import CliRunner

import qwn.cli
from qwn.config import Settings
from qwn.ingest import HEADING_SEP, Ingester, open_index
from qwn.models import Registry
from qwn.retrieve import EmptyScope, Retriever

runner = CliRunner()


@pytest.fixture
def notes(home):
    """30 notes in a/, 3 in b/, and a sibling folder a2/ that a prefix match would wrongly catch."""
    root = home / "notes"
    for folder, n in (("a", 30), ("b", 3), ("a2", 2)):
        (root / folder).mkdir(parents=True)
        for i in range(n):
            (root / folder / f"{i:02}.md").write_text(
                f"# {folder} {i}\n\nOrder PO-{i:05} of {folder}.\n"
            )
    settings = Settings(top_k=3, fts_k=3)
    registry = Registry(settings, overrides={"embedder": FakeEmbedder()})
    index = open_index(settings)
    Ingester(settings, index, registry.embedder).run([root])
    return root, Retriever(settings, index, registry)


def _paths(retriever: Retriever, doc_ids: list[str]) -> set[str]:
    return {d.path for d in retriever.index.documents() if d.doc_id in doc_ids}


def test_scope_resolves_files_and_folders_but_not_name_prefixes(notes):
    root, r = notes
    folder = r.scope([root / "a"])
    assert len(folder) == 30
    assert all("/a/" in p for p in _paths(r, folder))  # not a2/
    one = r.scope([root / "b" / "01.md"])
    assert _paths(r, one) == {str(root / "b" / "01.md")}
    assert len(r.scope([root / "b", root / "a2", root / "b"])) == 5  # union, no duplicates


def test_empty_scope_is_an_error(notes):
    root, r = notes
    with pytest.raises(EmptyScope, match="Nothing indexed under"):
        r.scope([root / "missing"])


def test_scoped_search_returns_k_in_scope_hits_when_out_of_scope_ones_are_nearer(notes):
    root, r = notes
    a_chunk = next(
        c for d in r.index.documents() if "/a/" in d.path for c in r.index.doc_chunks(d.doc_id)
    )
    blob = r.index.conn.execute(
        "SELECT embedding FROM vec_chunks WHERE rowid = ?", (a_chunk.rowid,)
    ).fetchone()[0]
    qvec = np.frombuffer(blob, dtype=np.float32)
    assert r.vector(qvec)[0].chunk.rowid == a_chunk.rowid  # unscoped: the a/ chunk is nearest

    b = r.scope([root / "b"])
    hits = r.vector(qvec, b)
    assert len(hits) == 3 and all("/b/" in h.path for h in hits)


def test_scoped_keyword_search_filters_by_document(notes):
    root, r = notes
    unscoped = r.keyword("PO-00001")
    assert {h.path.split("/")[-2] for h in unscoped} >= {"a", "b"}
    scoped = r.keyword("PO-00001", r.scope([root / "b"]))
    assert [h.path for h in scoped] == [str(root / "b" / "01.md")]


# qwn ask


@pytest.fixture
def cli(home, monkeypatch):
    def make_registry(settings):
        return Registry(
            settings,
            overrides={
                "embedder": FakeEmbedder(settings.embed_dim),
                "reranker": FakeReranker(),
                "generator": FakeGenerator(),
            },
            memory=FakeMemory(),
        )

    monkeypatch.setattr(qwn.cli, "make_registry", make_registry)
    return lambda *args: runner.invoke(qwn.cli.app, list(args))


def test_ask_answers_with_rendered_citations(cli, home):
    docs = make_corpus(home)
    cli("ingest", str(docs))
    r = cli("ask", "Which plan has single sign-on and an audit log?")
    assert r.exit_code == 0, r.output
    assert f"[pricing.md{HEADING_SEP}Enterprise]" in r.output and "[S1]" not in r.output
    assert "Cited:" in r.output

    r = cli("ask", "Which plan has single sign-on?", "--sources")
    assert "Sources:" in r.output and " *[pricing.md" in r.output


def test_ask_json_and_scope(cli, home):
    docs = make_corpus(home)
    cli("ingest", str(docs))
    r = cli("ask", "How much did revenue grow?", "--in", str(docs / "report.pdf"), "--json")
    assert r.exit_code == 0, r.output
    out = json.loads(r.output)
    assert out["abstained"] is False and out["invented_citations"] == 0
    assert {s["path"] for s in out["sources"]} == {str(docs / "report.pdf")}
    assert out["cited"][0]["cite"] == "report.pdf p.1"
    assert out["answer"].endswith("[report.pdf p.1]")
    assert set(out["timings_s"]) == {"embed", "search", "rerank", "generate"}


def test_ask_abstains_when_nothing_matches(cli, home):
    cli("ingest", str(make_corpus(home)))
    r = cli("ask", "What is the refund policy?", "--json")
    out = json.loads(r.output)
    assert out["abstained"] and out["cited"] == []


def test_ask_and_search_stop_on_an_empty_scope_or_index(cli, home):
    r = cli("ask", "anything?")
    assert r.exit_code == 1 and "index is empty" in r.output
    docs = make_corpus(home)
    cli("ingest", str(docs))
    for command in ("ask", "search"):
        r = cli(command, "anything?", "--in", str(home / "nowhere"))
        assert r.exit_code == 1 and "Nothing indexed under" in r.output


def test_search_in_scope(cli, home):
    docs = make_corpus(home)
    cli("ingest", str(docs))
    r = cli("search", "anything", "--in", str(docs / "chart.png"), "--json")
    assert r.exit_code == 0, r.output
    assert [row["path"] for row in json.loads(r.output)] == [str(docs / "chart.png")]


def test_ask_reports_model_errors_without_a_traceback(cli, home, monkeypatch):
    cli("ingest", str(make_corpus(home)))

    def boom(self, question, sources, *, greedy=False):
        raise RuntimeError("[metal] out of memory")

    monkeypatch.setattr(FakeGenerator, "answer", boom)
    r = cli("ask", "How much did revenue grow?")
    assert r.exit_code == 1
    assert "The model failed: RuntimeError: [metal] out of memory" in r.output


def test_eval_query_scope_resolves_against_the_set_root(notes):
    from qwn import eval as ev

    root, r = notes
    q = ev.Query("q", "?", [{"path": "b/01.md"}], ["markdown"], scope=["b"])
    assert len(ev._scope(r, q, root) or []) == 3
    assert ev._scope(r, ev.Query("q", "?", [], []), root) is None
    with pytest.raises(ev.EvalError, match="q: Nothing indexed under"):
        ev._scope(r, ev.Query("q", "?", [], [], scope=["nowhere"]), root)
