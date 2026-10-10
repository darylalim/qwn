"""`qwn ingest/search/status/eval` end to end, with fakes injected through make_registry."""

import json

import pytest
from fakes import FakeEmbedder, FakeGenerator, FakeGuard, FakeMemory, FakeReranker
from sample_corpus import make_corpus
from typer.testing import CliRunner

import qwn.cli
from qwn.interfaces import Box
from qwn.models import Registry

runner = CliRunner()


@pytest.fixture
def cli(home, monkeypatch):
    def make_registry(settings):
        return Registry(
            settings,
            overrides={
                "embedder": FakeEmbedder(settings.embed_dim),
                "reranker": FakeReranker(),
                "generator": FakeGenerator(),
                "guard": FakeGuard(),
            },
            memory=FakeMemory(),
        )

    monkeypatch.setattr(qwn.cli, "make_registry", make_registry)

    def invoke(*args: str, input: str | None = None):
        return runner.invoke(qwn.cli.app, list(args), input=input)

    return invoke


def test_ingest_search_status(cli, home):
    docs = make_corpus(home)
    r = cli("ingest", str(docs))
    assert r.exit_code == 0, r.output
    assert "Indexed 3, unchanged 0, moved 0, failed 0." in r.output

    r = cli("ingest", str(docs))
    assert "Indexed 0, unchanged 3" in r.output

    r = cli("search", "audit log PO-48213", "--json")
    assert r.exit_code == 0, r.output
    rows = json.loads(r.output)
    assert len(rows) == 5 and rows[0]["rank"] == 1
    assert rows[0]["path"].endswith("pricing.md") and rows[0]["rerank_score"] > 0

    r = cli("search", "revenue", "--no-rerank", "--rerank-k", "2")
    assert r.exit_code == 0 and "1. " in r.output and "3. " not in r.output

    r = cli("status")
    assert r.exit_code == 0, r.output
    assert "3 documents (1 pdf, 1 image, 1 text), 5 chunks" in r.output
    assert "26.8 GB recommended" in r.output


def test_ingest_dry_run_writes_nothing(cli, home):
    docs = make_corpus(home)
    r = cli("ingest", str(docs), "--dry-run")
    assert r.exit_code == 0, r.output
    assert "Would index 3, move 0, skip 0 unchanged." in r.output
    r = cli("status")
    assert "0 documents" in r.output


def test_prune_asks_then_removes(cli, home):
    docs = make_corpus(home)
    cli("ingest", str(docs))
    (docs / "chart.png").unlink()
    r = cli("ingest", str(docs), "--prune", input="n\n")
    assert "Will remove 1 documents" in r.output and "Removed" not in r.output
    r = cli("ingest", "--prune", "--yes")
    assert "Removed 1 documents." in r.output
    assert "2 documents" in cli("status").output


def test_ingest_needs_a_path_and_reports_bad_files(cli, home):
    r = cli("ingest")
    assert r.exit_code == 1 and "at least one PATH" in r.output
    (home / "deck.key").write_bytes(b"x")
    r = cli("ingest", str(home / "deck.key"))
    assert "deck.key: slides: export to PDF first" in r.output


def test_settings_change_needs_reindex(cli, home, monkeypatch):
    docs = make_corpus(home)
    cli("ingest", str(docs))
    monkeypatch.setenv("QWN_PDF_DPI", "100")
    r = cli("search", "anything")
    assert r.exit_code == 1 and "--reindex" in r.output
    assert "ingest_hash differs" in cli("status").output
    r = cli("ingest", "--reindex", str(docs))
    assert r.exit_code == 0 and "Indexed 3" in r.output


def test_search_on_empty_index(cli, home):
    r = cli("search", "anything")
    assert r.exit_code == 1 and "index is empty" in r.output


def _private_set(home):
    docs = make_corpus(home / "data")
    queries = [
        {"id": "p1", "query": "Revenue grew", "expected": [{"path": "docs/report.pdf", "page": 1}],
         "tags": ["text"]},
        {"id": "p2", "query": "Single sign-on", "expected": [
            {"path": "docs/pricing.md", "heading": "Enterprise"}], "tags": ["markdown", "exact"]},
        {"id": "p3", "query": "Bar chart", "expected": [{"path": "docs/chart.png"}],
         "tags": ["chart"]},
        {"id": "p4", "query": "What is the refund policy?", "expected": [],
         "tags": ["unanswerable"]},
    ]  # fmt: skip
    private = home / "eval" / "private"
    private.mkdir(parents=True)
    (private / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in queries))
    return docs


def test_eval_private_set_and_baseline(cli, home):
    docs = _private_set(home)
    cli("ingest", str(docs))
    r = cli("eval", "--set", "private", "--update-baseline")
    assert r.exit_code == 0, r.output
    assert "Exit criteria (phase 1: retrieval, phase 2: answers)" in r.output
    assert "recall@5 overall" in r.output and "citation_hit" in r.output
    baseline = json.loads((home / "eval" / "private" / "baseline.json").read_text())
    assert baseline["meta"]["set"] == "private"
    assert baseline["meta"]["options"] == {"rerank": True, "hybrid": True, "generate": True}
    rows = {r["id"]: r for r in baseline["per_query"]}
    assert rows["p1"]["citation_hit"] and not rows["p1"]["abstained"]
    assert rows["p1"]["cited"] == [{"path": "docs/report.pdf", "page": 1}]
    assert rows["p4"]["abstained"] and "citation_hit" not in rows["p4"]
    assert baseline["answer"]["summary"]["abstention"]["value"] == 1.0
    assert {r["id"] for r in baseline["per_query"]} == {"p1", "p2", "p3", "p4"}
    assert set(baseline["by_tag"]) == {"text", "markdown", "exact", "chart"}
    assert list((home / "eval" / "results").glob("*-private.json"))
    assert "Revenue grew" not in json.dumps(baseline["per_query"])  # no document text in results

    r = cli("eval", "--set", "private")
    assert r.exit_code == 0 and "No per-query changes vs baseline." in r.output

    r = cli("eval", "--set", "private", "--no-generate")
    assert r.exit_code == 1 and "settings_hash" in r.output  # generate is part of the settings
    r = cli("eval", "--set", "private", "--no-generate", "--force")
    assert "citation_hit" not in r.output and "Exit criteria (phase 1: retrieval)" in r.output

    r = cli("eval", "--set", "private", "--no-rerank")
    assert r.exit_code == 1 and "Not comparable with the baseline: settings_hash" in r.output
    r = cli("eval", "--set", "private", "--no-rerank", "--force")
    assert r.exit_code == 0


def test_eval_fails_up_front_on_unindexed_expected_paths(cli, home):
    docs = _private_set(home)
    cli("ingest", str(docs / "report.pdf"))
    r = cli("eval", "--set", "private")
    assert r.exit_code == 1
    assert "Expected files are not in the index" in r.output and "chart.png" in r.output


def test_eval_locate_scores_highlights_on_cited_pages(cli, home, monkeypatch):
    docs = _private_set(home)
    path = home / "eval" / "private" / "queries.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["region"] = [0.1, 0.05, 0.5, 0.1]  # p1: cited by the fake generator
    rows[2]["region"] = [0.0, 0.0, 0.2, 0.2]  # p3: the image isn't cited
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    generator = FakeGenerator(box=Box(0.0, 0.0, 0.6, 0.2))

    def make_registry(settings):
        return Registry(
            settings,
            overrides={
                "embedder": FakeEmbedder(settings.embed_dim),
                "reranker": FakeReranker(),
                "generator": generator,
                "guard": FakeGuard(),
            },
            memory=FakeMemory(),
        )

    monkeypatch.setattr(qwn.cli, "make_registry", make_registry)
    cli("ingest", str(docs))
    r = cli("eval", "--set", "private", "--locate", "--update-baseline")
    assert r.exit_code == 0, r.output
    assert "phase 6: highlighting" in r.output and "Highlights on cited pages" in r.output
    assert len(generator.located) == 1 and generator.located[0][1] == "Revenue grew 12 percent."
    baseline = json.loads((home / "eval" / "private" / "baseline.json").read_text())
    per = {r["id"]: r for r in baseline["per_query"]}
    assert per["p1"]["locate_hit"] and per["p1"]["locate_box"] == [0.0, 0.0, 0.6, 0.2]
    assert per["p3"]["locate"] == "not_cited" and "locate_hit" not in per["p3"]
    assert baseline["locate"]["summary"]["locate_hit"]["value"] == 1.0
    assert baseline["meta"]["options"] == {"rerank": True, "hybrid": True, "generate": True}

    generator.box = None  # the model can't say: a miss, and a "no usable box"
    r = cli("eval", "--set", "private", "--locate")
    assert "p1: highlight no longer hits the region" in r.output
    assert "✗ FAIL  locate_hit" in r.output and "✗ FAIL  no usable box" in r.output


def test_eval_locate_needs_answers(cli, home):
    r = cli("eval", "--locate", "--no-generate")
    assert r.exit_code == 1 and "drop --no-generate" in r.output
