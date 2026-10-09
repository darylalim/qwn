"""Ingest against a real SQLite + sqlite-vec index, with fake models."""

import logging
import shutil
import struct
import threading
import zlib
from pathlib import Path

import pytest
from fakes import FakeEmbedder
from pdf_fixture import write_pdf
from PIL import Image
from sample_corpus import MD, CountingEmbedder

from qwn.config import Settings
from qwn.index import IndexMismatch, page_dir
from qwn.ingest import HEADING_SEP, Ingester, ModelUnavailable, open_index
from qwn.models import ModelsInUse, Registry
from qwn.models_lock import PinnedModel


def test_ingest_lifecycle(ingester, index, corpus, embedder):
    report = ingester.run([corpus])
    assert sorted(Path(p).name for p in report.indexed) == ["chart.png", "pricing.md", "report.pdf"]
    assert not report.failed
    counts = index.counts()
    assert counts["documents"] == 3
    assert counts["chunks"] == counts["vectors"] == 2 + 1 + 2  # 2 pages, 1 image, 2 md sections

    pdf = index.document(str(corpus / "report.pdf"))
    assert pdf is not None and pdf.kind == "pdf" and pdf.n_chunks == 2
    pages = index.doc_chunks(pdf.doc_id)
    assert [c.page for c in pages] == [1, 2]
    assert "Revenue grew 12 percent." in pages[0].text
    assert pages[0].image_path == f"{page_dir(pdf.doc_id)}/1.webp"
    assert (index.root / pages[0].image_path).is_file()

    # re-ingest: nothing changed, nothing embedded
    calls = embedder.calls
    again = ingester.run([corpus])
    assert again.unchanged == 3 and not again.indexed
    assert embedder.calls == calls

    # modify one file: only it is re-indexed
    (corpus / "pricing.md").write_text(MD + "\n## Team\n\nTen users.\n")
    third = ingester.run([corpus])
    assert [Path(p).name for p in third.indexed] == ["pricing.md"] and third.unchanged == 2
    assert index.counts()["chunks"] == 6

    # delete a file and prune
    (corpus / "report.pdf").unlink()
    plan = ingester.prune_plan([corpus])
    assert [d.path for d in plan.remove] == [str(corpus / "report.pdf")]
    ingester.prune(plan.remove)
    assert index.counts()["documents"] == 2
    assert not (index.root / page_dir(pdf.doc_id)).exists()


def test_markdown_chunks_store_text_and_heading_path_without_header(ingester, index, corpus):
    ingester.run([corpus / "pricing.md"])
    doc = index.document(str(corpus / "pricing.md"))
    assert doc is not None
    chunks = index.doc_chunks(doc.doc_id)
    assert [c.heading_path for c in chunks] == [
        f"Pricing{HEADING_SEP}Starter",
        f"Pricing{HEADING_SEP}Enterprise",
    ]
    assert chunks[1].text.startswith("## Enterprise")
    assert "pricing.md" not in chunks[1].text  # the context header is for the models only


def test_crash_mid_commit_keeps_old_version(settings, index, corpus, registry):
    Ingester(settings, index, registry.embedder).run([corpus])
    doc = index.document(str(corpus / "report.pdf"))
    assert doc is not None
    before = index.doc_chunks(doc.doc_id)

    write_pdf(corpus / "report.pdf", [["Totally new text"], ["More"], ["Even more"]])
    broken = Ingester(settings, index, lambda: FakeEmbedder(dim=8))  # vec0 rejects 8-d rows
    report = broken.run([corpus / "report.pdf"])
    assert len(report.failed) == 1

    assert index.doc_chunks(doc.doc_id) == before
    assert index.counts()["vectors"] == index.counts()["chunks"]
    assert (index.root / before[0].image_path).is_file()  # old renders untouched
    assert not list((index.root / "pages").glob(".tmp-*"))


def test_moved_file_is_rekeyed_not_reembedded(ingester, index, corpus, embedder):
    ingester.run([corpus])
    old = index.document(str(corpus / "report.pdf"))
    assert old is not None
    calls = embedder.calls

    new_path = corpus / "archive" / "report-2025.pdf"
    new_path.parent.mkdir()
    (corpus / "report.pdf").rename(new_path)
    report = ingester.run([corpus])
    assert report.moved == [(str(corpus / "report.pdf"), str(new_path))]
    assert embedder.calls == calls

    new = index.document(str(new_path))
    assert new is not None and new.doc_id != old.doc_id and index.document(old.path) is None
    chunks = index.doc_chunks(new.doc_id)
    assert len(chunks) == 2
    assert all(c.image_path and c.image_path.startswith(page_dir(new.doc_id)) for c in chunks)
    assert all((index.root / c.image_path).is_file() for c in chunks if c.image_path)
    assert not (index.root / page_dir(old.doc_id)).exists()


def test_duplicate_copies_collapse_in_search(settings, ingester, index, corpus, registry):
    from qwn.retrieve import Retriever

    shutil.copy(corpus / "pricing.md", corpus / "pricing-copy.md")
    ingester.run([corpus])
    assert index.counts()["documents"] == 4
    hits = Retriever(settings, index, registry).search("audit log PO-48213", k=50)
    keys = [(h.chunk.sha256, h.chunk.page, h.chunk.chunk_idx) for h in hits]
    assert len(keys) == len(set(keys))
    assert sum(h.chunk.heading_path.endswith("Enterprise") for h in hits) == 1


def test_changed_settings_trigger_rebuild_message(settings, corpus, ingester):
    ingester.run([corpus])
    with pytest.raises(IndexMismatch, match="ingest_hash") as e:
        open_index(settings.model_copy(update={"pdf_dpi": 200}))
    assert "--reindex" in str(e.value)
    with pytest.raises(IndexMismatch, match="embed_dim"):
        open_index(settings.model_copy(update={"embed_dim": 512}))
    rebuilt = open_index(settings.model_copy(update={"pdf_dpi": 200}), reset=True)
    assert rebuilt.counts()["documents"] == 0
    assert not (rebuilt.root / "pages").exists()


def test_new_embed_revision_triggers_rebuild(settings, corpus, ingester, monkeypatch):
    ingester.run([corpus])
    monkeypatch.setattr("qwn.ingest.pinned", lambda repo: PinnedModel(repo, "f" * 40, "core", 2.7))
    with pytest.raises(IndexMismatch, match="embed_revision"):
        open_index(settings)


def test_exact_code_missed_by_vectors_is_found_by_fts(home, corpus):
    from qwn.retrieve import Retriever

    notes = corpus / "notes"
    for i in range(30):
        (notes / f"note-{i:02}.md").parent.mkdir(exist_ok=True)
        (notes / f"note-{i:02}.md").write_text(f"# Note {i}\n\nOrder PO-{50000 + i} shipped.\n")
    settings = Settings(top_k=3)
    registry = Registry(settings, overrides={"embedder": FakeEmbedder()})
    index = open_index(settings)
    Ingester(settings, index, registry.embedder).run([corpus])
    retriever = Retriever(settings, index, registry)
    target = "PO-48213"
    vector_only = retriever.search(target, hybrid=False, rerank=False, k=3)
    assert all(target not in h.chunk.text for h in vector_only)
    hybrid = retriever.search(target, hybrid=True, rerank=False, k=3)
    assert any(target in h.chunk.text for h in hybrid)


def test_cascaded_delete_empties_fts(ingester, index, corpus):
    ingester.run([corpus])
    conn = index.conn
    match = "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?"
    assert conn.execute(match, ('"PO-48213"',)).fetchone()[0] == 1
    doc = index.document(str(corpus / "pricing.md"))
    assert doc is not None
    index.delete([doc.doc_id])
    assert conn.execute(match, ('"PO-48213"',)).fetchone()[0] == 0
    conn.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES ('integrity-check', 1)")
    assert index.counts()["vectors"] == index.counts()["chunks"]


def _bomb_png(path: Path, w: int, h: int) -> Path:
    """A tiny PNG whose header declares w x h pixels."""
    Image.new("L", (1, 1)).save(path)
    data = bytearray(path.read_bytes())
    ihdr = 8 + 8  # signature + chunk length/type
    data[ihdr : ihdr + 8] = struct.pack(">II", w, h)
    crc = zlib.crc32(bytes(data[12 : ihdr + 13]))
    data[ihdr + 13 : ihdr + 17] = struct.pack(">I", crc)
    path.write_bytes(bytes(data))
    return path


def test_bad_inputs_are_recorded_and_the_run_continues(home, corpus, registry):
    bad = home / "bad"
    bad.mkdir()
    write_pdf(bad / "locked.pdf", [["secret"]], encrypted=True)
    good = write_pdf(bad / "fine.pdf", [["fine"]]).read_bytes()
    (bad / "truncated.pdf").write_bytes(good[:120])
    (bad / "empty.png").write_bytes(b"")
    Image.new("RGB", (40, 40)).save(bad / "warn-bomb.png")  # 1x-2x the limit: warning → error
    _bomb_png(bad / "huge.png", 100, 100)  # > 2x the limit
    (bad / "latin1.txt").write_bytes("caf\xe9 au lait".encode("latin-1"))
    (bad / "big.txt").write_text("x" * 200)
    (bad / "deck.pptx").write_bytes(b"PK")
    (bad / "notes.docx").write_bytes(b"PK")

    settings = Settings(max_image_pixels=1000, max_text_bytes=100)
    ingester = Ingester(settings, open_index(settings), registry.embedder)
    report = ingester.run([bad, bad / "deck.pptx", bad / "notes.docx", bad / "missing.pdf"])
    reasons = {Path(f.path).name: f.reason for f in report.failed}
    assert reasons["locked.pdf"] == "password-protected (not supported)"
    assert reasons["truncated.pdf"] == "unreadable PDF"
    assert reasons["empty.png"] == "empty file"
    assert reasons["warn-bomb.png"].startswith("image too large")
    assert reasons["huge.png"].startswith("image too large")
    assert reasons["big.txt"].startswith("text file too large")
    assert reasons["deck.pptx"] == "slides: export to PDF first"
    assert reasons["notes.docx"] == "unsupported file type (.docx)"
    assert reasons["missing.pdf"] == "not found"
    assert sorted(Path(p).name for p in report.indexed) == ["fine.pdf", "latin1.txt"]
    assert report.ignored == 2  # deck.pptx and notes.docx in the folder walk
    assert not list((ingester.pages).glob(".tmp-*"))


def test_pdf_over_page_limit_is_skipped(home, registry):
    path = write_pdf(home / "long.pdf", [["p"]] * 5)
    settings = Settings(max_pdf_pages=3)
    report = Ingester(settings, open_index(settings), registry.embedder).run([path])
    assert report.failed[0].reason == "too many pages (5 > max_pdf_pages 3)"


def test_file_changed_during_ingest_is_skipped(settings, index, corpus):
    target = corpus / "pricing.md"

    class Meddling(FakeEmbedder):
        def embed(self, items, *, is_query):
            target.write_text("changed")
            return super().embed(items, is_query=is_query)

    report = Ingester(settings, index, Meddling).run([target])
    assert report.failed[0].reason == "changed during ingest"
    assert index.counts()["documents"] == 0


def test_cancel_stops_between_files_and_leaves_no_partial_rows(settings, index, corpus, registry):
    cancel = threading.Event()
    cancel.set()
    report = Ingester(settings, index, registry.embedder).run([corpus], cancel=cancel)
    assert report.cancelled and not report.indexed
    assert index.counts()["documents"] == 0 and index.counts()["chunks"] == 0

    cancel.clear()

    def stop_after_first(i: int, n: int, path: Path) -> None:
        if i == 1:
            cancel.set()

    report = Ingester(settings, index, registry.embedder).run(
        [corpus], progress=stop_after_first, cancel=cancel
    )
    assert report.cancelled and len(report.indexed) == 1
    assert index.counts()["documents"] == 1


def test_concurrent_ingests_share_one_writer(settings, index, corpus):
    extra = corpus / "more"
    extra.mkdir()
    for i in range(6):
        (extra / f"n{i}.md").write_text(f"# N{i}\n\nbody {i}\n")
    reports = {}

    def run(name: str, paths: list[Path]) -> None:
        reports[name] = Ingester(settings, index, CountingEmbedder).run(paths)
        index.close()

    threads = [
        threading.Thread(target=run, args=("a", [corpus / "report.pdf", corpus / "chart.png"])),
        threading.Thread(target=run, args=("b", [extra])),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not reports["a"].failed and not reports["b"].failed
    assert index.counts()["documents"] == 8
    assert index.counts()["vectors"] == index.counts()["chunks"]


def test_prune_skips_unmounted_volumes_and_missing_roots(settings, ingester, index, corpus):
    import numpy as np

    from qwn.index import Document, doc_id_for

    ingester.run([corpus])
    volume_path = "/Volumes/qwn-test-not-mounted/scan.pdf"
    index.replace(
        Document(doc_id_for(volume_path), volume_path, "x", 0.0, "pdf", 0, "now"),
        [],
        np.zeros((0, settings.embed_dim), dtype=np.float32),
    )
    shutil.rmtree(corpus)

    everything = ingester.prune_plan([])
    assert everything.unmounted == {"qwn-test-not-mounted": 1}
    assert len(everything.remove) == 3

    named = ingester.prune_plan([corpus])
    assert named.missing_roots == {str(corpus): 3}
    assert named.remove == []


def test_model_load_failure_stops_the_run(settings, index, corpus):
    def in_use():
        raise ModelsInUse("Models are in use by another qwn process (pid 1).")

    with pytest.raises(ModelUnavailable, match="pid 1"):
        Ingester(settings, index, in_use).run([corpus])


def test_logs_never_contain_document_text_or_queries(settings, index, home, registry, caplog):
    from qwn.retrieve import Retriever

    sentinel = "ZEBRAQUARTZSENTINEL"
    doc = home / "secret.md"
    doc.write_text(f"# Heading\n\n{sentinel} body text.\n")
    caplog.set_level(logging.DEBUG)
    Ingester(settings, index, registry.embedder).run([doc])
    Retriever(settings, index, registry).search(f"where is {sentinel}?")
    assert caplog.records
    assert sentinel not in caplog.text
