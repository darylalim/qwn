"""The Library page's logic: uploads, deletes, re-index and background ingest (real index)."""

import threading

import pytest

from qwn.ingest import delete_documents, save_upload
from qwn.jobs import JobRegistry, JobRunning


def test_upload_name_clash_never_overwrites(settings):
    data = settings.data_dir
    first, outcome = save_upload(data, "report.pdf", b"one")
    assert (first.name, outcome) == ("report.pdf", "saved")
    same, outcome = save_upload(data, "report.pdf", b"one")
    assert (same, outcome) == (first, "duplicate")  # "already in your library"
    second, outcome = save_upload(data, "report.pdf", b"two")
    assert (second.name, outcome) == ("report (2).pdf", "saved")
    assert save_upload(data, "report.pdf", b"two") == (second, "duplicate")
    third, _ = save_upload(data, "report.pdf", b"three")
    assert third.name == "report (3).pdf"
    assert first.read_bytes() == b"one" and second.read_bytes() == b"two"


def test_upload_names_cannot_escape_the_library(settings):
    path, _ = save_upload(settings.data_dir, "../../etc/evil.md", b"x")
    assert path.parent == settings.data_dir and path.name == "evil.md"
    hidden, _ = save_upload(settings.data_dir, ".hidden.md", b"y")
    assert hidden.name == "hidden.md"  # folder walks would skip a dotfile


def test_delete_never_removes_files_outside_the_library(settings, index, ingester, corpus):
    upload, _ = save_upload(settings.data_dir, "notes.md", b"# Notes\n\nUploaded text.\n")
    ingester.run([corpus, upload])
    docs = index.documents()
    assert len(docs) == 4
    removed = delete_documents(index, docs, settings.data_dir)
    assert removed == [upload] and not upload.exists()
    assert all(f.exists() for f in corpus.iterdir())  # indexed in place: untouched
    assert index.documents() == [] and index.counts()["chunks"] == 0
    assert not any((index.root / "pages").glob("*/*"))  # page renders gone too


def test_force_reindexes_unchanged_files(ingester, corpus, embedder):
    ingester.run([corpus])
    calls = embedder.calls
    assert ingester.run([corpus]).unchanged == 3 and embedder.calls == calls
    report = ingester.run([corpus / "pricing.md"], force=True)
    assert len(report.indexed) == 1 and embedder.calls == calls + 1


def test_concurrent_ingest_requests_are_refused(ingester, corpus):
    jobs = JobRegistry()
    release = threading.Event()
    real = ingester.run

    def slow_run(paths, progress, cancel):
        release.wait(5)
        return real(paths, progress=progress, cancel=cancel)

    jobs.start("first", lambda p, c: slow_run([corpus], p, c))
    with pytest.raises(JobRunning):
        jobs.start("second", lambda p, c: real([corpus], progress=p, cancel=c))
    release.set()
    jobs.join(10)
    job = jobs.current()
    assert job is not None and job.state == "done" and job.report is not None
    assert len(job.report.indexed) == 3


def test_cancelled_job_leaves_the_index_unchanged(index, ingester, corpus):
    jobs = JobRegistry()

    def work(progress, cancel):
        cancel.set()  # cancelled before the first file
        return ingester.run([corpus], progress=progress, cancel=cancel)

    jobs.start("cancel", work)
    jobs.join(10)
    job = jobs.current()
    assert job is not None and job.state == "cancelled"
    assert index.documents() == []
