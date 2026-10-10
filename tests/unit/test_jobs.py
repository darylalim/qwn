import threading
from pathlib import Path

import pytest

from qwn.ingest import Failure, Report
from qwn.jobs import JobRegistry, JobRunning


def test_job_reports_progress_then_its_report():
    jobs = JobRegistry()
    seen = []

    def work(progress, cancel):
        for i in range(3):
            progress(i, 3, Path(f"/x/{i}.md"))
            seen.append(jobs.current())
        return Report(indexed=["a"], failed=[Failure("/x/b.md", "unreadable")])

    jobs.start("test", work)
    jobs.join(5)
    job = jobs.current()
    assert job is not None and job.state == "done" and job.finished
    assert [(s.done, s.current) for s in seen] == [(0, "0.md"), (1, "1.md"), (2, "2.md")]
    assert job.failures == [Failure("/x/b.md", "unreadable")]


def test_one_job_at_a_time():
    jobs = JobRegistry()
    release = threading.Event()
    jobs.start("first", lambda progress, cancel: (release.wait(5), Report())[1])
    with pytest.raises(JobRunning, match="Ingest already running"):
        jobs.start("second", lambda progress, cancel: Report())
    release.set()
    jobs.join(5)
    jobs.start("third", lambda progress, cancel: Report())  # finished jobs don't block
    jobs.join(5)
    job = jobs.current()
    assert job is not None and job.label == "third"


def test_cancel_sets_the_flag_the_work_checks():
    jobs = JobRegistry()
    started = threading.Event()

    def work(progress, cancel):
        started.set()
        cancel.wait(5)
        return Report(cancelled=cancel.is_set())

    jobs.start("cancel me", work)
    started.wait(5)
    jobs.cancel()
    jobs.join(5)
    job = jobs.current()
    assert job is not None and job.state == "cancelled"


def test_a_failing_run_is_recorded_not_raised():
    jobs = JobRegistry()

    def work(progress, cancel):
        raise RuntimeError("embedder unavailable")

    jobs.start("boom", work)
    jobs.join(5)
    job = jobs.current()
    assert job is not None and job.state == "failed"
    assert job.error == "embedder unavailable"
