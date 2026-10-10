"""Background jobs for the UI: one ingest at a time, in a worker thread (PLAN.md → Concurrency).

The UI owns one `JobRegistry` (an `st.cache_resource` object) and polls `current()` from a
fragment. Worker threads never touch Streamlit: they only update the `Job` they're given.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from qwn.ingest import Failure, Report

logger = logging.getLogger(__name__)

JobState = Literal["running", "done", "cancelled", "failed"]
Progress = Callable[[int, int, Path], None]
Work = Callable[[Progress, threading.Event], Report]


class JobRunning(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Ingest already running")


@dataclass(frozen=True)
class Job:
    """A snapshot: what the UI reads. The registry replaces it as the worker progresses."""

    id: int
    label: str
    state: JobState = "running"
    done: int = 0
    total: int = 0
    current: str = ""  # file name being processed
    started: float = field(default_factory=time.time)
    report: Report | None = None
    error: str | None = None  # the run itself failed (e.g. the embedder couldn't load)

    @property
    def finished(self) -> bool:
        return self.state != "running"

    @property
    def failures(self) -> list[Failure]:
        return self.report.failed if self.report is not None else []


class JobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Job | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_id = 1

    def current(self) -> Job | None:
        """The running job, or the last one to finish (None before the first)."""
        with self._lock:
            return self._job

    def start(self, label: str, work: Work) -> Job:
        """Run `work(progress, cancel)` in a worker thread; JobRunning if one is running."""
        with self._lock:
            if self._job is not None and not self._job.finished:
                raise JobRunning
            self._cancel = threading.Event()
            job = Job(self._next_id, label)
            self._next_id += 1
            self._job = job
            cancel = self._cancel
            self._thread = threading.Thread(
                target=self._run, args=(work, cancel), name=f"qwn-job-{job.id}", daemon=True
            )
            self._thread.start()
        logger.info("job %d started: %s", job.id, label)
        return job

    def cancel(self) -> None:
        """Ask the running job to stop before its next file."""
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _update(self, **changes: object) -> None:
        with self._lock:
            assert self._job is not None
            self._job = replace(self._job, **changes)

    def _run(self, work: Work, cancel: threading.Event) -> None:
        def progress(i: int, n: int, path: Path) -> None:
            self._update(done=i, total=n, current=path.name)

        try:
            report = work(progress, cancel)
        except Exception as e:
            logger.warning("job failed (%s)", type(e).__name__)
            self._update(state="failed", error=str(e) or type(e).__name__)
            return
        state: JobState = "cancelled" if report.cancelled else "done"
        self._update(state=state, report=report, current="")
        logger.info("job %s: %s", state, self._job.id if self._job else "?")
