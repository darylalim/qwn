"""One-time logging setup for the CLI and UI entry points (PLAN.md → Runtime robustness → Logging).

Library modules only call `logging.getLogger(__name__)`. Never log document text, questions or
answers: log ids, paths, counts, timings and error types instead.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_MARK = "_qwn_handler"


class _Stderr(logging.StreamHandler):
    """Writes to whatever sys.stderr is at emit time (test runners swap it)."""

    @property
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        pass


def configure(log_dir: Path) -> None:
    """File log at log_dir/qwn.log (5 MB x 3) for the `qwn` loggers, warnings also to stderr."""
    logger = logging.getLogger("qwn")
    for handler in [h for h in logger.handlers if getattr(h, _MARK, False)]:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(os.environ.get("QWN_LOG_LEVEL", "INFO").upper())

    log_dir.mkdir(parents=True, exist_ok=True)
    file = RotatingFileHandler(log_dir / "qwn.log", maxBytes=5_000_000, backupCount=3)
    file.setFormatter(logging.Formatter(FORMAT))
    err = _Stderr()
    err.setLevel(logging.WARNING)
    err.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    for handler in (file, err):
        setattr(handler, _MARK, True)
        logger.addHandler(handler)
