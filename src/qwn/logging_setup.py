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
CLEAR_LINE = "\r\x1b[2K"

# Third-party warnings that fire on every model call and say nothing a qwn user can act on:
# (logger, message prefix). mlx-vlm passes its template kwargs loosely to transformers'
# `apply_chat_template`; the output is the same either way.
KNOWN_NOISE = (("transformers.processing_utils", "Kwargs passed to `processor.__call__`"),)


class _DropPrefix(logging.Filter):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self.prefix)


class _Stderr(logging.StreamHandler):
    """Writes to whatever sys.stderr is at emit time (test runners swap it)."""

    @property
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, value: Any) -> None:
        pass

    def emit(self, record: logging.LogRecord) -> None:
        # On a terminal, replace a status line (`qwn chat`'s stage) instead of running on after it.
        if self.stream.isatty():
            self.stream.write(CLEAR_LINE)
        super().emit(record)


def configure(log_dir: Path) -> None:
    """File log at log_dir/qwn.log (5 MB x 3) for the `qwn` loggers, warnings also to stderr.

    Also drops the KNOWN_NOISE warnings, wherever they would have gone.
    """
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
    for name, prefix in KNOWN_NOISE:
        noisy = logging.getLogger(name)
        if not any(isinstance(f, _DropPrefix) and f.prefix == prefix for f in noisy.filters):
            noisy.addFilter(_DropPrefix(prefix))
