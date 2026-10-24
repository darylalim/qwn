import io
import logging
import sys

import pytest

from qwn import logging_setup


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def stderr(monkeypatch):
    def use(stream: io.StringIO) -> io.StringIO:
        monkeypatch.setattr(sys, "stderr", stream)
        return stream

    return use


def test_known_noise_is_dropped_other_transformers_warnings_kept(tmp_path, caplog):
    logging_setup.configure(tmp_path)
    logging_setup.configure(tmp_path)  # idempotent: one filter, not two
    noisy = logging.getLogger("transformers.processing_utils")
    assert len(noisy.filters) == 1
    with caplog.at_level(logging.WARNING, logger="transformers"):
        noisy.warning(
            "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, "
            "not in `**kwargs`"
        )
        noisy.warning("some other processor warning")
    assert [r.getMessage() for r in caplog.records] == ["some other processor warning"]


def test_warning_on_a_terminal_clears_the_status_line(tmp_path, stderr):
    out = stderr(_Tty())
    logging_setup.configure(tmp_path)
    logging.getLogger("qwn.test").warning("rewrite: no usable question in a %d-char reply", 3)
    assert out.getvalue() == (
        logging_setup.CLEAR_LINE + "WARNING: rewrite: no usable question in a 3-char reply\n"
    )


def test_warning_off_a_terminal_is_plain(tmp_path, stderr):
    out = stderr(io.StringIO())
    logging_setup.configure(tmp_path)
    logging.getLogger("qwn.test").warning("dropped invented citations: %s", [9])
    assert out.getvalue() == "WARNING: dropped invented citations: [9]\n"
