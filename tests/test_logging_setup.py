"""Tests for making the library's own logging reach a job's output.

Nothing configured the ``logging`` module before 2026-10-23, so the root logger
had no handler and Python's last-resort fallback applied: WARNING and above, to
stderr. Every ``logger.info`` in the library was discarded, which is why job
24882282 docked 2,000 molecules without printing one progress line.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from activelearning.main import _HANDLER_NAME, configure_logging

PROGRESS_LOGGER = "activelearning.applications.molecules.dock3_oracle"


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Put the logging module back as it was, whatever the test did to it."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    root_level = root.level
    library_level = logging.getLogger("activelearning").level
    yield
    root.handlers = handlers
    root.setLevel(root_level)
    logging.getLogger("activelearning").setLevel(library_level)


def test_the_library_logs_at_info(capsys: pytest.CaptureFixture[str]) -> None:
    """The docking progress line is the reason this exists."""
    configure_logging()

    logging.getLogger(PROGRESS_LOGGER).info("Docking progress: %d/%d", 500, 1000)

    captured = capsys.readouterr()
    assert "Docking progress: 500/1000" in captured.out


def test_records_go_to_stdout_not_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """stderr is RDKit's, which writes from C++ and ignores logging entirely.

    Keeping the library's records on stdout puts a job's log in the one file
    that is not already full of SMILES parse complaints.
    """
    configure_logging()

    logging.getLogger(PROGRESS_LOGGER).info("to stdout")

    captured = capsys.readouterr()
    assert "to stdout" in captured.out
    assert "to stdout" not in captured.err


def test_a_dependency_is_not_turned_verbose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """INFO at the root would bury the run's output in matplotlib and wandb."""
    configure_logging()
    dependency = logging.getLogger("matplotlib.font_manager")

    dependency.info("chatter")
    dependency.warning("still shown")

    captured = capsys.readouterr()
    assert "chatter" not in captured.out
    assert "still shown" in captured.out


def test_configuring_twice_does_not_duplicate_every_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A second call must not make each record print twice."""
    configure_logging()
    configure_logging()

    logging.getLogger(PROGRESS_LOGGER).info("once")

    captured = capsys.readouterr()
    assert captured.out.count("once") == 1
    root = logging.getLogger()
    installed = [h for h in root.handlers if h.name == _HANDLER_NAME]
    assert len(installed) == 1


def test_the_level_can_be_raised_for_a_quiet_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicit level wins, so a job can be made quiet without a code change."""
    configure_logging(logging.WARNING)

    logging.getLogger(PROGRESS_LOGGER).info("suppressed")
    logging.getLogger(PROGRESS_LOGGER).warning("kept")

    captured = capsys.readouterr()
    assert "suppressed" not in captured.out
    assert "kept" in captured.out


def test_the_level_comes_from_the_environment(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Debugging a running job should not need an edit and a resubmit."""
    monkeypatch.setenv("ACTIVELEARNING_LOG_LEVEL", "WARNING")
    configure_logging()

    logging.getLogger(PROGRESS_LOGGER).info("suppressed")

    assert "suppressed" not in capsys.readouterr().out
