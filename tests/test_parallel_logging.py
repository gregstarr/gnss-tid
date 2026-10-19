from __future__ import annotations

import io
import logging
import sys
from logging import StreamHandler

import pytest

from gnss_tid.parallel_logging import (
    WorkerChannel,
    _console_level,
    log_queue_listener,
    progress_queue_listener,
    worker_channels,
)


class ListHandler(StreamHandler):
    """StreamHandler that collects emitted records instead of writing."""

    def __init__(self, stream=None):
        super().__init__(stream=stream)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _record(level: int, msg: str) -> logging.LogRecord:
    return logging.LogRecord("test", level, __file__, 1, msg, None, None)


@pytest.fixture
def logging_env():
    """Save/restore root handlers and level; force root level to INFO."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    root.setLevel(logging.INFO)
    yield root
    root.handlers = original_handlers
    root.setLevel(original_level)


@pytest.fixture
def console(logging_env):
    """Attach a WARNING-level console handler (stderr) to the root logger."""
    handler = ListHandler(stream=sys.stderr)
    handler.setLevel(logging.WARNING)
    logging_env.addHandler(handler)
    yield handler
    logging_env.removeHandler(handler)


@pytest.fixture
def string_console(logging_env):
    """Attach a WARNING-level console handler backed by an in-memory buffer.

    The buffer keeps the handler from counting as a real console handler
    (stream must not be stdout/stderr), so ``logging_redirect_tqdm``
    leaves it attached to the root logger and the log queue listener
    forwards records to it.
    """
    buffer = io.StringIO()
    handler = ListHandler(stream=buffer)
    handler.setLevel(logging.WARNING)
    logging_env.addHandler(handler)
    yield handler
    logging_env.removeHandler(handler)


def _tqdm_standins() -> list[logging.Handler]:
    return [
        handler
        for handler in logging.getLogger().handlers
        if type(handler).__module__.startswith("tqdm")
    ]


def test_console_level_falls_back_to_warning(logging_env):
    logging_env.handlers = []
    assert _console_level() == logging.WARNING


def test_console_level_reads_console_handler(console):
    console.setLevel(logging.ERROR)
    assert _console_level() == logging.ERROR


def test_log_queue_listener_respects_handler_levels(console):
    with log_queue_listener() as queue:
        queue.put(_record(logging.INFO, "info should be dropped"))
        queue.put(_record(logging.WARNING, "warning should pass"))

    assert [record.getMessage() for record in console.records] == ["warning should pass"]


def test_progress_queue_listener_mirrors_console_level(console):
    console.setLevel(logging.ERROR)

    with progress_queue_listener(total=0, desc="test"):
        standins = _tqdm_standins()
        assert len(standins) == 1
        assert standins[0].level == logging.ERROR
        assert console not in logging.getLogger().handlers

    assert console in logging.getLogger().handlers
    assert console.level == logging.ERROR


def test_worker_channel_logs_respect_console_level(string_console):
    with worker_channels(total=0, desc="test") as channel, channel:
        channel.info("info should be dropped")
        channel.warning("warning should pass")
        channel.report()

    messages = [record.getMessage() for record in string_console.records]
    assert "info should be dropped" not in messages
    assert "warning should pass" in messages


def test_file_handler_still_receives_info(tmp_path, console):
    file_handler = logging.FileHandler(tmp_path / "test.log")
    file_handler.setLevel(logging.INFO)
    console.setLevel(logging.WARNING)
    logging.getLogger().addHandler(file_handler)

    try:
        with log_queue_listener() as queue:
            queue.put(_record(logging.INFO, "info to file"))
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()

    assert console.records == []
    assert "info to file" in (tmp_path / "test.log").read_text()


def test_worker_channel_unentered_uses_module_logger():
    channel = WorkerChannel()
    assert channel._logger() is logging.getLogger("gnss_tid.parallel_logging")
    assert channel.report() is None
