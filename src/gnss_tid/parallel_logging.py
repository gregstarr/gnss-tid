"""Logging helpers for joblib worker processes.

joblib spawns workers via ``loky``/``multiprocessing``, so log records emitted
inside a worker do not reach the parent process's handlers by default.  This
module provides two composable context managers that bridge that gap by
funnelling worker records through a shared ``multiprocessing.Queue``.

Usage
-----
Wrap the parallel section in the parent with ``log_queue_listener`` and pass
the yielded queue into each worker call.  Inside the worker, wrap the body
in ``worker_logger`` and use the yielded logger::

    from joblib import Parallel, delayed
    from gnss_tid.parallel_logging import log_queue_listener, worker_logger

    def worker(item, log_queue=None):
        with worker_logger(log_queue) as wlog:
            wlog.info("processing %s", item)
            ...

    with log_queue_listener() as q, Parallel(n_jobs=4) as parallel:
        parallel(delayed(worker)(item, q) for item in items)

The parent's root-logger handlers receive every worker record as if it had
been emitted locally.  When ``log_queue`` is ``None`` (e.g. when calling the
worker serially for debugging), ``worker_logger`` yields the module logger
unchanged so the same worker body works in both modes.
"""

import logging
from contextlib import contextmanager
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager
from os import getpid
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def worker_logger(log_queue: Any | None = None, level: int = logging.INFO):
    """Yield a logger that forwards records to a multiprocessing queue.

    When ``log_queue`` is ``None`` the module logger is yielded unchanged so
    the same code path works in serial callers.  When a queue is supplied,
    a per-PID logger is created and its handler is removed on exit.
    """
    if log_queue is None:
        yield logger
        return
    wlog = logging.getLogger(f"worker {getpid()}")
    handler = QueueHandler(log_queue)
    wlog.addHandler(handler)
    wlog.setLevel(level)
    try:
        yield wlog
    finally:
        wlog.removeHandler(handler)


@contextmanager
def log_queue_listener():
    """Yield a multiprocessing log queue with a running ``QueueListener``.

    The listener forwards records to whatever handlers are attached to the
    root logger and is stopped cleanly on exit.
    """
    q = Manager().Queue()
    listener = QueueListener(q, *logging.getLogger().handlers)
    listener.start()
    try:
        yield q
    finally:
        listener.stop()
