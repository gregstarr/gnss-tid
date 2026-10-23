"""IPC helpers for joblib worker processes.

joblib spawns workers via ``loky``/``multiprocessing``, so records emitted
inside a worker do not reach the parent process's handlers by default.  This
module provides two channels for bridging that gap:

* a **log queue** that forwards ``logging`` records back to the parent's
  root-logger handlers, and
* a **progress queue** whose tokens advance a ``tqdm`` bar on the parent.

Both are carried together in a :class:`WorkerChannel`, which is itself a
context manager exposing ``info`` / ``warning`` / ``report`` methods so the
worker only ever sees a single object.  :func:`worker_channels` opens both
listeners on the parent side and yields a populated ``WorkerChannel`` ready
to hand to ``delayed(...)``.

Usage
-----
Wrap the parallel section in the parent with :func:`worker_channels` and
pass the yielded channel into each worker call.  Inside the worker, enter
the channel once at the top and call its methods::

    from joblib import Parallel, delayed
    from gnss_tid.parallel_logging import WorkerChannel, worker_channels

    def worker(item, ch: WorkerChannel | None = None):
        with (ch or WorkerChannel()) as ch:
            ch.info("processing %s", item)
            ...
            ch.report()

    with worker_channels(total=len(items), desc="work") as ch, \\
         Parallel(n_jobs=4) as parallel:
        parallel(delayed(worker)(item, ch) for item in items)

A default-constructed ``WorkerChannel()`` has both queues set to ``None``;
``info`` / ``warning`` fall back to the module logger and ``report`` is a
no-op, so the same worker body works in serial callers that pass ``None``.
"""

import logging
import sys
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import get_context
from os import getpid
from typing import Any

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


@dataclass
class WorkerChannel:
    """IPC handle for a joblib worker — context manager + log/progress API.

    Carries the queues a worker uses to forward log records and progress
    ticks back to the parent process.  Either field may be ``None`` to
    disable that channel individually; ``WorkerChannel()`` with no args
    disables both (logs fall through to the module logger, progress is a
    no-op).
    """

    log_queue: Any = None
    progress_queue: Any = None

    def __enter__(self) -> "WorkerChannel":
        """Attach the worker-side queue handler.

        With a log queue, creates a ``worker <pid>`` logger at INFO level
        with a ``QueueHandler`` forwarding records to the parent-side queue;
        without one, the channel stays in its module-logger fallback state.
        """
        if self.log_queue is not None:
            self._wlog = logging.getLogger(f"worker {getpid()}")
            self._handler = QueueHandler(self.log_queue)
            self._wlog.addHandler(self._handler)
            self._wlog.setLevel(logging.INFO)
        return self

    def __exit__(self, *_exc) -> None:
        """Remove the queue handler, restoring module-logger fallback."""
        handler = getattr(self, "_handler", None)
        if handler is not None:
            self._wlog.removeHandler(handler)
            self._wlog = None
            self._handler = None

    def _logger(self) -> logging.Logger:
        """Return the entered channel's worker logger, else the module logger."""
        return getattr(self, "_wlog", None) or logger

    def info(self, msg: str, *args: Any) -> None:
        """Emit an INFO record through the channel's logger.

        Args:
            msg: Message or ``%``-style format string.
            *args: Format arguments substituted into ``msg``.
        """
        self._logger().info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        """Emit a WARNING record through the channel's logger.

        Args:
            msg: Message or ``%``-style format string.
            *args: Format arguments substituted into ``msg``.
        """
        self._logger().warning(msg, *args)

    def report(self, n: int = 1) -> None:
        """Advance the parent's progress bar by ``n`` ticks.

        Args:
            n: Number of ticks to add (default 1).

        No-op without a progress queue (serial callers or ``pbar=False``),
        so workers can report unconditionally.
        """
        if self.progress_queue is None:
            return
        self.progress_queue.put(n)


_spawn_manager = None


def _spawn_manager_queue() -> Any:
    """Return a multiprocessing queue from a per-process spawn-context manager.

    The manager server is spawned rather than forked, so creating queues
    while the parent is multi-threaded (e.g. inside an active joblib/loky
    executor) cannot fork a multi-threaded process into a deadlocked child.
    One manager server is cached per process: extra ``Queue()`` calls on a
    running manager are cheap (no fork, no new server), and the server lives
    until the process exits.

    Returns:
        A ``multiprocessing`` queue backed by the cached spawn-context
        manager server.
    """
    global _spawn_manager
    if _spawn_manager is None:
        _spawn_manager = get_context("spawn").Manager()
    return _spawn_manager.Queue()


@contextmanager
def log_queue_listener():
    """Run a parent-side ``QueueListener`` and yield its log queue.

    Workers put ``logging`` records on the returned queue (see
    :meth:`WorkerChannel.__enter__`); the listener drains it and re-emits
    each record through the handlers attached to the root logger, respecting
    each handler's own level — e.g. a WARNING console stays quiet while an
    INFO file handler still receives records.  The handler list is captured
    when the listener starts, so handlers attached later are not picked up,
    and the listener is stopped cleanly on exit.

    Yields:
        A multiprocessing queue accepting ``logging.LogRecord`` objects.
    """
    q = _spawn_manager_queue()
    listener = QueueListener(q, *logging.getLogger().handlers, respect_handler_level=True)
    listener.start()
    try:
        yield q
    finally:
        listener.stop()


def _console_level() -> int:
    """Return the level of the root logger's console handler, WARNING if none.

    A console handler is any ``StreamHandler`` writing to stdout or stderr.
    Used to mirror the replaced console handler's level onto the tqdm
    stand-in handler installed by ``logging_redirect_tqdm``, which copies
    formatter, stream and filters of the handler it replaces but not its
    level.

    Returns:
        Integer level of the console handler (e.g. ``logging.WARNING``),
        or ``logging.WARNING`` when no console handler is attached.
    """
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream in (
            sys.stdout,
            sys.stderr,
        ):
            return handler.level
    return logging.WARNING


@contextmanager
def progress_queue_listener(total: int, desc: str):
    """Drain worker progress ticks into a ``tqdm`` bar on the parent.

    Workers push integer tick counts onto the queue (see
    :meth:`WorkerChannel.report`).  A daemon thread on the parent process
    drains the queue and calls ``bar.update(n)`` for each value.  On exit a
    ``None`` sentinel stops the thread and the bar is closed.

    Console records during the bar's lifetime are emitted through the
    tqdm-aware stand-in handler installed by ``logging_redirect_tqdm``,
    which copies formatter, stream and filters of the console handler it
    replaces but not its level; the replaced handler's level is mirrored
    onto the stand-in (see :func:`_console_level`) so e.g. Hydra's
    ``console.level: WARNING`` keeps INFO records off the console while the
    bar is live.

    Args:
        total: Progress bar total, matched against reported tick counts.
        desc: Progress bar label.

    Yields:
        A multiprocessing queue accepting integer tick counts.
    """
    q = _spawn_manager_queue()
    level = _console_level()
    with logging_redirect_tqdm():
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream in (
                sys.stdout,
                sys.stderr,
            ):
                handler.setLevel(level)
        bar = tqdm(total=total, desc=desc)

        def _drain():
            while True:
                item = q.get()
                if item is None:
                    return
                bar.update(item)

        t = threading.Thread(target=_drain, daemon=True)
        t.start()
        try:
            yield q
        finally:
            q.put(None)
            t.join()
            bar.close()


@contextmanager
def worker_channels(total: int, desc: str, *, pbar: bool = True):
    """Open worker logging and optional progress listeners.

    Args:
        total: Number of work items for the progress bar.
        desc: Progress bar label.
        pbar: Whether to enable progress reporting (default True).

    Yields:
        WorkerChannel with a log queue and, when enabled, a progress queue.
        With pbar=False, report() is a no-op and no progress listener is started.
    """
    # progress_queue_listener swaps console handlers via logging_redirect_tqdm;
    # log_queue_listener captures the handler list at construction time, so it
    # must run *inside* the redirect to see the tqdm-aware handlers.
    with (
        progress_queue_listener(total, desc) if pbar else nullcontext() as pq,
        log_queue_listener() as lq,
    ):
        yield WorkerChannel(log_queue=lq, progress_queue=pq)
