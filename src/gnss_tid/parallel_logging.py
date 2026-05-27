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
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager
from os import getpid
from typing import Any

from tqdm import tqdm
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
        if self.log_queue is not None:
            self._wlog = logging.getLogger(f"worker {getpid()}")
            self._handler = QueueHandler(self.log_queue)
            self._wlog.addHandler(self._handler)
            self._wlog.setLevel(logging.INFO)
        return self

    def __exit__(self, *_exc) -> None:
        handler = getattr(self, "_handler", None)
        if handler is not None:
            self._wlog.removeHandler(handler)
            self._wlog = None
            self._handler = None

    def _logger(self) -> logging.Logger:
        return getattr(self, "_wlog", None) or logger

    def info(self, msg: str, *args: Any) -> None:
        self._logger().info(msg, *args)

    def warning(self, msg: str, *args: Any) -> None:
        self._logger().warning(msg, *args)

    def report(self, n: int = 1) -> None:
        if self.progress_queue is None:
            return
        self.progress_queue.put(n)


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


@contextmanager
def progress_queue_listener(total: int, desc: str):
    """Yield a multiprocessing queue whose tokens advance a ``tqdm`` bar.

    Workers push integer tick counts onto the queue (see
    :meth:`WorkerChannel.report`).  A daemon thread on the parent process
    drains the queue and calls ``bar.update(n)`` for each value.  On exit a
    ``None`` sentinel stops the thread and the bar is closed.
    """
    q = Manager().Queue()
    with logging_redirect_tqdm():
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
def worker_channels(total: int, desc: str):
    """Open both listeners together and yield a populated ``WorkerChannel``.

    Convenience wrapper for joblib drivers that always use both the log and
    progress channels with the same lifetime.
    """
    # progress_queue_listener swaps console handlers via logging_redirect_tqdm;
    # log_queue_listener captures the handler list at construction time, so it
    # must run *inside* the redirect to see the tqdm-aware handlers.
    with (
        progress_queue_listener(total, desc) as pq,
        log_queue_listener() as lq,
    ):
        yield WorkerChannel(log_queue=lq, progress_queue=pq)
