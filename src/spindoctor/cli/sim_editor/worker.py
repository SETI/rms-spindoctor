"""Run one render at a time off the UI thread, keeping only the latest request.

The editor runs two kinds of render in the background: the full render of the scene
on screen (:mod:`spindoctor.cli.sim_editor.full_render`) and the frame the navigator
reads (:mod:`spindoctor.cli.sim_editor.verdict`).  Each is a
:class:`LatestRequestRunner` with its own work function.

Thread model: only :meth:`_Job.run` runs off the UI thread.  It calls the work
function on an independent deep copy of the scene, and it touches neither a Qt object
nor any editor state.  It hands its result back through the job object itself, whose
fields it writes before it sets the job's ``threading.Event``; a ``QTimer`` on the UI
thread polls that event.  So every Qt object is created, used, and destroyed on the UI
thread, and a worker that outlives the window (the editor closed mid-render) only
writes into a job nobody reads.  The worker is an ordinary (non-daemon) thread, so the
interpreter waits for it before it shuts down: exiting the editor mid-render waits for
that one render to finish.  A daemon thread would be killed in the middle of the
render's numerical code at shutdown, which aborts the process.  A waiting request
never starts once the event loop has stopped, because only the UI thread's poll starts
one, so exit waits for at most one render per runner.  A failure is printed to
standard error on the UI thread, since the editor reports with ``print`` and carries
no logger.

Ordering: each request takes the next generation number.  At most one render runs at
a time; a request made while one runs replaces any request already waiting, and starts
when the running one finishes.  A finished render whose generation is older than the
latest request is discarded.  The render itself can't be interrupted, so a stale one
runs to the end before the waiting request starts.
"""

import copy
import sys
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

__all__ = ['POLL_INTERVAL_MS', 'LatestRequestRunner']

# How often the UI thread looks for a finished render while one runs.  Short enough
# that the result follows the render without a visible lag, long enough that the poll
# costs nothing next to the render.
POLL_INTERVAL_MS = 25


@dataclass
class _Job:
    """One background render: its scene, its generation, and, once done, its outcome.

    Parameters:
        generation: The request's generation number.
        sim_params: An independent deep copy of the scene, owned by the job.
        work: The function the worker calls with ``sim_params``.
        result: The work function's return value, set by :meth:`run` on success.
        error: ``'<ExceptionType>: <message>'``, set by :meth:`run` on failure.
        detail: The failure's formatted traceback, printed to standard error.
        done: Set by :meth:`run` after it has written the outcome fields.
    """

    generation: int
    sim_params: dict[str, Any]
    work: Callable[[dict[str, Any]], Any]
    result: Any = None
    error: str = ''
    detail: str = ''
    done: threading.Event = field(default_factory=threading.Event)

    def run(self) -> None:
        """Call the work function; record the outcome; then set ``done``.

        Runs on the worker thread.  Any exception, ``BaseException`` included,
        becomes the ``error`` text, so a failure always reaches the editor instead
        of being lost.  Nothing is re-raised: the thread ends as soon as this
        returns, Python delivers ``KeyboardInterrupt`` only to the main thread, and a
        ``SystemExit`` raised here would only end this thread.
        """
        try:
            self.result = self.work(self.sim_params)
        except BaseException as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.detail = traceback.format_exc()
        finally:
            self.done.set()


class LatestRequestRunner(QObject):
    """Runs a work function on a worker thread, latest request wins.

    Every method runs on the UI thread, and so do the ``ready`` and ``failed``
    signals, which carry only the result of the latest request.
    """

    ready = pyqtSignal(object, object)
    """Emitted with the request's scene and the work function's result."""

    failed = pyqtSignal(str)
    """Emitted with ``'<ExceptionType>: <message>'`` when the latest request fails."""

    def __init__(
        self,
        parent: QObject | None,
        *,
        work: Callable[[dict[str, Any]], Any],
        description: str,
    ) -> None:
        """Create an idle runner.

        Parameters:
            parent: The owning Qt object (the editor window); its deletion takes the
                poll timer with it, so a render that finishes later is never read.
            work: The function the worker calls with its copy of the scene.  It must
                not touch Qt objects or editor state.
            description: What the work is, for the message printed to standard error
                when it fails, such as ``'Full render'``.
        """
        super().__init__(parent)
        self._work = work
        self._description = description
        self._generation = 0
        self._running: _Job | None = None
        self._waiting: _Job | None = None
        self._poll = QTimer(self)
        self._poll.setInterval(POLL_INTERVAL_MS)
        self._poll.timeout.connect(self._collect)

    @property
    def is_busy(self) -> bool:
        """Whether a render is running (current or stale)."""
        return self._running is not None

    def request(self, sim_params: dict[str, Any]) -> None:
        """Ask for a render of ``sim_params``, superseding every earlier request.

        The scene is deep-copied here, so later edits can't reach the worker.  The
        render starts now when none is running, and otherwise waits, replacing any
        request already waiting.

        Parameters:
            sim_params: The editor's scene mapping.
        """
        self._generation += 1
        self._waiting = _Job(self._generation, copy.deepcopy(sim_params), self._work)
        if self._running is None:
            self._start_waiting()

    def cancel(self) -> None:
        """Drop the waiting request and the result of the running one."""
        self._generation += 1
        self._waiting = None

    def _start_waiting(self) -> None:
        """Start the waiting request on a new worker thread, or stop polling if none.

        A thread that fails to start reports the failure through ``failed`` (the
        waiting request is always the latest one) and leaves the runner idle.
        """
        job = self._waiting
        self._waiting = None
        if job is None:
            self._poll.stop()
            return
        # Non-daemon: the interpreter joins it at exit, where a daemon thread would
        # be killed mid-render and abort the process (see the module docstring).
        thread = threading.Thread(target=job.run, name='sim-editor-render')
        try:
            thread.start()
        except RuntimeError as exc:
            self._poll.stop()
            self.failed.emit(f'{type(exc).__name__}: {exc}')
            return
        self._running = job
        self._poll.start()

    def _collect(self) -> None:
        """On the UI thread, take a finished render, report it if current, start the next."""
        job = self._running
        if job is None or not job.done.is_set():
            return
        self._running = None
        self._start_waiting()
        if job.error:
            print(f'{self._description} failed:\n{job.detail}', file=sys.stderr)
        if job.generation != self._generation:
            return
        if job.error:
            self.failed.emit(job.error)
        else:
            self.ready.emit(job.sim_params, job.result)
