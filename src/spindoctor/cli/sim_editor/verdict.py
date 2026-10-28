"""The navigated frame's verdict, rendered off the UI thread.

The editor's preview is offset-free, but the status bar reports the navigator's
verdict on the frame the navigator reads, with the planted offset applied.  That
frame needs a second render, which can take as long as the preview, so
:class:`VerdictRunner` runs it on a worker thread: the preview appears at once, and
the verdict line fills in when the second render finishes.

Thread model: only :meth:`_VerdictJob.run` runs off the UI thread.  It calls
:func:`navigated_frame_summary` on an independent deep copy of the scene, and it
touches neither a Qt object nor any editor state.  It hands its result back through
the job object itself, whose fields it writes before it sets the job's
``threading.Event``; a ``QTimer`` on the UI thread polls that event.  So every Qt
object is created, used, and destroyed on the UI thread, and a worker that outlives
the window (the editor closed mid-render) only writes into a job nobody reads.  The
worker is an ordinary (non-daemon) thread, so the interpreter waits for it before it
shuts down: exiting the editor mid-render waits for that one render to finish.  A
daemon thread would be killed in the middle of the render's numerical code at
shutdown, which aborts the process.  A waiting request never starts once the event
loop has stopped, because only the UI thread's poll starts one, so exit waits for at
most one render.  A failure is printed to standard error on the UI thread, since the
editor reports with ``print`` and carries no logger.

Ordering: each request takes the next generation number.  At most one render runs
at a time; a request made while one runs replaces any request already waiting, and
starts when the running one finishes.  A finished render whose generation is older
than the latest request is discarded, so the status bar never shows the verdict for
an older scene.  The render itself can't be interrupted, so a stale one runs to the
end before the waiting request starts.
"""

import copy
import sys
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.instruments import frame_units_summary, resolve_sim_inst_config
from spindoctor.sim.render import render_combined_model

__all__ = ['VerdictRunner', 'navigated_frame_summary']

# How often the UI thread looks for a finished render while one runs.  Short enough
# that the verdict line follows the render without a visible lag, long enough that
# the poll costs nothing next to the render.
_POLL_INTERVAL_MS = 25


def navigated_frame_summary(sim_params: dict[str, Any]) -> str:
    """Render the frame the navigator reads and describe it for the status bar.

    The frame carries the planted offset and roll (``ignore_offset=False``), and the
    text is :func:`~spindoctor.sim.instruments.frame_units_summary` of that frame
    under the scene's resolved instrument block.

    Parameters:
        sim_params: The scene mapping.  The caller owns it; it is only read.

    Returns:
        One line of text naming the navigated frame's units, range, marker share,
        and the navigator's verdict when it would not call the frame clean.
    """
    image, _ = render_combined_model(sim_params, ignore_offset=False)
    inst_config = resolve_sim_inst_config(
        DEFAULT_CONFIG,
        sim_params.get('instrument'),
        sim_params.get('instrument_config'),
    )
    return frame_units_summary(image, inst_config)


@dataclass
class _VerdictJob:
    """One verdict render: its scene, its generation, and, once done, its outcome.

    Parameters:
        generation: The request's generation number.
        sim_params: An independent deep copy of the scene, owned by the job.
        summary: The status text, set by :meth:`run` on success.
        error: ``'<ExceptionType>: <message>'``, set by :meth:`run` on failure.
        detail: The failure's formatted traceback, printed to standard error.
        done: Set by :meth:`run` after it has written the outcome fields.
    """

    generation: int
    sim_params: dict[str, Any]
    summary: str = ''
    error: str = ''
    detail: str = ''
    done: threading.Event = field(default_factory=threading.Event)

    def run(self) -> None:
        """Render and summarize the scene; record the outcome; then set ``done``.

        Runs on the worker thread.  Any exception, ``BaseException`` included,
        becomes the ``error`` text, so a failure always reaches the status bar
        instead of leaving it blank.  Nothing is re-raised: the thread ends as soon
        as this returns, Python delivers ``KeyboardInterrupt`` only to the main
        thread, and a ``SystemExit`` raised here would only end this thread.
        """
        try:
            self.summary = navigated_frame_summary(self.sim_params)
        except BaseException as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self.detail = traceback.format_exc()
        finally:
            self.done.set()


class VerdictRunner(QObject):
    """Runs the navigated-frame verdict on a worker thread, latest request wins.

    Every method runs on the UI thread, and so do the ``ready`` and ``failed``
    signals, which carry only the result of the latest request.
    """

    ready = pyqtSignal(str)
    """Emitted with the status text when the latest request's render finishes."""

    failed = pyqtSignal(str)
    """Emitted with ``'<ExceptionType>: <message>'`` when the latest request fails."""

    def __init__(self, parent: QObject | None = None) -> None:
        """Create an idle runner.

        Parameters:
            parent: The owning Qt object (the editor window); its deletion takes the
                poll timer with it, so a render that finishes later is never read.
        """
        super().__init__(parent)
        self._generation = 0
        self._running: _VerdictJob | None = None
        self._waiting: _VerdictJob | None = None
        self._poll = QTimer(self)
        self._poll.setInterval(_POLL_INTERVAL_MS)
        self._poll.timeout.connect(self._collect)

    @property
    def is_busy(self) -> bool:
        """Whether a verdict render is running (current or stale)."""
        return self._running is not None

    def request(self, sim_params: dict[str, Any]) -> None:
        """Ask for the verdict on ``sim_params``, superseding every earlier request.

        The scene is deep-copied here, so later edits can't reach the worker.  The
        render starts now when none is running, and otherwise waits, replacing any
        request already waiting.

        Parameters:
            sim_params: The editor's scene mapping.
        """
        self._generation += 1
        self._waiting = _VerdictJob(self._generation, copy.deepcopy(sim_params))
        if self._running is None:
            self._start_waiting()

    def cancel(self) -> None:
        """Drop the waiting request and the result of the running one.

        The editor calls this when the scene has nothing for the worker to do (no
        planted offset, or a failed preview) and when the window closes.
        """
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
        thread = threading.Thread(target=job.run, name='sim-editor-verdict')
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
            print(f'Navigated-frame verdict render failed:\n{job.detail}', file=sys.stderr)
        if job.generation != self._generation:
            return
        if job.error:
            self.failed.emit(job.error)
        else:
            self.ready.emit(job.summary)
