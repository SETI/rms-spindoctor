"""The scene editor's navigated-frame verdict, rendered off the UI thread.

With a planted offset, the status bar first names the units with ``verdict
pending`` and then shows the navigated frame's summary when the worker's render
finishes.  Most tests replace the worker's render with a gated stand-in, so they
decide when each render finishes; one renders a real scene end to end.

Qt runs headless (offscreen platform).
"""

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6 import sip
    from PyQt6.QtWidgets import QApplication, QMessageBox
except (ImportError, OSError) as exc:
    pytest.skip(
        f'PyQt6/QtWidgets not available: {exc}',
        allow_module_level=True,
    )

try:
    if QApplication.instance() is None:
        QApplication([])
except Exception as exc:
    pytest.skip(
        f'PyQt6 QApplication init failed: {exc}',
        allow_module_level=True,
    )

from spindoctor.cli.sim_editor import CreateSimulatedImageModel
from spindoctor.cli.sim_editor import verdict as verdict_module
from spindoctor.config import DEFAULT_CONFIG
from spindoctor.sim.instruments import frame_units_summary, resolve_sim_inst_config

# Generous: a gated render finishes as soon as it is released, and the one real
# render is a 128 x 128 frame.
_TIMEOUT_SEC = 20.0


@pytest.fixture
def qapp() -> QApplication:
    """The shared (or freshly created) headless Qt application."""
    existing = QApplication.instance()
    if existing is None:
        return QApplication([])
    return cast(QApplication, existing)


@pytest.fixture
def model(qapp: QApplication) -> Any:
    """A freshly constructed simulated-image editor model on its default scene."""
    return CreateSimulatedImageModel()


def _process_until(condition: Callable[[], bool]) -> None:
    """Run the Qt event loop until ``condition`` holds, failing after the timeout."""
    deadline = time.monotonic() + _TIMEOUT_SEC
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail('timed out waiting for the verdict runner')
        QApplication.processEvents()
        time.sleep(0.005)


def _wait_for_verdict(model: Any) -> None:
    """Run the Qt event loop until no verdict render is running or waiting."""
    _process_until(lambda: not model._verdict_runner.is_busy)


class _GatedSummary:
    """A stand-in for the worker's render that finishes only when released.

    Each call records the scene's ``offset_u`` and the scene itself, and blocks on
    its own gate; the released call returns ``'summary of offset_u <value>'``, or
    raises ``error``.
    """

    def __init__(self, error: BaseException | None = None) -> None:
        """Create the stand-in, optionally failing every call with ``error``."""
        self.calls: list[float] = []
        self.scenes: list[dict[str, Any]] = []
        self.threads: list[threading.Thread] = []
        self._gates: list[threading.Event] = []
        self._lock = threading.Lock()
        self._error = error

    def __call__(self, sim_params: dict[str, Any]) -> str:
        """Record the call, wait for its gate, then answer or raise."""
        gate = threading.Event()
        with self._lock:
            self.calls.append(float(sim_params['offset_u']))
            self.scenes.append(sim_params)
            self.threads.append(threading.current_thread())
            self._gates.append(gate)
        if not gate.wait(_TIMEOUT_SEC):
            raise TimeoutError('the test never released this render')
        if self._error is not None:
            raise self._error
        return f'summary of offset_u {sim_params["offset_u"]:g}'

    def release(self, index: int) -> None:
        """Let call ``index`` finish, running the event loop until it has started."""
        _process_until(lambda: len(self._gates) > index)
        self._gates[index].set()


@pytest.fixture(autouse=True)
def _no_error_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail on an error dialog, which would otherwise wait offscreen for a click."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f'error dialog raised: {args[1:]!r}')

    monkeypatch.setattr(QMessageBox, 'critical', staticmethod(_raise))


@pytest.fixture
def gated(monkeypatch: pytest.MonkeyPatch) -> _GatedSummary:
    """Replace the worker's render with a gated stand-in."""
    stand_in = _GatedSummary()
    monkeypatch.setattr(verdict_module, 'navigated_frame_summary', stand_in)
    return stand_in


def _edit_offset(model: Any, offset_u: float) -> None:
    """Plant ``offset_u`` and re-render, as an edit of the offset spin box does."""
    model.sim_params['offset_u'] = offset_u
    model._update_image()


def _preview_summary(model: Any) -> str:
    """The status text of the current preview itself, under the scene's instrument."""
    inst_config = resolve_sim_inst_config(DEFAULT_CONFIG, model.sim_params['instrument'])
    return frame_units_summary(model._current_image, inst_config)


def _set_offscreen_body_scene(model: Any) -> None:
    """A 128 x 128 ``coiss_calib_nac`` frame whose planted offset moves its only body out."""
    model.sim_params.pop('noise', None)
    model.sim_params.update(
        {
            'size_v': 128,
            'size_u': 128,
            'instrument': 'coiss_calib_nac',
            'offset_u': 20.0,
            'bodies': [
                {
                    'name': 'RHEA',
                    'center_v': 64.0,
                    'center_u': 118.0,
                    'axis1': 6.0,
                    'axis2': 6.0,
                    'axis3': 6.0,
                    'illumination_angle': 0.0,
                    'phase_angle': 0.0,
                }
            ],
        }
    )


def test_units_label_says_the_verdict_is_pending_after_the_preview(model: Any) -> None:
    """The preview's render leaves the units and a pending verdict on the label."""
    _set_offscreen_body_scene(model)
    model._update_image()
    assert model._units_label.text() == 'I/F: verdict pending'


def test_units_label_reads_the_navigated_frame(model: Any) -> None:
    """The label says blank when the planted offset moves the only body out of the frame."""
    _set_offscreen_body_scene(model)
    model._update_image()
    _wait_for_verdict(model)
    assert '(peak below 0.0001: the navigator classifies it blank)' in model._units_label.text()


def test_a_stale_verdict_never_reaches_the_label(model: Any, gated: _GatedSummary) -> None:
    """A render that finishes after a newer edit leaves the label pending."""
    _edit_offset(model, 5.0)
    _edit_offset(model, 7.0)
    gated.release(0)
    # The newer request starts only once the stale render has been collected.
    _process_until(lambda: len(gated.calls) == 2)
    assert model._units_label.text() == 'DN: verdict pending'


def test_the_latest_edit_gets_its_verdict(model: Any, gated: _GatedSummary) -> None:
    """After the stale render, the newest edit's render runs and its summary is shown."""
    _edit_offset(model, 5.0)
    _edit_offset(model, 7.0)
    gated.release(0)
    gated.release(1)
    _wait_for_verdict(model)
    assert model._units_label.text() == 'summary of offset_u 7'


def test_rapid_edits_render_only_the_running_and_the_latest_scene(
    model: Any, gated: _GatedSummary
) -> None:
    """Edits made during a render replace each other; only the last one renders next."""
    _edit_offset(model, 5.0)
    for offset_u in (6.0, 7.0, 8.0):
        _edit_offset(model, offset_u)
    gated.release(0)
    gated.release(1)
    _wait_for_verdict(model)
    assert gated.calls == [5.0, 8.0]


def test_the_worker_gets_its_own_copy_of_the_scene(model: Any, gated: _GatedSummary) -> None:
    """Edits to the editor's scene after the request, nested ones included, miss the worker."""
    _edit_offset(model, 5.0)
    model.sim_params['offset_u'] = 9.0
    model.sim_params['noise']['read_noise_dn'] = 77.0
    model.sim_params['noise']['bias_dn'] = 3.0
    gated.release(0)
    _wait_for_verdict(model)
    worker_scene = gated.scenes[0]
    assert worker_scene['offset_u'] == 5.0
    assert worker_scene['noise'] == {
        'poisson': True,
        'read_noise_dn': 4.0,
        'cosmic_ray_events_per_pixel_per_sec': 0.0,
        'missing_data_rate': 0.0,
    }


def test_the_render_runs_off_the_ui_thread(model: Any, gated: _GatedSummary) -> None:
    """The navigated render runs on a thread other than the one driving the editor."""
    _edit_offset(model, 5.0)
    gated.release(0)
    _wait_for_verdict(model)
    assert gated.threads[0] is not threading.current_thread()


def test_the_label_is_written_only_on_the_ui_thread(
    model: Any, gated: _GatedSummary, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every write to the status label happens on the UI thread, the result's included."""
    label = model._units_label
    writers: list[threading.Thread] = []
    real_set_text = label.setText

    def _recording_set_text(text: str) -> None:
        writers.append(threading.current_thread())
        real_set_text(text)

    monkeypatch.setattr(label, 'setText', _recording_set_text)
    _edit_offset(model, 5.0)
    gated.release(0)
    _wait_for_verdict(model)
    assert label.text() == 'summary of offset_u 5'
    assert {thread.name for thread in writers} == {threading.current_thread().name}


def test_a_failed_verdict_render_reaches_the_label(
    model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception in the worker's render is reported on the status label."""
    failing = _GatedSummary(error=RuntimeError('verdict boom'))
    monkeypatch.setattr(verdict_module, 'navigated_frame_summary', failing)
    _edit_offset(model, 5.0)
    failing.release(0)
    _wait_for_verdict(model)
    assert model._units_label.text() == 'verdict render failed: RuntimeError: verdict boom'


def test_a_failed_verdict_render_prints_its_traceback(
    model: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An exception in the worker's render is printed to standard error with its traceback."""
    failing = _GatedSummary(error=RuntimeError('verdict boom'))
    monkeypatch.setattr(verdict_module, 'navigated_frame_summary', failing)
    _edit_offset(model, 5.0)
    failing.release(0)
    _wait_for_verdict(model)
    err = capsys.readouterr().err
    assert 'Navigated-frame verdict render failed' in err
    assert 'Traceback' in err
    assert 'RuntimeError: verdict boom' in err


def test_a_base_exception_in_the_worker_reaches_the_label(
    model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception outside Exception still reports, so the label never goes blank."""
    failing = _GatedSummary(error=SystemExit('worker exit'))
    monkeypatch.setattr(verdict_module, 'navigated_frame_summary', failing)
    _edit_offset(model, 5.0)
    failing.release(0)
    _wait_for_verdict(model)
    assert model._units_label.text() == 'verdict render failed: SystemExit: worker exit'


class _UnstartableThread(threading.Thread):
    """A thread whose start fails, as it does when the process can't make threads."""

    def start(self) -> None:
        """Fail the way ``threading.Thread.start`` does without a free thread."""
        raise RuntimeError("can't start new thread")


def test_a_thread_that_fails_to_start_reaches_the_label(
    model: Any, gated: _GatedSummary, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker thread that can't start reports the failure and leaves the runner idle."""
    monkeypatch.setattr(threading, 'Thread', _UnstartableThread)
    _edit_offset(model, 5.0)
    assert (
        model._units_label.text() == "verdict render failed: RuntimeError: can't start new thread"
    )
    assert model._verdict_runner.is_busy is False


def test_a_scene_without_an_offset_starts_no_verdict_render(
    model: Any, gated: _GatedSummary
) -> None:
    """An offset-free scene's status text comes from the preview, with no worker."""
    model._update_image()
    assert model._verdict_runner.is_busy is False
    assert gated.calls == []
    assert model._units_label.text() == _preview_summary(model)


def test_clearing_the_offset_drops_the_running_verdict(model: Any, gated: _GatedSummary) -> None:
    """A render for a scene whose offset was then removed never replaces the label."""
    _edit_offset(model, 5.0)
    _edit_offset(model, 0.0)
    gated.release(0)
    _wait_for_verdict(model)
    assert model._units_label.text() == _preview_summary(model)


def test_a_failed_preview_drops_the_running_verdict(
    model: Any, gated: _GatedSummary, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render that fails after a verdict was requested leaves the label empty."""
    _edit_offset(model, 5.0)

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError('preview failed')

    monkeypatch.setattr('spindoctor.cli.sim_editor.render_display.render_combined_model', _fail)
    monkeypatch.setattr(
        'spindoctor.cli.sim_editor.render_display.QMessageBox.critical',
        lambda *args, **kwargs: None,
    )
    _edit_offset(model, 6.0)
    gated.release(0)
    _wait_for_verdict(model)
    assert model._units_label.text() == ''


def test_closing_the_editor_drops_the_running_verdict(model: Any, gated: _GatedSummary) -> None:
    """A render that finishes after the window closed doesn't touch the label."""
    _edit_offset(model, 5.0)
    model.close()
    gated.release(0)
    _wait_for_verdict(model)
    assert model._units_label.text() == 'DN: verdict pending'


def test_deleting_the_editor_mid_render_is_safe(
    model: Any, gated: _GatedSummary, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render that finishes after the window was destroyed ends without an error.

    The poll timer must go with the window.  The event loop then runs well past the
    poll interval, so a timer that outlived it and fired on the deleted runner
    would report through ``sys.excepthook``, which PyQt uses for a slot's exception.
    """
    errors: list[str] = []
    monkeypatch.setattr(threading, 'excepthook', lambda args: errors.append(repr(args.exc_value)))
    monkeypatch.setattr(sys, 'excepthook', lambda *args: errors.append(repr(args[1])))
    _edit_offset(model, 5.0)
    poll_timer = model._verdict_runner._poll
    sip.delete(model)
    assert sip.isdeleted(poll_timer)
    gated.release(0)
    gated.threads[0].join(_TIMEOUT_SEC)
    deadline = time.monotonic() + 8 * verdict_module._POLL_INTERVAL_MS / 1000.0
    while time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    assert errors == []


# Exits the editor's event loop the way ``main`` does while a verdict render runs.
_EXIT_MID_RENDER = """
import os, sys
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import scipy.fft
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from spindoctor.cli.sim_editor import CreateSimulatedImageModel, verdict
real = verdict.navigated_frame_summary
def slow(sim_params):
    # Spend the render in scipy's C++ FFT, which releases the GIL; a daemon
    # thread killed there at shutdown aborts the process, as the review found
    # inside the renderer's fftconvolve.
    a = np.random.default_rng(0).random((1024, 1024))
    for _ in range(40):
        a = scipy.fft.ifft2(scipy.fft.fft2(a)).real
    return real(sim_params)
verdict.navigated_frame_summary = slow
window = CreateSimulatedImageModel()
window.show()
window.sim_params['offset_u'] = 3.0
window._update_image()
window.sim_params['offset_u'] = 4.0
window._update_image()
assert window._verdict_runner.is_busy
QTimer.singleShot(50, window.close)
sys.exit(app.exec())
"""


def test_exiting_the_editor_mid_render_exits_cleanly() -> None:
    """Closing the editor and exiting while a verdict renders ends the process with status 0."""
    src = Path(verdict_module.__file__).resolve().parents[3]
    env = {
        **os.environ,
        'PYTHONPATH': os.pathsep.join([str(src), *filter(None, [os.environ.get('PYTHONPATH')])]),
        'QT_QPA_PLATFORM': 'offscreen',
    }
    result = subprocess.run(
        [sys.executable, '-c', _EXIT_MID_RENDER],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
