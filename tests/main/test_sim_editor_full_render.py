"""The scene editor's two-pass picture: a fast preview, then the full render when idle.

Each edit draws the simulator's fast preview.  Once the editor has been idle, the full
render runs on a worker thread and replaces the preview, if the scene is unchanged.
Most tests replace the worker's render with a gated stand-in, so they decide when it
finishes, and ask for the full render directly, as the idle timer would.

Qt runs headless (offscreen platform).
"""

import os
import threading
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
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

from PIL import Image

from spindoctor.cli.sim_editor import CreateSimulatedImageModel
from spindoctor.cli.sim_editor import full_render as full_render_module
from spindoctor.cli.sim_editor.full_render import FullRender
from spindoctor.cli.sim_editor.main_window import build_arg_parser
from spindoctor.cli.sim_editor.render_display import _dn_to_display_uint8
from spindoctor.sim.render import render_combined_model
from tests.main.sim_editor_waits import (
    NEVER_IDLE_SEC,
    TIMEOUT_SEC,
    process_until,
    render_fully,
    wait_for_background_renders,
)


@pytest.fixture
def qapp() -> QApplication:
    """The shared (or freshly created) headless Qt application."""
    existing = QApplication.instance()
    if existing is None:
        return QApplication([])
    return cast(QApplication, existing)


@pytest.fixture(autouse=True)
def _no_error_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail on an error dialog, which would otherwise wait offscreen for a click."""

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f'error dialog raised: {args[1:]!r}')

    monkeypatch.setattr(QMessageBox, 'critical', staticmethod(_raise))


def _set_scene(model: Any) -> None:
    """Give the editor a small Cassini scene whose preview differs from its full render.

    The camera's defaults put a PSF on the frame, so the full render draws on a finer
    grid than the preview does, and the body's limb is anti-aliased only in the full
    render.
    """
    model.sim_params.pop('noise', None)
    model.sim_params.update(
        {
            'size_v': 64,
            'size_u': 64,
            'instrument': 'coiss_nac',
            'artifacts': {'instrument_defaults': True},
            'bodies': [
                {
                    'name': 'RHEA',
                    'center_v': 30.3,
                    'center_u': 33.7,
                    'axis1': 30.0,
                    'axis2': 30.0,
                    'axis3': 30.0,
                    'illumination_angle': 20.0,
                    'phase_angle': 30.0,
                }
            ],
        }
    )


@pytest.fixture
def model(qapp: QApplication) -> Any:
    """An editor on the small Cassini scene, showing its fast preview."""
    window = CreateSimulatedImageModel(full_render_delay_sec=NEVER_IDLE_SEC)
    _set_scene(window)
    window._update_image()
    return window


class _GatedFullRender:
    """A stand-in for the worker's full render that finishes only when released.

    Each call records its scene and blocks on its own gate.  The released call returns
    the real full render of its scene, or raises ``error``.
    """

    def __init__(self, error: BaseException | None = None) -> None:
        """Create the stand-in, optionally failing every call with ``error``."""
        self.scenes: list[dict[str, Any]] = []
        self._gates: list[threading.Event] = []
        self._lock = threading.Lock()
        self._error = error
        self._real = full_render_module.render_full_frame

    def __call__(self, sim_params: dict[str, Any]) -> FullRender:
        """Record the call, wait for its gate, then render or raise."""
        gate = threading.Event()
        with self._lock:
            self.scenes.append(sim_params)
            self._gates.append(gate)
        if not gate.wait(TIMEOUT_SEC):
            raise TimeoutError('the test never released this render')
        if self._error is not None:
            raise self._error
        return self._real(sim_params)

    def release(self, index: int) -> None:
        """Let call ``index`` finish, running the event loop until it has started."""
        process_until(lambda: len(self._gates) > index)
        self._gates[index].set()


@pytest.fixture
def gated(monkeypatch: pytest.MonkeyPatch) -> _GatedFullRender:
    """Replace the worker's full render with a gated stand-in."""
    stand_in = _GatedFullRender()
    monkeypatch.setattr(full_render_module, 'render_full_frame', stand_in)
    return stand_in


@pytest.fixture
def failing(monkeypatch: pytest.MonkeyPatch) -> _GatedFullRender:
    """Replace the worker's full render with a gated stand-in that fails."""
    stand_in = _GatedFullRender(error=RuntimeError('full boom'))
    monkeypatch.setattr(full_render_module, 'render_full_frame', stand_in)
    return stand_in


def _fail_full_render(model: Any, failing: _GatedFullRender) -> None:
    """Start the full render of the editor's scene and let it fail."""
    model._start_full_renders()
    failing.release(0)
    wait_for_background_renders(model)


def _preview(model: Any) -> Any:
    """The fast preview of the editor's scene, as the simulator renders it."""
    image, _ = render_combined_model(model.sim_params, ignore_offset=True, preview=True)
    return image


def _full(model: Any) -> Any:
    """The full render of the editor's scene, as the simulator renders it."""
    image, _ = render_combined_model(model.sim_params, ignore_offset=True)
    return image


def test_an_edit_shows_the_fast_preview(model: Any) -> None:
    """An edit puts the simulator's fast preview of the scene on screen."""
    assert np.array_equal(model._current_image, _preview(model))


def test_the_status_bar_says_the_preview_is_on_screen(model: Any) -> None:
    """After an edit, the status bar says the image is the fast preview."""
    assert model._image_kind_label.text() == 'Fast preview'


def test_the_status_bar_says_the_full_render_is_running(
    model: Any, gated: _GatedFullRender
) -> None:
    """While the full render runs, the status bar says so beside the preview."""
    model._start_full_renders()
    assert model._image_kind_label.text() == 'Fast preview (full render running)'
    gated.release(0)
    wait_for_background_renders(model)


def test_the_full_render_replaces_the_preview(model: Any) -> None:
    """A finished full render of the unchanged scene replaces the preview."""
    render_fully(model)
    assert np.array_equal(model._current_image, _full(model))


def test_the_status_bar_says_the_full_render_is_on_screen(model: Any) -> None:
    """Once the full render replaces the preview, the status bar says so."""
    render_fully(model)
    assert model._image_kind_label.text() == 'Full render'


def test_a_failed_full_render_reaches_the_status_bar(
    model: Any, failing: _GatedFullRender, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed full render leaves the preview on screen and says why."""
    _fail_full_render(model, failing)
    assert model._image_kind_label.text() == (
        'Fast preview (full render failed: RuntimeError: full boom)'
    )
    assert 'Full render failed' in capsys.readouterr().err


def test_a_long_full_render_failure_is_shortened(
    model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status bar shows only the start of a long failure message's first line."""
    failing = _GatedFullRender(error=RuntimeError('x' * 200 + '\nsecond line'))
    monkeypatch.setattr(full_render_module, 'render_full_frame', failing)
    _fail_full_render(model, failing)
    shortened = 'RuntimeError: ' + 'x' * 63 + '...'
    assert model._image_kind_label.text() == f'Fast preview (full render failed: {shortened})'


def test_a_failed_full_render_leaves_no_pending_verdict(
    model: Any, failing: _GatedFullRender
) -> None:
    """Without a planted offset, the units line stops waiting once the full render fails."""
    _fail_full_render(model, failing)
    assert model._units_label.text() == 'DN: no verdict, the full render failed'


def test_a_failed_full_render_ends_the_saturation_wait(
    model: Any, failing: _GatedFullRender
) -> None:
    """The saturation overlay stops waiting once the full render fails."""
    model._saturation_overlay_check.setChecked(True)
    _fail_full_render(model, failing)
    assert model._saturation_label.text() == 'Saturated: unknown, the full render failed'


def test_an_edit_discards_a_running_full_render(model: Any, gated: _GatedFullRender) -> None:
    """A full render started before an edit never replaces the edit's preview."""
    model._start_full_renders()
    model.sim_params['bodies'][0]['center_u'] = 40.2
    model._update_image()
    gated.release(0)
    wait_for_background_renders(model)
    assert np.array_equal(model._current_image, _preview(model))


def test_an_edit_drops_the_failure_of_a_running_full_render(
    model: Any, failing: _GatedFullRender
) -> None:
    """A full render started before an edit doesn't report its failure after the edit."""
    model._start_full_renders()
    model.sim_params['bodies'][0]['center_u'] = 40.2
    model._update_image()
    failing.release(0)
    wait_for_background_renders(model)
    assert model._image_kind_label.text() == 'Fast preview'


def test_a_refused_edit_drops_the_failure_of_a_running_full_render(
    model: Any, failing: _GatedFullRender
) -> None:
    """A full render started before a refused edit doesn't report its failure after it."""
    model._start_full_renders()
    model.sim_params['bodies'][0]['minnaert_k'] = 0.0
    model._update_image()
    failing.release(0)
    wait_for_background_renders(model)
    assert model._image_kind_label.text() == 'Fast preview'


def test_a_full_render_of_a_changed_scene_is_not_shown(model: Any, gated: _GatedFullRender) -> None:
    """A full render finishing after an edit, before the edit's preview, is not shown."""
    preview = model._current_image.copy()
    model._start_full_renders()
    model.sim_params['bodies'][0]['center_u'] = 40.2
    gated.release(0)
    wait_for_background_renders(model)
    assert np.array_equal(model._current_image, preview)


def test_no_full_render_starts_before_an_edit_is_previewed(
    model: Any, gated: _GatedFullRender
) -> None:
    """If the idle timer fires before an edit's preview, no full render starts."""
    model.sim_params['bodies'][0]['center_u'] = 40.2
    model._start_full_renders()
    assert model._full_render_runner.is_busy is False


def test_the_idle_timer_starts_the_full_render(qapp: QApplication) -> None:
    """Once the editor has been idle for its delay, the full render replaces the preview."""
    window: Any = CreateSimulatedImageModel(full_render_delay_sec=0.0)
    _set_scene(window)
    window._updater.immediate_update()
    process_until(lambda: window._image_kind_label.text() == 'Full render')
    assert np.array_equal(window._current_image, _full(window))


def test_editing_the_midtime_still_ends_on_the_full_render(qapp: QApplication) -> None:
    """A midtime typed before the idle timer fires still ends on the scene's full render.

    The midtime changes no pixel, but the full render is shown only for the scene being
    edited, so the edit has to draw a new preview and restart the idle timer.
    """
    window: Any = CreateSimulatedImageModel(full_render_delay_sec=0.0)
    _set_scene(window)
    window._updater.immediate_update()
    window._midtime_edit.setText('2005-01-01T00:00:00')
    process_until(lambda: window._image_kind_label.text() == 'Full render')
    assert window._displayed_params == window.sim_params


def test_the_saturation_overlay_waits_for_the_full_render(model: Any) -> None:
    """While the preview is on screen, the saturation overlay says it is waiting."""
    model._saturation_overlay_check.setChecked(True)
    assert model._saturation_label.text() == 'Saturated: waiting for the full render'


def test_saving_writes_the_full_render(
    model: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Saving while the preview is on screen writes the stretched full render."""
    path = tmp_path / 'saved.png'
    monkeypatch.setattr(
        QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(path), 'PNG'))
    )
    model._save_image()
    with Image.open(path) as saved:
        pixels = np.asarray(saved)
    assert np.array_equal(pixels, _dn_to_display_uint8(_full(model)))


def test_the_full_render_delay_option_parses_seconds() -> None:
    """The command line's full-render delay is read in seconds."""
    args = build_arg_parser().parse_args(['--full-render-delay', '2.5'])
    assert args.full_render_delay == 2.5


def test_the_full_render_delay_option_refuses_a_negative_delay(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A negative full-render delay is refused with a message saying why."""
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(['--full-render-delay', '-1'])
    assert 'must be a finite number of seconds, 0 or more' in capsys.readouterr().err


def test_the_full_render_delay_option_refuses_a_delay_a_timer_cannot_hold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A delay past a Qt timer's longest interval is refused with a message saying why."""
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(['--full-render-delay', '2147483.648'])
    assert 'must be at most 2147483.647 seconds' in capsys.readouterr().err


def test_the_full_render_delay_defaults_to_one_second() -> None:
    """Without the option, the editor waits 1 s before the full render."""
    args = build_arg_parser().parse_args([])
    assert args.full_render_delay == 1.0
