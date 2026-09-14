"""Cursor-coordinate tests for the ``sd_create_simulated_image`` scene editor.

A scene states every position as a pixel corner, and a Qt event position is the
same measure, so the editor converts nothing on the way in: it keeps the
position continuous, floors it to find the pixel a click lands in, and crosses
to pixel centric only where it meets the renderer's own hit-test metadata, which
states a rendered star position that way.

Qt runs headless (offscreen platform).
"""

import os
from typing import Any, cast

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication
except (ImportError, OSError) as exc:
    pytest.skip(f'PyQt6/QtWidgets not available: {exc}', allow_module_level=True)

try:
    if QApplication.instance() is None:
        QApplication([])
except Exception as exc:
    pytest.skip(f'PyQt6 QApplication init failed: {exc}', allow_module_level=True)

from PyQt6.QtCore import QPointF

from spindoctor.cli.sim_editor import CreateSimulatedImageModel

_SIZE = 40


@pytest.fixture
def qapp() -> QApplication:
    """The shared (or freshly created) headless Qt application."""
    existing = QApplication.instance()
    if existing is None:
        return QApplication([])
    return cast(QApplication, existing)


@pytest.fixture
def model(qapp: QApplication) -> Any:
    """An editor model with a small frame and a known preview image."""
    editor = CreateSimulatedImageModel()
    editor.sim_params['size_v'] = _SIZE
    editor.sim_params['size_u'] = _SIZE
    v_grid, u_grid = np.mgrid[0:_SIZE, 0:_SIZE]
    editor._current_image = (100.0 * v_grid + u_grid).astype(np.float64)
    return editor


def _status_text(model: Any) -> str:
    """The status-bar text the model last wrote."""
    return str(model._status_label.text())


def test_label_position_stays_continuous_at_zoom_one(model: Any) -> None:
    """At zoom 1 a cursor inside a pixel keeps its fractional position."""
    model._zoom_factor = 1.0
    img_v, _img_u = model._label_pos_to_image_vu(QPointF(3.5, 2.5))
    assert img_v == pytest.approx(2.5, abs=1e-9)


def test_label_position_divides_by_the_zoom(model: Any) -> None:
    """A label position is divided by the zoom without being quantized first."""
    model._zoom_factor = 4.0
    _img_v, img_u = model._label_pos_to_image_vu(QPointF(9.0, 3.0))
    assert img_u == pytest.approx(2.25, abs=1e-9)


def test_readout_prints_a_pixel_centre(model: Any) -> None:
    """The readout can name the center of a pixel, which is what a scene uses."""
    model._zoom_factor = 1.0
    model._update_status_bar(QPointF(4.5, 3.5))
    assert 'V, U:     3.50,     4.50' in _status_text(model)


def test_readout_samples_the_pixel_containing_the_position(model: Any) -> None:
    """The value shown is the pixel containing the printed position."""
    model._zoom_factor = 1.0
    model._update_status_bar(QPointF(4.9, 3.9))
    assert 'Value: 304.000000' in _status_text(model)


def test_mask_hit_test_holds_to_the_end_of_a_pixel(model: Any) -> None:
    """A click in the far corner of a pixel still selects what that pixel shows."""
    model._add_body_tab()
    mask = np.zeros((_SIZE, _SIZE), dtype=bool)
    mask[6, 8] = True
    model._last_meta = {'body_masks': [mask], 'inventory': {}}
    model._select_model_at(6.99, 8.99)
    assert model._selected_model_key == ('body', 0)


def test_mask_hit_test_does_not_reach_the_previous_pixel(model: Any) -> None:
    """A click in the pixel before a mask pixel does not select what it shows."""
    model._add_body_tab()
    mask = np.zeros((_SIZE, _SIZE), dtype=bool)
    mask[6, 8] = True
    model._last_meta = {'body_masks': [mask], 'inventory': {}}
    model._select_model_at(5.6, 7.6)
    assert model._selected_model_key is None


# A sigma-free star offers a one-pixel-radius click target about its rendered
# position.  The renderer states that position pixel centric, so a rendered
# ``(10.0, 20.0)`` is at pixel corner ``(10.5, 20.5)`` and the target reaches
# from ``9.5`` to ``11.5`` in V.
_STAR_INFO = [{'center_v': 10.0, 'center_u': 20.0, 'sigma': 0.0}]


def test_star_hit_test_reaches_past_the_rendered_position(model: Any) -> None:
    """A click 0.9 px beyond the rendered star's position is inside its target."""
    model._add_star_tab()
    model._last_meta = {'star_info': _STAR_INFO}
    model._select_model_at(11.4, 20.5)
    assert model._selected_model_key == ('star', 0)


def test_star_hit_test_misses_short_of_the_rendered_position(model: Any) -> None:
    """A click 1.4 px short of the rendered star's position is outside its target."""
    model._add_star_tab()
    model._last_meta = {'star_info': _STAR_INFO}
    model._select_model_at(9.1, 20.5)
    assert model._selected_model_key is None
