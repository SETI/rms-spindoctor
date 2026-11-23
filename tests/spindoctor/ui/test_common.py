"""Tests for ``spindoctor.ui.common``."""

import os
from collections.abc import Callable
from typing import Any

import pytest

# If this file is collected without ``tests/spindoctor/ui/conftest.py`` (unusual), still
# prefer headless Qt before importing bindings.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication, QFormLayout
except (ImportError, OSError) as exc:
    pytest.skip(
        f'PyQt6/QtWidgets not available: {exc}',
        allow_module_level=True,
    )

try:
    from spindoctor.ui.common import build_stretch_controls
except (ImportError, OSError) as exc:
    pytest.skip(
        f'spindoctor.ui.common import failed (Qt dependency): {exc}',
        allow_module_level=True,
    )


def _stretch_callbacks() -> tuple[
    Callable[[float], None],
    Callable[[float], None],
    Callable[[float], None],
]:
    def _black(_x: float) -> None:
        pass

    def _white(_x: float) -> None:
        pass

    def _gamma(_x: float) -> None:
        pass

    return _black, _white, _gamma


def _base_stretch_kwargs() -> dict[str, Any]:
    on_b, on_w, on_g = _stretch_callbacks()
    return {
        'img_min': 0.0,
        'img_max': 1.0,
        'black_init': 0.1,
        'white_init': 0.9,
        'gamma_init': 1.0,
        'on_black_changed': on_b,
        'on_white_changed': on_w,
        'on_gamma_changed': on_g,
    }


def test_build_stretch_controls_accepts_valid_numeric_args(qapp: QApplication) -> None:
    """Typical finite numeric arguments construct sliders and labels."""
    form = QFormLayout()
    result = build_stretch_controls(form, **_base_stretch_kwargs())
    assert result['slider_black'].value() == 100
    assert result['slider_white'].value() == 900


def test_build_stretch_controls_rejects_bool_black_init(qapp: QApplication) -> None:
    """``bool`` is rejected even though it is a subclass of ``int``."""
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['black_init'] = True
    with pytest.raises(TypeError, match='black_init must be int or float, not bool'):
        build_stretch_controls(form, **kw)


def test_build_stretch_controls_rejects_non_numeric_img_max(qapp: QApplication) -> None:
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['img_max'] = 'bad'
    with pytest.raises(TypeError, match='img_max must be int or float, not str'):
        build_stretch_controls(form, **kw)


def test_build_stretch_controls_rejects_non_finite_gamma_init(qapp: QApplication) -> None:
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['gamma_init'] = float('inf')
    with pytest.raises(ValueError, match=r'gamma_init must be a finite number'):
        build_stretch_controls(form, **kw)


def test_build_stretch_controls_rejects_bool_value_label_min_width(qapp: QApplication) -> None:
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['value_label_min_width'] = True
    with pytest.raises(TypeError, match='value_label_min_width must be int, not bool'):
        build_stretch_controls(form, **kw)


def test_build_stretch_controls_rejects_non_positive_value_label_min_width(
    qapp: QApplication,
) -> None:
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['value_label_min_width'] = 0
    with pytest.raises(ValueError, match='value_label_min_width must be > 0, got 0'):
        build_stretch_controls(form, **kw)


def test_build_stretch_controls_rejects_negative_slider_horizontal_stretch(
    qapp: QApplication,
) -> None:
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['slider_horizontal_stretch'] = -1
    with pytest.raises(ValueError, match='slider_horizontal_stretch must be >= 0, got -1'):
        build_stretch_controls(form, **kw)


def test_build_stretch_controls_allows_degenerate_img_range(qapp: QApplication) -> None:
    """``img_max <= img_min`` still uses internal ``img_min + 1.0`` upper bound."""
    form = QFormLayout()
    kw = _base_stretch_kwargs()
    kw['img_min'] = 5.0
    kw['img_max'] = 5.0
    kw['black_init'] = 5.0
    kw['white_init'] = 6.0
    result = build_stretch_controls(form, **kw)
    assert result['slider_black'].value() == 0
    assert result['slider_white'].value() == 1000
