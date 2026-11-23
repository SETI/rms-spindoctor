"""The vidicon beam bend and residual image."""

import numpy as np
import pytest

from spindoctor.sim.render.detector.vidicon import add_residual_image, apply_beam_bend

_SIZE = 64


def test_beam_bend_displaces_a_bright_edge() -> None:
    """The beam bend warps the image near a bright horizontal limb (v shift)."""
    signal = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    signal[32:, :] = 1.0
    before = signal.copy()
    apply_beam_bend(signal, amplitude_px=2.0, full_brightness_dn=1.0)
    # The vertical displacement field moves the limb; the frame changes near it.
    assert not np.array_equal(signal, before)
    assert float(np.abs(signal - before)[28:36, :].sum()) > 0.0


def test_beam_bend_disabled_at_zero_amplitude() -> None:
    """A zero bend amplitude leaves the image untouched."""
    signal = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    signal[32:, :] = 1.0
    before = signal.copy()
    apply_beam_bend(signal, amplitude_px=0.0, full_brightness_dn=1.0)
    assert np.array_equal(signal, before)


def test_residual_image_adds_a_displaced_ghost() -> None:
    """The self_offset ghost adds a shifted, scaled copy of the frame."""
    signal = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    signal[30, 30] = 100.0
    add_residual_image(signal, amplitude=0.1, prior='self_offset', offset_v=5, offset_u=5)
    assert float(signal[35, 35]) == pytest.approx(10.0)  # ghost of the bright pixel
    assert float(signal[30, 30]) == pytest.approx(100.0)  # original intact


def test_residual_image_disabled_at_zero_amplitude() -> None:
    """A zero ghost amplitude is a no-op."""
    signal = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    signal[30, 30] = 100.0
    before = signal.copy()
    add_residual_image(signal, amplitude=0.0, prior='self_offset', offset_v=5, offset_u=5)
    assert np.array_equal(signal, before)
