"""The two frame-fixed optics centres land on the same point.

``optics.distortion.center_v`` / ``center_u`` and ``optics.stray_light.center_v``
/ ``center_u`` are both stated the way every other scene position is, in
pixel-corner coordinates on the detector grid, and both name where a whole-frame
field sits.  The fields are evaluated on the oversampled grid in pixel-centric
coordinates, so each converts, and they must convert the same way: an author
writing the same number in both blocks is naming one point.

The two arrive there by different code, so these tests measure where each field
actually ends up rather than comparing the arithmetic to itself.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.forward.distortion import apply_distortion
from spindoctor.sim.forward.optics import apply_optics, apply_stray_light
from spindoctor.sim.forward.stages import new_sim_frame
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX

_SIZE = 64
_OVERSAMPLE = 4
_CENTER = 20.0


def _expected_oversampled_center(detector_center: float) -> float:
    """Where a stated centre lands on the oversampled grid.

    Parameters:
        detector_center: The centre as a scene states it.

    Returns:
        The same point in pixel-centric coordinates on the oversampled grid.
    """
    return detector_center * _OVERSAMPLE - PIXEL_CENTER_TO_CORNER_PX


def _scene(optics: dict[str, Any]) -> dict[str, Any]:
    """A minimal scene carrying one optics block.

    Parameters:
        optics: The ``optics`` mapping to render with.

    Returns:
        The scene mapping.
    """
    return {'random_seed': 1, 'optics': optics}


def test_the_radial_bump_peaks_on_the_stated_centre() -> None:
    """A radial field's peak is at the centre it was given, in the grid's own terms."""
    img = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    apply_stray_light(img, amplitude=0.5, model='radial', center_v=11.0, center_u=29.0)
    peak_v, peak_u = np.unravel_index(int(np.argmax(img)), img.shape)
    assert (int(peak_v), int(peak_u)) == (11, 29)


def test_the_radial_bump_defaults_to_the_frame_centre() -> None:
    """With no centre the bump sits at the middle of the frame.

    On an even-sized frame that middle falls between two samples, so the two
    straddling rows carry the same value.  A centre half a pixel off would make
    one of them the unique peak, which is what distinguishes the two.
    """
    img = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    apply_stray_light(img, amplitude=0.5, model='radial')
    low, high = _SIZE // 2 - 1, _SIZE // 2
    assert float(img[low, low]) == pytest.approx(float(img[high, high]))
    assert float(img[low, high]) == pytest.approx(float(img[high, low]))


def test_stray_light_converts_its_centre_onto_the_oversampled_grid() -> None:
    """The stated centre reaches the grid the field is evaluated on."""
    frame = new_sim_frame(_SIZE, _SIZE, oversample=_OVERSAMPLE)
    apply_optics(
        frame,
        params=_scene(
            {
                'stray_light': {
                    'model': 'radial',
                    'amplitude': 0.5,
                    'center_v': _CENTER,
                    'center_u': _CENTER,
                }
            }
        ),
        rng=np.random.default_rng(1),
    )
    peak_v, peak_u = np.unravel_index(int(np.argmax(frame.signal)), frame.signal.shape)
    expected = _expected_oversampled_center(_CENTER)
    assert int(peak_v) == pytest.approx(expected, abs=0.5)
    assert int(peak_u) == pytest.approx(expected, abs=0.5)


def _warp_fixed_point(distortion: dict[str, Any]) -> float:
    """Measure the v coordinate a radial warp leaves in place.

    A warp resamples the plane, so feeding it a ramp whose value *is* its own v
    coordinate makes the result read out the source coordinate each output
    sample was drawn from.  The difference between that and the output
    coordinate is zero exactly at the centre and changes sign across it, so the
    centre comes out of a zero crossing -- measured to well under a pixel, and
    without the test naming the number it is checking.

    Parameters:
        distortion: The distortion block to apply.

    Returns:
        The v coordinate the warp holds fixed, on the oversampled grid.
    """
    size = _SIZE * _OVERSAMPLE
    frame = new_sim_frame(_SIZE, _SIZE, oversample=_OVERSAMPLE)
    frame.signal[:] = np.arange(size, dtype=np.float64)[:, None]
    apply_distortion(frame, params=_scene({}), oversample=_OVERSAMPLE, distortion=distortion)

    rows = np.arange(size, dtype=np.float64)
    column = round(_expected_oversampled_center(_CENTER))
    drift = frame.signal[:, column] - rows
    inner = slice(size // 4, 3 * size // 4)
    v = rows[inner]
    d = drift[inner]
    crossings = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
    assert crossings.size == 1, f'expected one zero crossing, found {crossings.size}'
    i = int(crossings[0])
    return float(v[i] - d[i] * (v[i + 1] - v[i]) / (d[i + 1] - d[i]))


def test_distortion_holds_its_stated_centre_fixed() -> None:
    """The point a radial warp leaves in place is the centre the scene stated."""
    fixed = _warp_fixed_point({'k1': 0.6, 'k2': 0.0, 'center_v': _CENTER, 'center_u': _CENTER})
    assert fixed == pytest.approx(_expected_oversampled_center(_CENTER), abs=0.05)


def test_distortion_defaults_to_the_frame_centre() -> None:
    """With no centre stated the warp holds the middle of the frame fixed."""
    fixed = _warp_fixed_point({'k1': 0.6, 'k2': 0.0})
    assert fixed == pytest.approx((_SIZE * _OVERSAMPLE - 1) / 2.0, abs=0.05)
