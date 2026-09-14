"""The two frame-fixed optics centers land on the same point.

``optics.distortion.center_v`` / ``center_u`` and ``optics.stray_light.center_v``
/ ``center_u`` are both stated the way every other scene position is, in
pixel-corner coordinates on the detector grid, and both name where a whole-frame
field sits.  The fields are evaluated on the oversampled grid in pixel-centric
coordinates, so each converts, and they must convert the same way: an author
writing the same number in both blocks is naming one point.

The two arrive there by different code, so these tests measure where each field
actually ends up rather than comparing the arithmetic to itself.
"""

import math
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
    """Where a stated center lands on the oversampled grid.

    Parameters:
        detector_center: The center as a scene states it.

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


def test_the_radial_bump_is_symmetric_about_the_stated_center() -> None:
    """A radial field is equidistant-valued either side of the center it was given.

    An ``argmax`` cannot settle this.  The field is smooth, so a center half a
    sample off leaves the same sample brightest (and at exactly half a sample the
    two straddling samples tie, which ``argmax`` breaks toward the lower one).
    Equality of two samples placed symmetrically about the stated center moves
    the moment the center does, in either direction.
    """
    img = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    apply_stray_light(img, amplitude=0.5, model='radial', center_v=11.0, center_u=29.0)
    for gap in (1, 2, 3):
        assert float(img[11 - gap, 29]) == pytest.approx(float(img[11 + gap, 29]))
        assert float(img[11, 29 - gap]) == pytest.approx(float(img[11, 29 + gap]))


def test_the_radial_bump_defaults_to_the_frame_center() -> None:
    """With no center the bump sits at the middle of the frame.

    On an even-sized frame that middle falls between two samples, so the two
    straddling rows carry the same value.  A center half a pixel off would make
    one of them the unique peak, which is what distinguishes the two.
    """
    img = np.zeros((_SIZE, _SIZE), dtype=np.float64)
    apply_stray_light(img, amplitude=0.5, model='radial')
    low, high = _SIZE // 2 - 1, _SIZE // 2
    assert float(img[low, low]) == pytest.approx(float(img[high, high]))
    # Not the transpose of the line above: this one holds the column fixed, so it
    # is sensitive to the v center alone.  Comparing ``img[low, high]`` against
    # ``img[high, low]`` instead would be true of any radial field whose two
    # center coordinates are equal, whatever they are, and so could not fail.
    assert float(img[low, low]) == pytest.approx(float(img[high, low]))


def test_stray_light_converts_its_center_onto_the_oversampled_grid() -> None:
    """The stated center reaches the grid the field is evaluated on.

    Measured by symmetry rather than by a peak.  The conversion moves the center
    by half a subsample, which is exactly the distance an integer ``argmax``
    cannot resolve: against an expected center of ``N + 0.5`` both ``N`` and
    ``N + 1`` sit within half a sample, so a peak test accepts the conversion
    omitted, doubled, or applied with the wrong sign.  Samples placed
    symmetrically about the expected center are equal only where it actually is.
    """
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
    expected = _expected_oversampled_center(_CENTER)
    lo = math.floor(expected)
    hi = lo + 1
    col = lo
    assert float(frame.signal[lo, col]) == pytest.approx(float(frame.signal[hi, col]))
    for gap in (1, 2, 3):
        assert float(frame.signal[lo - gap, col]) == pytest.approx(
            float(frame.signal[hi + gap, col])
        )
        assert float(frame.signal[col, lo - gap]) == pytest.approx(
            float(frame.signal[col, hi + gap])
        )


def _warp_fixed_point(distortion: dict[str, Any]) -> float:
    """Measure the v coordinate a radial warp leaves in place.

    A warp resamples the plane, so feeding it a ramp whose value *is* its own v
    coordinate makes the result read out the source coordinate each output
    sample was drawn from.  The difference between that and the output
    coordinate is zero exactly at the center and changes sign across it, so the
    center comes out of a zero crossing -- measured to well under a pixel, and
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


def test_distortion_holds_its_stated_center_fixed() -> None:
    """The point a radial warp leaves in place is the center the scene stated."""
    fixed = _warp_fixed_point({'k1': 0.6, 'k2': 0.0, 'center_v': _CENTER, 'center_u': _CENTER})
    assert fixed == pytest.approx(_expected_oversampled_center(_CENTER), abs=0.05)


def test_distortion_defaults_to_the_frame_center() -> None:
    """With no center stated the warp holds the middle of the frame fixed."""
    fixed = _warp_fixed_point({'k1': 0.6, 'k2': 0.0})
    assert fixed == pytest.approx((_SIZE * _OVERSAMPLE - 1) / 2.0, abs=0.05)
