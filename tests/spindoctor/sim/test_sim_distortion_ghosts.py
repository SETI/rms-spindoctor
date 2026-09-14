"""Residual geometric distortion and ghost reflections in the optics stage.

Distortion tests: a radial k1 warp moves a corner point by the predicted
amount while the center stays put, the non-radial field is deterministic per
seed, and a disabled block is the identity.  Ghost tests: a bright source
casts a displaced, amplitude-scaled copy.
"""

from typing import Any

import numpy as np

from spindoctor.sim.forward.distortion import apply_distortion
from spindoctor.sim.forward.ghosts import apply_ghosts
from spindoctor.sim.forward.stages import new_sim_frame


def _distort(signal: np.ndarray, distortion: dict[str, Any], *, seed: int = 7) -> np.ndarray:
    """Apply a distortion block to a fresh frame's signal and return it."""
    frame = new_sim_frame(signal.shape[0], signal.shape[1])
    frame.signal[:] = signal
    apply_distortion(
        frame, params={'optics': {'distortion': distortion}, 'random_seed': seed}, oversample=1
    )
    return frame.signal


def _peak(image: np.ndarray) -> tuple[int, int]:
    """Row, column of the brightest pixel."""
    idx = int(np.argmax(image))
    return idx // image.shape[1], idx % image.shape[1]


def test_disabled_distortion_is_identity() -> None:
    """An all-zero distortion block leaves the signal bit-identical."""
    signal = np.zeros((40, 40), dtype=np.float64)
    signal[10, 30] = 1.0
    out = _distort(signal, {'k1': 0.0, 'k2': 0.0})
    assert np.array_equal(out, signal)


def test_center_pixel_is_unmoved() -> None:
    """A source at the optical center does not move under a radial warp."""
    signal = np.zeros((41, 41), dtype=np.float64)
    signal[20, 20] = 1.0
    out = _distort(signal, {'k1': 0.2, 'k2': 0.0, 'center_v': 20.0, 'center_u': 20.0})
    assert _peak(out) == (20, 20)


def test_k1_warp_moves_a_corner_point_inward_by_the_prediction() -> None:
    """A positive k1 pulls an off-center point toward the center by k1*rho^2."""
    size = 41
    signal = np.zeros((size, size), dtype=np.float64)
    qv, qu = 8, 8
    signal[qv, qu] = 1.0
    center = 20.0
    k1 = 0.1
    out = _distort(signal, {'k1': k1, 'k2': 0.0, 'center_v': center, 'center_u': center})
    # The sampled warp places the input feature at p where warp(p) = q, so to
    # first order the output radius shrinks by k1 * rho^2.
    rho_ref = 0.5 * float(np.hypot(size, size))
    rv, ru = qv - center, qu - center
    rho2 = (rv**2 + ru**2) / rho_ref**2
    predicted_v = center + rv * (1.0 - k1 * rho2)
    peak_v, _ = _peak(out)
    assert abs(peak_v - predicted_v) < 0.7


def test_nonradial_field_is_deterministic_per_seed() -> None:
    """The non-radial wander repeats exactly for the same seed."""
    signal = np.zeros((40, 40), dtype=np.float64)
    signal[15:25, 15:25] = 1.0
    first = _distort(signal, {'k1': 0.0, 'k2': 0.0, 'nonradial_rms_px': 1.5}, seed=11)
    second = _distort(signal, {'k1': 0.0, 'k2': 0.0, 'nonradial_rms_px': 1.5}, seed=11)
    assert np.array_equal(first, second)


def test_nonradial_field_differs_between_seeds() -> None:
    """A different seed draws a different wander."""
    signal = np.zeros((40, 40), dtype=np.float64)
    signal[15:25, 15:25] = 1.0
    first = _distort(signal, {'k1': 0.0, 'k2': 0.0, 'nonradial_rms_px': 1.5}, seed=11)
    other = _distort(signal, {'k1': 0.0, 'k2': 0.0, 'nonradial_rms_px': 1.5}, seed=12)
    assert not np.array_equal(first, other)


def test_ghost_appears_at_offset_with_expected_amplitude() -> None:
    """A bright source casts a displaced copy scaled by the ghost amplitude."""
    frame = new_sim_frame(60, 60)
    frame.signal[30, 30] = 1.0
    apply_ghosts(
        frame,
        ghosts=[{'dv_px': 8.0, 'du_px': 6.0, 'amplitude': 0.1, 'defocus_sigma': 1.0}],
        oversample=1,
    )
    # The original source survives and a faint ghost sits at the offset.
    assert float(frame.signal[30, 30]) > 0.5
    ghost_region = frame.signal[34:43, 32:41]
    assert abs(float(ghost_region.sum()) - 0.1) < 0.02


def test_ghost_zero_amplitude_is_a_noop() -> None:
    """A zero-amplitude ghost contributes nothing."""
    frame = new_sim_frame(30, 30)
    frame.signal[15, 15] = 1.0
    before = frame.signal.copy()
    apply_ghosts(
        frame,
        ghosts=[{'dv_px': 5.0, 'du_px': 0.0, 'amplitude': 0.0, 'defocus_sigma': 1.0}],
        oversample=1,
    )
    assert np.array_equal(frame.signal, before)
