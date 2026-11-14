"""Residual geometric distortion and ghost reflections in the optics stage.

Distortion tests: a radial k1 warp moves a corner point by the predicted
amount while the center stays put, the non-radial field is deterministic per
seed, and a disabled block is the identity.  The warp resamples only the lit
clusters of a plane; those tests compare it against one resampling of the whole
frame.  Ghost tests: a bright source casts a displaced, amplitude-scaled copy.
"""

from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from spindoctor.sim.forward import distortion as distortion_module
from spindoctor.sim.forward.distortion import apply_distortion
from spindoctor.sim.forward.ghosts import apply_ghosts
from spindoctor.sim.forward.stages import new_sim_frame
from spindoctor.sim.seeds import derive_effect_seed


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


def _whole_frame_warp(signal: np.ndarray, distortion: dict[str, Any], *, seed: int) -> np.ndarray:
    """Warp a plane by resampling the whole frame at once, at an oversampling of 1.

    This is the warp the cropped one must reproduce: every source position built on
    the full grid and one cubic interpolation over the whole plane.
    """
    size_v, size_u = signal.shape
    center_v = float(distortion.get('center_v', size_v / 2.0)) - 0.5
    center_u = float(distortion.get('center_u', size_u / 2.0)) - 0.5
    rho_ref = 0.5 * float(np.hypot(size_v, size_u))
    vv, uu = np.mgrid[0:size_v, 0:size_u].astype(np.float64)
    rv = vv - center_v
    ru = uu - center_u
    rho2 = (rv**2 + ru**2) / rho_ref**2
    factor = 1.0 + distortion['k1'] * rho2 + distortion['k2'] * rho2**2
    src_v = center_v + rv * factor
    src_u = center_u + ru * factor
    rms = float(distortion.get('nonradial_rms_px', 0.0))
    if rms > 0.0:
        field_seed = derive_effect_seed(seed, 'optics/distortion')
        field_v, field_u = distortion_module._nonradial_field((size_v, size_u), rms, field_seed)
        src_v = src_v + field_v
        src_u = src_u + field_u
    warped: np.ndarray = ndimage.map_coordinates(
        signal, np.stack([src_v, src_u]), order=3, mode='constant', cval=0.0
    )
    return warped


def _disc(signal: np.ndarray, center_v: float, center_u: float, radius: float) -> None:
    """Light a uniform disc in a plane in place."""
    vv, uu = np.mgrid[0 : signal.shape[0], 0 : signal.shape[1]]
    signal[(vv - center_v) ** 2 + (uu - center_u) ** 2 < radius**2] = 0.6


def _separate_clusters() -> np.ndarray:
    """Two discs and a star that lies inside the first disc's crop but is its own cluster."""
    signal = np.zeros((512, 512), dtype=np.float64)
    _disc(signal, 100.0, 100.0, 40.0)
    _disc(signal, 400.0, 380.0, 50.0)
    signal[100, 200] = 5.0
    return signal


def _cut_by_the_frame_edge() -> np.ndarray:
    """Discs cut off by the top edge and a corner, where blocks overhang the frame."""
    signal = np.zeros((500, 470), dtype=np.float64)
    _disc(signal, 0.0, 256.0, 60.0)
    _disc(signal, 499.0, 469.0, 30.0)
    return signal


def _small_discs() -> np.ndarray:
    """Two small discs on a wide frame, placed where the strong warps below move them far."""
    signal = np.zeros((512, 1024), dtype=np.float64)
    _disc(signal, 256.0, 100.0, 8.0)
    _disc(signal, 256.0, 500.0, 8.0)
    return signal


@pytest.mark.parametrize(
    ('make_signal', 'distortion'),
    [
        pytest.param(_separate_clusters, {'k1': 0.02, 'k2': -0.01}, id='separate_clusters'),
        pytest.param(_cut_by_the_frame_edge, {'k1': 0.02, 'k2': -0.01}, id='frame_edge'),
        pytest.param(
            _separate_clusters,
            {'k1': 0.0, 'k2': 0.0, 'nonradial_rms_px': 1.5},
            id='nonradial_wander',
        ),
        # Each warp moves one disc farther than the crop's margin: the radial warp
        # moves the outer disc about 115 pixels, and the wander moves the inner one
        # about 140.  Only the output crop's reach covers where the disc lands.
        pytest.param(_small_discs, {'k1': 1.5, 'k2': 0.0}, id='far_radial'),
        pytest.param(
            _small_discs,
            {'k1': 0.0, 'k2': 0.0, 'nonradial_rms_px': 120.0},
            id='far_wander',
        ),
    ],
)
def test_cropped_warp_matches_the_whole_frame_warp(
    make_signal: Any, distortion: dict[str, Any]
) -> None:
    """Warping only the lit clusters gives the whole-frame warp, apart from rounding."""
    signal = make_signal()
    expected = _whole_frame_warp(signal, distortion, seed=7)
    out = _distort(signal, distortion, seed=7)
    assert float(np.max(np.abs(out - expected))) <= 1e-20 * float(np.max(np.abs(expected)))


@pytest.mark.parametrize(
    ('k1', 'k2', 'center_v', 'center_u'),
    [
        pytest.param(1.0, -0.75, 255.5, 255.5, id='largest_between_center_and_corner'),
        pytest.param(0.02, 0.01, 50.0, 400.0, id='off_center'),
    ],
)
def test_largest_radial_displacement_is_the_largest_over_every_pixel(
    k1: float, k2: float, center_v: float, center_u: float
) -> None:
    """The bound on the displacement equals its largest value over the whole frame."""
    shape = (512, 512)
    rho_ref = 0.5 * float(np.hypot(*shape))
    vv, uu = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    r = np.hypot(vv - center_v, uu - center_u)
    s = (r / rho_ref) ** 2
    brute = float(np.max(np.abs(r * (k1 * s + k2 * s**2))))
    bound = distortion_module._largest_radial_displacement(
        shape, center_v, center_u, rho_ref, k1, k2
    )
    assert brute <= bound <= brute + 0.01


def test_a_fully_lit_plane_is_warped_in_one_piece() -> None:
    """A plane lit everywhere gives exactly what one resampling of the frame gives."""
    signal = np.random.default_rng(3).uniform(0.0, 1.0, size=(300, 300))
    distortion = {'k1': 0.02, 'k2': -0.01, 'center_v': 100.0, 'center_u': 200.0}
    out = _distort(signal, distortion)
    assert np.array_equal(out, _whole_frame_warp(signal, distortion, seed=7))


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
