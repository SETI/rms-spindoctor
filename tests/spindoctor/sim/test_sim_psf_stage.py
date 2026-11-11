"""The whole-scene PSF kernel and its application.

The optics-stage PSF is a separately-normalized Gaussian core plus Moffat wing.
These tests cover kernel normalization, the exact wing energy fraction, that a
delta reproduces the kernel, that an edge blurs into a PSF-shaped ramp, and
that an elliptical core respects its axis sigmas.  The blur convolves each
cluster of lit pixels on its own crop; those tests compare it against one
convolution of the whole frame.
"""

from collections.abc import Callable

import numpy as np
import pytest
from scipy.signal import fftconvolve

from spindoctor.sim.forward.psf import apply_psf, psf_kernel
from spindoctor.sim.instrument_resolvers import resolve_psf_truncation_px


def _kernel(**over: float) -> np.ndarray:
    """A default kernel with optional parameter overrides."""
    params: dict[str, float] = {'sigma_v': 0.7, 'sigma_u': 0.7, 'w': 0.1, 'r0': 2.0, 'n': 3.0}
    params.update(over)
    return psf_kernel(
        params['sigma_v'],
        params['sigma_u'],
        params['w'],
        params['r0'],
        params['n'],
        truncation_px=8,
        oversample=2,
    )


def test_kernel_sums_to_one() -> None:
    """The composed kernel conserves flux (unit sum)."""
    assert abs(float(_kernel().sum()) - 1.0) < 1e-9


def test_wing_fraction_is_the_exact_mixing_weight() -> None:
    """w is exactly the wing energy fraction: the kernel is (1-w)*core + w*wing."""
    core = _kernel(w=0.0)
    wing = _kernel(w=1.0)
    mixed = _kernel(w=0.3)
    assert np.allclose(mixed, 0.7 * core + 0.3 * wing, atol=1e-12)


def test_core_and_wing_each_normalize_to_unity() -> None:
    """Each term integrates to 1, so the mixing weight means what it says."""
    assert abs(float(_kernel(w=0.0).sum()) - 1.0) < 1e-9
    assert abs(float(_kernel(w=1.0).sum()) - 1.0) < 1e-9


def test_delta_reproduces_the_kernel() -> None:
    """Convolving a centered delta returns the kernel itself."""
    kernel = _kernel()
    size = kernel.shape[0]
    signal = np.zeros((size, size), dtype=np.float64)
    signal[size // 2, size // 2] = 1.0
    point_e = np.zeros_like(signal)
    apply_psf(
        signal,
        point_e,
        sigma_v=0.7,
        sigma_u=0.7,
        w=0.1,
        r0=2.0,
        n=3.0,
        truncation_px=8,
        oversample=2,
    )
    assert np.allclose(signal, kernel, atol=1e-10)


def test_edge_becomes_a_monotonic_ramp() -> None:
    """A step edge in the frame interior blurs into a smooth, monotonic ramp."""
    signal = np.zeros((40, 40), dtype=np.float64)
    signal[:, 20:35] = 1.0
    point_e = np.zeros_like(signal)
    apply_psf(
        signal,
        point_e,
        sigma_v=0.7,
        sigma_u=0.7,
        w=0.0,
        r0=2.0,
        n=3.0,
        truncation_px=8,
        oversample=1,
    )
    profile = signal[20, :]
    # The rising edge near u=20 is a smooth, monotonic ramp from dark to lit.
    transition = profile[15:26]
    assert np.all(np.diff(transition) >= -1e-9)
    assert float(profile[16]) < 0.05
    assert float(profile[24]) > 0.95


def test_elliptical_core_is_wider_along_the_larger_sigma() -> None:
    """sigma_v > sigma_u spreads a point wider along v than along u."""
    signal = np.zeros((60, 60), dtype=np.float64)
    signal[30, 30] = 1.0
    point_e = np.zeros_like(signal)
    apply_psf(
        signal,
        point_e,
        sigma_v=2.0,
        sigma_u=0.8,
        w=0.0,
        r0=2.0,
        n=3.0,
        truncation_px=8,
        oversample=1,
    )
    vv, uu = np.mgrid[0:60, 0:60]
    total = float(signal.sum())
    var_v = float((signal * (vv - 30.0) ** 2).sum()) / total
    var_u = float((signal * (uu - 30.0) ** 2).sum()) / total
    assert var_v > var_u


def test_point_source_plane_is_also_blurred() -> None:
    """A star in the point-source plane inherits the same kernel."""
    signal = np.zeros((40, 40), dtype=np.float64)
    point_e = np.zeros((40, 40), dtype=np.float64)
    point_e[20, 20] = 100.0
    apply_psf(
        signal,
        point_e,
        sigma_v=1.0,
        sigma_u=1.0,
        w=0.0,
        r0=2.0,
        n=3.0,
        truncation_px=8,
        oversample=1,
    )
    assert float(point_e[20, 20]) < 100.0
    assert float(point_e[20, 21]) > 0.0
    assert abs(float(point_e.sum()) - 100.0) < 1e-6


def _blur(signal: np.ndarray) -> np.ndarray:
    """Blur a copy of a signal plane with the default kernel, no stars."""
    out = signal.copy()
    apply_psf(
        out,
        np.zeros_like(out),
        sigma_v=0.7,
        sigma_u=0.7,
        w=0.1,
        r0=2.0,
        n=3.0,
        truncation_px=8,
        oversample=2,
    )
    return out


def _ring_around_a_star() -> np.ndarray:
    """An annulus whose box holds a separate star, on a frame not whole blocks in size."""
    signal = np.zeros((300, 280), dtype=np.float64)
    vv, uu = np.mgrid[0:300, 0:280]
    radius = np.hypot(vv - 150.0, uu - 140.0)
    signal[(radius > 90.0) & (radius < 100.0)] = 0.4
    signal[150, 140] = 7.0
    return signal


def _cut_by_the_frame_edge() -> np.ndarray:
    """Content touching the top edge and the bottom-right corner."""
    signal = np.zeros((300, 280), dtype=np.float64)
    signal[0:20, 100:160] = 0.5
    signal[290:300, 260:280] = 0.8
    return signal


@pytest.mark.parametrize(
    'make_signal',
    [
        pytest.param(_ring_around_a_star, id='cluster_inside_another_box'),
        pytest.param(_cut_by_the_frame_edge, id='frame_edge'),
    ],
)
def test_cluster_blur_matches_the_whole_frame_convolution(
    make_signal: Callable[[], np.ndarray],
) -> None:
    """Blurring each lit cluster on its own crop gives the whole-frame blur to rounding."""
    signal = make_signal()
    expected = fftconvolve(signal, _kernel(), mode='same')
    out = _blur(signal)
    assert float(np.max(np.abs(out - expected))) <= 1e-14 * float(np.max(expected))


def test_a_fully_lit_plane_is_blurred_in_one_piece() -> None:
    """A plane lit everywhere gives exactly one whole-frame convolution."""
    signal = np.random.default_rng(5).uniform(0.0, 1.0, size=(120, 100))
    assert np.array_equal(_blur(signal), fftconvolve(signal, _kernel(), mode='same'))


def test_coiss_uses_a_wider_truncation_window() -> None:
    """Cassini cameras get the documented wider PSF window."""
    assert resolve_psf_truncation_px('coiss_nac') == 32
    assert resolve_psf_truncation_px('gossi') == 16
    assert resolve_psf_truncation_px(None) == 16
