"""Unit tests for a kernel's encircled energy and for carrying a kernel to a resampled grid."""

from __future__ import annotations

import numpy as np
import pytest
from util.star_photometry.star_field import (
    QUANTIZATION_VARIANCE_DN2,
    KernelStarFit,
    StarCutout,
)
from util.star_photometry.star_psf_summary import (
    kernel_encircled_energy,
    kernel_encircled_energy_like_stars,
    resampled_kernel,
)

# A kernel of the simulator's form, narrow enough that pixel sampling matters.
KERNEL = {'sigma_v': 0.36, 'sigma_u': 0.36, 'w': 0.25, 'r0': 2.0, 'n': 5.0}
TRUNCATION_PX = 16
SIZE = 27
CENTERS = [(13.0, 13.0), (13.3, 12.8), (12.6, 13.45), (13.25, 13.25)]


def _like_stars(flux: float, *, dn_max: float) -> tuple[float, float]:
    """Return the kernel's radii on copies of four stars at :data:`CENTERS`.

    The stars carry no noise but rounding: their gain is huge and their sky shows only
    the variance of rounding.  Each sits on a sky of 3 DN.

    Parameters:
        flux: Each star's flux, in DN.
        dn_max: The largest DN the camera records.

    Returns:
        ``(ee50, ee80)`` in pixels.
    """
    cutouts = [
        StarCutout(
            image=np.zeros((SIZE, SIZE)),
            valid=np.ones((SIZE, SIZE), dtype=bool),
            v=v,
            u=u,
            sky_var=QUANTIZATION_VARIANCE_DN2,
            gain_e_per_dn=1.0e12,
        )
        for v, u in CENTERS
    ]
    fits = [
        KernelStarFit(center_v=v, center_u=u, flux=flux, background=3.0, residuals=np.zeros(0))
        for v, u in CENTERS
    ]
    return kernel_encircled_energy_like_stars(
        KERNEL,
        cutouts,
        fits,
        truncation_px=TRUNCATION_PX,
        dn_max=dn_max,
        r_max=8.0,
        n_bins=16,
        seed=1258,
        draws=2,
    )


def _noise_free() -> tuple[float, float]:
    """Return the kernel's noise-free radii at :data:`CENTERS`.

    Returns:
        ``(ee50, ee80)`` in pixels.
    """
    return kernel_encircled_energy(
        KERNEL, CENTERS, size=SIZE, truncation_px=TRUNCATION_PX, r_max=8.0, n_bins=16
    )


def test_a_bright_unsaturated_star_keeps_its_noise_free_radii() -> None:
    """Copies of a star bright enough that rounding is negligible measure as noise-free."""
    ee80 = _like_stars(1.0e6, dn_max=1.0e9)[1]
    assert ee80 == pytest.approx(_noise_free()[1], abs=0.005)


def test_rounding_narrows_a_faint_star() -> None:
    """Rounding to whole DN takes the faint wing off a star of a few tens of DN."""
    ee80 = _like_stars(100.0, dn_max=1.0e9)[1]
    assert ee80 < _noise_free()[1] - 0.1


def test_the_converter_ceiling_widens_a_saturated_star() -> None:
    """A star clipped at the converter's ceiling loses its peak, so half its light spreads."""
    ee50 = _like_stars(1.0e6, dn_max=255.0)[0]
    assert ee50 > _noise_free()[0] + 0.5


# A Gaussian core alone, for the variance arithmetic of a resample: spread over its pixel,
# a Gaussian of sigma s has variance s**2 + 1/12, stretching scales that by STRETCH**2,
# and bilinear interpolation adds a triangle of variance STRETCH**2 / 6.
GAUSSIAN = {'sigma_v': 0.7, 'sigma_u': 0.7, 'w': 0.0, 'r0': 2.0, 'n': 3.0}
STRETCH = 1.17


def _resampled_sigma(*, bilinear: bool) -> float:
    """Return the core sigma of :data:`GAUSSIAN` carried to a grid stretched by STRETCH.

    Parameters:
        bilinear: Whether the resample interpolates bilinearly.

    Returns:
        The output kernel's core sigma.
    """
    return resampled_kernel(GAUSSIAN, scale=STRETCH, bilinear=bilinear).kernel['sigma_v']


def test_a_kernel_carried_to_its_own_grid_keeps_its_core() -> None:
    """A kernel carried to a grid of the same pixels, without interpolation, keeps its core."""
    kernel = {'sigma_v': 0.8, 'sigma_u': 0.8, 'w': 0.3, 'r0': 2.0, 'n': 4.0}
    carried = resampled_kernel(kernel, scale=1.0, bilinear=False).kernel
    assert carried['sigma_v'] == pytest.approx(0.8, abs=1e-3)


def test_a_kernel_carried_to_its_own_grid_keeps_its_wing() -> None:
    """A kernel carried to a grid of the same pixels keeps its wing fraction."""
    kernel = {'sigma_v': 0.8, 'sigma_u': 0.8, 'w': 0.3, 'r0': 2.0, 'n': 4.0}
    carried = resampled_kernel(kernel, scale=1.0, bilinear=False).kernel
    assert carried['w'] == pytest.approx(0.3, abs=1e-3)


def test_the_wing_radius_stretches_with_the_grid() -> None:
    """The output kernel's wing radius is the input's in output pixels."""
    carried = resampled_kernel(GAUSSIAN, scale=STRETCH, bilinear=False).kernel
    assert carried['r0'] == pytest.approx(2.0 * STRETCH)


def test_a_stretched_core_keeps_the_light_spread_over_the_input_pixel() -> None:
    """The output core spread over an output pixel holds the stretched input's variance."""
    expected = (STRETCH**2 * (0.7**2 + 1 / 12) - 1 / 12) ** 0.5
    assert _resampled_sigma(bilinear=False) == pytest.approx(expected, abs=0.01)


def test_bilinear_interpolation_widens_the_core_by_its_triangle() -> None:
    """Bilinear interpolation adds the variance of a triangle one input pixel each way."""
    expected = (STRETCH**2 * (0.7**2 + 1 / 12 + 1 / 6) - 1 / 12) ** 0.5
    assert _resampled_sigma(bilinear=True) == pytest.approx(expected, abs=0.01)
