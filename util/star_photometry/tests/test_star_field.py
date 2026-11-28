"""Unit tests for the PSF kernel fit of star cutouts."""

from __future__ import annotations

import numpy as np
import pytest
from util.star_photometry.star_field import StarCutout, fit_psf_kernel, kernel_star_image

from spindoctor.sim.render.optics.psf import apply_psf

# A kernel of the simulator's form, narrow enough that pixel sampling matters.
KERNEL = {'sigma_v': 0.45, 'sigma_u': 0.45, 'w': 0.2, 'r0': 2.0, 'n': 5.0}
TRUNCATION_PX = 8
OVERSAMPLE = 4
SIZE = 21


def test_a_kernel_star_is_the_simulators_render() -> None:
    """A star drawn for the fit equals the simulator's render of a star at that place.

    The simulator puts the star's light on one render-grid point, blurs the plane with the
    kernel, and averages each block of render-grid points into one detector pixel.
    """
    plane = np.zeros((SIZE * OVERSAMPLE, SIZE * OVERSAMPLE))
    point_v, point_u = 10 * OVERSAMPLE + 1, 10 * OVERSAMPLE + 2
    plane[point_v, point_u] = OVERSAMPLE**2
    apply_psf(
        np.zeros_like(plane),
        plane,
        **KERNEL,
        truncation_px=TRUNCATION_PX,
        oversample=OVERSAMPLE,
    )
    rendered = plane.reshape(SIZE, OVERSAMPLE, SIZE, OVERSAMPLE).mean(axis=(1, 3))
    # A render-grid point k has its center at (k + 0.5) / oversample - 0.5.
    center_v = (point_v + 0.5) / OVERSAMPLE - 0.5
    center_u = (point_u + 0.5) / OVERSAMPLE - 0.5
    drawn = kernel_star_image(
        KERNEL, SIZE, center_v, center_u, truncation_px=TRUNCATION_PX, oversample=OVERSAMPLE
    )
    np.testing.assert_allclose(drawn, rendered, rtol=0.0, atol=1e-12)


def test_the_fit_recovers_a_planted_kernel() -> None:
    """Noisy stars drawn with a known core and wing give that core and wing back."""
    rng = np.random.default_rng(1258)
    cutouts = []
    for _ in range(8):
        center_v, center_u = 10.0 + rng.uniform(-0.5, 0.5, size=2)
        star = 4000.0 * kernel_star_image(
            KERNEL, SIZE, center_v, center_u, truncation_px=TRUNCATION_PX
        )
        image = star + 5.0 + rng.normal(0.0, 1.0, star.shape)
        cutouts.append(
            StarCutout(
                image=image,
                valid=np.ones(image.shape, dtype=bool),
                v=round(center_v),
                u=round(center_u),
                sky_var=1.0,
                gain_e_per_dn=1.0e9,
            )
        )
    fit = fit_psf_kernel(
        cutouts, r0=2.0, n=5.0, start_sigma=0.7, start_w=0.05, truncation_px=TRUNCATION_PX
    )
    assert fit.kernel['sigma_v'] == pytest.approx(KERNEL['sigma_v'], abs=0.01)
    assert fit.kernel['w'] == pytest.approx(KERNEL['w'], abs=0.01)
