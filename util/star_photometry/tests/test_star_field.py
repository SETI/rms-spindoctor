"""Unit tests for the PSF kernel fit of star cutouts."""

from __future__ import annotations

import numpy as np
import pytest
from util.star_photometry.star_field import StarCutout, fit_psf_kernel, kernel_star_image

from spindoctor.sim.render.optics.psf import apply_psf
from spindoctor.sim.render.radiance.stars import render_stars

# A kernel of the simulator's form, narrow enough that pixel sampling matters.
KERNEL = {'sigma_v': 0.45, 'sigma_u': 0.45, 'w': 0.2, 'r0': 2.0, 'n': 5.0}
TRUNCATION_PX = 8
OVERSAMPLE = 4
SIZE = 21


def _stars(
    kernel: dict[str, float],
    *,
    flux: float,
    gain_e_per_dn: float,
    seed: int,
    hit_dn: float = 0.0,
) -> list[StarCutout]:
    """Return eight noisy stars drawn with a kernel, each near the middle of its cutout.

    Each star's electrons take shot noise at the gain, and the sky of 5 DN takes noise
    of 1 DN.

    Parameters:
        kernel: The kernel the stars are drawn with.
        flux: Each star's flux, in DN.
        gain_e_per_dn: Electrons per DN.
        seed: The seed of the star positions and the noise.
        hit_dn: A hit of this many DN is added 4 px below each star's center.

    Returns:
        The cutouts, each starting its fit at the whole pixel nearest its star.
    """
    rng = np.random.default_rng(seed)
    cutouts = []
    for _ in range(8):
        center_v, center_u = 10.0 + rng.uniform(-0.5, 0.5, size=2)
        star = flux * kernel_star_image(
            kernel, SIZE, center_v, center_u, truncation_px=TRUNCATION_PX
        )
        image = rng.poisson(star * gain_e_per_dn) / gain_e_per_dn
        image = image + 5.0 + rng.normal(0.0, 1.0, star.shape)
        image[round(center_v) + 4, round(center_u)] += hit_dn
        cutouts.append(
            StarCutout(
                image=image,
                valid=np.ones(image.shape, dtype=bool),
                v=round(center_v),
                u=round(center_u),
                sky_var=1.0,
                gain_e_per_dn=gain_e_per_dn,
            )
        )
    return cutouts


def test_a_kernel_star_is_the_simulators_render() -> None:
    """A star drawn for the fit equals the simulator's render of a star at that place.

    The star sits between render-grid points, where the simulator shares its light
    between the four around it, then blurs the plane with the kernel and averages each
    block of render-grid points into one detector pixel.
    """
    # The simulator's star position is a pixel corner coordinate on the render grid.
    corner_v, corner_u = 10.37, 9.81
    plane = np.zeros((SIZE * OVERSAMPLE, SIZE * OVERSAMPLE))
    render_stars(
        plane,
        [{'name': 'S', 'v': corner_v * OVERSAMPLE, 'u': corner_u * OVERSAMPLE, 'vmag': 0.0}],
        0.0,
        shift_u=0.0,
        zero_point=1.0,
        exposure_sec=1.0,
        rendered_sigma=1.0,
        oversample=OVERSAMPLE,
    )
    apply_psf(
        np.zeros_like(plane),
        plane,
        **KERNEL,
        truncation_px=TRUNCATION_PX,
        oversample=OVERSAMPLE,
    )
    rendered = plane.reshape(SIZE, OVERSAMPLE, SIZE, OVERSAMPLE).mean(axis=(1, 3))
    # The center of array element [i, j] is the pixel corner coordinate (i + 0.5, j + 0.5).
    drawn = kernel_star_image(
        KERNEL,
        SIZE,
        corner_v - 0.5,
        corner_u - 0.5,
        truncation_px=TRUNCATION_PX,
        oversample=OVERSAMPLE,
    )
    np.testing.assert_allclose(drawn, rendered, rtol=0.0, atol=1e-12)


def test_the_fit_recovers_a_planted_kernel() -> None:
    """Noisy stars drawn with a known core and wing give that core and wing back."""
    cutouts = _stars(KERNEL, flux=4000.0, gain_e_per_dn=1.0e9, seed=1258)
    fit = fit_psf_kernel(
        cutouts, r0=2.0, n=5.0, start_sigma=0.7, start_w=0.05, truncation_px=TRUNCATION_PX
    )
    assert fit.kernel['sigma_v'] == pytest.approx(KERNEL['sigma_v'], abs=0.01)
    assert fit.kernel['w'] == pytest.approx(KERNEL['w'], abs=0.01)


def test_the_fit_weights_each_pixel_by_its_shot_noise() -> None:
    """Stars whose shot noise outweighs the sky's fit to a reduced chi-square near 1."""
    cutouts = _stars(KERNEL, flux=4000.0, gain_e_per_dn=1.0, seed=1259)
    fit = fit_psf_kernel(
        cutouts, r0=2.0, n=5.0, start_sigma=0.7, start_w=0.05, truncation_px=TRUNCATION_PX
    )
    assert fit.reduced_chi2 == pytest.approx(1.0, abs=0.1)


def test_the_fit_leaves_out_hits() -> None:
    """A hit beside each star is left out with its neighbors, and the fit ignores it."""
    cutouts = _stars(KERNEL, flux=4000.0, gain_e_per_dn=1.0e9, seed=1258, hit_dn=200.0)
    fit = fit_psf_kernel(
        cutouts, r0=2.0, n=5.0, start_sigma=0.7, start_w=0.05, truncation_px=TRUNCATION_PX
    )
    # Each hit goes with the four pixels beside it.
    assert fit.outlier_pixels == 8 * 5
    # A hit left in the final fit raises its reduced chi-square far above 1.
    assert fit.reduced_chi2 == pytest.approx(1.0, abs=0.1)


def test_the_fit_does_not_spread_from_an_invalid_hit() -> None:
    """A hit on a pixel already left out takes none of its neighbors with it."""
    cutouts = _stars(KERNEL, flux=4000.0, gain_e_per_dn=1.0e9, seed=1258, hit_dn=200.0)
    for cutout in cutouts:
        cutout.valid[round(cutout.v) + 4, round(cutout.u)] = False
    fit = fit_psf_kernel(
        cutouts, r0=2.0, n=5.0, start_sigma=0.7, start_w=0.05, truncation_px=TRUNCATION_PX
    )
    assert fit.outlier_pixels == 0


@pytest.mark.parametrize(
    'gain_e_per_dn', [pytest.param(1.0e9, id='sky noise'), pytest.param(1.0, id='shot noise')]
)
def test_the_fit_keeps_a_wing_its_kernel_misses(gain_e_per_dn: float) -> None:
    """A broad wing the fixed wing form cannot follow is misfit, but none of it is lost.

    The stars are bright, so a cut against the model alone would take the wing for hits.
    Where shot noise outweighs the sky's, a cut against the model's shot noise alone
    would too.
    """
    broad = {'sigma_v': 0.45, 'sigma_u': 0.45, 'w': 0.25, 'r0': 3.0, 'n': 3.0}
    cutouts = _stars(broad, flux=5000.0, gain_e_per_dn=gain_e_per_dn, seed=1260)
    fit = fit_psf_kernel(
        cutouts, r0=2.0, n=5.0, start_sigma=0.7, start_w=0.05, truncation_px=TRUNCATION_PX
    )
    assert fit.outlier_pixels == 0
