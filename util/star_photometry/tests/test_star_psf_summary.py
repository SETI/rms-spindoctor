"""Unit tests for a kernel's encircled energy on stars rendered like real ones."""

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
