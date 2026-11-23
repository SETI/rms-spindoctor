"""Clearing the star stage's caches.

The star field and the sky-count plane are each cached on their parameters, and
:func:`~spindoctor.sim.render.radiance.stars.clear_star_caches` drops both, so a
determinism test can render them again from scratch.
"""

import numpy as np

from spindoctor.sim.render.radiance import stars
from spindoctor.sim.render.radiance.stars import clear_star_caches, render_sky_counts, render_stars


def _fill_star_caches() -> None:
    """Render one star and one sky-count plane, which fills both caches."""
    render_stars(
        np.zeros((32, 32)),
        [{'v': 16.0, 'u': 16.0, 'vmag': 8.0}],
        0.0,
        shift_u=0.0,
        zero_point=1.0e7,
        exposure_sec=1.0,
        rendered_sigma=1.0,
    )
    render_sky_counts(
        np.zeros((32, 32)),
        seed=5,
        a=-3.1,
        b=0.34,
        density_factor=2000.0,
        pixel_scale_arcsec=1.237,
        faint_cutoff_mag=15.0,
        zero_point=1.0e7,
        exposure_sec=1.0,
        diffuse_flux_per_px=0.0,
        oversample=1,
    )


def test_clear_star_caches_empties_both_caches() -> None:
    """After a render fills them, clearing leaves the star and sky-count caches empty."""
    _fill_star_caches()
    assert stars._render_stars_cached.cache_info().currsize > 0
    assert stars._render_sky_counts_cached.cache_info().currsize > 0
    clear_star_caches()
    assert stars._render_stars_cached.cache_info().currsize == 0
    assert stars._render_sky_counts_cached.cache_info().currsize == 0
