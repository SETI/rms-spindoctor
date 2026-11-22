"""Clearing every render cache.

:func:`~spindoctor.sim.render.driver.clear_render_caches` drops the scene cache, the
validation cache, and the caches each render stage keeps, so a determinism test can
render from cold caches.
"""

from typing import Any

import pytest
from tests.spindoctor.sim.scenes import frame_scene, star_entry

from spindoctor.sim.render.bodies import mesh as body_mesh
from spindoctor.sim.render.bodies import relief
from spindoctor.sim.render.bodies import render as body_render
from spindoctor.sim.render.driver import _render_scene_cached, clear_render_caches, render_scene
from spindoctor.sim.render.optics import psf
from spindoctor.sim.render.radiance import stars
from spindoctor.sim.scene import validate

# Every cache clear_render_caches drops, by name.
_CACHES: dict[str, Any] = {
    'scene': _render_scene_cached,
    'validation': validate._validate_scene_json,
    'stars': stars._render_stars_cached,
    'sky counts': stars._render_sky_counts_cached,
    'ellipsoid shapes': body_render._render_ellipsoid_shape_cached,
    'mesh shapes': body_mesh._render_mesh_shape_cached,
    'relief grid': relief._unit_relief_grid,
    'albedo grid': relief._unit_albedo_grid,
    'relief slope': relief._unit_relief_slope,
    'relief outline': relief._outline_envelope,
    'psf kernel': psf.psf_kernel,
}


def _scene_filling_every_cache() -> dict[str, Any]:
    """A scene whose render fills every render cache.

    It holds a star, a background sky, a PSF, an ellipsoid with relief and an albedo
    texture, and a mesh body.
    """
    ellipsoid = {
        'name': 'E',
        'center_v': 20.0,
        'center_u': 20.0,
        'semi_a_px': 12.0,
        'semi_b_px': 12.0,
        'limb_relief_rms': 0.02,
        'albedo_texture': {'rms': 0.2, 'corr_px': 4.0},
        'range_km': 1.0e6,
    }
    mesh = {
        'name': 'M',
        'shape_model': 'polyhedral_mesh',
        'center_v': 44.0,
        'center_u': 44.0,
        'semi_a_px': 10.0,
        'semi_b_px': 9.0,
        'semi_c_px': 8.0,
        'range_km': 2.0e6,
    }
    return frame_scene(
        size=64,
        seed=3,
        stars=[star_entry(50.0, 10.0, 8.0)],
        sky_counts={'density_factor': 500.0},
        optics={'psf': {'sigma_v': 0.8, 'sigma_u': 0.8}},
        bodies=[ellipsoid, mesh],
    )


def _fill_caches() -> None:
    """Clear every render cache, then render the scene that fills them all."""
    clear_render_caches()
    render_scene(_scene_filling_every_cache())


@pytest.mark.parametrize('cache', sorted(_CACHES))
def test_render_fills_the_cache(cache: str) -> None:
    """The scene's render leaves an entry in the cache (guards the clearing test).

    Parameters:
        cache: The key into ``_CACHES`` naming the cache.
    """
    _fill_caches()
    assert _CACHES[cache].cache_info().currsize > 0


@pytest.mark.parametrize('cache', sorted(_CACHES))
def test_clear_render_caches_empties_the_cache(cache: str) -> None:
    """After the render, clearing every render cache leaves this cache empty.

    Parameters:
        cache: The key into ``_CACHES`` naming the cache.
    """
    _fill_caches()
    clear_render_caches()
    assert _CACHES[cache].cache_info().currsize == 0
