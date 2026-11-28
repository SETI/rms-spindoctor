"""The ellipsoid path's shape cache.

Each body's drawing is cached at the reference center until the cache is cleared.
"""

import numpy as np

from spindoctor.sim.render.bodies import render as body_render
from spindoctor.sim.scene.validate import parse_body


def _render_body() -> None:
    """Render one smooth body, index 3, at oversample 2 onto a 64 x 96 frame."""
    params = {'name': 'b', 'semi_a_px': 20.0, 'semi_b_px': 18.0, 'semi_c_px': 16.0}
    img = np.zeros((64, 96))
    body_render.render_single_body(
        img,
        parse_body(params),
        0.0,
        shift_u=0.0,
        body_streams_seed=7,
        body_index=3,
        oversample=2,
    )


def test_clear_ellipsoid_shape_caches_empties_the_shape_cache() -> None:
    """After a body is drawn, clearing leaves the ellipsoid shape cache empty."""
    _render_body()
    assert body_render._render_ellipsoid_shape_cached.cache_info().currsize > 0
    body_render.clear_ellipsoid_shape_caches()
    assert body_render._render_ellipsoid_shape_cached.cache_info().currsize == 0
