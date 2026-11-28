"""The mesh renderer's shape cache.

Each body's drawing is cached at the reference center until the cache is cleared.
"""

import numpy as np

from spindoctor.sim.render.bodies import mesh as body_mesh
from spindoctor.sim.scene.validate import parse_body


def _render_mesh_body() -> None:
    """Render one mesh body with relief, index 3, onto a 64 x 64 frame."""
    params = {
        'name': 'lump',
        'shape_model': 'polyhedral_mesh',
        'semi_a_px': 20.0,
        'semi_b_px': 18.0,
        'semi_c_px': 16.0,
        'limb_relief_rms': 0.02,
    }
    body_mesh.render_single_mesh_body(
        np.zeros((64, 64)),
        parse_body(params),
        center_v=32.0,
        center_u=32.0,
        semi_a_px=20.0,
        semi_b_px=18.0,
        semi_c_px=16.0,
        illumination_angle=0.0,
        phase_angle=0.5,
        anti_aliasing=0.0,
        body_streams_seed=7,
        body_index=3,
    )


def test_clear_mesh_shape_caches_empties_the_shape_cache() -> None:
    """After a mesh body is drawn, clearing leaves the mesh shape cache empty."""
    _render_mesh_body()
    assert body_mesh._render_mesh_shape_cached.cache_info().currsize > 0
    body_mesh.clear_mesh_shape_caches()
    assert body_mesh._render_mesh_shape_cached.cache_info().currsize == 0
