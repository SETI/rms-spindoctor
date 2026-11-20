"""The mesh renderer's fallback seed and its shape cache.

A mesh body drawn with neither a scene seed nor a seed of its own seeds its relief
and its pose scatter from a digest of its geometry, index, and name.  The digest
hashes the key's ``repr``, so the test compares that.  Each body's
drawing is cached at the reference center until the cache is cleared.
"""

import numpy as np
import pytest

from spindoctor.sim.render.bodies import mesh as body_mesh
from spindoctor.sim.render.bodies.spec import BodySpec, body_identity_seed, body_spec_from_params


def _record_fallback_keys(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    """Record the fallback key of every identity seed the mesh renderer asks for.

    Parameters:
        monkeypatch: The test's monkeypatch fixture.

    Returns:
        The list the fallback keys are appended to, in call order.
    """
    keys: list[tuple[object, ...]] = []

    def recording(
        scene_seed: int | None,
        body: BodySpec,
        body_index: int,
        *,
        fallback_key: tuple[object, ...],
    ) -> int:
        """Record the fallback key and return the real identity seed.

        Parameters:
            scene_seed: The scene's seed for the bodies' streams.
            body: The body.
            body_index: The body's position in the scene's body list.
            fallback_key: The values the fallback digest hashes.

        Returns:
            The body's identity seed.
        """
        keys.append(fallback_key)
        return body_identity_seed(scene_seed, body, body_index, fallback_key=fallback_key)

    monkeypatch.setattr(body_mesh, 'body_identity_seed', recording)
    return keys


def _render_seedless_mesh_body() -> None:
    """Render one seedless mesh body with relief, index 3, onto a 64 x 64 frame."""
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
        body_spec_from_params(params, index=3),
        center_v=32.0,
        center_u=32.0,
        semi_a_px=20.0,
        semi_b_px=18.0,
        semi_c_px=16.0,
        illumination_angle=0.0,
        phase_angle=0.5,
        anti_aliasing=0.0,
        body_index=3,
    )


def test_mesh_fallback_key_is_the_widths_index_and_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mesh seed falls back on the full widths, index, and name."""
    keys = _record_fallback_keys(monkeypatch)
    _render_seedless_mesh_body()
    assert repr(keys) == "[(40.0, 36.0, 32.0, 3, 'LUMP')]"


def test_clear_mesh_shape_caches_empties_the_shape_cache() -> None:
    """After a mesh body is drawn, clearing leaves the mesh shape cache empty."""
    _render_seedless_mesh_body()
    assert body_mesh._render_mesh_shape_cached.cache_info().currsize > 0
    body_mesh.clear_mesh_shape_caches()
    assert body_mesh._render_mesh_shape_cached.cache_info().currsize == 0
