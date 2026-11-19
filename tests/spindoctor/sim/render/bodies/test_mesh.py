"""The mesh renderer's fallback seed and its shape cache.

A mesh body drawn with neither a scene seed nor a seed of its own seeds its relief
and its pose scatter from a digest of its geometry, index, and name, keeping the
digest's low 31 bits.  The body's terrain depends on exactly that seed.  Each body's
drawing is cached at the reference center until the cache is cleared.
"""

import numpy as np
import pytest

from spindoctor.sim.render.bodies import mesh as body_mesh
from spindoctor.sim.render.bodies.spec import BodySpec, body_identity_seed, body_spec_from_params


def _record_fallbacks(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[object, ...], bool]]:
    """Record the fallback key and mask of every identity seed the mesh renderer asks for.

    Parameters:
        monkeypatch: The test's monkeypatch fixture.

    Returns:
        The list the ``(fallback_key, mask_fallback)`` pairs are appended to.
    """
    calls: list[tuple[tuple[object, ...], bool]] = []

    def recording(
        scene_seed: int | None,
        body: BodySpec,
        body_index: int,
        *,
        fallback_key: tuple[object, ...],
        mask_fallback: bool = False,
    ) -> int:
        """Record the fallback key and mask and return the real identity seed.

        Parameters:
            scene_seed: The scene's seed for the bodies' streams.
            body: The body.
            body_index: The body's position in the scene's body list.
            fallback_key: The values the fallback digest hashes.
            mask_fallback: Whether the fallback digest keeps its low 31 bits.

        Returns:
            The body's identity seed.
        """
        calls.append((fallback_key, mask_fallback))
        return body_identity_seed(
            scene_seed, body, body_index, fallback_key=fallback_key, mask_fallback=mask_fallback
        )

    monkeypatch.setattr(body_mesh, 'body_identity_seed', recording)
    return calls


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
        ref_center_v=32.0,
        ref_center_u=32.0,
        body_index=3,
    )


def test_mesh_fallback_is_the_masked_digest_of_widths_index_and_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mesh seed falls back on the full widths, index, and name, masked to 31 bits."""
    calls = _record_fallbacks(monkeypatch)
    _render_seedless_mesh_body()
    assert calls == [((40.0, 36.0, 32.0, 3, 'LUMP'), True)]


def test_clear_mesh_shape_caches_empties_the_shape_cache() -> None:
    """After a mesh body is drawn, clearing leaves the mesh shape cache empty."""
    _render_seedless_mesh_body()
    assert body_mesh._render_mesh_shape_cached.cache_info().currsize > 0
    body_mesh.clear_mesh_shape_caches()
    assert body_mesh._render_mesh_shape_cached.cache_info().currsize == 0
