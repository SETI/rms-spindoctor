"""The ellipsoid path's fallback seeds and its shape cache.

A body drawn with neither a scene seed nor a seed of its own seeds its craters, its
relief, and its albedo noise from a digest.  The craters' digest hashes the body's
widths and center, and the other streams' digest its widths, index, and name.  The
digest hashes the keys' ``repr``, so the tests compare that: ``16`` and ``16.0`` are
different keys.  Each body's drawing is cached at the reference center until the
cache is cleared.
"""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render.bodies import render as body_render
from spindoctor.sim.render.bodies.spec import BodySpec, body_identity_seed, body_spec_from_params


def _record_fallback_keys(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    """Record the fallback key of every identity seed the dispatcher asks for.

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

    monkeypatch.setattr(body_render, 'body_identity_seed', recording)
    return keys


def _render_seedless_body(**keys: Any) -> None:
    """Render one seedless body, index 3, at oversample 2 onto a 64 x 96 frame.

    Parameters:
        **keys: The body's surface keys.
    """
    params = {'name': 'b', 'semi_a_px': 20.0, 'semi_b_px': 18.0, 'semi_c_px': 16.0, **keys}
    img = np.zeros((64, 96))
    body_render.render_single_body(
        img,
        body_spec_from_params(params, index=3),
        0.0,
        offset_u=0.0,
        body_index=3,
        ref_center_v=32.0,
        ref_center_u=48.0,
        oversample=2,
    )


def test_crater_fallback_key_is_the_widths_and_the_detector_grid_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The crater seed falls back on the full widths and the detector-grid center."""
    keys = _record_fallback_keys(monkeypatch)
    _render_seedless_body(crater_fill=0.4)
    assert repr(keys) == '[(40.0, 36.0, 32.0, (16.0, 24.0))]'


def test_surface_fallback_key_is_the_widths_index_and_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The relief and albedo seeds fall back on the full widths, index, and name."""
    keys = _record_fallback_keys(monkeypatch)
    _render_seedless_body(limb_relief_rms=0.02)
    assert repr(keys) == "[(40.0, 36.0, 32.0, 3, 'B')]"


def test_clear_ellipsoid_shape_caches_empties_the_shape_cache() -> None:
    """After a body is drawn, clearing leaves the ellipsoid shape cache empty."""
    _render_seedless_body()
    assert body_render._render_ellipsoid_shape_cached.cache_info().currsize > 0
    body_render.clear_ellipsoid_shape_caches()
    assert body_render._render_ellipsoid_shape_cached.cache_info().currsize == 0
