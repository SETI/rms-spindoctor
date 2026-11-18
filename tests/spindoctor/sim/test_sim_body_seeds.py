"""Per-body crater seeds in combined scenes (SIM-1).

Every body in a combined scene derives its own crater sub-seed from the scene's
crater seed plus a stable per-body identity (scene index and name), so two
bodies with identical geometry render independent crater patterns and occupy
distinct entries in the body shape cache instead of colliding on one.
"""

from typing import Any

import numpy as np
from tests.spindoctor.sim.scenes import frame_scene, quiet_noise

from spindoctor.sim.render.bodies import render as body_render
from spindoctor.sim.render.driver import clear_render_caches, render_combined_model

_BODY_GEOMETRY: dict[str, Any] = {
    'semi_a_px': 20.0,
    'semi_b_px': 16.0,
    'semi_c_px': 16.0,
    'crater_fill': 0.4,
}


def _clear_render_caches() -> None:
    """Drop every render-path lru_cache so RNG paths re-run."""
    clear_render_caches()


def _twin_body_scene() -> dict[str, Any]:
    """A noiseless scene with two cratered bodies of identical geometry."""
    return frame_scene(
        size=(96, 160),
        seed=42,
        noise=quiet_noise(),
        bodies=[
            {'name': 'ALPHA', 'center_v': 48.0, 'center_u': 44.0, **_BODY_GEOMETRY},
            {'name': 'BETA', 'center_v': 48.0, 'center_u': 116.0, **_BODY_GEOMETRY},
        ],
    )


def _body_crops(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-sized crops centered on the two bodies (integer centers, so the
    sub-pixel positioning shift is exact and the crops are directly comparable)."""
    return img[26:70, 22:66], img[26:70, 94:138]


def test_twin_bodies_get_different_crater_patterns() -> None:
    """Two identical-geometry bodies in one scene render distinct craters."""
    _clear_render_caches()
    img, _ = render_combined_model(_twin_body_scene())
    crop_alpha, crop_beta = _body_crops(img)
    assert not np.array_equal(crop_alpha, crop_beta)


def test_twin_bodies_are_both_rendered() -> None:
    """Both body crops contain lit pixels (guards the difference test)."""
    _clear_render_caches()
    img, _ = render_combined_model(_twin_body_scene())
    crop_alpha, crop_beta = _body_crops(img)
    assert crop_alpha.max() > 0.0
    assert crop_beta.max() > 0.0


def test_twin_bodies_occupy_distinct_shape_cache_entries() -> None:
    """The shape cache holds one entry per body rather than colliding on one."""
    _clear_render_caches()
    render_combined_model(_twin_body_scene())
    info = body_render._render_body_shape_cached.cache_info()
    assert info.currsize == 2


def test_twin_bodies_do_not_share_a_cached_shape_array() -> None:
    """The second body never receives the first body's cached array (no hits)."""
    _clear_render_caches()
    render_combined_model(_twin_body_scene())
    info = body_render._render_body_shape_cached.cache_info()
    assert info.hits == 0
