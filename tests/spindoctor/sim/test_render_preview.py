"""The scene editor's fast preview render (``render_combined_model(..., preview=True)``).

The preview draws on the detector grid, without limb anti-aliasing or the distortion
warp.  These tests render one scene that carries a PSF, a distortion warp, a body, and
a star, and check that the preview has the full render's image shape, is
deterministic, leaves the caller's scene alone, and never comes back from the cache in
place of a full render.
"""

import copy
from typing import Any

import numpy as np
from tests.spindoctor.sim.scenes import frame_scene, star_entry

from spindoctor.sim.render.driver import clear_render_caches, render_combined_model


def _scene() -> dict[str, Any]:
    """A Cassini frame with a PSF, a distortion warp, a body, and a star."""
    body = {
        'name': 'B',
        'center_v': 30.3,
        'center_u': 22.6,
        'semi_a_px': 13.0,
        'semi_b_px': 12.0,
        'semi_c_px': 12.0,
        'illumination_angle': 30.0,
        'phase_angle': 40.0,
    }
    return frame_scene(
        size=(64, 48),
        seed=11,
        optics={
            'psf': {'sigma_v': 1.2, 'sigma_u': 1.2},
            'distortion': {'k1': 0.02, 'k2': 0.0},
        },
        bodies=[body],
        stars=[star_entry(8.5, 40.5, 5.0)],
    )


def _preview(scene: dict[str, Any]) -> Any:
    """The preview image of ``scene``."""
    image, _ = render_combined_model(scene, preview=True)
    return image


def _full(scene: dict[str, Any]) -> Any:
    """The full render's image of ``scene``."""
    image, _ = render_combined_model(scene)
    return image


def test_the_preview_has_the_full_render_shape() -> None:
    """The preview is on the detector grid, the same size as the full render."""
    assert _preview(_scene()).shape == _full(_scene()).shape


def test_the_preview_is_deterministic() -> None:
    """Two previews of one scene, with the caches cleared between, are identical."""
    first = _preview(_scene())
    clear_render_caches()
    assert np.array_equal(_preview(_scene()), first)


def test_the_preview_leaves_the_scene_unchanged() -> None:
    """Rendering a preview doesn't modify the caller's scene."""
    scene = _scene()
    before = copy.deepcopy(scene)
    _preview(scene)
    assert scene == before


def test_the_preview_draws_on_the_detector_grid() -> None:
    """The preview of a PSF scene is the same as with its oversampling set to 1."""
    detector_grid = _scene()
    detector_grid['oversample'] = 1
    assert np.array_equal(_preview(_scene()), _preview(detector_grid))


def test_the_preview_skips_the_distortion_warp() -> None:
    """The preview of a scene is the same with or without its distortion block."""
    undistorted = _scene()
    del undistorted['optics']['distortion']
    assert np.array_equal(_preview(_scene()), _preview(undistorted))


def test_the_preview_draws_limbs_without_anti_aliasing() -> None:
    """The preview of a body is the same at any anti-aliasing setting."""
    sharp = _scene()
    sharp['bodies'][0]['anti_aliasing'] = 0.0
    assert np.array_equal(_preview(_scene()), _preview(sharp))


def test_a_full_render_after_a_preview_is_the_full_render() -> None:
    """A preview in the cache is never handed out for a full render of the scene."""
    clear_render_caches()
    expected = _full(_scene())
    clear_render_caches()
    preview = _preview(_scene())
    full = _full(_scene())
    assert np.array_equal(full, expected)
    assert not np.array_equal(full, preview)
