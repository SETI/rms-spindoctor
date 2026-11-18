"""Ring/body depth ordering: range_km is the only compositing key.

A ring_system that carries ``range_km`` orders per pixel against the
bodies' physical ``range_km``, so mixed scenes composite physically.
There is no fallback ordering: objects whose painted pixels overlap must
all carry an explicit ``range_km``, and a scene that stacks them without
one fails loudly instead of being ordered by a guess.  Non-overlapping
objects need no ranges.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene, quiet_noise, ring_system_block

from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.scene.errors import SimSceneValidationError


def _scene(ring_range_km: float | None) -> dict[str, Any]:
    """A body at 1e6 km overlapped by a face-on ring band at the given range.

    Face-on (B = 90) the line-of-sight depth is uniform across the ring
    plane, so the whole band sits exactly at ``ring_range_km`` and the
    depth ordering against the body is unambiguous everywhere.
    """
    feature = {
        'name': 'R',
        'kind': 'ringlet',
        'tau': 1.5,
        'width_px': 8.0,
        'orbit': {'a_px': 8.0, 'ae_px': 0.0, 'long_peri_deg': 0.0, 'rate_peri_deg_per_day': 0.0},
    }
    ring_system = ring_system_block(30.0, 30.0, [feature])
    if ring_range_km is not None:
        ring_system['range_km'] = ring_range_km
        ring_system['km_per_pixel'] = 1000.0
    return frame_scene(
        size=60,
        seed=5,
        noise=quiet_noise(),
        bodies=[disc_body(30.0, 30.0, 10.0, phase_angle=0.0, range_km=1.0e6)],
        ring_system=ring_system,
    )


def test_ring_with_larger_range_km_renders_behind_the_body() -> None:
    """A ring at range_km 2e6 sorts behind a body at 1e6: the body occludes it."""
    body_only = {k: v for k, v in _scene(2.0e6).items() if k != 'ring_system'}
    body_img, body_meta = render_scene(body_only)
    behind_img, behind_meta = render_scene(_scene(2.0e6))
    mask = behind_meta['body_masks'][0]
    assert bool(mask.any())
    assert np.array_equal(behind_img[mask], body_img[body_meta['body_masks'][0]])


def test_ring_with_smaller_range_km_renders_in_front_of_the_body() -> None:
    """A ring at range_km 5e5 screens the same body and changes its pixels."""
    body_only = {k: v for k, v in _scene(5.0e5).items() if k != 'ring_system'}
    body_img, _ = render_scene(body_only)
    front_img, front_meta = render_scene(_scene(5.0e5))
    mask = front_meta['body_masks'][0]
    assert not np.array_equal(front_img[mask], body_img[mask])


def test_overlapping_ring_without_range_km_is_an_error() -> None:
    """A ring system with no range_km overlapping a body has no defined stacking."""
    scene = _scene(None)
    with pytest.raises(SimSceneValidationError, match=r'overlap.*range_km'):
        render_scene(scene)


def test_overlapping_bodies_without_range_km_are_an_error() -> None:
    """Two overlapping bodies must both carry explicit depths."""
    scene = {k: v for k, v in _scene(2.0e6).items() if k != 'ring_system'}
    del scene['bodies'][0]['range_km']
    scene['bodies'].append(dict(scene['bodies'][0], name='B2', center_v=34.0))
    with pytest.raises(SimSceneValidationError, match=r"'B2' and 'B' overlap"):
        render_scene(scene)


def test_non_overlapping_objects_need_no_range_km() -> None:
    """Paint order is unobservable for disjoint objects, so no ranges needed."""
    scene = _scene(None)
    del scene['bodies'][0]['range_km']
    # Move the body clear of the band (ring outer edge 16 px from center).
    scene['bodies'][0]['center_v'] = 30.0
    scene['bodies'][0]['center_u'] = 52.0
    scene['bodies'][0]['semi_a_px'] = 4.0
    scene['bodies'][0]['semi_b_px'] = 4.0
    scene['bodies'][0]['semi_c_px'] = 4.0
    _img, meta = render_scene(scene)
    assert bool(meta['body_masks'][0].any())
    assert bool(meta['ring_masks'][0].any())
