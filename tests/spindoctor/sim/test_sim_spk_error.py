"""Planted spacecraft-ephemeris (SPK) error: parallax on bodies and rings.

A scene-level SPK error displaces every body and ring feature by the planted
image-plane vector scaled by ``reference_range_km / range_km`` (near objects
move more), while stars stay put.  The navigator predicts from the unshifted
catalog geometry: the drawn error never crosses the information boundary.
"""

from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import frame_scene, quiet_noise, star_entry

from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.scene.boundary import build_nav_params
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.validate import validate_sim_params


def _centroid_v(image: np.ndarray, v_slice: slice, u_slice: slice) -> float:
    """Intensity-weighted v centroid over a sub-window."""
    sub = image[v_slice, u_slice]
    vv = np.mgrid[v_slice.start : v_slice.stop, u_slice.start : u_slice.stop][0]
    total = float(sub.sum())
    return float((sub * vv).sum()) / total


def _body_scene(*, range_km: float, spk: dict[str, Any] | None) -> dict[str, Any]:
    """A single interior body at a given physical range, optional SPK error."""
    scene = frame_scene(
        size=60,
        seed=2,
        noise=quiet_noise(),
        bodies=[
            {
                'name': 'B',
                'center_v': 30.0,
                'center_u': 30.0,
                'semi_a_px': 9.0,
                'semi_b_px': 9.0,
                'range_km': range_km,
            }
        ],
    )
    if spk is not None:
        scene['spk_error'] = spk
    return scene


def test_near_body_displaces_more_than_far_body() -> None:
    """Parallax scales as 1/range: the nearer body shifts twice as far."""
    spk = {'dv_px': 4.0, 'du_px': 0.0, 'reference_range_km': 100000.0}
    base_near, _ = render_scene(_body_scene(range_km=100000.0, spk=None))
    near, _ = render_scene(_body_scene(range_km=100000.0, spk=spk))
    base_far, _ = render_scene(_body_scene(range_km=200000.0, spk=None))
    far, _ = render_scene(_body_scene(range_km=200000.0, spk=spk))
    window = (slice(20, 45), slice(20, 41))
    shift_near = _centroid_v(near, *window) - _centroid_v(base_near, *window)
    shift_far = _centroid_v(far, *window) - _centroid_v(base_far, *window)
    assert abs(shift_near - 4.0) < 0.3
    assert abs(shift_far - 2.0) < 0.3


def test_stars_are_not_displaced_by_spk_error() -> None:
    """The star field stays put while the body parallax-shifts."""
    spk = {'dv_px': 5.0, 'du_px': 0.0, 'reference_range_km': 100000.0}
    scene = _body_scene(range_km=100000.0, spk=spk)
    scene['stars'] = [star_entry(12.0, 48.0, 0.0)]
    plain = dict(scene)
    plain.pop('spk_error')
    with_spk, _ = render_scene(scene)
    without, _ = render_scene(plain)
    star_window = (slice(6, 19), slice(42, 55))
    assert abs(_centroid_v(with_spk, *star_window) - _centroid_v(without, *star_window)) < 0.05


def test_spk_error_is_absent_from_nav_params() -> None:
    """The planted spacecraft error never reaches the navigator's view."""
    scene = _body_scene(
        range_km=100000.0,
        spk={'dv_px': 3.0, 'du_px': 1.0, 'reference_range_km': 100000.0},
    )
    nav = build_nav_params(scene)
    assert 'spk_error' not in nav


def test_spk_error_requires_range_km_on_bodies() -> None:
    """A body-bearing scene with SPK error must give each body a range_km."""
    scene = frame_scene(
        size=32,
        bodies=[{'name': 'B', 'center_v': 16.0, 'center_u': 16.0, 'semi_a_px': 4.0}],
        spk_error={'dv_px': 1.0, 'du_px': 0.0, 'reference_range_km': 1000.0},
    )
    with pytest.raises(SimSceneValidationError, match='range_km'):
        validate_sim_params(scene)
