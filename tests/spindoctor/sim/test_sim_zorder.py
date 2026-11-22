"""Ring/body depth ordering: range_km is the only compositing key.

A ring_system that carries ``range_km`` orders per pixel against the
bodies' physical ``range_km``, so mixed scenes composite physically.
There is no fallback ordering: objects whose painted pixels overlap must
all carry an explicit ``range_km``, and a scene that stacks them without
one fails loudly instead of being ordered by a guess.  Non-overlapping
objects need no ranges.

The ring-over-body tests check the composed radiance of an inclined ring
around a planet against the transmission-screen closed form: the near arm
dims the disc by ``exp(-tau/mu)`` and adds its own light, a gap and a faint
ring let the disc show through, and the far arm hides behind the disc.
"""

import math
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene, quiet_noise, ring_system_block

from spindoctor.sim.render.driver import render_scene
from spindoctor.sim.render.radiance.compose import compose_scene_radiance
from spindoctor.sim.render.stages import new_sim_frame
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.sim.scene.validate import validate_sim_params
from spindoctor.support.types import NDArrayFloatType


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


# ---------------------------------------------------------------------------
# Ring over body: an inclined ring around a planet, composited by depth.
# ---------------------------------------------------------------------------

# A planet of radius 40 px at the center of a 128 px frame, with a ring around
# it opened 30 degrees toward the observer and the Sun.  Planet and ring share
# one range, so the ring's near arm (below the center, larger v) lies in front
# of the planet and its far arm behind.  The sheet spans ring-plane radii 50 to
# 90 px with a cleared gap from 65 to 75 px; the 30-degree opening halves the
# radii along v, so both arms cross the disc.
_RB_SIZE = 128
_RB_CENTER = 64.0
_RB_RADIUS = 40.0
_RB_OPENING_DEG = 30.0
_RB_TAU = 0.5
_RB_FAINT_TAU = 0.02
_RB_ALBEDO = 0.6

# Probe pixels, each at least 4 px from the limb (wholly on or wholly off the
# disc) and at least 6 ring-plane px from every ring edge, where the sheet's
# optical depth is exactly its tau.  Ring-plane radius is
# hypot(du, dv / sin(30 deg)) about the ring center, at the pixel center.
# Near arm, ring-plane radius 57, inside the sheet, 28.5 px inside the limb:
_RB_FRONT = (92, 64)
# Near arm, ring-plane radius 71, inside the gap, 4.5 px inside the limb:
_RB_GAP = (99, 64)
# Far arm, ring-plane radius 57, inside the sheet, 28.5 px inside the limb:
_RB_BEHIND = (35, 64)
# Far arm, ring-plane radius 58.4, inside the sheet, 7.7 px outside the limb:
_RB_BESIDE = (44, 107)


def _rb_orbit(a_px: float) -> dict[str, float]:
    """A circular, non-precessing ring orbit of semimajor axis ``a_px``."""
    return {'a_px': a_px, 'ae_px': 0.0, 'long_peri_deg': 0.0, 'rate_peri_deg_per_day': 0.0}


def _rb_radiance(*, body: bool, ring: bool, tau: float = _RB_TAU) -> NDArrayFloatType:
    """Compose the ring-over-body scene's noise-free radiance.

    Parameters:
        body: Whether the scene holds the planet.
        ring: Whether the scene holds the ring system.
        tau: The optical depth of the sheet; the gap clears exactly this much.

    Returns:
        The composed signal plane at oversample 1.
    """
    blocks: dict[str, Any] = {}
    if body:
        blocks['bodies'] = [
            disc_body(
                _RB_CENTER,
                _RB_CENTER,
                _RB_RADIUS,
                illumination_angle=0.0,
                phase_angle=0.0,
                range_km=1.0e6,
            )
        ]
    if ring:
        features = [
            {
                'name': 'SHEET',
                'kind': 'ringlet',
                'tau': tau,
                'width_px': 40.0,
                'albedo': _RB_ALBEDO,
                'phase_g': 0.0,
                'orbit': _rb_orbit(50.0),
            },
            {'name': 'GAP', 'kind': 'gap', 'tau': tau, 'width_px': 10.0, 'orbit': _rb_orbit(65.0)},
        ]
        blocks['ring_system'] = ring_system_block(
            _RB_CENTER,
            _RB_CENTER,
            features,
            opening_deg_obs=_RB_OPENING_DEG,
            opening_deg_sun=_RB_OPENING_DEG,
            range_km=1.0e6,
            km_per_pixel=1000.0,
        )
    scene = validate_sim_params(frame_scene(size=_RB_SIZE, seed=3, exposure_sec=1.0, **blocks))
    frame = new_sim_frame(_RB_SIZE, _RB_SIZE)
    compose_scene_radiance(frame, params=scene)
    return frame.signal


def _rb_sheet_light(tau: float) -> float:
    """The sheet's own light at a pixel of optical depth ``tau``, in closed form.

    Lit single scattering with an isotropic phase function:
    ``A/4 * mu0/(mu0 + mu) * (1 - exp(-tau * (1/mu0 + 1/mu)))``.
    """
    mu = mu0 = math.sin(math.radians(_RB_OPENING_DEG))
    return _RB_ALBEDO / 4.0 * mu0 / (mu0 + mu) * (1.0 - math.exp(-tau * (1.0 / mu0 + 1.0 / mu)))


def _rb_transmission(tau: float) -> float:
    """The fraction ``exp(-tau/mu)`` of the light behind the sheet that passes it."""
    return math.exp(-tau / math.sin(math.radians(_RB_OPENING_DEG)))


def test_a_ring_in_front_of_a_body_dims_it_and_adds_its_own_light() -> None:
    """The near arm shows its own light plus the disc dimmed by exp(-tau/mu)."""
    planet = _rb_radiance(body=True, ring=False)
    both = _rb_radiance(body=True, ring=True)
    expected = _rb_sheet_light(_RB_TAU) + _rb_transmission(_RB_TAU) * planet[_RB_FRONT]
    assert float(planet[_RB_FRONT]) > 0.0
    assert float(both[_RB_FRONT]) == pytest.approx(expected, rel=1e-12)


def test_a_gap_in_front_of_a_body_shows_it_undimmed() -> None:
    """Through a cleared gap in the near arm the disc keeps its full light."""
    planet = _rb_radiance(body=True, ring=False)
    both = _rb_radiance(body=True, ring=True)
    assert float(planet[_RB_GAP]) > 0.0
    assert float(both[_RB_GAP]) == pytest.approx(float(planet[_RB_GAP]), rel=1e-12)


def test_a_ring_behind_a_body_is_hidden_by_it() -> None:
    """Where the disc wholly covers the far arm, only the disc's light shows."""
    planet = _rb_radiance(body=True, ring=False)
    both = _rb_radiance(body=True, ring=True)
    assert float(both[_RB_BEHIND]) == pytest.approx(float(planet[_RB_BEHIND]), rel=1e-12)


def test_a_ring_behind_a_body_shows_beside_it() -> None:
    """Off the disc the far arm shows its own light over the empty sky."""
    both = _rb_radiance(body=True, ring=True)
    assert float(both[_RB_BESIDE]) == pytest.approx(_rb_sheet_light(_RB_TAU), rel=1e-12)


def test_a_faint_ring_in_front_of_a_body_shows_it_through() -> None:
    """A faint near arm passes nearly all of the disc's light and adds a little."""
    planet = _rb_radiance(body=True, ring=False)
    both = _rb_radiance(body=True, ring=True, tau=_RB_FAINT_TAU)
    expected = _rb_sheet_light(_RB_FAINT_TAU) + _rb_transmission(_RB_FAINT_TAU) * planet[_RB_FRONT]
    assert float(both[_RB_FRONT]) == pytest.approx(expected, rel=1e-12)
