"""Tests of what the realism match takes from the largest body of a real frame.

The real frame here is a stand-in: its offset backplane is a sphere's analytic angles at
the image's pixels, so nothing reads holdings or SPICE and the file runs in the plain
unit tier.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import scipy.ndimage

from spindoctor.feature.geometry import LimbPolyline
from spindoctor.sim.render.bodies.photometry import shade_surface
from tests.integration import sim_realism_body
from tests.integration.sim_realism_body import (
    EDGE_CLEARANCE_PX,
    disc_is_clipped,
    offset_backplane,
    real_body,
    real_normal_albedo,
)
from tests.integration.sim_realism_scenes import MATCHED_BODY_LAW, matched_scene

_SIZE = 100
_CENTER = 50.0
_RADIUS = 25.0
_PHASE_DEG = 30.0
_METADATA = {'body:MOON': {'predicted_diameter_px': 50.0, 'phase_angle_deg': _PHASE_DEG}}


def _sphere_angles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The sphere's mask and its incidence and emission angles at the image's pixels.

    The Sun lies toward -v and +u, so the disc is not symmetric across either axis.
    """
    vv, uu = np.mgrid[0:_SIZE, 0:_SIZE].astype(np.float64)
    dv = (vv - _CENTER) / _RADIUS
    du = (uu - _CENTER) / _RADIUS
    on_body = dv**2 + du**2 < 1.0
    nz = np.sqrt(np.clip(1.0 - dv**2 - du**2, 0.0, 1.0))
    phase = math.radians(_PHASE_DEG)
    toward_sun = (-dv + du) / math.sqrt(2.0)
    mu0 = np.clip(toward_sun * math.sin(phase) + nz * math.cos(phase), -1.0, 1.0)
    return on_body, np.arccos(mu0), np.arccos(nz)


class _Backplane:
    """The three backplanes of one body that the measurement reads, offset applied."""

    def __init__(self) -> None:
        self.on_body, self.incidence, self.emission = _sphere_angles()

    def where_intercepted(self, _body: str) -> Any:
        return SimpleNamespace(vals=self.on_body)

    def incidence_angle(self, _body: str) -> Any:
        return SimpleNamespace(mvals=np.ma.array(self.incidence, mask=~self.on_body))

    def emission_angle(self, _body: str) -> Any:
        return SimpleNamespace(mvals=np.ma.array(self.emission, mask=~self.on_body))


def _real_frame(albedo: float, *, rim_if: float | None = None) -> Any:
    """A frame of one body and its offset backplane.

    Parameters:
        albedo: The normal albedo the disc is drawn at.
        rim_if: When given, the I/F of every body pixel that one step less of erosion
            than the measurement takes would keep out.
    """
    backplane = _Backplane()
    shaded = shade_surface(
        np.cos(backplane.incidence),
        np.cos(backplane.emission),
        law=MATCHED_BODY_LAW,
        phase_angle=math.radians(_PHASE_DEG),
    )
    surface = albedo * np.where(backplane.on_body, shaded, 0.0)
    if rim_if is not None:
        kept = scipy.ndimage.binary_erosion(backplane.on_body, iterations=EDGE_CLEARANCE_PX - 1)
        surface[backplane.on_body & ~kept] = rim_if
    return SimpleNamespace(bp=backplane, data=surface)


def _interior_in_image() -> np.ndarray:
    """The disc pixels the measurement reads."""
    on_body, _, _ = _sphere_angles()
    return np.asarray(
        scipy.ndimage.binary_erosion(on_body, iterations=EDGE_CLEARANCE_PX), dtype=bool
    )


def _clipped_mask(n_clipped: int) -> np.ndarray:
    """A mask that clips the first ``n_clipped`` disc pixels the measurement reads."""
    rows, cols = np.nonzero(_interior_in_image())
    mask = np.zeros((_SIZE, _SIZE), dtype=bool)
    mask[rows[:n_clipped], cols[:n_clipped]] = True
    return mask


def _limb_feature(model: str) -> Any:
    """A feature carrying a limb polyline from ``model``."""
    geometry = LimbPolyline(
        vertices_vu=np.zeros((2, 2)),
        normals_vu=np.ones((2, 2)),
        sigma_normal_per_vertex_px=np.full(2, 0.5),
        sigma_tangent_per_vertex_px=np.full(2, 0.5),
        bbox_extfov_vu=(0, 0, 1, 1),
    )
    return SimpleNamespace(source_model=model, geometry=geometry)


def test_the_disc_measures_the_albedo_it_is_drawn_at() -> None:
    """A body drawn at 0.6 times its law measures 0.6."""
    frame = _real_frame(0.6)
    albedo = real_normal_albedo(frame.data, frame.bp, 'MOON', phase_angle_deg=_PHASE_DEG)
    assert albedo == pytest.approx(0.6, rel=1e-12)


def test_the_albedo_leaves_out_the_pixels_something_hides() -> None:
    """Disc pixels a ring covers, drawn twice as bright, leave the albedo at 0.6.

    The covered rows hold more than half of the disc, so they would set the median.
    """
    frame = _real_frame(0.6)
    under_ring = np.zeros((_SIZE, _SIZE), dtype=bool)
    under_ring[:52, :] = True
    frame.data[under_ring] *= 2.0
    body = real_body(
        frame,
        [],
        _METADATA,
        backplane=frame.bp,
        calibrated=True,
        clipped=None,
        hidden=lambda _name: under_ring,
    )
    assert body is not None
    assert body.normal_albedo == pytest.approx(0.6, rel=1e-12)


def test_full_scale_pixels_that_something_hides_do_not_clip_the_disc() -> None:
    """A saturated body over more than half of the disc leaves the disc unclipped."""
    frame = _real_frame(0.6)
    in_front = np.zeros((_SIZE, _SIZE), dtype=bool)
    in_front[:52, :] = True
    body = real_body(
        frame,
        [],
        _METADATA,
        backplane=frame.bp,
        calibrated=True,
        clipped=in_front,
        hidden=lambda _name: in_front,
    )
    assert body is not None
    assert body.disc_clipped is False


def test_the_offset_backplane_applies_the_offset_unrounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image pixel (2, 5) at offset (0.3, -1.7) samples the geometry at (1.7, 6.7).

    The geometry's grid is in pixel-corner (u, v), so that point is (7.2, 2.2).
    """
    monkeypatch.setattr(sim_realism_body, 'Meshgrid', lambda _fov, uv: uv)
    monkeypatch.setattr(sim_realism_body, 'Backplane', lambda _obs, *, meshgrid: meshgrid)
    grid: Any = offset_backplane(SimpleNamespace(data=np.zeros((4, 6)), fov=None), (0.3, -1.7))
    assert tuple(grid.vals[2, 5]) == pytest.approx((7.2, 2.2))


def test_the_edge_of_the_disc_does_not_count() -> None:
    """A bright rim just inside the edge, where optics blur, leaves the albedo alone."""
    frame = _real_frame(0.6, rim_if=9.0)
    albedo = real_normal_albedo(frame.data, frame.bp, 'MOON', phase_angle_deg=_PHASE_DEG)
    assert albedo == pytest.approx(0.6, rel=1e-12)


def test_a_disc_with_one_percent_of_its_pixels_at_full_scale_is_clipped() -> None:
    """1% of the disc's pixels at full scale makes the disc clipped."""
    n_disc = int(np.count_nonzero(_interior_in_image()))
    clipped = _clipped_mask(math.ceil(0.01 * n_disc))
    assert disc_is_clipped(_real_frame(0.6).bp, 'MOON', clipped=clipped) is True


def test_a_disc_with_fewer_pixels_at_full_scale_is_not_clipped() -> None:
    """Just under 1% of the disc's pixels at full scale, as hot pixels give, is not."""
    n_disc = int(np.count_nonzero(_interior_in_image()))
    clipped = _clipped_mask(math.ceil(0.01 * n_disc) - 1)
    assert disc_is_clipped(_real_frame(0.6).bp, 'MOON', clipped=clipped) is False


def _clipped_real_body() -> Any:
    """The largest body of a frame whose whole disc reads the full scale."""
    frame = _real_frame(0.6)
    return real_body(
        frame,
        [_limb_feature('body:MOON')],
        _METADATA,
        backplane=frame.bp,
        calibrated=True,
        clipped=_interior_in_image(),
    )


def test_a_clipped_disc_gives_no_albedo() -> None:
    """A clipped disc's I/F is the clip level, so it measures no albedo."""
    body = _clipped_real_body()
    assert body is not None
    assert body.normal_albedo is None


def test_a_clipped_disc_gives_no_limb_peak() -> None:
    """A clipped disc's limb rises to a plateau, so no model gives its gradient peak."""
    body = _clipped_real_body()
    assert body is not None
    assert body.limb_peak_model is None


def test_a_body_whose_limb_the_navigator_does_not_place_is_marked() -> None:
    """A frame whose only limb belongs to another body has none for the largest body."""
    frame = _real_frame(0.6)
    body = real_body(
        frame,
        [_limb_feature('body:OTHER')],
        _METADATA,
        backplane=frame.bp,
        calibrated=True,
        clipped=None,
    )
    assert body is not None
    assert body.limb_emitted is False


@pytest.mark.parametrize('scene_class', ['body_full_fov', 'ring_plus_body'])
def test_a_matched_body_takes_the_measured_normal_albedo(scene_class: str) -> None:
    """A matched limb or ring scene gives its body the normal albedo it is handed."""
    scene = matched_scene(
        'N1', scene_class, 'coiss_calib_nac', 0.5, size_vu=(256, 256), normal_albedo=0.37
    )
    assert scene['bodies'][0]['normal_albedo'] == 0.37


@pytest.mark.parametrize('scene_class', ['body_full_fov', 'ring_plus_body'])
def test_a_matched_body_is_lit_from_the_direction_it_is_given(scene_class: str) -> None:
    """A matched limb or ring scene lights its body from the real body's direction."""
    scene = matched_scene(
        'N1', scene_class, 'coiss_calib_nac', 0.5, size_vu=(256, 256), illumination_deg=212.5
    )
    assert scene['bodies'][0]['illumination_angle'] == 212.5


def test_an_unmeasured_matched_body_stays_white() -> None:
    """Without a measured albedo, the matched body leaves the key out."""
    scene = matched_scene('N1', 'ring_plus_body', 'coiss_calib_nac', 0.5, size_vu=(256, 256))
    assert 'normal_albedo' not in scene['bodies'][0]
