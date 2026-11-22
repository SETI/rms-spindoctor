"""Tests of the realism match's normal albedo measured on a real frame.

The real frame here is a stand-in: its backplanes are a sphere's analytic angles, so
nothing reads holdings or SPICE and the file runs in the plain unit tier.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render.bodies.photometry import shade_surface
from tests.integration.sim_realism_albedo import real_normal_albedo
from tests.integration.sim_realism_scenes import matched_scene

_SIZE = 100
_CENTER = 50.0
_RADIUS = 25.0
_PHASE_DEG = 30.0
_SHIFT_VU = (4, -3)


def _sphere_angles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The predicted sphere's mask and its incidence and emission angles, lit from -v."""
    vv, uu = np.mgrid[0:_SIZE, 0:_SIZE].astype(np.float64)
    dv = (vv - _CENTER) / _RADIUS
    du = (uu - _CENTER) / _RADIUS
    on_body = dv**2 + du**2 < 1.0
    nz = np.sqrt(np.clip(1.0 - dv**2 - du**2, 0.0, 1.0))
    phase = math.radians(_PHASE_DEG)
    mu0 = np.clip(-dv * math.sin(phase) + nz * math.cos(phase), -1.0, 1.0)
    return on_body, np.arccos(mu0), np.arccos(nz)


class _Backplane:
    """The three backplanes of one body that the measurement reads."""

    def __init__(self) -> None:
        self.on_body, self.incidence, self.emission = _sphere_angles()

    def where_intercepted(self, _body: str) -> Any:
        return SimpleNamespace(vals=self.on_body)

    def incidence_angle(self, _body: str) -> Any:
        return SimpleNamespace(mvals=np.ma.array(self.incidence, mask=~self.on_body))

    def emission_angle(self, _body: str) -> Any:
        return SimpleNamespace(mvals=np.ma.array(self.emission, mask=~self.on_body))


def _real_frame(albedo: float) -> Any:
    """A frame whose body lies 4 px down and 3 px left of where it is predicted."""
    backplane = _Backplane()
    shaded = shade_surface(
        np.cos(backplane.incidence), np.cos(backplane.emission), law='lommel_seeliger'
    )
    data = np.zeros((_SIZE, _SIZE))
    data[
        max(_SHIFT_VU[0], 0) : _SIZE + min(_SHIFT_VU[0], 0),
        max(_SHIFT_VU[1], 0) : _SIZE + min(_SHIFT_VU[1], 0),
    ] = (
        albedo
        * np.where(backplane.on_body, shaded, 0.0)[
            max(-_SHIFT_VU[0], 0) : _SIZE + min(-_SHIFT_VU[0], 0),
            max(-_SHIFT_VU[1], 0) : _SIZE + min(-_SHIFT_VU[1], 0),
        ]
    )
    return SimpleNamespace(bp=backplane, data=data)


def test_the_disc_is_read_where_the_verified_offset_puts_it() -> None:
    """A body drawn at 0.6 times its law, 4 px down and 3 px left, measures 0.6."""
    albedo = real_normal_albedo(
        _real_frame(0.6), 'MOON', offset_vu=(4.2, -2.9), phase_angle_deg=_PHASE_DEG
    )
    assert albedo == pytest.approx(0.6, rel=1e-12)


@pytest.mark.parametrize('scene_class', ['body_full_fov', 'ring_plus_body'])
def test_a_matched_body_takes_the_measured_normal_albedo(scene_class: str) -> None:
    """A matched limb or ring scene gives its body the normal albedo it is handed."""
    scene = matched_scene(
        'N1', scene_class, 'coiss_calib_nac', 0.5, size_vu=(256, 256), normal_albedo=0.37
    )
    assert scene['bodies'][0]['normal_albedo'] == 0.37


def test_an_unmeasured_matched_body_stays_white() -> None:
    """Without a measured albedo, the matched body leaves the key out."""
    scene = matched_scene('N1', 'ring_plus_body', 'coiss_calib_nac', 0.5, size_vu=(256, 256))
    assert 'normal_albedo' not in scene['bodies'][0]
