"""A planted camera roll turns everything in the field of view with the frame.

A planted roll is a rotation of the camera frame about the boresight, so a rolled
render is the unrolled render turned about the frame center, with every position
and every direction the scene states turned by the same angle in the same sense.
A +90 degree roll maps the pixel lattice onto itself, so most tests here compare a
rolled render with ``np.rot90(unrolled, 1)``, the quarter turn the ring and star roll
tests in ``test_sim_rotation.py`` pin as the sense of a positive roll.  Each case holds
one directional quantity, and each fixture differs from its own half turn, so a
quantity that does not turn, or turns the other way, fails the comparison.

An ellipsoid looks the same after a half turn, so a quarter turn checks where an
elongated ellipsoid lands and the line its long axis lies on, but cannot tell which
way its orientation turned.  A 30 degree roll of its outline settles that, and so do
the cases with surface detail, which a half turn does change.

The renders agree to within one DN.  They agree exactly when measured, but a quarter
turn computed in floating point puts a coordinate a few units in the last place off
a whole pixel, which could move a pixel across a rounding boundary.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from tests.spindoctor.sim.scenes import disc_body, frame_scene, quiet_noise, star_entry

from spindoctor.sim.render.driver import render_scene
from spindoctor.support.types import NDArrayFloatType

_SIZE = 96
_CENTER = _SIZE / 2.0
_DN_TOLERANCE = 1.0
# The sky field keeps this many pixels clear of the frame edge out of its comparison:
# a rolled frame also shows the stars just beyond its edge, whose light reaches the
# outermost pixels, and the unrolled frame draws none there.
_SKY_EDGE_PX = 3


def _scene(roll_deg: float, **blocks: Any) -> dict[str, Any]:
    """A noiseless square frame holding ``blocks``, rolled by ``roll_deg``."""
    params = frame_scene(size=_SIZE, seed=5, exposure_sec=1.0, noise=quiet_noise(), **blocks)
    params['offset_rotation_deg'] = roll_deg
    return params


def _render(roll_deg: float, blocks: dict[str, Any]) -> NDArrayFloatType:
    """The frame holding ``blocks`` rendered at ``roll_deg``."""
    img, _meta = render_scene(_scene(roll_deg, **blocks))
    return np.asarray(img, dtype=np.float64)


def _elongated_ellipsoid() -> dict[str, Any]:
    """A fully lit elongated ellipsoid off the frame center, turned in the plane.

    Off the frame center, the case checks that the body turns about the frame center
    as it turns about its own.
    """
    return {
        'bodies': [
            {
                'name': 'B',
                'center_v': 40.0,
                'center_u': 54.0,
                'semi_a_px': 22.0,
                'semi_b_px': 9.0,
                'semi_c_px': 9.0,
                'rotation_z': 25.0,
                'phase_angle': 0.0,
            }
        ]
    }


def _half_lit_sphere() -> dict[str, Any]:
    """A sphere at the frame center lit from 30 degrees at 70 degrees phase."""
    return {
        'bodies': [disc_body(_CENTER, _CENTER, 20.0, illumination_angle=30.0, phase_angle=70.0)]
    }


def _mesh_body() -> dict[str, Any]:
    """A lumpy mesh body at the frame center in a general pose, lit from the side."""
    return {
        'bodies': [
            {
                'name': 'H',
                'shape_model': 'polyhedral_mesh',
                'center_v': _CENTER,
                'center_u': _CENTER,
                'semi_a_px': 24.0,
                'semi_b_px': 14.0,
                'semi_c_px': 12.0,
                'mesh_lumpiness': 0.4,
                'mesh_seed': 3,
                'pose_euler_deg': [10.0, 20.0, 30.0],
                'illumination_angle': 40.0,
                'phase_angle': 30.0,
            }
        ]
    }


def _cratered_ellipsoid() -> dict[str, Any]:
    """A fully lit cratered ellipsoid at the frame center, turned in the plane.

    The outline is elongated so that the rolled body covers other pixels than the
    unrolled one, and only crater centers drawn for the unrolled body land where
    the turned craters belong.
    """
    return {
        'bodies': [
            {
                'name': 'B',
                'center_v': _CENTER,
                'center_u': _CENTER,
                'semi_a_px': 30.0,
                'semi_b_px': 16.0,
                'semi_c_px': 16.0,
                'rotation_z': 25.0,
                'phase_angle': 0.0,
                'crater_fill': 0.4,
                'crater_max_radius': 0.2,
            }
        ]
    }


def _relief_sphere() -> dict[str, Any]:
    """A sphere at the frame center with limb relief, lit from the side."""
    return {
        'bodies': [
            disc_body(
                _CENTER,
                _CENTER,
                24.0,
                illumination_angle=20.0,
                phase_angle=40.0,
                limb_relief_rms=0.03,
                limb_relief_corr_deg=20.0,
            )
        ]
    }


def _textured_sphere() -> dict[str, Any]:
    """A fully lit sphere at the frame center with an albedo texture and a spot."""
    spot = {'lat_deg': 20.0, 'lon_deg': 30.0, 'radius_deg': 15.0, 'albedo_factor': 0.4}
    return {
        'bodies': [
            disc_body(
                _CENTER,
                _CENTER,
                24.0,
                phase_angle=0.0,
                albedo_texture={'rms': 0.1, 'corr_px': 4.0, 'spots': [spot]},
            )
        ]
    }


def _banded_sphere() -> dict[str, Any]:
    """A fully lit banded sphere at the frame center with a storm, turned in the plane."""
    storm = {'lat_deg': 25.0, 'lon_deg': 20.0, 'radius_deg': 15.0, 'albedo_factor': 0.4}
    return {
        'bodies': [
            disc_body(
                _CENTER,
                _CENTER,
                24.0,
                phase_angle=0.0,
                rotation_z=25.0,
                disc_texture={'band_amplitude': 0.3, 'band_wavenumber': 5.0, 'storms': [storm]},
            )
        ]
    }


def _transit_sphere() -> dict[str, Any]:
    """A fully lit sphere at the frame center crossed by a moon and its shadow."""
    return {
        'bodies': [
            disc_body(
                _CENTER,
                _CENTER,
                24.0,
                phase_angle=0.0,
                transits=[
                    {
                        'moon': {
                            'dv_px': -8.0,
                            'du_px': 5.0,
                            'radius_px': 4.0,
                            'albedo_factor': 1.5,
                        },
                        'shadow': {'dv_px': 6.0, 'du_px': 9.0, 'radius_px': 4.0, 'darkness': 0.8},
                    }
                ],
            )
        ]
    }


def _hazy_sphere() -> dict[str, Any]:
    """A sphere at the frame center with a tilted, clouded, ramped haze, lit from the side."""
    return {
        'bodies': [
            disc_body(
                _CENTER,
                _CENTER,
                18.0,
                illumination_angle=30.0,
                phase_angle=50.0,
                atmosphere={
                    'scale_height_px': 2.0,
                    'tau_ref': 0.5,
                    'axis_tilt_deg': 15.0,
                    'interior_ramp_amplitude': 0.1,
                    'cloud_blobs': [{'center_vu': [-6.0, 4.0], 'sigma_px': 2.0, 'amplitude': 0.3}],
                },
            )
        ]
    }


def _smeared_star() -> dict[str, Any]:
    """A star off the frame center whose image drifts during the exposure."""
    return {'stars': [star_entry(30.5, 60.5, 2.0, psf_sigma=1.0, move_v=2.0, move_u=7.0)]}


def _star_with_companion() -> dict[str, Any]:
    """A star off the frame center with an unresolved companion."""
    companion = {'sep_px': 6.0, 'angle_deg': 30.0, 'delta_mag': 0.5}
    return {'stars': [star_entry(30.5, 60.5, 2.0, psf_sigma=1.0, companion=companion)]}


def _star_with_catalog_error() -> dict[str, Any]:
    """A star off the frame center drawn away from its catalog position."""
    return {
        'stars': [
            star_entry(30.5, 60.5, 2.0, psf_sigma=1.0, catalog_error_v=3.0, catalog_error_u=-5.0)
        ]
    }


def _scene_smear() -> dict[str, Any]:
    """A star off the frame center smeared by a whole-scene drift."""
    return {
        'stars': [star_entry(30.5, 60.5, 2.0, psf_sigma=1.0)],
        'optics': {'smear': [{'dv_px': 3.0, 'du_px': 8.0, 'object_class': 'all'}]},
    }


_QUARTER_TURN_CASES: dict[str, Callable[[], dict[str, Any]]] = {
    'ellipsoid_orientation': _elongated_ellipsoid,
    'lit_side': _half_lit_sphere,
    'mesh_pose': _mesh_body,
    'craters': _cratered_ellipsoid,
    'relief': _relief_sphere,
    'albedo_texture': _textured_sphere,
    'disc_texture': _banded_sphere,
    'transits': _transit_sphere,
    'atmosphere': _hazy_sphere,
    'star_smear': _smeared_star,
    'companion': _star_with_companion,
    'catalog_error': _star_with_catalog_error,
    'scene_smear': _scene_smear,
}


@pytest.mark.parametrize(
    'make_blocks', list(_QUARTER_TURN_CASES.values()), ids=list(_QUARTER_TURN_CASES)
)
def test_quarter_roll_renders_the_unrolled_frame_turned(
    make_blocks: Callable[[], dict[str, Any]],
) -> None:
    """A +90 degree roll renders the unrolled frame turned a quarter turn.

    The first assertion proves the fixture can tell the two senses apart: it
    differs from its own half turn, so it differs from at least one of the two
    quarter turns.

    Parameters:
        make_blocks: Builds the scene blocks holding the quantity under test.
    """
    blocks = make_blocks()
    unrolled = _render(0.0, blocks)
    rolled = _render(90.0, blocks)
    assert np.abs(unrolled - np.rot90(unrolled, 2)).max() > _DN_TOLERANCE
    assert np.abs(rolled - np.rot90(unrolled, 1)).max() <= _DN_TOLERANCE


def _dense_sky() -> dict[str, Any]:
    """A sky field dense enough to put stars all over a small frame."""
    return {
        'sky_counts': {'a': 1.0, 'b': 0.34, 'density_factor': 50.0},
        'optics': {'psf': {'sigma_v': 0.8}},
    }


def test_quarter_roll_turns_the_sky_field() -> None:
    """A +90 degree roll renders the unrolled sky field turned a quarter turn."""
    unrolled = _render(0.0, _dense_sky())
    rolled = _render(90.0, _dense_sky())
    inner = slice(_SKY_EDGE_PX, _SIZE - _SKY_EDGE_PX)
    turned = np.rot90(unrolled, 1)
    assert np.abs(unrolled - np.rot90(unrolled, 2))[inner, inner].max() > _DN_TOLERANCE
    assert np.abs(rolled - turned)[inner, inner].max() <= _DN_TOLERANCE


def test_rolled_sky_field_fills_the_corners() -> None:
    """A 45 degree roll brings sky into the corners the unrolled frame did not cover.

    The frame turned 45 degrees about its center leaves a triangle at each corner
    of the frame that lay outside the unrolled frame.  Each must hold sky stars.
    """
    rolled = _render(45.0, _dense_sky())
    background = float(np.median(_render(45.0, {})))
    vv, uu = np.mgrid[0:_SIZE, 0:_SIZE]
    corner_reach = _SIZE / 4.0 - _SKY_EDGE_PX
    corner_sums = []
    for corner_v, corner_u in ((0, 0), (0, _SIZE), (_SIZE, 0), (_SIZE, _SIZE)):
        in_corner = np.abs(vv + 0.5 - corner_v) + np.abs(uu + 0.5 - corner_u) < corner_reach
        corner_sums.append(float(np.sum(rolled[in_corner] - background)))
    assert min(corner_sums) > 0.0


def _light_angle_deg(img: NDArrayFloatType) -> float:
    """The direction of the center of light from the frame center, from +v toward +u."""
    vv, uu = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    weights = np.clip(img - np.median(img), 0.0, None)
    total = float(weights.sum())
    dv = float((vv * weights).sum()) / total + 0.5 - _CENTER
    du = float((uu * weights).sum()) / total + 0.5 - _CENTER
    return math.degrees(math.atan2(du, dv))


def _long_axis_deg(img: NDArrayFloatType) -> float:
    """The long-axis direction of the lit pixels, from +v toward +u, in (-90, 90]."""
    vv, uu = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    weights = np.clip(img - np.median(img), 0.0, None)
    total = float(weights.sum())
    mean_v = float((vv * weights).sum()) / total
    mean_u = float((uu * weights).sum()) / total
    cov_vv = float((weights * (vv - mean_v) ** 2).sum()) / total
    cov_uu = float((weights * (uu - mean_u) ** 2).sum()) / total
    cov_vu = float((weights * (vv - mean_v) * (uu - mean_u)).sum()) / total
    return 0.5 * math.degrees(math.atan2(2.0 * cov_vu, cov_vv - cov_uu))


_ROLL_DEG = 30.0
_ANGLE_TOLERANCE_DEG = 0.1


def test_roll_turns_the_lit_side_by_the_roll() -> None:
    """A 30 degree roll turns a half-lit sphere's center of light by 30 degrees.

    The sense is the one a star takes about the frame center, from ``+v`` toward
    ``+u``.
    """
    blocks = {'bodies': [disc_body(_CENTER, _CENTER, 20.0, phase_angle=90.0)]}
    turned = _light_angle_deg(_render(_ROLL_DEG, blocks)) - _light_angle_deg(_render(0.0, blocks))
    assert abs((turned + 180.0) % 360.0 - 180.0 - _ROLL_DEG) < _ANGLE_TOLERANCE_DEG


def test_roll_turns_an_ellipsoid_outline_by_the_roll() -> None:
    """A 30 degree roll turns a fully lit elongated ellipsoid's long axis by 30 degrees."""
    body = {
        'name': 'B',
        'center_v': _CENTER,
        'center_u': _CENTER,
        'semi_a_px': 30.0,
        'semi_b_px': 12.0,
        'semi_c_px': 12.0,
        'phase_angle': 0.0,
    }
    blocks = {'bodies': [body]}
    turned = _long_axis_deg(_render(_ROLL_DEG, blocks)) - _long_axis_deg(_render(0.0, blocks))
    assert abs(turned - _ROLL_DEG) < _ANGLE_TOLERANCE_DEG
