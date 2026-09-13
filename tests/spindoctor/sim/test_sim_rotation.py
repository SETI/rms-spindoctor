"""Camera-roll rendering and the per-scene ``fit_camera_rotation`` override.

A planted camera roll rotates the rendered scene about the boresight before the
translation offset; the simulated star NavModel predicts the unrolled geometry,
so a star technique recovers the roll.  These tests cover the renderer geometry
(a star lands at its analytically rotated position; a ring system's center and
node angle roll together; a star, a ring and a body placed alike turn about one
point, the frame's uv centre) and the scene-level ``fit_camera_rotation``
override that lets a scene exercise the 3-DoF path on any emulated camera,
independent of that camera's real rotation-fitting flag.
"""

from collections.abc import Callable

import numpy as np
import pytest

from spindoctor.nav_orchestrator.instrument_config import instrument_settings_from_obs
from spindoctor.obs.obs_inst_sim import ObsSim
from spindoctor.sim.render import render_combined_model

_SIZE = 128
_CENTER = _SIZE / 2.0


def _noiseless_params(**overrides: object) -> dict[str, object]:
    """coiss_nac sim params with detector noise suppressed for exact geometry."""
    params: dict[str, object] = {
        'size_v': _SIZE,
        'size_u': _SIZE,
        'random_seed': 7,
        'instrument': 'coiss_nac',
        'exposure_sec': 1.0,
        'offset_v': 0.0,
        'offset_u': 0.0,
        'bodies': [],
        'noise': {'poisson': False, 'read_noise_dn': 0.0, 'bias_dn': 0.0},
    }
    params.update(overrides)
    return params


def _centroid(img: np.ndarray) -> tuple[float, float]:
    """Brightness-weighted centroid of a single-source image."""
    ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    weights = np.clip(img - np.median(img), 0.0, None)
    total = float(weights.sum())
    return float((ys * weights).sum() / total), float((xs * weights).sum() / total)


def test_roll_rotates_star_about_boresight() -> None:
    """A 90 deg roll lands a star at its analytically rotated position.

    A scene states a position as a pixel corner, so the star at ``(40.5, 90.5)``
    is ``(-23.5, 26.5)`` from the frame's uv centre ``(64, 64)``; a +90 deg roll
    (matrix ``[[0, -1], [1, 0]]`` in ``(v, u)``) maps that to ``(-26.5, -23.5)``,
    putting the star at uv ``(37.5, 40.5)`` -- the centre of pixel ``(37, 40)``,
    which is where the rendered centroid must land.
    """
    params = _noiseless_params(
        offset_rotation_deg=90.0,
        stars=[{'name': 'S', 'v': 40.5, 'u': 90.5, 'vmag': 2.0, 'psf_sigma': 2.0}],
    )
    img, _meta = render_combined_model(params)
    centroid_v, centroid_u = _centroid(img)
    assert abs(centroid_v - 37.0) < 0.05
    assert abs(centroid_u - 40.0) < 0.05


def test_zero_roll_leaves_star_in_place() -> None:
    """A zero roll renders the star at its catalog position (no displacement)."""
    params = _noiseless_params(
        offset_rotation_deg=0.0,
        stars=[{'name': 'S', 'v': 40.5, 'u': 90.5, 'vmag': 2.0, 'psf_sigma': 2.0}],
    )
    img, _meta = render_combined_model(params)
    centroid_v, centroid_u = _centroid(img)
    assert abs(centroid_v - 40.0) < 0.05
    assert abs(centroid_u - 90.0) < 0.05


def test_star_record_keeps_unrolled_position() -> None:
    """The emitted star record carries the unrolled catalog ``(v, u)``.

    The roll is applied to the rendered image only; the NavModel must predict the
    unrolled geometry so a technique recovers the roll rather than cancelling it.
    The record states the position in the scene's own pixel corner coordinates.
    """
    params = _noiseless_params(
        offset_rotation_deg=30.0,
        stars=[{'name': 'S', 'v': 40.5, 'u': 90.5, 'vmag': 2.0}],
    )
    _img, meta = render_combined_model(params)
    star = meta['stars'][0]
    assert star.v == 40.5
    assert star.u == 90.5


def test_roll_rotates_ring_center_and_node_together() -> None:
    """A camera roll rotates the ring center about the boresight and adds to the node.

    Rendering an off-center inclined ring system under a +30 deg roll must
    equal rendering the same system with no roll but the roll pre-applied by
    hand -- the center rotated about the frame center by the same rotation
    matrix the body and star paths use, and the roll added to the node angle.
    The comparison is bitwise: both paths must feed identical placement into
    the shared projection.
    """
    roll_deg = 30.0
    center_v, center_u = 44.0, 76.0
    node_deg = 20.0
    geometry: dict[str, float] = {
        'center_v': center_v,
        'center_u': center_u,
        'opening_deg_obs': 35.0,
        'opening_deg_sun': 35.0,
        'node_deg': node_deg,
    }
    features: list[dict[str, object]] = [
        {
            'kind': 'ringlet',
            'tau': 1.0,
            'width': 6.0,
            'orbit': {'a': 25.0, 'ae': 1.0, 'long_peri': 15.0, 'rate_peri': 0.0},
        }
    ]
    rolled_params = _noiseless_params(
        offset_rotation_deg=roll_deg,
        ring_system={'geometry': geometry, 'features': features},
    )
    rolled, _meta = render_combined_model(rolled_params)

    # Pre-apply the roll by hand, with the same float operations the renderer
    # uses for bodies and stars: rotate the center about the boresight and
    # add the roll to the node angle.
    roll_cos = float(np.cos(np.radians(roll_deg)))
    roll_sin = float(np.sin(np.radians(roll_deg)))
    rel_v = center_v - _CENTER
    rel_u = center_u - _CENTER
    pre_rolled_geometry: dict[str, float] = {
        **geometry,
        'center_v': _CENTER + roll_cos * rel_v - roll_sin * rel_u,
        'center_u': _CENTER + roll_sin * rel_v + roll_cos * rel_u,
        'node_deg': node_deg + roll_deg,
    }
    unrolled_params = _noiseless_params(
        offset_rotation_deg=0.0,
        ring_system={'geometry': pre_rolled_geometry, 'features': features},
    )
    unrolled, _meta2 = render_combined_model(unrolled_params)
    np.testing.assert_array_equal(rolled, unrolled)


def _ring_system_at(center_v: float, center_u: float) -> dict[str, object]:
    """A face-on circular ringlet centred on ``(center_v, center_u)``."""
    return {
        'geometry': {
            'center_v': center_v,
            'center_u': center_u,
            'opening_deg_obs': 90.0,
            'opening_deg_sun': 90.0,
            'node_deg': 0.0,
        },
        'features': [{'kind': 'ringlet', 'tau': 1.0, 'width': 4.0, 'orbit': {'a': 20.0}}],
    }


def _sphere_at(center_v: float, center_u: float) -> dict[str, object]:
    """A head-on lit sphere centred on ``(center_v, center_u)``."""
    return {
        'center_v': center_v,
        'center_u': center_u,
        'axis1': 20.0,
        'axis2': 20.0,
        'axis3': 20.0,
        'phase_angle': 0.0,
        'illumination_angle': 0.0,
    }


_PIVOT_PROBE_V = 40.5
_PIVOT_PROBE_U = 90.5
# A +90 deg roll is the sharpest probe of the pivot: it maps the pixel lattice
# onto itself, so the rasterized ring and sphere centroids carry no
# discretization bias to confuse the comparison, and it turns a half-pixel split
# between two pivots into a full pixel of disagreement.
_PIVOT_PROBE_ROLL_DEG = 90.0
_PIVOT_TOLERANCE_PX = 0.01

_GEOMETRY_FACTORIES: dict[str, Callable[[], dict[str, object]]] = {
    'ring': lambda: {'ring_system': _ring_system_at(_PIVOT_PROBE_V, _PIVOT_PROBE_U)},
    'body': lambda: {'bodies': [_sphere_at(_PIVOT_PROBE_V, _PIVOT_PROBE_U)]},
}


@pytest.mark.parametrize(
    'make_geometry', list(_GEOMETRY_FACTORIES.values()), ids=list(_GEOMETRY_FACTORIES)
)
def test_roll_turns_stars_and_geometry_about_one_point(
    make_geometry: Callable[[], dict[str, object]],
) -> None:
    """A rolled star and a rolled ring / sphere placed alike land on one another.

    Each source is placed at the same scene position and rolled by the same angle
    in its own scene, so the two rendered centroids coincide only if both paths
    turn about the same point.  Reading ``size / 2`` as a pixel centric
    coordinate for the star field while the geometry reads it as a pixel corner
    pivots the star field half a detector pixel away from the geometry, which at
    this roll angle separates the two centroids by a full pixel.

    Parameters:
        make_geometry: Builds the scene keys placing the ring or sphere.
    """
    star_img, _star_meta = render_combined_model(
        _noiseless_params(
            offset_rotation_deg=_PIVOT_PROBE_ROLL_DEG,
            stars=[
                {
                    'name': 'S',
                    'v': _PIVOT_PROBE_V,
                    'u': _PIVOT_PROBE_U,
                    'vmag': 2.0,
                    'psf_sigma': 2.0,
                }
            ],
        )
    )
    star_v, star_u = _centroid(star_img)
    geometry_img, _geometry_meta = render_combined_model(
        _noiseless_params(offset_rotation_deg=_PIVOT_PROBE_ROLL_DEG, **make_geometry())
    )
    geometry_v, geometry_u = _centroid(geometry_img)
    assert abs(geometry_v - star_v) < _PIVOT_TOLERANCE_PX
    assert abs(geometry_u - star_u) < _PIVOT_TOLERANCE_PX


def test_fit_camera_rotation_override_enables_3dof() -> None:
    """The scene override turns on rotation fitting for an emulated coiss_nac."""
    obs = ObsSim.from_file(
        '/tmp/rot_override.json',
        sim_params=_noiseless_params(fit_camera_rotation=True),
    )
    assert instrument_settings_from_obs(obs).fit_camera_rotation is True


def test_fit_camera_rotation_defaults_to_instrument() -> None:
    """Without the override the obs uses the instrument's own flag (coiss: off)."""
    obs = ObsSim.from_file('/tmp/rot_default.json', sim_params=_noiseless_params())
    assert instrument_settings_from_obs(obs).fit_camera_rotation is False
