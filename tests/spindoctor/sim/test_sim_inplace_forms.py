"""The simulator's in-place closed forms against their plain expressions, bit for bit.

The renderer builds several closed forms over whole oversampled frames in place,
with ``out=`` arguments and in-place operators, because a fresh frame-sized
temporary costs more than the arithmetic in it.  Each is meant to keep the
operation and the operand order of a plain expression, so that every render keeps
its bits; regrouping a step (``a * (b / c)`` for ``a * b / c``) changes the last bit
of some values while every tolerance-based test still passes.  Each test here
compares one in-place form with the plain expression it replaces, written out
below, on decimal-valued inputs: values such as 0.3, whose binary expansions use
every bit, so a regrouped step rounds differently somewhere in the array.

The forms also keep what the plain expressions accept and return: scalars,
operands that broadcast against each other, and float32 operands, which keep
float32 where the plain expression does.  The comparison covers the result's
Python type, dtype, shape, and bytes.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render.bodies.photometry import (
    _lommel_seeliger,
    incidence_cosines,
    shade_surface,
)
from spindoctor.sim.render.bodies.placement import BodyCoverage, uncovered_fraction
from spindoctor.sim.render.rings.photometry import ring_reflection_factor
from spindoctor.sim.render.rings.profiles import FeatureBoundaries, feature_tau
from spindoctor.sim.render.rings.render import _GRID_STRIP_ROWS, _ring_plane_grid
from spindoctor.sim.shared.ellipsoid import (
    EllipsoidProjection,
    project_ellipsoid,
    surface_normals,
    surface_unit_coords,
)
from spindoctor.sim.shared.lighting import illumination_vector
from spindoctor.sim.shared.ring import (
    RingOrbit,
    compute_edge_radii_array,
    compute_edge_wave_dr,
    compute_orbit_radii,
    ring_orbit_from_mapping,
    ring_plane_from_sky,
    ring_radial_scale,
)
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.stable_math import stable_hypot


def _decimal(low: float, high: float, shape: tuple[int, ...], *, seed: int) -> np.ndarray:
    """Random values rounded to four decimals, which use every bit of a float64."""
    return np.round(np.random.default_rng(seed).uniform(low, high, shape), 4)


def _assert_identical(result: Any, expected: Any) -> None:
    """The result has the expected Python type, dtype, shape, and bytes."""
    assert type(result) is type(expected)
    assert np.asarray(result).dtype == np.asarray(expected).dtype
    assert np.shape(result) == np.shape(expected)
    assert np.asarray(result).tobytes() == np.asarray(expected).tobytes()


def _assert_all_identical(results: tuple[Any, ...], expected: tuple[Any, ...]) -> None:
    """Each result of a tuple is identical to its expected value."""
    assert len(results) == len(expected)
    for result, value in zip(results, expected, strict=True):
        _assert_identical(result, value)


# The plain expressions the in-place forms replace.


def _plain_ring_plane_from_sky(
    dv: Any, du: Any, *, opening_deg_obs: float, node_deg: float
) -> tuple[Any, Any, Any, Any]:
    """Sky offsets to ring-plane radius, longitude, x, and y."""
    sin_b = math.sin(math.radians(opening_deg_obs))
    node = math.radians(node_deg)
    x = du * math.cos(node) + -dv * math.sin(node)
    y = (-du * math.sin(node) + -dv * math.cos(node)) / sin_b
    return stable_hypot(x, y), np.mod(np.arctan2(y, x), 2.0 * math.pi), x, y


def _plain_ring_radial_scale(r: Any, x: Any, y: Any, *, opening_deg_obs: float) -> Any:
    """The ring-plane radial distance per image pixel."""
    sin_b = math.sin(math.radians(opening_deg_obs))
    scale = np.sqrt(x * x + (y * y) / (sin_b * sin_b))
    return np.where(r > 0.0, scale / np.where(r > 0.0, r, 1.0), 1.0)


_EDGE = {'a': 300.3, 'ae': 3.7, 'long_peri': 10.1, 'rate_peri': 0.3, 'epoch': 0.0}
_TIME = 86400.0 * 1.7


def _plain_edge_radii(angles: Any) -> Any:
    """The precessing mode-1 ellipse of ``_EDGE`` at ``_TIME``."""
    days = (_TIME - _EDGE['epoch']) / 86400.0
    long_peri = math.radians(_EDGE['long_peri'] + _EDGE['rate_peri'] * days)
    e = _EDGE['ae'] / _EDGE['a']
    return _EDGE['a'] * (1.0 - e * e) / (1.0 + e * np.cos(angles - long_peri))


def _orbit() -> RingOrbit:
    """An orbit with two m-modes and an edge wave."""
    return ring_orbit_from_mapping(
        {
            'a_px': 300.3,
            'ae_px': 3.7,
            'long_peri_deg': 10.1,
            'modes': [{'m': 3, 'amp_px': 0.53, 'peri_deg': 20.7}, {'m': 5, 'amp_px': 0.21}],
            'edge_wave': {
                'amp_px': 1.3,
                'wavelength_px': 40.1,
                'edge_wave_damping_deg': 50.0,
                'lam0_deg': 30.0,
            },
        }
    )


def _plain_orbit_radii(lam: Any, orbit: RingOrbit) -> Any:
    """The orbit's radii at ``lam`` at time 0."""
    e = orbit.ae / orbit.a
    r = orbit.a * (1.0 - e * e) / (1.0 + e * np.cos(lam - math.radians(orbit.long_peri)))
    for mode in orbit.modes:
        r = r - mode.amp * np.cos(mode.m * (lam - math.radians(mode.peri)))
    assert orbit.edge_wave is not None
    return r + compute_edge_wave_dr(lam, orbit.edge_wave, a=orbit.a)


def _plain_reflection(tau: Any, mu: Any, mu0: Any, *, lit: bool) -> Any:
    """The single-scattering geometric factor."""
    if lit:
        return mu0 / (mu0 + mu) * (1.0 - np.exp(-tau * (1.0 / mu0 + 1.0 / mu)))
    if abs(mu0 - mu) < 1e-6:
        return (tau / mu) * np.exp(-tau / mu)
    return mu0 / (mu0 - mu) * (np.exp(-tau / mu0) - np.exp(-tau / mu))


_WAVE: dict[str, Any] = {
    'kind': 'wave',
    'tau': 0.83,
    'wavelength_px': 12.3,
    'density_wave_damping_px': 30.7,
}


def _plain_wave(r: Any, r_edge: Any, *, os: int) -> Any:
    """The density wave's damped sine at radii ``r``."""
    x = np.clip(r - r_edge, 0.0, None)
    damping = _WAVE['density_wave_damping_px'] * os
    wavelength = _WAVE['wavelength_px'] * os
    return _WAVE['tau'] * np.exp(-x / damping) * np.sin(2.0 * math.pi * x / wavelength)


_LIGHT = {'illumination_angle': 0.37, 'phase_angle': 0.71}


def _plain_incidence(normal_v: Any, normal_u: Any, normal_z: Any) -> Any:
    """The incidence cosine under ``_LIGHT``."""
    illum_v, illum_u, illum_z = illumination_vector(**_LIGHT)
    return normal_v * illum_v + normal_u * illum_u + normal_z * illum_z


def _plain_lommel_seeliger(mu0: Any, mu: Any) -> Any:
    """``2 mu0 / (mu0 + mu)``, 0 where degenerate."""
    denominator = mu0 + mu
    with np.errstate(divide='ignore', invalid='ignore'):
        value = np.where(denominator > 0.0, 2.0 * mu0 / denominator, 0.0)
    return np.nan_to_num(value, nan=0.0)


def _plain_lambert(cos_i: Any, cos_e: Any, *, surge: float) -> Any:
    """The Lambert law times a surge factor on the visible hemisphere."""
    return np.where(cos_e > 0.0, np.clip(cos_i, 0.0, 1.0) * surge, 0.0)


def _plain_normals(
    v_rot: Any, u_rot: Any, dist_sq: Any, projection: EllipsoidProjection
) -> tuple[Any, Any, Any]:
    """Unit surface normals in image coordinates."""
    geometry = projection.geometry
    if geometry.sin_lat == 0.0:
        z_coords = geometry.semi_c * np.sqrt(np.maximum(0.0, 1.0 - dist_sq))
        normal_v = v_rot / geometry.semi_a**2
        normal_u = u_rot / geometry.semi_b**2
        normal_z = z_coords / geometry.semi_c**2
    else:
        x, y, z = surface_unit_coords(v_rot, u_rot, dist_sq, geometry=geometry)
        normal_v, normal_u, normal_z = geometry.to_view(
            x / geometry.semi_a, y / geometry.semi_b, z / geometry.semi_c
        )
        normal_z = np.where(dist_sq >= 1.0, 0.0, normal_z)
    magnitude = np.maximum(np.sqrt(normal_v**2 + normal_u**2 + normal_z**2), 1e-10)
    normal_v = normal_v / magnitude
    normal_u = normal_u / magnitude
    normal_z = normal_z / magnitude
    cos_rz, sin_rz = projection.cos_rz, projection.sin_rz
    return normal_v * cos_rz + normal_u * sin_rz, -normal_v * sin_rz + normal_u * cos_rz, normal_z


def _projection(sub_observer_lat: float) -> EllipsoidProjection:
    """A tilted, rotated ellipsoid at a sub-pixel center, taller than two strips."""
    return project_ellipsoid(
        (301, 57),
        (150.3, 28.7),
        25.3,
        semi_b_px=23.1,
        semi_c_px=21.7,
        rotation_z=0.37,
        sub_observer_lat=sub_observer_lat,
    )


# Bit-for-bit agreement over full arrays of decimal values.

_FULL = _decimal(-60.0, 60.0, (40, 37), seed=1)
_SECOND = _decimal(-60.0, 60.0, (40, 37), seed=8)
_LAM = _decimal(0.0, 2.0 * math.pi, (500,), seed=2)
_TAU = _decimal(0.0, 3.0, (40, 37), seed=3)
_COS = _decimal(-1.0, 1.0, (3, 40, 37), seed=4)


def test_ring_plane_from_sky_is_its_plain_expression() -> None:
    """The ring-plane radius, longitude, x, and y have the plain expressions' bits."""
    kwargs: dict[str, float] = {'opening_deg_obs': 31.3, 'node_deg': 17.9}
    results = ring_plane_from_sky(_FULL, _SECOND, **kwargs)
    expected = _plain_ring_plane_from_sky(_FULL, _SECOND, **kwargs)
    _assert_all_identical(results, expected)


def test_ring_radial_scale_is_its_plain_expression() -> None:
    """The radial scale has the plain expression's bits, 1 where the radius is 0."""
    r = np.abs(_FULL)
    r[3, 4] = 0.0
    result = ring_radial_scale(r, 0.3 * _FULL, 0.9 * _FULL, opening_deg_obs=31.3)
    _assert_identical(
        result, _plain_ring_radial_scale(r, 0.3 * _FULL, 0.9 * _FULL, opening_deg_obs=31.3)
    )


def test_the_ring_plane_grid_is_the_whole_grid_mapped_at_once() -> None:
    """Mapped a strip of rows at a time, the grid has the whole-grid mapping's bits."""
    shape = (_GRID_STRIP_ROWS + 37, 41)
    kwargs: dict[str, float] = {'opening_deg_obs': 23.7, 'node_deg': 11.3}
    r, lam, radial_scale, y = _ring_plane_grid(
        shape, center_v=150.3, center_u=20.7, with_y=True, **kwargs
    )
    v_grid, u_grid = np.meshgrid(
        np.arange(shape[0], dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX,
        np.arange(shape[1], dtype=np.float64) + PIXEL_CENTER_TO_CORNER_PX,
        indexing='ij',
    )
    r_all, lam_all, x_all, y_all = _plain_ring_plane_from_sky(
        v_grid - 150.3, u_grid - 20.7, **kwargs
    )
    scale_all = _plain_ring_radial_scale(r_all, x_all, y_all, opening_deg_obs=23.7)
    _assert_all_identical((r, lam, radial_scale, y), (r_all, lam_all, scale_all, y_all))


def test_edge_radii_are_their_plain_expression() -> None:
    """The precessing ellipse's radii have the plain expression's bits."""
    result = compute_edge_radii_array(_LAM, time=_TIME, **_EDGE)
    _assert_identical(result, _plain_edge_radii(_LAM))


def test_orbit_radii_are_their_plain_expression() -> None:
    """Radii with m-modes and an edge wave have the plain expression's bits."""
    orbit = _orbit()
    _assert_identical(
        compute_orbit_radii(_LAM, orbit, epoch=0.0, time=0.0), _plain_orbit_radii(_LAM, orbit)
    )


@pytest.mark.parametrize(
    ('mu', 'mu0', 'lit'),
    [
        pytest.param(0.37, 0.61, True, id='lit'),
        pytest.param(0.37, 0.61, False, id='unlit'),
        pytest.param(0.37, 0.37, False, id='unlit_equal_mu'),
    ],
)
def test_ring_reflection_factor_is_its_plain_expression(mu: float, mu0: float, lit: bool) -> None:
    """Each form of the reflection factor has the plain expression's bits."""
    result = ring_reflection_factor(_TAU, mu, mu0, lit=lit)
    _assert_identical(result, _plain_reflection(_TAU, mu, mu0, lit=lit))


def test_density_wave_is_its_plain_expression() -> None:
    """The density wave's profile has the plain expression's bits."""
    r = _decimal(100.0, 900.0, (2000,), seed=5)
    r_edge = np.full_like(r, 120.3)
    result = feature_tau(_WAVE, r, FeatureBoundaries(inner=r_edge, outer=None), os=3)
    _assert_identical(result, _plain_wave(r, r_edge, os=3))


def test_incidence_cosines_are_their_plain_expression() -> None:
    """The incidence cosines have the plain expression's bits."""
    result = incidence_cosines(_COS[0], _COS[1], _COS[2], **_LIGHT)
    _assert_identical(result, _plain_incidence(_COS[0], _COS[1], _COS[2]))


def test_lommel_seeliger_is_its_plain_expression() -> None:
    """The Lommel-Seeliger term has the plain expression's bits, 0 where degenerate."""
    mu0, mu = np.clip(_COS[0], 0.0, 1.0), np.clip(_COS[1], 0.0, 1.0)
    _assert_identical(_lommel_seeliger(mu0, mu), _plain_lommel_seeliger(mu0, mu))


def test_shading_is_its_plain_expression() -> None:
    """The shaded surface has the plain expression's bits, 0 off the visible side."""
    result = shade_surface(_COS[0], _COS[1], law='lambert', phase_angle=0.07, surge_amplitude=0.3)
    surge = (1.0 + 0.3 * math.exp(-0.07 / math.radians(6.0))) / 1.3
    _assert_identical(result, _plain_lambert(_COS[0], _COS[1], surge=surge))


@pytest.mark.parametrize('sub_observer_lat', [0.0, 0.43], ids=['equatorial', 'tilted'])
def test_surface_normals_are_their_plain_expression(sub_observer_lat: float) -> None:
    """Normals worked in strips of rows have the plain expressions' bits, at either latitude."""
    proj = _projection(sub_observer_lat)
    results = surface_normals(
        proj.v_rot,
        proj.u_rot,
        proj.ellipse_dist_sq,
        geometry=proj.geometry,
        cos_rz=proj.cos_rz,
        sin_rz=proj.sin_rz,
    )
    expected = _plain_normals(proj.v_rot, proj.u_rot, proj.ellipse_dist_sq, proj)
    _assert_all_identical(results, expected)


def test_ellipse_distance_is_its_plain_expression() -> None:
    """The outline distance, its second term added in strips, has the plain bits."""
    proj = _projection(0.43)
    expected = (proj.v_rot / proj.semi_a_outline) ** 2 + (proj.u_rot / proj.semi_b) ** 2
    _assert_identical(proj.ellipse_dist_sq, expected)


def test_starlight_scaled_over_the_covered_box_is_the_whole_frame_scaled() -> None:
    """Scaling starlight over the bodies' box gives the frame of transmissions' bits."""
    frame = _decimal(0.0, 9.0, (40, 37), seed=9)
    coverages = [
        BodyCoverage(
            box_v=slice(3, 20), box_u=slice(5, 30), coverage=_decimal(0, 1, (17, 25), seed=10)
        ),
        BodyCoverage(
            box_v=slice(12, 35), box_u=slice(2, 18), coverage=_decimal(0, 1, (23, 16), seed=11)
        ),
        BodyCoverage(box_v=slice(0, 0), box_u=slice(0, 0), coverage=np.zeros((0, 0))),
    ]
    transmission = np.ones(frame.shape)
    for coverage in coverages:
        transmission[coverage.box_v, coverage.box_u] *= coverage.transmission
    box, fraction = uncovered_fraction(coverages)
    scaled = frame.copy()
    scaled[box] *= fraction
    _assert_identical(scaled, frame * transmission)


# The inputs the plain expressions accept, and what they return for them.

_COLUMN = _decimal(-0.9, 0.9, (5, 1), seed=6)
_ROW = _decimal(-0.9, 0.9, (1, 4), seed=7)

# The functions, typed to take anything: their annotations state the arrays the
# renderer passes, while the plain expressions also accept scalars and broadcasting.
_loose_ring_plane_from_sky: Callable[..., Any] = ring_plane_from_sky
_loose_surface_normals: Callable[..., Any] = surface_normals
_loose_compute_orbit_radii: Callable[..., Any] = compute_orbit_radii
_loose_ring_reflection_factor: Callable[..., Any] = ring_reflection_factor
_loose_incidence_cosines: Callable[..., Any] = incidence_cosines
_loose_lommel_seeliger: Callable[..., Any] = _lommel_seeliger
_loose_shade_surface: Callable[..., Any] = shade_surface
_loose_feature_tau: Callable[..., Any] = feature_tau
_LooseFeatureBoundaries: Callable[..., Any] = FeatureBoundaries


@pytest.mark.parametrize(
    ('function', 'plain'),
    [
        pytest.param(
            lambda: _loose_ring_plane_from_sky(3.1, 4.3, opening_deg_obs=30.0, node_deg=10.0),
            lambda: _plain_ring_plane_from_sky(3.1, 4.3, opening_deg_obs=30.0, node_deg=10.0),
            id='ring_plane_from_sky_scalars',
        ),
        pytest.param(
            lambda: _loose_ring_plane_from_sky(
                np.array(3.1), np.array(4.3), opening_deg_obs=30.0, node_deg=10.0
            ),
            lambda: _plain_ring_plane_from_sky(
                np.array(3.1), np.array(4.3), opening_deg_obs=30.0, node_deg=10.0
            ),
            id='ring_plane_from_sky_0d',
        ),
        pytest.param(
            lambda: _loose_ring_plane_from_sky(
                _COLUMN * 50.0, _ROW * 50.0, opening_deg_obs=30.0, node_deg=10.0
            ),
            lambda: _plain_ring_plane_from_sky(
                _COLUMN * 50.0, _ROW * 50.0, opening_deg_obs=30.0, node_deg=10.0
            ),
            id='ring_plane_from_sky_column_and_row',
        ),
        pytest.param(
            lambda: _loose_surface_normals(
                3.1,
                4.2,
                0.3,
                geometry=_projection(0.43).geometry,
                cos_rz=_projection(0.43).cos_rz,
                sin_rz=_projection(0.43).sin_rz,
            ),
            lambda: _plain_normals(3.1, 4.2, 0.3, _projection(0.43)),
            id='surface_normals_scalars',
        ),
    ],
)
def test_tuple_results_match_the_plain_expressions(
    function: Callable[[], tuple[Any, ...]], plain: Callable[[], tuple[Any, ...]]
) -> None:
    """Scalar and broadcast inputs give the plain expressions' types, shapes, and bits."""
    _assert_all_identical(function(), plain())


def _f32(values: np.ndarray) -> np.ndarray:
    """The values as float32."""
    return values.astype(np.float32)


def _float32_radial_scale(project: Callable[..., Any], scale: Callable[..., Any]) -> Any:
    """The radial scale of float32 offsets: a float64 radius over float32 x and y."""
    r, _, x, y = project(_f32(_FULL), _f32(_SECOND), opening_deg_obs=31.3, node_deg=17.9)
    return scale(r, x, y, opening_deg_obs=31.3)


@pytest.mark.parametrize(
    ('function', 'plain'),
    [
        pytest.param(
            lambda: _loose_compute_orbit_radii(0.3, _orbit(), epoch=0.0, time=0.0),
            lambda: _plain_orbit_radii(0.3, _orbit()),
            id='orbit_radii_scalar',
        ),
        pytest.param(
            lambda: _loose_compute_orbit_radii(_f32(_LAM), _orbit(), epoch=0.0, time=0.0),
            lambda: _plain_orbit_radii(_f32(_LAM), _orbit()),
            id='orbit_radii_float32',
        ),
        pytest.param(
            lambda: _float32_radial_scale(ring_plane_from_sky, ring_radial_scale),
            lambda: _float32_radial_scale(_plain_ring_plane_from_sky, _plain_ring_radial_scale),
            id='radial_scale_float32_projection',
        ),
        pytest.param(
            lambda: _loose_ring_reflection_factor(0.5, 0.3, 0.4, lit=True),
            lambda: _plain_reflection(0.5, 0.3, 0.4, lit=True),
            id='reflection_scalars',
        ),
        pytest.param(
            lambda: _loose_ring_reflection_factor(_TAU[0], _COLUMN + 1.0, 0.4, lit=True),
            lambda: _plain_reflection(_TAU[0], _COLUMN + 1.0, 0.4, lit=True),
            id='reflection_mu_larger_than_tau',
        ),
        pytest.param(
            lambda: _loose_ring_reflection_factor(_f32(_TAU), 0.37, 0.61, lit=False),
            lambda: _plain_reflection(_f32(_TAU), 0.37, 0.61, lit=False),
            id='reflection_float32',
        ),
        pytest.param(
            lambda: _loose_incidence_cosines(0.1, 0.2, 0.9, **_LIGHT),
            lambda: _plain_incidence(0.1, 0.2, 0.9),
            id='incidence_scalars',
        ),
        pytest.param(
            lambda: _loose_incidence_cosines(_COLUMN, _ROW, 0.9, **_LIGHT),
            lambda: _plain_incidence(_COLUMN, _ROW, 0.9),
            id='incidence_column_and_row',
        ),
        pytest.param(
            lambda: _loose_lommel_seeliger(np.float64(0.5), np.float64(0.2)),
            lambda: _plain_lommel_seeliger(np.float64(0.5), np.float64(0.2)),
            id='lommel_seeliger_scalars',
        ),
        pytest.param(
            lambda: _loose_lommel_seeliger(np.abs(_COLUMN), np.abs(_ROW)),
            lambda: _plain_lommel_seeliger(np.abs(_COLUMN), np.abs(_ROW)),
            id='lommel_seeliger_column_and_row',
        ),
        pytest.param(
            lambda: _loose_shade_surface(0.3, 0.6, law='lambert'),
            lambda: _plain_lambert(0.3, 0.6, surge=1.0),
            id='shading_scalars',
        ),
        pytest.param(
            lambda: _loose_shade_surface(_COLUMN, _ROW, law='lambert'),
            lambda: _plain_lambert(_COLUMN, _ROW, surge=1.0),
            id='shading_column_and_row',
        ),
        pytest.param(
            lambda: _loose_feature_tau(
                _WAVE, 150.7, _LooseFeatureBoundaries(inner=120.3, outer=None), os=3
            ),
            lambda: _plain_wave(150.7, 120.3, os=3),
            id='density_wave_scalar',
        ),
    ],
)
def test_results_match_the_plain_expressions(
    function: Callable[[], Any], plain: Callable[[], Any]
) -> None:
    """Scalar, broadcast, and float32 inputs give the plain expressions' results."""
    _assert_identical(function(), plain())
