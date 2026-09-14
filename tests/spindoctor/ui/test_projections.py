"""Tests for ``spindoctor.ui.mosaic_viewer.projections``.

Covers pure-math behavior of every public function:

- ``lonlat_to_display`` -- forward projection
- ``display_to_lonlat`` -- inverse projection
- ``sphere_pixel_to_lonlat`` -- ray-cast sphere inverse

No PyQt6 dependency; all tests operate on ``numpy`` arrays.
"""

import math

import numpy as np
import pytest

from spindoctor.ui.mosaic_viewer.projections import (
    ProjectionKind,
    ProjectionParams,
    display_to_lonlat,
    lonlat_to_display,
    sphere_pixel_to_lonlat,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CX = 300.0
_CY = 200.0
_SCALE = 150.0


def _params(kind: ProjectionKind) -> ProjectionParams:
    """Return standard 2-D projection params for the given kind."""
    return ProjectionParams(kind=kind, cx=_CX, cy=_CY, scale=_SCALE)


def _params_3d(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> ProjectionParams:
    """Return standard SPHERE_3D projection params."""
    return ProjectionParams(
        kind=ProjectionKind.SPHERE_3D,
        cx=_CX,
        cy=_CY,
        scale=_SCALE,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
    )


def _assert_lon_allclose(
    lon_actual: np.ndarray,
    lon_expected: np.ndarray,
    *,
    atol: float,
    err_msg: str = '',
) -> None:
    """Assert that two longitude arrays agree within ``atol`` degrees modulo 360.

    Uses the minimum absolute angular difference so that values near 0 / 360
    boundaries are compared correctly.

    Parameters:
        lon_actual: Computed longitude array (deg).
        lon_expected: Reference longitude array (deg).
        atol: Absolute tolerance in degrees.
        err_msg: Optional label for the assertion message.
    """
    diff = np.mod(lon_actual - lon_expected, 360.0)
    # Map to [-180, 180] so values near 360 wrap to near 0
    diff = np.where(diff > 180.0, diff - 360.0, diff)
    np.testing.assert_allclose(diff, 0.0, atol=atol, err_msg=err_msg)


def _sample_lonlat(
    n: int, lat_min: float, lat_max: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of n random (lon, lat) in ([0, 360), [lat_min, lat_max]).

    Parameters:
        n: Number of sample points.
        lat_min: Minimum latitude (exclusive of poles).
        lat_max: Maximum latitude (exclusive of poles).
        rng: NumPy random generator for reproducibility.

    Returns:
        Tuple ``(lon_deg, lat_deg)``.
    """
    lon = rng.uniform(0.0, 360.0, size=n)
    lat = rng.uniform(lat_min, lat_max, size=n)
    return lon, lat


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


def test_polar_n_round_trip() -> None:
    """Forward then inverse POLAR_N recovers (lon, lat) within 1e-6 deg.

    Samples 30 points with latitude strictly between 0 and 90 degrees.
    The pole itself is excluded because longitude is undefined there.
    """
    rng = np.random.default_rng(seed=42)
    lon, lat = _sample_lonlat(30, lat_min=1.0, lat_max=89.0, rng=rng)
    params = _params(ProjectionKind.POLAR_N)

    vx, vy, vis = lonlat_to_display(lon, lat, params)
    lon_rt, lat_rt, valid = display_to_lonlat(vx, vy, params)

    assert np.all(vis)
    assert np.all(valid)
    np.testing.assert_allclose(lat_rt, lat, atol=1e-6, err_msg='POLAR_N lat round-trip')
    _assert_lon_allclose(lon_rt, lon, atol=1e-6, err_msg='POLAR_N lon round-trip')


def test_polar_s_round_trip() -> None:
    """Forward then inverse POLAR_S recovers (lon, lat) within 1e-6 deg.

    Samples 30 points with latitude strictly between -90 and 0 degrees.
    """
    rng = np.random.default_rng(seed=43)
    lon, lat = _sample_lonlat(30, lat_min=-89.0, lat_max=-1.0, rng=rng)
    params = _params(ProjectionKind.POLAR_S)

    vx, vy, vis = lonlat_to_display(lon, lat, params)
    lon_rt, lat_rt, valid = display_to_lonlat(vx, vy, params)

    assert np.all(vis)
    assert np.all(valid)
    np.testing.assert_allclose(lat_rt, lat, atol=1e-6, err_msg='POLAR_S lat round-trip')
    _assert_lon_allclose(lon_rt, lon, atol=1e-6, err_msg='POLAR_S lon round-trip')


def test_mollweide_round_trip() -> None:
    """Forward then inverse MOLLWEIDE recovers (lon, lat) within 1e-4 deg.

    Samples 30 random points over the full globe, avoiding the poles where
    Mollweide compresses longitude information.
    """
    rng = np.random.default_rng(seed=44)
    lon, lat = _sample_lonlat(30, lat_min=-89.0, lat_max=89.0, rng=rng)
    params = _params(ProjectionKind.MOLLWEIDE)

    vx, vy, vis = lonlat_to_display(lon, lat, params)
    lon_rt, lat_rt, valid = display_to_lonlat(vx, vy, params)

    assert np.all(vis)
    assert np.all(valid)
    np.testing.assert_allclose(lat_rt, lat, atol=1e-4, err_msg='MOLLWEIDE lat round-trip')
    _assert_lon_allclose(lon_rt, lon, atol=1e-4, err_msg='MOLLWEIDE lon round-trip')


# ---------------------------------------------------------------------------
# Polar projection specific geometry
# ---------------------------------------------------------------------------


def test_polar_n_pole_maps_to_centre() -> None:
    """POLAR_N: the north pole (lat=90) projects to exactly (cx, cy)."""
    params = _params(ProjectionKind.POLAR_N)
    lon = np.array([0.0])
    lat = np.array([90.0])

    vx, vy, vis = lonlat_to_display(lon, lat, params)

    assert vis[0]
    assert vx[0] == pytest.approx(_CX)
    assert vy[0] == pytest.approx(_CY)


def test_polar_n_equator_lon0_maps_to_right() -> None:
    """POLAR_N: equator at lon=0 has r = tan(pi/4) = 1.0 normalized, so vx = cx + scale.

    At lon=0 the point lies directly above the pole in the screen-up direction,
    but the formula uses sin(lon) for xn and -cos(lon) for yn.  With lon=0,
    sin=0 and -cos=-1, so the point is at xn=0, yn=-1, meaning vx=cx, vy=cy-scale.
    At lon=90 instead: sin=1, -cos=0, giving xn=1, yn=0, so vx=cx+scale, vy=cy.
    """
    params = _params(ProjectionKind.POLAR_N)

    # lon=0, lat=0 → top of polar circle: vx=cx, vy=cy-scale
    vx0, vy0, vis0 = lonlat_to_display(np.array([0.0]), np.array([0.0]), params)
    assert vis0[0]
    assert vx0[0] == pytest.approx(_CX, abs=1e-9)
    assert vy0[0] == pytest.approx(_CY - _SCALE, abs=1e-9)

    # lon=90, lat=0 → right of polar circle: vx=cx+scale, vy=cy
    vx90, vy90, vis90 = lonlat_to_display(np.array([90.0]), np.array([0.0]), params)
    assert vis90[0]
    assert vx90[0] == pytest.approx(_CX + _SCALE, abs=1e-9)
    assert vy90[0] == pytest.approx(_CY, abs=1e-9)


def test_polar_n_equator_radius_equals_one_normalized() -> None:
    """POLAR_N: equator radius in normalized units equals tan(pi/4) = 1.0."""
    params = _params(ProjectionKind.POLAR_N)
    lon = np.array([0.0])
    lat = np.array([0.0])

    vx, vy, _ = lonlat_to_display(lon, lat, params)
    xn = (vx[0] - _CX) / _SCALE
    yn = (vy[0] - _CY) / _SCALE
    r = math.sqrt(xn**2 + yn**2)

    assert r == pytest.approx(math.tan(math.pi / 4.0), abs=1e-9)


# ---------------------------------------------------------------------------
# Mollweide specific geometry
# ---------------------------------------------------------------------------


def test_mollweide_origin_maps_to_centre() -> None:
    """MOLLWEIDE: lon=0, lat=0 projects to exactly (cx, cy)."""
    params = _params(ProjectionKind.MOLLWEIDE)
    vx, vy, vis = lonlat_to_display(np.array([0.0]), np.array([0.0]), params)

    assert vis[0]
    assert vx[0] == pytest.approx(_CX)
    assert vy[0] == pytest.approx(_CY)


def test_mollweide_lon180_maps_to_left_edge() -> None:
    """MOLLWEIDE: lon=180, lat=0 projects to the left edge (xn = -2*sqrt(2)).

    The forward formula normalizes longitude to [-180, 180] via
    ``mod(lon + 180, 360) - 180``.  At lon=180 this yields -180 deg (i.e.
    lon_r = -pi), which maps to xn = -2*sqrt(2) -- the left boundary of the
    ellipse.
    """
    params = _params(ProjectionKind.MOLLWEIDE)
    sqrt2 = math.sqrt(2.0)

    vx, vy, vis = lonlat_to_display(np.array([180.0]), np.array([0.0]), params)
    xn = (vx[0] - _CX) / _SCALE

    assert vis[0]
    assert xn == pytest.approx(-2.0 * sqrt2, abs=1e-6)
    assert vy[0] == pytest.approx(_CY, abs=1e-6)


# ---------------------------------------------------------------------------
# SPHERE_3D forward projection
# ---------------------------------------------------------------------------


def test_sphere3d_front_hemisphere_centre_maps_to_centre() -> None:
    """SPHERE_3D: lon=0, lat=0 with yaw=pitch=0 projects to (cx, cy), visible=True."""
    params = _params_3d()
    vx, vy, vis = lonlat_to_display(np.array([0.0]), np.array([0.0]), params)

    assert vis[0]
    assert vx[0] == pytest.approx(_CX)
    assert vy[0] == pytest.approx(_CY)


def test_sphere3d_back_hemisphere_is_not_visible() -> None:
    """SPHERE_3D: lon=180, lat=0 with yaw=pitch=0 is on back hemisphere, visible=False."""
    params = _params_3d()
    _vx, _vy, vis = lonlat_to_display(np.array([180.0]), np.array([0.0]), params)

    assert not vis[0]


def test_sphere3d_north_pole_visible_no_yaw() -> None:
    """SPHERE_3D: north pole (lat=90) is visible when pitch=0."""
    params = _params_3d(yaw_deg=0.0, pitch_deg=0.0)
    _vx, vy, vis = lonlat_to_display(np.array([0.0]), np.array([90.0]), params)

    assert vis[0]
    assert vy[0] == pytest.approx(_CY - _SCALE, abs=1e-9)


def test_sphere3d_south_pole_visible_no_yaw() -> None:
    """SPHERE_3D: south pole (lat=-90) is visible when pitch=0."""
    params = _params_3d(yaw_deg=0.0, pitch_deg=0.0)
    _vx, vy, vis = lonlat_to_display(np.array([0.0]), np.array([-90.0]), params)

    assert vis[0]
    assert vy[0] == pytest.approx(_CY + _SCALE, abs=1e-9)


# ---------------------------------------------------------------------------
# sphere_pixel_to_lonlat
# ---------------------------------------------------------------------------


def test_sphere_pixel_off_disk_returns_hit_false() -> None:
    """sphere_pixel_to_lonlat: pixels outside sphere disk return hit=False."""
    params = _params_3d()
    # Place pixel more than one radius away from center in x
    vx = np.array([_CX + _SCALE + 1.0])
    vy = np.array([_CY])

    _lon, _lat, hit = sphere_pixel_to_lonlat(vx, vy, params)

    assert not hit[0]


def test_sphere_pixel_centre_returns_lon0_lat0() -> None:
    """sphere_pixel_to_lonlat: the display center maps back to lon=0, lat=0."""
    params = _params_3d()
    vx = np.array([_CX])
    vy = np.array([_CY])

    lon_deg, lat_deg, hit = sphere_pixel_to_lonlat(vx, vy, params)

    assert hit[0]
    assert lat_deg[0] == pytest.approx(0.0, abs=1e-9)
    # lon is degenerate at (lat=0, lon=0) but the camera faces lon=0
    assert lon_deg[0] == pytest.approx(0.0, abs=1e-9)


def test_sphere_pixel_on_disk_boundary_returns_hit_true() -> None:
    """sphere_pixel_to_lonlat: a pixel exactly on the limb (r=1) returns hit=True."""
    params = _params_3d()
    # Place pixel exactly one radius to the right of center
    vx = np.array([_CX + _SCALE])
    vy = np.array([_CY])

    _lon, _lat, hit = sphere_pixel_to_lonlat(vx, vy, params)

    assert hit[0]


# ---------------------------------------------------------------------------
# SPHERE_3D inverse round-trip
# ---------------------------------------------------------------------------


def test_sphere3d_round_trip_front_hemisphere() -> None:
    """lonlat_to_display then sphere_pixel_to_lonlat recovers (lon, lat) within 1e-4 deg.

    Only front-hemisphere points (those with visible=True at yaw=pitch=0) are
    included, because back-hemisphere points project off the visible disk.
    """
    rng = np.random.default_rng(seed=45)
    # Draw many candidates and keep front-hemisphere ones
    lon_all, lat_all = _sample_lonlat(200, lat_min=-85.0, lat_max=85.0, rng=rng)
    params = _params_3d()

    _vx_all, _vy_all, vis_all = lonlat_to_display(lon_all, lat_all, params)
    lon = lon_all[vis_all]
    lat = lat_all[vis_all]

    assert len(lon) >= 30, 'Expected at least 30 front-hemisphere samples'

    vx, vy, vis = lonlat_to_display(lon, lat, params)
    assert np.all(vis)

    lon_rt, lat_rt, hit = sphere_pixel_to_lonlat(vx, vy, params)

    assert np.all(hit)
    np.testing.assert_allclose(lat_rt, lat, atol=1e-4, err_msg='SPHERE_3D lat round-trip')
    _assert_lon_allclose(lon_rt, lon, atol=1e-4, err_msg='SPHERE_3D lon round-trip')


def test_sphere3d_round_trip_with_yaw_and_pitch() -> None:
    """SPHERE_3D round-trip holds when camera is not centered on lon=0, lat=0.

    Uses yaw=45 deg and pitch=30 deg to exercise the rotation matrix path.
    """
    rng = np.random.default_rng(seed=46)
    params = _params_3d(yaw_deg=45.0, pitch_deg=30.0)

    lon_all, lat_all = _sample_lonlat(200, lat_min=-80.0, lat_max=80.0, rng=rng)
    _vx_all, _vy_all, vis_all = lonlat_to_display(lon_all, lat_all, params)
    lon = lon_all[vis_all]
    lat = lat_all[vis_all]

    assert len(lon) >= 20, 'Expected at least 20 visible samples with yaw/pitch'

    vx, vy, _vis = lonlat_to_display(lon, lat, params)
    lon_rt, lat_rt, hit = sphere_pixel_to_lonlat(vx, vy, params)

    assert np.all(hit)
    np.testing.assert_allclose(lat_rt, lat, atol=1e-4)
    _assert_lon_allclose(lon_rt, lon, atol=1e-4)
