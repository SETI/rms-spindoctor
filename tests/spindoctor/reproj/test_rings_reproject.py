"""Spec tests for ``RingMosaic.reproject`` and its argument validators.

``reproject()`` is exercised hermetically (CI has no SPICE kernels or holdings) by
installing a synthetic pinhole geometry: a fake ``oops.backplane.Backplane`` whose
ring radius/longitude backplanes are exact linear functions of the pixel indices,
plus a matching fake ``RingMosaic.longitude_radius_to_pixels`` implementing the
analytic inverse. Every expected value below is derived from that closed-form
geometry and from the documented contracts:

- ``reproject()`` always returns a *sparse* result: only longitude columns holding
  at least one valid pixel are stored, ``longitude_antimask`` marks the full-circle
  bins present, and ``count_nonzero(longitude_antimask) == img.shape[1]``
  (dev guide, "Ring sparse storage").
- Longitude bin ``b`` corresponds to longitude ``b * longitude_resolution`` (rad);
  radius row ``r`` corresponds to ``radius_inner + r * radius_resolution`` (km).
- With an ``orbit_model``, longitudes are co-rotating and radii are signed offsets
  (km) from the orbital radius at each (longitude, time), so an eccentric ring
  reprojects to a straight line (dev guide, "Ring radius and longitude semantics").
- Validators raise the documented ``TypeError`` / ``ValueError`` with the documented
  message content.

The synthetic image is 20x20 with ``data[v, u] = 100 v + u`` so any indexing error
changes the observed values. Geometry: pixel ``(v, u)`` sees longitude
``LON0 + (u - 0.5) * LON_RES / 2`` and (absolute mode) radius
``990 + (v - 0.5) * 2.5`` km. With the default mosaic grid (radii 1000..1020 km at
5 km, longitude pi/16 rad) the reprojection covers bins 8..14 sampling pixels
``u = 4 + 2 c``, ``v = 4 + 2 r``.
"""

import dataclasses
import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, ClassVar

import julian
import numpy as np
import numpy.ma as ma
import oops.fov
import pytest
from polymath import Scalar

from spindoctor.reproj.ring_orbit_model import RingOrbitModel
from spindoctor.reproj.rings import (
    _MAX_LONGITUDE,
    RingMosaic,
    _validate_reproject_longitude_range,
    _validate_reproject_radius_range,
    _validate_reproject_zoom_amt,
)
from spindoctor.support.types import NDArrayFloatType

# ---------------------------------------------------------------------------
# Synthetic geometry constants (see module docstring)
# ---------------------------------------------------------------------------

_N = 20  # image size (pixels per side)
_LON_RES = math.pi / 16  # mosaic longitude resolution (rad/bin)
_RAD_RES = 5.0  # mosaic radius resolution (km/bin)
_N_FULL_LON = math.floor(_MAX_LONGITUDE / _LON_RES) + 1  # 32 bins in 0..2pi
_LON0 = 6 * _LON_RES  # longitude seen by pixel u = 0.5
_LON_PIX = _LON_RES / 2.0  # rad per pixel (2 pixels per longitude bin)
_RAD0 = 990.0  # radius seen by pixel v = 0.5 (absolute mode)
_RAD_PIX = 2.5  # km per pixel (2 pixels per radius bin)
_EPOCH_UTC = '2000-01-01T12:00:00'
_EPOCH_ET = float(julian.tdb_from_tai(julian.tai_from_iso(_EPOCH_UTC)))


def _lon_of_u(u: NDArrayFloatType) -> NDArrayFloatType:
    """Longitude (rad) seen at fractional pixel column u.

    Parameters:
        u: Fractional pixel column array.

    Returns:
        Longitude array in radians.
    """
    return _LON0 + (u - 0.5) * _LON_PIX


def _u_of_lon(lon: NDArrayFloatType) -> NDArrayFloatType:
    """Fractional pixel column at which longitude lon (rad) is seen.

    Parameters:
        lon: Longitude array in radians.

    Returns:
        Fractional pixel column array.
    """
    return (lon - _LON0) / _LON_PIX + 0.5


def _rad_of_v(v: NDArrayFloatType) -> NDArrayFloatType:
    """Absolute ring radius (km) seen at fractional pixel row v.

    Parameters:
        v: Fractional pixel row array.

    Returns:
        Radius array in km.
    """
    return _RAD0 + (v - 0.5) * _RAD_PIX


def _v_of_rad(rad: NDArrayFloatType) -> NDArrayFloatType:
    """Fractional pixel row at which absolute radius rad (km) is seen.

    Parameters:
        rad: Radius array in km.

    Returns:
        Fractional pixel row array.
    """
    return (rad - _RAD0) / _RAD_PIX + 0.5


class _FakeObs:
    """Minimal Observation stand-in exposing what reproject() touches.

    Parameters:
        data: Optional image array; defaults to the ``100 v + u`` pattern.
        midtime: Observation mid-time (ET seconds).
    """

    def __init__(self, *, data: Any = None, midtime: float = 0.0) -> None:
        """Initialize with the default synthetic image unless data is given."""
        self.midtime = midtime
        if data is None:
            vv, uu = np.meshgrid(np.arange(_N), np.arange(_N), indexing='ij')
            data = (vv * 100 + uu).astype(np.float64)
        self.data = data
        self.data_shape_uv = (_N, _N)
        self.fov = oops.fov.FlatFOV((0.001, 0.001), (_N, _N))


def _make_backplane_class(
    *,
    model: RingOrbitModel | None,
    shadow_u_columns: tuple[int, ...],
) -> type[Any]:
    """Build a fake oops Backplane class implementing the synthetic geometry.

    In absolute mode (``model is None``) the radius backplane is
    ``_rad_of_v(v)``. In offset mode the radius backplane is
    ``model.radius_at_longitude(lon, obs.midtime) + (v - 7.5) * _RAD_PIX`` so pixel
    rows hold constant *offsets* from the orbit while absolute radii vary with
    longitude. When a meshgrid is supplied (uv_range), the arrays cover only the
    restricted region, exactly as a real Backplane would.

    Parameters:
        model: Orbit model for offset-mode radii, or None for absolute radii.
        shadow_u_columns: Pixel columns reported as inside the planet shadow.

    Returns:
        A class with the Backplane methods reproject() calls.
    """

    class _FakeRingBackplane:
        """Fake Backplane producing analytic ring backplanes over the pixel grid."""

        shadow_calls: ClassVar[int] = 0

        def __init__(self, obs: Any, meshgrid: Any = None) -> None:
            """Build backplane arrays for the full frame or the meshgrid region."""
            if meshgrid is None:
                u0, v0, nv, nu = 0, 0, _N, _N
            else:
                uv = meshgrid.uv.vals
                u0 = int(uv[0, 0, 0] - 0.5)
                v0 = int(uv[0, 0, 1] - 0.5)
                nv, nu = uv.shape[0], uv.shape[1]
            vv, uu = np.meshgrid(np.arange(v0, v0 + nv), np.arange(u0, u0 + nu), indexing='ij')
            lons = _lon_of_u(uu.astype(np.float64))
            if model is None:
                radius = _rad_of_v(vv.astype(np.float64))
            else:
                radius = (
                    model.radius_at_longitude(lons, obs.midtime)
                    + (vv.astype(np.float64) - 7.5) * _RAD_PIX
                )
            self._radius = Scalar(radius)
            self._longitude = Scalar(lons)
            self._radial_res = Scalar(vv.astype(np.float64) + 2.0)
            self._shape = (nv, nu)
            self._shadow = np.isin(uu, np.asarray(shadow_u_columns, dtype=np.int64))

        def ring_radius(self, name: str) -> Scalar:
            """Ring radius backplane."""
            return self._radius

        def ring_longitude(self, name: str) -> Scalar:
            """Inertial ring longitude backplane."""
            return self._longitude

        def ring_radial_resolution(self, name: str) -> Scalar:
            """Radial resolution backplane (2 + v, so per-column means are testable)."""
            return self._radial_res

        def ring_angular_resolution(self, name: str) -> Scalar:
            """Constant angular resolution backplane."""
            return Scalar(np.full(self._shape, 0.001))

        def phase_angle(self, name: str) -> Scalar:
            """Constant phase angle backplane."""
            return Scalar(np.full(self._shape, 0.5))

        def emission_angle(self, name: str) -> Scalar:
            """Constant emission angle backplane."""
            return Scalar(np.full(self._shape, 0.3))

        def incidence_angle(self, name: str) -> Scalar:
            """Constant incidence angle backplane."""
            return Scalar(np.full(self._shape, 0.4))

        def where_inside_shadow(self, name: str, shadow_name: str) -> Scalar:
            """Shadow mask backplane; counts calls for omit_shadow assertions."""
            type(self).shadow_calls += 1
            return Scalar(self._shadow)

    return _FakeRingBackplane


def _make_lr2p(
    model: RingOrbitModel | None,
) -> Callable[..., tuple[NDArrayFloatType, NDArrayFloatType]]:
    """Build the analytic inverse mapping used in place of longitude_radius_to_pixels.

    Parameters:
        model: Orbit model matching the fake backplane's mode, or None.

    Returns:
        A function with the longitude_radius_to_pixels signature returning (u, v).
    """

    def _fake_lr2p(
        obs: Any,
        longitude: Any,
        radius: Any,
        *,
        orbit_model: RingOrbitModel | None = None,
        ring_body_name: str = 'saturn:ring',
    ) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """Convert (longitude, radius) to fractional (u, v) via the synthetic geometry."""
        lon: NDArrayFloatType = np.atleast_1d(np.asarray(Scalar(longitude).vals, dtype=np.float64))
        rad = np.atleast_1d(np.asarray(Scalar(radius).vals, dtype=np.float64))
        if orbit_model is not None:
            lon = orbit_model.corotating_to_inertial(lon, obs.midtime)
        u = _u_of_lon(lon)
        if model is None:
            v = _v_of_rad(rad)
        else:
            v = (rad - model.radius_at_longitude(lon, obs.midtime)) / _RAD_PIX + 7.5
        return u, v

    return _fake_lr2p


def _install_geometry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: RingOrbitModel | None = None,
    shadow_u_columns: tuple[int, ...] = (),
) -> type[Any]:
    """Install the fake Backplane and inverse mapping for one test.

    Parameters:
        monkeypatch: The pytest monkeypatch fixture (restores on teardown).
        model: Orbit model for offset-mode geometry, or None for absolute mode.
        shadow_u_columns: Pixel columns reported as shadowed.

    Returns:
        The fake Backplane class (exposes ``shadow_calls``).
    """
    bp_cls = _make_backplane_class(model=model, shadow_u_columns=shadow_u_columns)
    monkeypatch.setattr('oops.backplane.Backplane', bp_cls)
    monkeypatch.setattr(RingMosaic, 'longitude_radius_to_pixels', staticmethod(_make_lr2p(model)))
    return bp_cls


def _make_mosaic(**kwargs: Any) -> RingMosaic:
    """Return the standard absolute-radius test mosaic (1000..1020 km, pi/16 rad).

    Parameters:
        kwargs: Extra keyword arguments forwarded to the RingMosaic constructor.

    Returns:
        A RingMosaic for SATURN on the shared synthetic grid.
    """
    return RingMosaic(
        body_name='SATURN',
        radius_inner=1000.0,
        radius_outer=1020.0,
        longitude_resolution=_LON_RES,
        radius_resolution=_RAD_RES,
        **kwargs,
    )


def _make_offset_mosaic(model: RingOrbitModel) -> RingMosaic:
    """Return the offset-radius test mosaic (-10..+10 km about the orbit).

    Parameters:
        model: The orbit model defining the co-rotating frame.

    Returns:
        A RingMosaic for SATURN with offset radius semantics.
    """
    return RingMosaic(
        body_name='SATURN',
        radius_inner=-10.0,
        radius_outer=10.0,
        longitude_resolution=_LON_RES,
        radius_resolution=_RAD_RES,
        orbit_model=model,
    )


def _make_model(*, e: float = 0.0, w0: float = 0.0, mean_motion: float = 0.0) -> RingOrbitModel:
    """Return an orbit model with a = 1010 km centred on the synthetic ring.

    Parameters:
        e: Eccentricity.
        w0: Longitude of pericenter at J2000 (rad).
        mean_motion: Mean motion (rad/day) of the co-rotating frame.

    Returns:
        A RingOrbitModel anchored at the shared test epoch.
    """
    return RingOrbitModel(
        name='reproject-test-model',
        a=1010.0,
        e=e,
        w0=w0,
        dw=0.0,
        mean_motion=mean_motion,
        epoch_utc=_EPOCH_UTC,
    )


def _expected_absolute_img() -> NDArrayFloatType:
    """Expected 5x7 image for absolute mode: rows sample v = 4 + 2 r, bins 8..14."""
    out = np.empty((5, 7), dtype=np.float64)
    for r in range(5):
        for c in range(7):
            out[r, c] = (4 + 2 * r) * 100 + (4 + 2 * c)
    return out


def _expected_offset_img() -> NDArrayFloatType:
    """Expected 5x7 image for offset mode: rows sample v = 3 + 2 r, bins 8..14."""
    out = np.empty((5, 7), dtype=np.float64)
    for r in range(5):
        for c in range(7):
            out[r, c] = (3 + 2 * r) * 100 + (4 + 2 * c)
    return out


# ---------------------------------------------------------------------------
# Validator contracts
# ---------------------------------------------------------------------------


class TestValidateLongitudeRange:
    """Documented error contract of _validate_reproject_longitude_range."""

    @pytest.mark.parametrize('bad', ['abc', {'a': 1}, 1.5, None])
    def test_non_sequence_rejected(self, bad: object) -> None:
        """Anything but a tuple/list raises TypeError naming the expected type."""
        with pytest.raises(TypeError, match='must be a tuple or list of two numbers'):
            _validate_reproject_longitude_range(bad)

    @pytest.mark.parametrize('bad', [(1.0,), (1.0, 2.0, 3.0), ()])
    def test_wrong_length_rejected(self, bad: tuple[float, ...]) -> None:
        """Sequences without exactly two elements raise ValueError."""
        with pytest.raises(ValueError, match='exactly two elements'):
            _validate_reproject_longitude_range(bad)

    def test_bool_endpoint_rejected(self) -> None:
        """bool endpoints are rejected even though bool subclasses int."""
        with pytest.raises(TypeError, match='start must be int or float'):
            _validate_reproject_longitude_range((True, 1.0))

    def test_string_endpoint_rejected(self) -> None:
        """String endpoints raise TypeError naming the offending element."""
        with pytest.raises(TypeError, match='end must be int or float'):
            _validate_reproject_longitude_range((0.5, '1.0'))

    @pytest.mark.parametrize('bad', [(math.nan, 1.0), (0.5, math.inf)])
    def test_non_finite_rejected(self, bad: tuple[float, float]) -> None:
        """NaN or infinite endpoints raise ValueError."""
        with pytest.raises(ValueError, match='must be finite'):
            _validate_reproject_longitude_range(bad)

    def test_negative_start_rejected(self) -> None:
        """Longitudes below 0 raise ValueError."""
        with pytest.raises(ValueError, match='start must satisfy'):
            _validate_reproject_longitude_range((-0.1, 1.0))

    def test_end_beyond_max_longitude_rejected(self) -> None:
        """An end of exactly 2 pi exceeds _MAX_LONGITUDE and raises ValueError."""
        with pytest.raises(ValueError, match='end must satisfy'):
            _validate_reproject_longitude_range((0.0, 2.0 * math.pi))

    def test_start_greater_than_end_rejected(self) -> None:
        """start > end raises ValueError (ranges do not wrap through 0)."""
        with pytest.raises(ValueError, match='start <= end'):
            _validate_reproject_longitude_range((2.0, 1.0))

    def test_seam_crossing_range_rejected(self) -> None:
        """A range crossing the 2 pi -> 0 seam is rejected, not wrapped."""
        with pytest.raises(ValueError, match='start <= end'):
            _validate_reproject_longitude_range((6.0, 0.5))

    def test_valid_list_returns_floats(self) -> None:
        """A valid list is returned as a (start, end) float tuple."""
        assert _validate_reproject_longitude_range([0.25, 0.5]) == (0.25, 0.5)

    def test_numpy_endpoints_accepted(self) -> None:
        """NumPy floating endpoints are accepted and converted to Python floats."""
        out = _validate_reproject_longitude_range((np.float32(0.25), np.float64(0.5)))
        assert out == (float(np.float32(0.25)), 0.5)

    def test_equal_endpoints_accepted(self) -> None:
        """A zero-width range (start == end) is allowed."""
        assert _validate_reproject_longitude_range((1.0, 1.0)) == (1.0, 1.0)


class TestValidateRadiusRange:
    """Documented error contract of _validate_reproject_radius_range."""

    def test_non_sequence_rejected(self) -> None:
        """Anything but a tuple/list raises TypeError."""
        with pytest.raises(TypeError, match='must be a tuple or list of two numbers'):
            _validate_reproject_radius_range('abc')

    def test_wrong_length_rejected(self) -> None:
        """Sequences without exactly two elements raise ValueError."""
        with pytest.raises(ValueError, match='exactly two elements'):
            _validate_reproject_radius_range((1.0, 2.0, 3.0))

    def test_bool_endpoint_rejected(self) -> None:
        """bool endpoints raise TypeError naming the offending element."""
        with pytest.raises(TypeError, match='inner must be int or float'):
            _validate_reproject_radius_range((True, 5.0))

    def test_non_finite_rejected(self) -> None:
        """NaN endpoints raise ValueError."""
        with pytest.raises(ValueError, match='must be finite'):
            _validate_reproject_radius_range((math.nan, 5.0))

    def test_zero_width_range_rejected(self) -> None:
        """inner == outer violates the strict inner < outer requirement."""
        with pytest.raises(ValueError, match='inner < outer'):
            _validate_reproject_radius_range((5.0, 5.0))

    def test_inverted_range_rejected(self) -> None:
        """inner > outer raises ValueError explaining both radius conventions."""
        with pytest.raises(ValueError, match='inner < outer'):
            _validate_reproject_radius_range((7.0, 3.0))

    def test_signed_offsets_accepted(self) -> None:
        """Negative inner values are valid (offset semantics with an orbit model)."""
        assert _validate_reproject_radius_range((-1000.0, 1000.0)) == (-1000.0, 1000.0)

    def test_integer_endpoints_accepted(self) -> None:
        """Plain ints are accepted and converted to floats."""
        assert _validate_reproject_radius_range((1000, 1020)) == (1000.0, 1020.0)


class TestValidateZoomAmt:
    """Documented error contract of _validate_reproject_zoom_amt."""

    def test_scalar_int_duplicated(self) -> None:
        """A single int applies to both axes."""
        assert _validate_reproject_zoom_amt(3) == (3, 3)

    def test_numpy_integer_accepted(self) -> None:
        """NumPy integers are accepted as int-like."""
        assert _validate_reproject_zoom_amt(np.int32(4)) == (4, 4)

    @pytest.mark.parametrize('pair', [(2, 3), [2, 3]])
    def test_two_element_sequence_accepted(self, pair: object) -> None:
        """A (radial, longitude) pair is returned in order."""
        assert _validate_reproject_zoom_amt(pair) == (2, 3)

    @pytest.mark.parametrize('bad', [(2,), (1, 2, 3), []])
    def test_wrong_length_sequence_rejected(self, bad: object) -> None:
        """Sequences without exactly two elements raise ValueError."""
        with pytest.raises(ValueError, match='exactly two elements'):
            _validate_reproject_zoom_amt(bad)

    @pytest.mark.parametrize('bad', [True, np.bool_(True)])
    def test_bool_scalar_rejected(self, bad: object) -> None:
        """bool scalars raise TypeError despite bool subclassing int."""
        with pytest.raises(TypeError, match='not bool'):
            _validate_reproject_zoom_amt(bad)

    def test_bool_element_rejected(self) -> None:
        """A bool inside the pair raises TypeError naming the element."""
        with pytest.raises(TypeError, match='element 2 must be int-like'):
            _validate_reproject_zoom_amt((2, True))

    def test_float_scalar_rejected(self) -> None:
        """Fractional zoom factors raise TypeError."""
        with pytest.raises(TypeError, match='must be int'):
            _validate_reproject_zoom_amt(2.5)

    def test_float_element_rejected(self) -> None:
        """A float inside the pair raises TypeError naming the element."""
        with pytest.raises(TypeError, match='element 2 must be int or numpy integer'):
            _validate_reproject_zoom_amt((2, 2.5))

    def test_zero_accepted_by_validator(self) -> None:
        """Zero passes validation; downstream treats it as zoom 1 with spline order 0.

        The reproject() docstring only defines positive (zoom) and negative (spline
        order) values, so zero falls into an undocumented gap; this test pins the
        validator's current behavior.
        """
        assert _validate_reproject_zoom_amt(0) == (0, 0)

    def test_negative_passes_validator_for_downstream_spline_check(self) -> None:
        """Negative values pass validation; reproject() raises NotImplementedError later."""
        assert _validate_reproject_zoom_amt(-2) == (-2, -2)


# ---------------------------------------------------------------------------
# reproject() argument errors that need no geometry
# ---------------------------------------------------------------------------


class TestReprojectArgumentErrors:
    """Errors raised before any oops geometry is touched."""

    def test_margin_below_one_rejected(self) -> None:
        """margin < 1 raises ValueError before the observation is accessed."""
        mosaic = _make_mosaic()
        with pytest.raises(ValueError, match='margin must be >= 1'):
            mosaic.reproject(object(), margin=0)

    def test_per_call_orbit_model_mismatch_rejected(self) -> None:
        """A per-call orbit model differing from the constructor's raises ValueError."""
        mosaic = _make_mosaic()
        with pytest.raises(ValueError, match='must match the orbit_model passed to'):
            mosaic.reproject(object(), orbit_model=_make_model())

    def test_negative_zoom_spline_order_not_implemented(self) -> None:
        """Negative zoom_amt (spline order) raises the documented NotImplementedError."""
        mosaic = _make_mosaic()
        data = ma.MaskedArray(np.zeros((4, 4)))
        with pytest.raises(NotImplementedError, match='Spline interpolation'):
            mosaic.reproject(object(), data, zoom_amt=-1)

    def test_invalid_longitude_range_rejected_through_reproject(self) -> None:
        """reproject() surfaces the longitude_range validator error unchanged."""
        mosaic = _make_mosaic()
        data = ma.MaskedArray(np.zeros((4, 4)))
        with pytest.raises(ValueError, match='start <= end'):
            mosaic.reproject(object(), data, longitude_range=(2.0, 1.0))

    def test_invalid_radius_range_rejected_through_reproject(self) -> None:
        """reproject() surfaces the radius_range validator error unchanged."""
        mosaic = _make_mosaic()
        data = ma.MaskedArray(np.zeros((4, 4)))
        with pytest.raises(ValueError, match='inner < outer'):
            mosaic.reproject(object(), data, radius_range=(1020.0, 1000.0))


# ---------------------------------------------------------------------------
# Constructor and static grid utilities
# ---------------------------------------------------------------------------


class TestConstructorAndGridValidation:
    """RingMosaic constructor and generate_* error contracts."""

    def test_inverted_radius_bounds_rejected(self) -> None:
        """radius_inner >= radius_outer raises ValueError."""
        with pytest.raises(ValueError, match='must be less than radius_outer'):
            RingMosaic(body_name='SATURN', radius_inner=1020.0, radius_outer=1000.0)

    def test_zero_longitude_resolution_rejected(self) -> None:
        """longitude_resolution == 0 raises ValueError."""
        with pytest.raises(ValueError, match='longitude_resolution must be positive'):
            RingMosaic(
                body_name='SATURN',
                radius_inner=1000.0,
                radius_outer=1020.0,
                longitude_resolution=0.0,
            )

    def test_full_circle_longitude_resolution_rejected(self) -> None:
        """longitude_resolution >= 2 pi raises ValueError."""
        with pytest.raises(ValueError, match='longitude_resolution must be positive'):
            RingMosaic(
                body_name='SATURN',
                radius_inner=1000.0,
                radius_outer=1020.0,
                longitude_resolution=7.0,
            )

    def test_zero_radius_resolution_rejected(self) -> None:
        """radius_resolution <= 0 raises ValueError."""
        with pytest.raises(ValueError, match='radius_resolution must be positive'):
            RingMosaic(
                body_name='SATURN',
                radius_inner=1000.0,
                radius_outer=1020.0,
                radius_resolution=0.0,
            )

    def test_generate_longitudes_rejects_non_positive_resolution(self) -> None:
        """generate_longitudes raises ValueError for resolution <= 0."""
        with pytest.raises(ValueError, match='longitude_resolution must be positive'):
            RingMosaic.generate_longitudes(longitude_resolution=0.0)

    def test_generate_radii_rejects_non_positive_resolution(self) -> None:
        """generate_radii raises ValueError for resolution <= 0."""
        with pytest.raises(ValueError, match='radius_resolution must be positive'):
            RingMosaic.generate_radii(1000.0, 1020.0, radius_resolution=-1.0)

    def test_generate_radii_rejects_inverted_bounds(self) -> None:
        """generate_radii raises ValueError when outer < inner."""
        with pytest.raises(ValueError, match='must be >= radius_inner'):
            RingMosaic.generate_radii(1020.0, 1000.0)

    def test_to_bounded_reuses_longitude_validator(self) -> None:
        """to_bounded rejects inverted ranges with the reproject() validator message."""
        with pytest.raises(ValueError, match='start <= end'):
            _make_mosaic().to_bounded(longitude_range=(2.0, 1.0))


# ---------------------------------------------------------------------------
# Full reprojection through the synthetic geometry
# ---------------------------------------------------------------------------


class TestReprojectAbsoluteGeometry:
    """reproject() against the absolute-radius (no orbit model) geometry."""

    def test_antimask_marks_expected_bins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The sparse antimask marks exactly bins 8..14 of the full-circle grid."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs())
        np.testing.assert_array_equal(np.where(res.longitude_antimask)[0], np.arange(8, 15))

    def test_antimask_length_is_full_circle_bin_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The antimask always spans all full-circle longitude bins."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs())
        assert res.longitude_antimask.shape == (_N_FULL_LON,)

    def test_sparsity_invariant_columns_match_antimask(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """count_nonzero(longitude_antimask) equals the sparse column count."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs())
        assert int(np.count_nonzero(res.longitude_antimask)) == res.img.shape[1]

    def test_image_values_match_analytic_geometry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each cell holds data[4 + 2 r, 4 + 2 c]: bin/radius edges map to those pixels."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs())
        np.testing.assert_array_equal(ma.getdata(res.img), _expected_absolute_img())

    def test_image_fully_valid_for_unmasked_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No cell is masked when every sampled pixel is valid."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs())
        assert not ma.getmaskarray(res.img).any()

    def test_result_grid_fields_echo_mosaic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolutions and default radius bounds are copied from the mosaic."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs())
        assert res.body_name == 'SATURN'
        assert res.longitude_resolution == _LON_RES
        assert res.radius_resolution == _RAD_RES
        assert res.radius_inner == 1000.0
        assert res.radius_outer == 1020.0
        assert res.orbit_model is None
        assert res.photometric_model_name is None

    def test_time_is_observation_midtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The result records obs.midtime."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs(midtime=777.5))
        assert res.time == 777.5

    def test_image_name_is_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The image_name label is stored on the result."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs(), image_name='N12345')
        assert res.image_name == 'N12345'

    def test_mean_radial_resolution_averages_over_radius_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-column mean radial resolution is mean(v + 2) over sampled rows = 10."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs())
        np.testing.assert_allclose(res.mean_radial_resolution, 10.0)

    def test_constant_geometry_means_pass_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constant phase/emission/angular-resolution backplanes yield constant means."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs())
        np.testing.assert_allclose(res.mean_phase, 0.5, rtol=1e-6)
        np.testing.assert_allclose(res.mean_emission, 0.3, rtol=1e-6)
        np.testing.assert_allclose(res.mean_angular_resolution, 0.001, rtol=1e-6)

    def test_incidence_is_mean_of_sampled_pixels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The scalar incidence is the mean of the constant incidence backplane."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs())
        assert res.incidence == pytest.approx(0.4)

    def test_default_dtypes_follow_mosaic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """img is float64 and metadata arrays are float32 by default."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs())
        assert ma.getdata(res.img).dtype == np.dtype(np.float64)
        assert res.mean_radial_resolution.dtype == np.dtype(np.float32)

    def test_custom_dtypes_follow_mosaic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mosaic-level image/metadata dtypes propagate to the result arrays."""
        _install_geometry(monkeypatch)
        mosaic = _make_mosaic(image_dtype=np.float32, metadata_dtype=np.float64)
        res = mosaic.reproject(_FakeObs())
        assert ma.getdata(res.img).dtype == np.dtype(np.float32)
        assert res.mean_phase.dtype == np.dtype(np.float64)
        assert res.image_dtype == np.dtype(np.float32)
        assert res.metadata_dtype == np.dtype(np.float64)

    def test_explicit_data_matches_obs_data_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Passing data=obs.data explicitly reproduces the data=None result."""
        _install_geometry(monkeypatch)
        mosaic = _make_mosaic()
        obs = _FakeObs()
        res_default = mosaic.reproject(obs)
        res_explicit = mosaic.reproject(obs, ma.MaskedArray(obs.data))
        np.testing.assert_array_equal(ma.getdata(res_default.img), ma.getdata(res_explicit.img))

    def test_longitude_range_restricts_bins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A [9, 12] bin-edge range keeps only bins whose pixels fall inside it."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs(), longitude_range=(9 * _LON_RES, 12 * _LON_RES))
        np.testing.assert_array_equal(np.where(res.longitude_antimask)[0], np.array([9, 10, 11]))

    def test_non_grid_aligned_start_keeps_columns_grid_aligned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sub-bin longitude_range start does not shift columns off the global grid.

        The binning origin is the floor of the start snapped to the global grid, so
        the surviving bins are identical to the grid-aligned request and column k
        still means longitude bin_index * longitude_resolution.
        """
        _install_geometry(monkeypatch)
        mosaic = _make_mosaic()
        aligned = mosaic.reproject(_FakeObs(), longitude_range=(9 * _LON_RES, 12 * _LON_RES))
        offset = mosaic.reproject(_FakeObs(), longitude_range=(9.3 * _LON_RES, 12 * _LON_RES))
        np.testing.assert_array_equal(offset.longitude_antimask, aligned.longitude_antimask)
        np.testing.assert_array_equal(ma.getdata(offset.img), ma.getdata(aligned.img))

    def test_single_column_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A narrow longitude range produces a valid single-column sparse result."""
        _install_geometry(monkeypatch)
        mosaic = _make_mosaic()
        res = mosaic.reproject(_FakeObs(), longitude_range=(9 * _LON_RES, 9.9 * _LON_RES))
        assert res.img.shape == (5, 1)
        np.testing.assert_array_equal(np.where(res.longitude_antimask)[0], np.array([9]))
        mosaic.add(res)
        assert mosaic.to_sparse().img.shape == (5, 1)

    def test_custom_radius_range_echoed_and_resized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """radius_range overrides the mosaic bounds and sizes the radius axis."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs(), radius_range=(1000.0, 1010.0))
        assert res.radius_inner == 1000.0
        assert res.radius_outer == 1010.0
        assert res.img.shape[0] == 3
        np.testing.assert_array_equal(ma.getdata(res.img), _expected_absolute_img()[:3, :])

    def test_custom_radius_range_result_rejected_by_add(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A result on different radius bounds cannot be accumulated into the mosaic."""
        _install_geometry(monkeypatch)
        mosaic = _make_mosaic()
        res = mosaic.reproject(_FakeObs(), radius_range=(1000.0, 1010.0))
        with pytest.raises(ValueError, match='radius bounds mismatch'):
            mosaic.add(res)

    def test_margin_excludes_edge_pixels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A large margin removes bins whose sample pixels fall within it."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs(), margin=8)
        np.testing.assert_array_equal(np.where(res.longitude_antimask)[0], np.array([10, 11]))

    def test_result_accumulates_into_mosaic_at_correct_bins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """add() places the sparse columns at their global longitude bins."""
        _install_geometry(monkeypatch)
        mosaic = _make_mosaic()
        res = mosaic.reproject(_FakeObs(), image_name='img_a')
        mosaic.add(res)
        full = mosaic.to_full()
        np.testing.assert_array_equal(ma.getdata(full.img)[:, 8:15], _expected_absolute_img())
        assert full.contributing_image_names == ('img_a',)

    def test_masked_input_column_drops_longitude_bin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Masking every pixel a bin samples removes that bin from the result."""
        _install_geometry(monkeypatch)
        obs = _FakeObs()
        data = ma.MaskedArray(obs.data.copy())
        data[:, 4:6] = ma.masked  # u = 4 is bin 8's only sampled column
        res = _make_mosaic().reproject(obs, data)
        np.testing.assert_array_equal(np.where(res.longitude_antimask)[0], np.arange(9, 15))
        np.testing.assert_array_equal(ma.getdata(res.img), _expected_absolute_img()[:, 1:])

    def test_single_masked_pixel_masks_single_cell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One masked pixel masks exactly the cell that samples it; the column stays."""
        _install_geometry(monkeypatch)
        obs = _FakeObs()
        data = ma.MaskedArray(obs.data.copy())
        data[4, 6] = ma.masked  # sampled by radius row 0 of bin 9 (column 1)
        res = _make_mosaic().reproject(obs, data)
        mask = ma.getmaskarray(res.img)
        assert mask[0, 1]
        assert int(mask.sum()) == 1
        assert res.img.shape == (5, 7)

    def test_shadowed_bin_dropped_when_omit_shadow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pixels inside the shadow are masked, dropping fully shadowed bins."""
        _install_geometry(monkeypatch, shadow_u_columns=(8,))  # bin 10 samples u = 8
        res = _make_mosaic().reproject(_FakeObs(), omit_shadow=True)
        np.testing.assert_array_equal(
            np.where(res.longitude_antimask)[0], np.array([8, 9, 11, 12, 13, 14])
        )

    def test_shadow_ignored_when_omit_shadow_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """omit_shadow=False keeps shadowed pixels and never computes the shadow."""
        bp_cls = _install_geometry(monkeypatch, shadow_u_columns=(8,))
        res = _make_mosaic().reproject(_FakeObs(), omit_shadow=False)
        np.testing.assert_array_equal(np.where(res.longitude_antimask)[0], np.arange(8, 15))
        assert bp_cls.shadow_calls == 0

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_empty_intersection_returns_empty_sparse_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A radius window missing the visible ring yields zero sparse columns."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs(), radius_range=(5000.0, 5100.0))
        assert res.img.shape[1] == 0
        assert not res.longitude_antimask.any()
        assert res.mean_phase.shape == (0,)
        assert math.isnan(res.incidence)

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_zero_width_longitude_range_returns_empty_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """start == end selects no pixels and yields an empty sparse result."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs(), longitude_range=(9 * _LON_RES, 9 * _LON_RES))
        assert res.img.shape == (5, 0)
        assert not res.longitude_antimask.any()

    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_adding_empty_result_is_a_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """add() of a zero-column result records no image and stores no columns."""
        _install_geometry(monkeypatch)
        mosaic = _make_mosaic()
        res = mosaic.reproject(_FakeObs(), longitude_range=(9 * _LON_RES, 9 * _LON_RES))
        mosaic.add(res)
        data = mosaic.to_sparse()
        assert not data.longitude_antimask.any()
        assert data.contributing_image_names == ()

    def test_zoom_without_orbit_model_not_implemented(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zoom in inertial (absolute-radius) mode raises NotImplementedError."""
        _install_geometry(monkeypatch)
        with pytest.raises(NotImplementedError, match='non-corotating inertial radius'):
            _make_mosaic().reproject(_FakeObs(), zoom_amt=2)

    def test_add_overflow_after_uint16_images(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """add() raises OverflowError once the uint16 image counter is exhausted."""
        _install_geometry(monkeypatch)
        mosaic = _make_mosaic()
        res = mosaic.reproject(_FakeObs())
        mosaic._image_count = np.iinfo(np.uint16).max + 1
        with pytest.raises(OverflowError, match='exceeds uint16 max'):
            mosaic.add(res)

    def test_add_rejects_antimask_column_count_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """add() rejects results whose antimask true-count disagrees with img columns."""
        _install_geometry(monkeypatch)
        mosaic = _make_mosaic()
        res = mosaic.reproject(_FakeObs())
        corrupt_antimask = res.longitude_antimask.copy()
        corrupt_antimask[0] = True  # one more True than sparse columns
        corrupt = dataclasses.replace(res, longitude_antimask=corrupt_antimask)
        with pytest.raises(ValueError, match='one True entry per sparse column'):
            mosaic.add(corrupt)

    def test_uv_range_with_shadow_masking_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """uv_range with the default omit_shadow=True must not crash."""
        _install_geometry(monkeypatch)
        res = _make_mosaic().reproject(_FakeObs(), uv_range=(2, 17, 2, 17))
        assert res.img.shape[0] == 5

    def test_uv_range_preserves_reprojected_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Restricting uv_range must not change the values of surviving cells."""
        _install_geometry(monkeypatch)
        mosaic = _make_mosaic()
        res = mosaic.reproject(_FakeObs(), uv_range=(2, 17, 2, 17), omit_shadow=False)
        bins = np.where(res.longitude_antimask)[0]
        np.testing.assert_array_equal(bins, np.arange(9, 14))
        # Analytic expectation: cell (r, bin b) samples data[4 + 2 r, (b - 6) * 2].
        expected = np.empty(res.img.shape, dtype=np.float64)
        for r in range(expected.shape[0]):
            for c, b in enumerate(bins):
                expected[r, c] = (4 + 2 * r) * 100 + (b - 6) * 2
        valid = ~ma.getmaskarray(res.img)
        assert valid.any()
        np.testing.assert_array_equal(ma.getdata(res.img)[valid], expected[valid])


class TestReprojectWithOrbitModel:
    """Offset-radius and co-rotating longitude semantics."""

    def test_offset_rows_straighten_circular_orbit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a circular orbit model, offset rows sample constant-offset pixels."""
        model = _make_model()
        _install_geometry(monkeypatch, model=model)
        res = _make_offset_mosaic(model).reproject(_FakeObs())
        np.testing.assert_array_equal(np.where(res.longitude_antimask)[0], np.arange(8, 15))
        np.testing.assert_array_equal(ma.getdata(res.img), _expected_offset_img())

    def test_offset_rows_straighten_eccentric_orbit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An eccentric ring reprojects to the same straight rows as a circular one.

        The absolute radius sampled varies with longitude, but each output row holds
        a constant signed offset from the orbit, per the documented offset semantics.
        """
        model = _make_model(e=0.05, w0=0.3)
        _install_geometry(monkeypatch, model=model)
        res = _make_offset_mosaic(model).reproject(_FakeObs())
        np.testing.assert_array_equal(ma.getdata(res.img), _expected_offset_img())

    def test_result_radius_bounds_are_offsets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The result echoes the signed offset bounds, not absolute radii."""
        model = _make_model()
        _install_geometry(monkeypatch, model=model)
        res = _make_offset_mosaic(model).reproject(_FakeObs())
        assert res.radius_inner == -10.0
        assert res.radius_outer == 10.0
        assert res.orbit_model == model

    def test_corotating_longitudes_shift_antimask_bins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One day past epoch with mean_motion -2 bins/day, bins shift by +2."""
        model = _make_model(mean_motion=-2 * _LON_RES)
        _install_geometry(monkeypatch, model=model)
        obs = _FakeObs(midtime=_EPOCH_ET + 86400.0)
        res = _make_offset_mosaic(model).reproject(obs)
        np.testing.assert_array_equal(np.where(res.longitude_antimask)[0], np.arange(10, 17))
        np.testing.assert_array_equal(ma.getdata(res.img), _expected_offset_img())

    def test_corotating_bins_wrap_across_longitude_seam(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A co-rotating shift of 20 bins wraps the result across the 2 pi -> 0 seam."""
        model = _make_model(mean_motion=-20 * _LON_RES)
        _install_geometry(monkeypatch, model=model)
        obs = _FakeObs(midtime=_EPOCH_ET + 86400.0)
        res = _make_offset_mosaic(model).reproject(obs)
        corot_bins = np.where(res.longitude_antimask)[0]
        np.testing.assert_array_equal(corot_bins, np.array([0, 1, 2, 28, 29, 30, 31]))
        # Columns are sorted by co-rotating bin; recover each column's inertial bin.
        expected = np.empty(res.img.shape, dtype=np.float64)
        for c, b in enumerate(corot_bins):
            inertial_bin = (b - 20) % _N_FULL_LON
            for r in range(expected.shape[0]):
                expected[r, c] = (3 + 2 * r) * 100 + (4 + 2 * (inertial_bin - 8))
        np.testing.assert_array_equal(ma.getdata(res.img), expected)

    def test_per_call_value_equal_orbit_model_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A distinct but value-equal per-call orbit model is accepted."""
        model = _make_model()
        _install_geometry(monkeypatch, model=model)
        twin = _make_model()
        assert twin is not model
        res = _make_offset_mosaic(model).reproject(_FakeObs(), orbit_model=twin)
        assert res.orbit_model == model

    def test_zoom_preserves_constant_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """(2, 2) sub-pixel zoom returns the unzoomed grid shape and exact values.

        With a constant input image the zoomed samples all read the same value, so
        the unzoom averaging must return it exactly.
        """
        model = _make_model()
        _install_geometry(monkeypatch, model=model)
        obs = _FakeObs(data=np.full((_N, _N), 7.0))
        res = _make_offset_mosaic(model).reproject(obs, zoom_amt=(2, 2))
        assert res.img.shape == (5, 7)
        np.testing.assert_array_equal(np.where(res.longitude_antimask)[0], np.arange(8, 15))
        valid = ~ma.getmaskarray(res.img)
        assert valid.any()
        np.testing.assert_array_equal(ma.getdata(res.img)[valid], 7.0)


# ---------------------------------------------------------------------------
# orbit_pixels and the coordinate-conversion empty path
# ---------------------------------------------------------------------------


class _FakeExtBpObs:
    """Observation stand-in for orbit_pixels: exposes ext_bp and extdata shape."""

    def __init__(self) -> None:
        """Build the analytic full-frame backplane as the extended backplane."""
        bp_cls = _make_backplane_class(model=None, shadow_u_columns=())
        self.ext_bp = bp_cls(self)
        self.extdata_shape_uv = (_N, _N)
        self.midtime = 0.0


class TestOrbitPixels:
    """orbit_pixels filtering against the synthetic geometry."""

    def test_orbit_inside_image_yields_constant_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A circular orbit at 1010 km maps to the constant fractional row 8.5."""
        monkeypatch.setattr(
            RingMosaic, 'longitude_radius_to_pixels', staticmethod(_make_lr2p(None))
        )
        u_pix, v_pix = RingMosaic.orbit_pixels(_FakeExtBpObs(), _make_model())
        assert len(u_pix) > 0
        np.testing.assert_allclose(v_pix, 8.5)

    def test_returned_pixels_are_inside_the_fov(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All returned (u, v) pairs lie within the extended data bounds."""
        monkeypatch.setattr(
            RingMosaic, 'longitude_radius_to_pixels', staticmethod(_make_lr2p(None))
        )
        u_pix, v_pix = RingMosaic.orbit_pixels(_FakeExtBpObs(), _make_model())
        assert bool(np.all(u_pix >= 0))
        assert bool(np.all(u_pix <= _N - 1))
        assert bool(np.all(v_pix >= 0))
        assert bool(np.all(v_pix <= _N - 1))

    def test_orbit_outside_radius_range_yields_no_pixels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An orbit whose radius never intersects the backplane returns empty arrays."""
        monkeypatch.setattr(
            RingMosaic, 'longitude_radius_to_pixels', staticmethod(_make_lr2p(None))
        )
        far_model = RingOrbitModel(
            name='far',
            a=5000.0,
            e=0.0,
            w0=0.0,
            dw=0.0,
            mean_motion=0.0,
            epoch_utc=_EPOCH_UTC,
        )
        u_pix, v_pix = RingMosaic.orbit_pixels(_FakeExtBpObs(), far_model)
        assert len(u_pix) == 0
        assert len(v_pix) == 0


class TestLongitudeRadiusToPixelsEmptyPath:
    """The real longitude_radius_to_pixels short-circuits on empty input."""

    def test_empty_input_returns_empty_pixels(self) -> None:
        """Empty longitude/radius arrays return empty pixel arrays without oops calls."""
        u, v = RingMosaic.longitude_radius_to_pixels(None, np.zeros(0), np.zeros(0))
        assert u.shape == (0,)
        assert v.shape == (0,)

    def test_empty_input_with_orbit_model(self) -> None:
        """The empty path also works after the co-rotating conversion."""
        obs = SimpleNamespace(midtime=0.0)
        u, v = RingMosaic.longitude_radius_to_pixels(
            obs, np.zeros(0), np.zeros(0), orbit_model=_make_model()
        )
        assert u.shape == (0,)
        assert v.shape == (0,)
