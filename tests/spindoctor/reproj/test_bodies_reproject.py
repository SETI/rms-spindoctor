"""Spec-first unit tests for the ``BodyMosaic.reproject()`` pipeline.

These tests exercise the full ``reproject()`` inner loop hermetically (no SPICE,
no holdings) by combining a synthetic observation whose ``uv_from_coords`` is an
exact linear map with an ``override_backplane`` fake whose latitude/longitude
backplanes are the inverse of that map.  Each detector pixel ``(v, u)`` is placed
at the CENTER of latitude bin ``lat0 + v`` and longitude bin ``lon0 + u``, so the
expected reprojection grid follows directly from the documented floor/ceil bin
arithmetic: ``lat_idx_range == (lat0, lat0 + nv)`` and the extra ceil row/column
is masked.

Contract sources: the ``reproject()`` / ``BodyMosaic`` docstrings in
``src/spindoctor/reproj/bodies.py`` and ``docs/dev_guide/dev_guide_reprojection.rst``
(grid semantics, dtype propagation, photometric correction, masking rules).
"""

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import polymath
import pytest
from tests.spindoctor.reproj.test_bodies import _make_repro

from spindoctor.reproj.bodies import BodyMosaic, BodyReprojResult
from spindoctor.reproj.cartographic_model import create_cartographic_model
from spindoctor.reproj.photometric_model import LambertModel

_LAT_RES = 0.1  # rad/pixel
_LON_RES = 0.1  # rad/pixel
_LAT0 = 10  # full-grid latitude bin of detector row 0
_LON0 = 20  # full-grid longitude bin of detector column 0
_HALF_PI = math.pi / 2.0
_DEF_INCIDENCE = 0.3  # rad
_DEF_EMISSION = 0.2  # rad
_DEF_PHASE = 0.7  # rad
_DEF_CENTER_RES = 5.0  # km/pixel
_N_FULL_LAT = math.floor(math.pi / _LAT_RES) + 1  # 32
_N_FULL_LON = math.floor(2.0 * math.pi / _LON_RES) + 1  # 63


class FakeBackplane:
    """Backplane stand-in mapping detector pixel (v, u) to bin centers.

    Latitude of pixel ``(v, u)`` is ``(lat0 + v_offset + v + 0.5) * _LAT_RES - pi/2``
    and longitude is ``(lon0 + u_offset + u + 0.5) * _LON_RES``, i.e. each pixel
    sits exactly at the center of one full-grid bin.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        lat0: int = _LAT0,
        lon0: int = _LON0,
        v_offset: int = 0,
        u_offset: int = 0,
        lat_mask: npt.NDArray[np.bool_] | bool = False,
        lon_mask: npt.NDArray[np.bool_] | bool = False,
        incidence: npt.NDArray[np.float64] | None = None,
        emission: npt.NDArray[np.float64] | None = None,
        phase: npt.NDArray[np.float64] | None = None,
        center_res: float = _DEF_CENTER_RES,
        sub_solar: tuple[float, float] = (0.1, 0.2),
        sub_observer: tuple[float, float] = (0.3, 0.4),
    ) -> None:
        """Build the fake backplane arrays.

        Parameters:
            shape: (n_v, n_u) shape of the backplane arrays (detector or subimage).
            lat0: Full-grid latitude bin of detector row 0.
            lon0: Full-grid longitude bin of detector column 0.
            v_offset: Detector row of this backplane's row 0 (for subimage backplanes).
            u_offset: Detector column of this backplane's column 0.
            lat_mask: Mask for the latitude backplane (True = no surface point).
            lon_mask: Mask for the longitude backplane.
            incidence: Per-pixel incidence angle (rad); scalar default 0.3.
            emission: Per-pixel emission angle (rad); scalar default 0.2.
            phase: Per-pixel phase angle (rad); scalar default 0.7.
            center_res: Scalar center resolution (km/pixel).
            sub_solar: (longitude, latitude) sub-solar point (rad).
            sub_observer: (longitude, latitude) sub-observer point (rad).
        """
        nv, nu = shape
        v_idx, u_idx = np.meshgrid(np.arange(nv), np.arange(nu), indexing='ij')
        lat = (lat0 + v_offset + v_idx + 0.5) * _LAT_RES - _HALF_PI
        lon = (lon0 + u_offset + u_idx + 0.5) * _LON_RES
        self._lat = ma.MaskedArray(lat, mask=lat_mask)
        self._lon = ma.MaskedArray(lon, mask=lon_mask)
        inc = incidence if incidence is not None else np.full(shape, _DEF_INCIDENCE)
        emi = emission if emission is not None else np.full(shape, _DEF_EMISSION)
        pha = phase if phase is not None else np.full(shape, _DEF_PHASE)
        self._incidence = ma.MaskedArray(inc)
        self._emission = ma.MaskedArray(emi)
        self._phase = ma.MaskedArray(pha)
        self._center_res = center_res
        self._sub_solar = sub_solar
        self._sub_observer = sub_observer

    def latitude(self, name: str, lat_type: str = 'centric') -> SimpleNamespace:
        """Return the latitude backplane wrapper.

        Parameters:
            name: Body name (ignored).
            lat_type: Latitude type (ignored).
        """
        return SimpleNamespace(mvals=self._lat)

    def longitude(
        self, name: str, direction: str = 'east', lon_type: str = 'centric'
    ) -> SimpleNamespace:
        """Return the longitude backplane wrapper.

        Parameters:
            name: Body name (ignored).
            direction: Longitude direction (ignored).
            lon_type: Longitude type (ignored).
        """
        return SimpleNamespace(mvals=self._lon)

    def incidence_angle(self, name: str) -> SimpleNamespace:
        """Return the incidence backplane wrapper.

        Parameters:
            name: Body name (ignored).
        """
        return SimpleNamespace(mvals=self._incidence)

    def emission_angle(self, name: str) -> SimpleNamespace:
        """Return the emission backplane wrapper.

        Parameters:
            name: Body name (ignored).
        """
        return SimpleNamespace(mvals=self._emission)

    def phase_angle(self, name: str) -> SimpleNamespace:
        """Return the phase backplane wrapper.

        Parameters:
            name: Body name (ignored).
        """
        return SimpleNamespace(mvals=self._phase)

    def center_resolution(self, name: str) -> SimpleNamespace:
        """Return the scalar center-resolution wrapper.

        Parameters:
            name: Body name (ignored).
        """
        return SimpleNamespace(vals=self._center_res)

    def sub_solar_longitude(self, name: str) -> SimpleNamespace:
        """Return the sub-solar longitude wrapper.

        Parameters:
            name: Body name (ignored).
        """
        return SimpleNamespace(vals=self._sub_solar[0])

    def sub_solar_latitude(self, name: str) -> SimpleNamespace:
        """Return the sub-solar latitude wrapper.

        Parameters:
            name: Body name (ignored).
        """
        return SimpleNamespace(vals=self._sub_solar[1])

    def sub_observer_longitude(self, name: str) -> SimpleNamespace:
        """Return the sub-observer longitude wrapper.

        Parameters:
            name: Body name (ignored).
        """
        return SimpleNamespace(vals=self._sub_observer[0])

    def sub_observer_latitude(self, name: str) -> SimpleNamespace:
        """Return the sub-observer latitude wrapper.

        Parameters:
            name: Body name (ignored).
        """
        return SimpleNamespace(vals=self._sub_observer[1])


class FakeObs:
    """Observation stand-in with an exact linear lat/lon -> (u, v) mapping.

    Inverts the :class:`FakeBackplane` geometry: bin edge ``lat = bin * _LAT_RES - pi/2``
    maps to full-frame pixel coordinate ``v = bin - lat0`` (and similarly for u), with
    rounding to defeat float jitter and optional fractional offsets and per-bin masking.
    """

    def __init__(
        self,
        data: npt.NDArray[np.float64],
        *,
        lat0: int = _LAT0,
        lon0: int = _LON0,
        midtime: float = 1000.0,
        frac_u: float = 0.0,
        frac_v: float = 0.0,
        masked_lon_bin: int | None = None,
    ) -> None:
        """Build the fake observation.

        Parameters:
            data: Full-frame detector image, shape (n_v, n_u).
            lat0: Full-grid latitude bin of detector row 0 (must match the backplane).
            lon0: Full-grid longitude bin of detector column 0.
            midtime: Observation midtime (TDB seconds).
            frac_u: Fractional pixel offset added to every u coordinate.
            frac_v: Fractional pixel offset added to every v coordinate.
            masked_lon_bin: If set, the returned UV pair is masked for samples at
                this full-grid longitude bin (simulates an invalid surface point).
        """
        self.data = data
        self.midtime = midtime
        self._lat0 = lat0
        self._lon0 = lon0
        self._frac_u = frac_u
        self._frac_v = frac_v
        self._masked_lon_bin = masked_lon_bin

    def uv_from_coords(self, surface: Any, coords: Any) -> Any:
        """Map (longitude, latitude) arrays to a polymath UV Pair.

        Parameters:
            surface: Body surface object (ignored).
            coords: (longitude, latitude) polymath Scalars in radians.
        """
        lon, lat = coords
        lon_scalar = polymath.Scalar(lon)
        lat_scalar = polymath.Scalar(lat)
        lon_v = np.asarray(lon_scalar.vals, dtype=np.float64)
        lat_v = np.asarray(lat_scalar.vals, dtype=np.float64)
        lon_bins = np.round(lon_v / _LON_RES)
        u = lon_bins - self._lon0 + self._frac_u
        v = np.round((lat_v + _HALF_PI) / _LAT_RES) - self._lat0 + self._frac_v
        vals = np.stack([u, v], axis=-1)
        if self._masked_lon_bin is not None:
            return polymath.Pair(vals, lon_bins.astype(int) == self._masked_lon_bin)
        return polymath.Pair(vals)


def _make_mosaic(**overrides: Any) -> BodyMosaic:
    """Return a BodyMosaic wired for the fake linear geometry.

    Parameters:
        overrides: Keyword arguments overriding the defaults (squashed latlon type,
            zero edge margin, 0.1 rad resolutions, body MIMAS).
    """
    kwargs: dict[str, Any] = {
        'body_name': 'MIMAS',
        'lat_resolution': _LAT_RES,
        'lon_resolution': _LON_RES,
        'latlon_type': 'squashed',
        'edge_margin': 0,
    }
    kwargs.update(overrides)
    return BodyMosaic(**kwargs)


def _reproject(mosaic: BodyMosaic, obs: FakeObs, bp: FakeBackplane, **kwargs: Any) -> Any:
    """Run mosaic.reproject with oops.Body.lookup stubbed out.

    Parameters:
        mosaic: The BodyMosaic under test.
        obs: Fake observation.
        bp: Fake backplane passed as ``override_backplane``.
        kwargs: Extra keyword arguments forwarded to ``reproject``.
    """
    with patch('oops.Body.lookup', return_value=SimpleNamespace(surface=object())):
        return mosaic.reproject(obs, override_backplane=bp, **kwargs)


def _scene(nv: int = 6, nu: int = 8) -> tuple[npt.NDArray[np.float64], FakeObs, FakeBackplane]:
    """Return a standard non-square scene (data, obs, backplane).

    Parameters:
        nv: Number of detector rows.
        nu: Number of detector columns.
    """
    data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
    return data, FakeObs(data), FakeBackplane((nv, nu))


# =========================================================================
# latitude_longitude_to_pixels
# =========================================================================


class TestLatitudeLongitudeToPixels:
    """Contract tests for BodyMosaic.latitude_longitude_to_pixels."""

    def test_empty_input_returns_empty_pair(self) -> None:
        """Zero-length inputs return a UV Pair with shape (0, 2)."""
        mosaic = _make_mosaic()
        uv = mosaic.latitude_longitude_to_pixels(object(), [], [])
        assert uv.vals.shape == (0, 2)

    def test_squashed_east_passes_coords_unchanged(self) -> None:
        """With latlon_type='squashed' and east longitudes, coords reach uv unchanged."""
        received: list[tuple[Any, Any]] = []

        class _Obs:
            """Observation recording the coordinates given to uv_from_coords."""

            def uv_from_coords(self, surface: Any, coords: Any) -> Any:
                """Record coords and return a zero UV Pair.

                Parameters:
                    surface: Body surface object (ignored).
                    coords: (longitude, latitude) polymath Scalars.
                """
                lon, lat = coords
                received.append((np.asarray(lon.vals), np.asarray(lat.vals)))
                return polymath.Pair(np.zeros((2, 2)))

        mosaic = _make_mosaic()
        with patch('oops.Body.lookup', return_value=SimpleNamespace(surface=object())):
            mosaic.latitude_longitude_to_pixels(_Obs(), [0.1, -0.4], [1.0, 2.0])
        np.testing.assert_allclose(received[0][0], [1.0, 2.0])
        np.testing.assert_allclose(received[0][1], [0.1, -0.4])

    def test_west_direction_negates_longitude_mod_2pi(self) -> None:
        """lon_direction='west' converts longitudes to (-lon) mod 2*pi before uv."""
        received: list[Any] = []

        class _Obs:
            """Observation recording the longitude given to uv_from_coords."""

            def uv_from_coords(self, surface: Any, coords: Any) -> Any:
                """Record the longitude and return a zero UV Pair.

                Parameters:
                    surface: Body surface object (ignored).
                    coords: (longitude, latitude) polymath Scalars.
                """
                received.append(np.asarray(coords[0].vals))
                return polymath.Pair(np.zeros((2, 2)))

        mosaic = _make_mosaic(lon_direction='west')
        with patch('oops.Body.lookup', return_value=SimpleNamespace(surface=object())):
            mosaic.latitude_longitude_to_pixels(_Obs(), [0.0, 0.0], [1.0, 2.0])
        np.testing.assert_allclose(received[0], [2.0 * math.pi - 1.0, 2.0 * math.pi - 2.0])

    def test_centric_converts_via_surface_and_feeds_converted_lon(self) -> None:
        """latlon_type='centric' converts lon first, then lat using the converted lon."""

        class _Surface:
            """Surface stub converting centric coords with recordable arguments."""

            def __init__(self) -> None:
                """Initialize the lat-conversion argument recorder."""
                self.lat_call_lon: npt.NDArray[np.float64] | None = None

            def lon_from_centric(self, lon: Any) -> Any:
                """Return lon + 0.5 as the 'squashed' longitude.

                Parameters:
                    lon: Centric longitude Scalar.
                """
                return polymath.Scalar.as_scalar(np.asarray(lon.vals) + 0.5)

            def lat_from_centric(self, lat: Any, lon: Any) -> Any:
                """Record lon and return lat + 0.25 as the 'squashed' latitude.

                Parameters:
                    lat: Centric latitude Scalar.
                    lon: Longitude Scalar (should be the converted one).
                """
                self.lat_call_lon = np.asarray(lon.vals).copy()
                return polymath.Scalar.as_scalar(np.asarray(lat.vals) + 0.25)

        received: list[tuple[Any, Any]] = []

        class _Obs:
            """Observation recording the coordinates given to uv_from_coords."""

            def uv_from_coords(self, surface: Any, coords: Any) -> Any:
                """Record coords and return a zero UV Pair.

                Parameters:
                    surface: Body surface object (ignored).
                    coords: (longitude, latitude) polymath Scalars.
                """
                received.append((np.asarray(coords[0].vals), np.asarray(coords[1].vals)))
                return polymath.Pair(np.zeros((2, 2)))

        surface = _Surface()
        mosaic = _make_mosaic(latlon_type='centric')
        with patch('oops.Body.lookup', return_value=SimpleNamespace(surface=surface)):
            mosaic.latitude_longitude_to_pixels(_Obs(), [0.1, 0.2], [1.0, 2.0])
        np.testing.assert_allclose(received[0][0], [1.5, 2.5])
        np.testing.assert_allclose(received[0][1], [0.35, 0.45])
        assert surface.lat_call_lon is not None
        np.testing.assert_allclose(surface.lat_call_lon, [1.5, 2.5])


# =========================================================================
# navigation_uncertainty validation
# =========================================================================


class TestNavigationUncertaintyValidation:
    """reproject() rejects invalid navigation_uncertainty per its docstring."""

    def test_bool_raises_type_error(self) -> None:
        """A bool is not accepted as navigation_uncertainty."""
        _data, obs, bp = _scene()
        mosaic = _make_mosaic()
        with pytest.raises(TypeError, match='navigation_uncertainty must be a real number'):
            _reproject(mosaic, obs, bp, navigation_uncertainty=True)

    def test_negative_raises_value_error(self) -> None:
        """A negative value raises ValueError."""
        _data, obs, bp = _scene()
        mosaic = _make_mosaic()
        with pytest.raises(ValueError, match='must be finite and >= 0'):
            _reproject(mosaic, obs, bp, navigation_uncertainty=-0.1)

    def test_infinite_raises_value_error(self) -> None:
        """An infinite value raises ValueError."""
        _data, obs, bp = _scene()
        mosaic = _make_mosaic()
        with pytest.raises(ValueError, match='must be finite and >= 0'):
            _reproject(mosaic, obs, bp, navigation_uncertainty=math.inf)

    def test_nan_raises_value_error(self) -> None:
        """A NaN value raises ValueError."""
        _data, obs, bp = _scene()
        mosaic = _make_mosaic()
        with pytest.raises(ValueError, match='must be finite and >= 0'):
            _reproject(mosaic, obs, bp, navigation_uncertainty=math.nan)


# =========================================================================
# Grid construction and pixel sampling
# =========================================================================


class TestReprojectGrid:
    """Grid semantics: bin index ranges, shapes, and value placement."""

    def test_lat_idx_range_follows_floor_ceil_of_coverage(self) -> None:
        """lat_idx_range spans floor(min lat bin) to ceil(max lat bin) of valid pixels."""
        _data, obs, bp = _scene(nv=6, nu=8)
        result = _reproject(_make_mosaic(), obs, bp)
        assert result.lat_idx_range == (_LAT0, _LAT0 + 6)

    def test_lon_idx_range_follows_floor_ceil_of_coverage(self) -> None:
        """lon_idx_range spans floor(min lon bin) to ceil(max lon bin) of valid pixels."""
        _data, obs, bp = _scene(nv=6, nu=8)
        result = _reproject(_make_mosaic(), obs, bp)
        assert result.lon_idx_range == (_LON0, _LON0 + 8)

    def test_img_shape_matches_idx_ranges(self) -> None:
        """img covers exactly the inclusive lat/lon index ranges (non-square)."""
        _data, obs, bp = _scene(nv=6, nu=8)
        result = _reproject(_make_mosaic(), obs, bp)
        assert result.img.shape == (7, 9)

    def test_img_values_sampled_from_mapped_detector_pixels(self) -> None:
        """Cell (r, c) holds data[v, u] of the detector pixel its bin maps to."""
        data, obs, bp = _scene(nv=6, nu=8)
        result = _reproject(_make_mosaic(), obs, bp)
        np.testing.assert_allclose(result.img.data[:6, :8], data)

    def test_bins_with_no_detector_pixel_are_masked(self) -> None:
        """The extra ceil row and column (mapping off-detector) are masked."""
        _data, obs, bp = _scene(nv=6, nu=8)
        result = _reproject(_make_mosaic(), obs, bp)
        mask = ma.getmaskarray(result.img)
        assert bool(mask[6, :].all())
        assert bool(mask[:, 8].all())
        assert not mask[:6, :8].any()

    def test_metadata_arrays_share_img_window(self) -> None:
        """resolution/eff_resolution/phase/emission/incidence match the img window."""
        _data, obs, bp = _scene(nv=6, nu=8)
        result = _reproject(_make_mosaic(), obs, bp)
        assert result.resolution.shape == result.img.shape
        assert result.eff_resolution.shape == result.img.shape
        assert result.phase.shape == result.img.shape
        assert result.emission.shape == result.img.shape
        assert result.incidence.shape == result.img.shape

    def test_data_defaults_to_obs_data(self) -> None:
        """When data is not given, obs.data is reprojected."""
        data, obs, bp = _scene()
        explicit = _reproject(_make_mosaic(), obs, bp, data=data)
        default = _reproject(_make_mosaic(), obs, bp)
        np.testing.assert_allclose(default.img.filled(-1.0), explicit.img.filled(-1.0))

    def test_explicit_data_overrides_obs_data(self) -> None:
        """An explicit data array is used instead of obs.data."""
        data, obs, bp = _scene()
        other = data + 100.0
        result = _reproject(_make_mosaic(), obs, bp, data=other)
        np.testing.assert_allclose(result.img.data[:6, :8], other)

    def test_scalar_metadata_on_result(self) -> None:
        """time, image_name, and sub-solar/sub-observer scalars come from obs and bp."""
        _data, obs, bp = _scene()
        result = _reproject(_make_mosaic(), obs, bp, image_name='img_stem')
        assert result.time == pytest.approx(1000.0)
        assert result.image_name == 'img_stem'
        assert result.sub_solar_lon == pytest.approx(0.1)
        assert result.sub_solar_lat == pytest.approx(0.2)
        assert result.sub_observer_lon == pytest.approx(0.3)
        assert result.sub_observer_lat == pytest.approx(0.4)

    def test_grid_configuration_echoed_on_result(self) -> None:
        """Resolutions, latlon_type, lon_direction, and body_name echo the mosaic."""
        _data, obs, bp = _scene()
        result = _reproject(_make_mosaic(), obs, bp)
        assert result.body_name == 'MIMAS'
        assert result.lat_resolution == pytest.approx(_LAT_RES)
        assert result.lon_resolution == pytest.approx(_LON_RES)
        assert result.latlon_type == 'squashed'
        assert result.lon_direction == 'east'

    def test_pole_adjacent_coverage_clips_to_grid(self) -> None:
        """Latitude bins beyond the top of the full grid are clipped to n_full_lat - 1."""
        nv, nu = 4, 4
        lat0 = _N_FULL_LAT - 4  # ceil of the max bin would exceed the grid
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data, lat0=lat0)
        bp = FakeBackplane((nv, nu), lat0=lat0)
        result = _reproject(_make_mosaic(), obs, bp)
        assert result.lat_idx_range == (lat0, _N_FULL_LAT - 1)
        assert not ma.getmaskarray(result.img)[3, 0]

    def test_longitude_edge_coverage_clips_to_grid(self) -> None:
        """Longitude bins beyond the top of the full grid are clipped to n_full_lon - 1."""
        nv, nu = 4, 4
        lon0 = _N_FULL_LON - 3
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data, lon0=lon0)
        bp = FakeBackplane((nv, nu), lon0=lon0)
        result = _reproject(_make_mosaic(), obs, bp)
        assert result.lon_idx_range == (lon0, _N_FULL_LON - 1)


# =========================================================================
# Masking rules and limits
# =========================================================================


class TestReprojectMasking:
    """Zero data, backplane masks, uv masks, edge margins, and angle limits."""

    def test_zero_data_pixel_is_masked(self) -> None:
        """Pixels with data == 0 are treated as invalid and their bins are masked."""
        data, obs, bp = _scene()
        data[2, 3] = 0.0
        result = _reproject(_make_mosaic(), obs, bp)
        assert bool(ma.getmaskarray(result.img)[2, 3])
        assert not ma.getmaskarray(result.img)[2, 2]

    def test_mask_bad_areas_expands_zero_regions(self) -> None:
        """mask_bad_areas=True grows each zero-pixel region by one pixel in all directions."""
        data, obs, bp = _scene()
        data[2, 3] = 0.0
        result = _reproject(_make_mosaic(), obs, bp, mask_bad_areas=True)
        mask = ma.getmaskarray(result.img)
        assert bool(mask[1, 2])
        assert bool(mask[3, 4])
        assert not mask[0, 3]

    def test_backplane_masked_pixel_is_masked(self) -> None:
        """A masked latitude backplane sample invalidates the corresponding bin."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        lat_mask = np.zeros((nv, nu), dtype=bool)
        lat_mask[1, 1] = True
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu), lat_mask=lat_mask)
        result = _reproject(_make_mosaic(), obs, bp)
        assert bool(ma.getmaskarray(result.img)[1, 1])
        assert not ma.getmaskarray(result.img)[1, 2]

    def test_masked_uv_sample_is_masked(self) -> None:
        """Bins whose lat/lon do not map onto the detector (masked UV) are masked."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data, masked_lon_bin=_LON0 + 2)
        bp = FakeBackplane((nv, nu))
        result = _reproject(_make_mosaic(), obs, bp)
        mask = ma.getmaskarray(result.img)
        assert bool(mask[:, 2].all())
        assert not mask[:6, 3].any()

    def test_edge_margin_discards_border_pixels(self) -> None:
        """edge_margin masks that many detector border pixels before binning."""
        data, obs, bp = _scene(nv=6, nu=8)
        result = _reproject(_make_mosaic(edge_margin=1), obs, bp)
        assert result.lat_idx_range == (_LAT0 + 1, _LAT0 + 5)
        assert result.lon_idx_range == (_LON0 + 1, _LON0 + 7)
        assert result.img.data[0, 0] == pytest.approx(data[1, 1])

    def test_max_incidence_masks_over_limit_pixels(self) -> None:
        """Pixels with incidence beyond the mosaic max_incidence are excluded."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        incidence = np.full((nv, nu), _DEF_INCIDENCE)
        incidence[2, 2] = 1.0
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu), incidence=incidence)
        result = _reproject(_make_mosaic(max_incidence=0.5), obs, bp)
        assert bool(ma.getmaskarray(result.img)[2, 2])
        assert not ma.getmaskarray(result.img)[2, 3]

    def test_max_emission_masks_over_limit_pixels(self) -> None:
        """Pixels with emission beyond the mosaic max_emission are excluded."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        emission = np.full((nv, nu), _DEF_EMISSION)
        emission[3, 1] = 1.2
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu), emission=emission)
        result = _reproject(_make_mosaic(max_emission=0.5), obs, bp)
        assert bool(ma.getmaskarray(result.img)[3, 1])
        assert not ma.getmaskarray(result.img)[3, 2]

    def test_max_resolution_masks_coarse_pixels(self) -> None:
        """Pixels whose center_res / cos(emission) exceeds max_resolution are excluded."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        emission = np.full((nv, nu), _DEF_EMISSION)
        emission[3, 3] = 1.2  # resolution 5 / cos(1.2) ~ 13.8 km/pixel
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu), emission=emission)
        result = _reproject(_make_mosaic(max_resolution=10.0), obs, bp)
        assert bool(ma.getmaskarray(result.img)[3, 3])
        assert not ma.getmaskarray(result.img)[3, 4]

    def test_fully_masked_backplane_returns_empty_placeholder(self) -> None:
        """A body entirely off the detector yields the minimal fully-masked result."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu), lat_mask=np.ones((nv, nu), dtype=bool))
        result = _reproject(_make_mosaic(), obs, bp, image_name='off_fov')
        assert result.lat_idx_range == (0, 1)
        assert result.lon_idx_range == (0, 1)
        assert result.img.shape == (2, 2)
        assert bool(ma.getmaskarray(result.img).all())
        assert result.time == pytest.approx(1000.0)
        assert result.image_name == 'off_fov'
        assert result.sub_solar_lon == pytest.approx(0.1)

    def test_all_zero_data_returns_fully_masked_result(self) -> None:
        """An observation with no nonzero pixels produces a fully masked 2x2 result."""
        nv, nu = 6, 8
        data = np.zeros((nv, nu))
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu))
        result = _reproject(_make_mosaic(), obs, bp)
        assert result.lat_idx_range == (0, 1)
        assert result.lon_idx_range == (0, 1)
        assert bool(ma.getmaskarray(result.img).all())

    def test_single_valid_pixel(self) -> None:
        """A single nonzero pixel yields exactly one valid cell with its value."""
        nv, nu = 6, 8
        data = np.zeros((nv, nu))
        data[2, 3] = 7.5
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu))
        result = _reproject(_make_mosaic(), obs, bp)
        mask = ma.getmaskarray(result.img)
        assert int((~mask).sum()) == 1
        assert result.img.data[0, 0] == pytest.approx(7.5)
        assert result.lat_idx_range == (_LAT0 + 2, _LAT0 + 3)
        assert result.lon_idx_range == (_LON0 + 3, _LON0 + 4)

    def test_single_valid_pixel_at_frame_corner(self) -> None:
        """A body reduced to the (0, 0) detector corner still reprojects correctly."""
        nv, nu = 6, 8
        data = np.zeros((nv, nu))
        data[0, 0] = 3.25
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu))
        result = _reproject(_make_mosaic(), obs, bp)
        assert result.img.data[0, 0] == pytest.approx(3.25)
        assert int((~ma.getmaskarray(result.img)).sum()) == 1


# =========================================================================
# Resolution, effective resolution, photometry, dtypes
# =========================================================================


class TestReprojectResolutionAndPhotometry:
    """Resolution formulas, navigation uncertainty scaling, and photometric models."""

    def test_resolution_is_center_resolution_over_cos_emission(self) -> None:
        """resolution == center_resolution / cos(emission) at each mapped pixel."""
        _data, obs, bp = _scene()
        result = _reproject(_make_mosaic(), obs, bp)
        expected = _DEF_CENTER_RES / math.cos(_DEF_EMISSION)
        np.testing.assert_allclose(result.resolution.data[:6, :8], expected, rtol=1e-6)

    def test_eff_resolution_equals_resolution_at_zero_uncertainty(self) -> None:
        """With navigation_uncertainty=0 the effective resolution equals resolution."""
        _data, obs, bp = _scene()
        result = _reproject(_make_mosaic(), obs, bp, navigation_uncertainty=0.0)
        np.testing.assert_allclose(
            result.eff_resolution.data[:6, :8], result.resolution.data[:6, :8]
        )

    def test_eff_resolution_scales_with_uncertainty(self) -> None:
        """eff_resolution == resolution * (1 + navigation_uncertainty)."""
        _data, obs, bp = _scene()
        result = _reproject(_make_mosaic(), obs, bp, navigation_uncertainty=0.5)
        np.testing.assert_allclose(
            result.eff_resolution.data[:6, :8], result.resolution.data[:6, :8] * 1.5, rtol=1e-6
        )

    def test_geometry_angles_copied_to_grid(self) -> None:
        """phase/emission/incidence carry the detector backplane values per cell."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        incidence = np.linspace(0.1, 0.6, nv * nu).reshape(nv, nu)
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu), incidence=incidence)
        result = _reproject(_make_mosaic(), obs, bp)
        np.testing.assert_allclose(result.incidence.data[:6, :8], incidence, rtol=1e-6)
        np.testing.assert_allclose(result.phase.data[:6, :8], _DEF_PHASE, rtol=1e-6)
        np.testing.assert_allclose(result.emission.data[:6, :8], _DEF_EMISSION, rtol=1e-6)

    def test_photometric_model_divides_by_cos_incidence(self) -> None:
        """A Lambert model corrects sampled data by max(cos(incidence), clamp)."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        incidence = np.linspace(0.1, 1.4, nv * nu).reshape(nv, nu)
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu), incidence=incidence)
        mosaic = _make_mosaic(photometric_model=LambertModel())
        result = _reproject(mosaic, obs, bp)
        expected = data / np.maximum(np.cos(incidence.astype(np.float32)), 0.01)
        np.testing.assert_allclose(result.img.data[:6, :8], expected, rtol=1e-5)

    def test_photometric_model_name_recorded(self) -> None:
        """photometric_model_name is the model's name when a model is configured."""
        _data, obs, bp = _scene()
        result = _reproject(_make_mosaic(photometric_model=LambertModel()), obs, bp)
        assert result.photometric_model_name == 'lambert'

    def test_photometric_model_name_none_without_model(self) -> None:
        """photometric_model_name is None when no model is configured."""
        _data, obs, bp = _scene()
        result = _reproject(_make_mosaic(), obs, bp)
        assert result.photometric_model_name is None

    def test_dtype_propagation(self) -> None:
        """img uses image_dtype; geometry arrays use metadata_dtype; time is a float."""
        _data, obs, bp = _scene()
        mosaic = _make_mosaic(image_dtype=np.float32, metadata_dtype=np.float64)
        result = _reproject(mosaic, obs, bp)
        assert result.img.dtype == np.dtype(np.float32)
        assert result.resolution.dtype == np.dtype(np.float64)
        assert result.eff_resolution.dtype == np.dtype(np.float64)
        assert result.phase.dtype == np.dtype(np.float64)
        assert result.image_dtype == np.dtype(np.float32)
        assert result.metadata_dtype == np.dtype(np.float64)
        assert isinstance(result.time, float)


# =========================================================================
# mask_only mode
# =========================================================================


class TestReprojectMaskOnly:
    """mask_only=True returns the valid-pixel coverage mask only."""

    def test_mask_only_img_is_unmasked_coverage(self) -> None:
        """img holds 1.0 at covered bins and 0.0 elsewhere, with no mask bits set."""
        _data, obs, bp = _scene(nv=6, nu=8)
        result = _reproject(_make_mosaic(), obs, bp, mask_only=True)
        assert not ma.getmaskarray(result.img).any()
        np.testing.assert_allclose(result.img.data[:6, :8], 1.0)
        np.testing.assert_allclose(result.img.data[6, :], 0.0)
        np.testing.assert_allclose(result.img.data[:, 8], 0.0)

    def test_mask_only_metadata_fully_masked(self) -> None:
        """All geometry fields are fully masked in mask_only mode."""
        _data, obs, bp = _scene()
        result = _reproject(_make_mosaic(), obs, bp, mask_only=True)
        assert bool(ma.getmaskarray(result.resolution).all())
        assert bool(ma.getmaskarray(result.eff_resolution).all())
        assert bool(ma.getmaskarray(result.phase).all())
        assert bool(ma.getmaskarray(result.emission).all())
        assert bool(ma.getmaskarray(result.incidence).all())

    def test_mask_only_ignores_data_values(self) -> None:
        """Coverage is based on geometry only; zero data still counts as covered."""
        nv, nu = 6, 8
        data = np.zeros((nv, nu))
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu))
        result = _reproject(_make_mosaic(), obs, bp, mask_only=True)
        np.testing.assert_allclose(result.img.data[:6, :8], 1.0)

    def test_mask_only_photometric_model_name_is_none(self) -> None:
        """mask_only results carry photometric_model_name=None even with a model set."""
        _data, obs, bp = _scene()
        mosaic = _make_mosaic(photometric_model=LambertModel())
        result = _reproject(mosaic, obs, bp, mask_only=True)
        assert result.photometric_model_name is None

    def test_mask_only_outside_fov_placeholder(self) -> None:
        """mask_only with the body off the detector returns a 2x2 all-zero coverage."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu), lat_mask=np.ones((nv, nu), dtype=bool))
        result = _reproject(_make_mosaic(), obs, bp, mask_only=True)
        assert result.img.shape == (2, 2)
        np.testing.assert_allclose(result.img.data, 0.0)
        assert bool(ma.getmaskarray(result.resolution).all())


# =========================================================================
# Subimage handling
# =========================================================================


class TestReprojectSubimage:
    """subimage_edges with a backplane covering only the subimage."""

    def test_subimage_backplane_samples_subimage_pixels(self) -> None:
        """A subimage-shaped backplane maps bins to subimage-relative data samples."""
        nv, nu = 8, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data)
        bp = FakeBackplane((4, 4), v_offset=2, u_offset=2)
        result = _reproject(_make_mosaic(), obs, bp, subimage_edges=(2, 5, 2, 5))
        assert result.lat_idx_range == (_LAT0 + 2, _LAT0 + 6)
        assert result.lon_idx_range == (_LON0 + 2, _LON0 + 6)
        np.testing.assert_allclose(result.img.data[:4, :4], data[2:6, 2:6])

    def test_mask_only_full_frame_backplane_with_subimage(self) -> None:
        """mask_only supports subimage_edges with a full-frame backplane."""
        nv, nu = 8, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu))
        result = _reproject(_make_mosaic(), obs, bp, subimage_edges=(2, 5, 2, 5), mask_only=True)
        np.testing.assert_allclose(result.img.data[2:6, 2:6], 1.0)

    def test_full_frame_backplane_with_subimage_edges(self) -> None:
        """A full-frame backplane plus subimage_edges reprojects the subimage."""
        nv, nu = 8, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu))
        result = _reproject(_make_mosaic(), obs, bp, subimage_edges=(2, 5, 2, 5))
        assert result.img.data[0, 0] == pytest.approx(data[2, 2])


# =========================================================================
# zoom
# =========================================================================


class TestReprojectZoom:
    """The zoom parameter's documented sub-pixel interpolation."""

    def test_zoom_one_matches_direct_sampling(self) -> None:
        """zoom=1 samples the original image directly."""
        data, obs, bp = _scene()
        result = _reproject(_make_mosaic(zoom=1), obs, bp)
        np.testing.assert_allclose(result.img.data[:6, :8], data)

    def test_zoom_enables_sub_pixel_sampling(self) -> None:
        """zoom > 1 must change sampling when bins map to fractional pixel positions."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data, frac_u=0.75, frac_v=0.75)
        bp = FakeBackplane((nv, nu))
        r1 = _reproject(_make_mosaic(zoom=1), obs, bp)
        r4 = _reproject(_make_mosaic(zoom=4), obs, bp)
        assert not np.array_equal(r1.img.filled(-1.0), r4.img.filled(-1.0))


# =========================================================================
# add() fed by reproject() results
# =========================================================================


class TestAddWithReprojectResults:
    """Accumulation semantics of add() using real reproject() outputs."""

    def test_reproject_add_round_trip(self) -> None:
        """Adding a reprojection and reading it back preserves values and coverage."""
        data, obs, bp = _scene(nv=6, nu=8)
        mosaic = _make_mosaic()
        mosaic.add(_reproject(mosaic, obs, bp))
        out = mosaic.to_bounded()
        np.testing.assert_allclose(out.img.data[:6, :8], data)

    def test_overlapping_add_better_resolution_wins(self) -> None:
        """When coverage overlaps, the image with finer effective resolution wins."""
        nv, nu = 6, 8
        coarse = np.full((nv, nu), 1.0)
        fine = np.full((nv, nu), 9.0)
        mosaic = _make_mosaic()
        mosaic.add(_reproject(mosaic, FakeObs(coarse), FakeBackplane((nv, nu), center_res=5.0)))
        mosaic.add(_reproject(mosaic, FakeObs(fine), FakeBackplane((nv, nu), center_res=2.0)))
        out = mosaic.to_bounded()
        np.testing.assert_allclose(out.img.data[:6, :8], 9.0)

    def test_overlapping_add_worse_resolution_loses(self) -> None:
        """A later image with coarser effective resolution does not overwrite."""
        nv, nu = 6, 8
        fine = np.full((nv, nu), 9.0)
        coarse = np.full((nv, nu), 1.0)
        mosaic = _make_mosaic()
        mosaic.add(_reproject(mosaic, FakeObs(fine), FakeBackplane((nv, nu), center_res=2.0)))
        mosaic.add(_reproject(mosaic, FakeObs(coarse), FakeBackplane((nv, nu), center_res=5.0)))
        out = mosaic.to_bounded()
        np.testing.assert_allclose(out.img.data[:6, :8], 9.0)

    def test_disjoint_adds_union_coverage(self) -> None:
        """Two disjoint reprojections both appear in the mosaic at their own bins."""
        nv, nu = 3, 3
        d1 = np.full((nv, nu), 4.0)
        d2 = np.full((nv, nu), 6.0)
        mosaic = _make_mosaic()
        mosaic.add(_reproject(mosaic, FakeObs(d1), FakeBackplane((nv, nu))))
        obs2 = FakeObs(d2, lat0=_LAT0, lon0=_LON0 + 10)
        mosaic.add(_reproject(mosaic, obs2, FakeBackplane((nv, nu), lon0=_LON0 + 10)))
        out = mosaic.to_full()
        assert out.img.data[_LAT0, _LON0] == pytest.approx(4.0)
        assert out.img.data[_LAT0, _LON0 + 10] == pytest.approx(6.0)

    def test_no_valid_pixel_result_still_counts_image(self) -> None:
        """An empty (off-detector) reprojection still records its image name."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data)
        bp_off = FakeBackplane((nv, nu), lat_mask=np.ones((nv, nu), dtype=bool))
        mosaic = _make_mosaic()
        mosaic.add(_reproject(mosaic, obs, bp_off, image_name='empty_one'))
        mosaic.add(_reproject(mosaic, obs, FakeBackplane((nv, nu)), image_name='real_one'))
        out = mosaic.to_full()
        assert out.contributing_image_names == ('empty_one', 'real_one')
        assert int(out.image_number.data[_LAT0, _LON0]) == 1

    def test_bounds_reflect_valid_data_only(self) -> None:
        """bounds covers only bins that actually hold valid data."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data)
        bp_off = FakeBackplane((nv, nu), lat_mask=np.ones((nv, nu), dtype=bool))
        mosaic = _make_mosaic()
        mosaic.add(_reproject(mosaic, obs, bp_off))  # fully masked (0..1, 0..1) window
        mosaic.add(_reproject(mosaic, obs, FakeBackplane((nv, nu))))
        bounds = mosaic.bounds
        assert bounds is not None
        (lat_min, _), (lon_min, _) = bounds
        assert lat_min == pytest.approx(_LAT0 * _LAT_RES - _HALF_PI)
        assert lon_min == pytest.approx(_LON0 * _LON_RES)

    def test_add_mismatched_grid_shape_raises(self) -> None:
        """A repro whose img shape disagrees with its idx ranges fails to add."""
        good = _make_repro(lat_range=(5, 6), lon_range=(10, 11))
        bad = BodyReprojResult(
            body_name=good.body_name,
            img=good.img,
            lat_resolution=good.lat_resolution,
            lon_resolution=good.lon_resolution,
            lat_idx_range=(5, 7),  # claims 3 rows but img has 2
            lon_idx_range=(10, 11),
            latlon_type=good.latlon_type,
            lon_direction=good.lon_direction,
            resolution=good.resolution,
            eff_resolution=good.eff_resolution,
            phase=good.phase,
            emission=good.emission,
            incidence=good.incidence,
            time=good.time,
            photometric_model_name=None,
            image_dtype=good.image_dtype,
            metadata_dtype=good.metadata_dtype,
        )
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        with pytest.raises(IndexError, match='boolean index'):
            mosaic.add(bad)

    def test_add_photometric_model_mismatch_raises(self) -> None:
        """Adding a photometrically corrected repro to a plain mosaic raises ValueError."""
        _data, obs, bp = _scene()
        phot_mosaic = _make_mosaic(photometric_model=LambertModel())
        repro = _reproject(phot_mosaic, obs, bp)
        plain_mosaic = _make_mosaic()
        with pytest.raises(ValueError, match='photometric_model_name mismatch'):
            plain_mosaic.add(repro)

    def test_add_at_uint16_capacity_succeeds(self) -> None:
        """The 65,536th image (number 65535) is still accepted."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mosaic._image_count = np.iinfo(np.uint16).max
        mosaic.add(_make_repro(lat_range=(5, 5), lon_range=(10, 10)))
        out = mosaic.to_bounded()
        assert int(out.image_number.data[0, 0]) == np.iinfo(np.uint16).max

    def test_add_beyond_uint16_capacity_raises_overflow(self) -> None:
        """Exceeding the uint16 image-number capacity raises OverflowError."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        mosaic._image_count = np.iinfo(np.uint16).max + 1
        with pytest.raises(OverflowError, match='exceeds uint16 max'):
            mosaic.add(_make_repro(lat_range=(5, 5), lon_range=(10, 10)))

    def test_copy_slop_copies_neighbors_of_replaced_pixels(self) -> None:
        """copy_slop=1 also copies pixels adjacent to each replaced pixel."""
        mosaic = BodyMosaic(body_name='MIMAS', lat_resolution=_LAT_RES, lon_resolution=_LON_RES)
        first = _make_repro(
            lat_range=(5, 7), lon_range=(10, 12), img_values=1.0, eff_res_values=1.0
        )
        mosaic.add(first)
        eff2 = np.full((3, 3), 5.0, dtype=np.float32)
        eff2[1, 1] = 0.1  # only the center wins on resolution
        second = _make_repro(
            lat_range=(5, 7), lon_range=(10, 12), img_values=9.0, eff_res_values=eff2
        )
        mosaic.add(second, copy_slop=1)
        out = mosaic.to_bounded()
        assert out.img.data[1, 1] == pytest.approx(9.0)
        assert out.img.data[0, 1] == pytest.approx(9.0)


# =========================================================================
# Inverse-mapping consistency (reproject -> mosaic -> cartographic model)
# =========================================================================


class TestInverseMappingConsistency:
    """Projecting a reprojected mosaic back onto the image recovers the data."""

    def test_reproject_then_project_back_recovers_image(self) -> None:
        """create_cartographic_model on the mosaic reproduces the observed image.

        Detector pixels sit at bin CENTERS, so projecting back samples the mosaic
        halfway between grid rows/columns; the expected model is the bilinear
        average of the four surrounding mosaic cells (zero-filled at the masked
        ceil row/column).
        """
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu))
        mosaic = _make_mosaic()
        repro = _reproject(mosaic, obs, bp)
        mosaic.add(repro)
        # Window the bounded view to the full reprojection extent (including
        # the masked ceil row/column) so the bilinear expectation below sees
        # zero-filled cells there; the default window is the valid-data bounds,
        # which exclude the masked edge bins.
        bounded = mosaic.to_bounded(
            lat_range=(
                repro.lat_idx_range[0] * _LAT_RES - math.pi / 2.0,
                repro.lat_idx_range[1] * _LAT_RES - math.pi / 2.0,
            ),
            lon_range=(
                repro.lon_idx_range[0] * _LON_RES,
                repro.lon_idx_range[1] * _LON_RES,
            ),
        )

        with patch('oops.backplane.Backplane', new=lambda o: bp):
            carto = create_cartographic_model(
                bounded, obs, body_name='MIMAS', latlon_type='squashed'
            )

        assert carto is not None
        padded = np.zeros((nv + 1, nu + 1))
        padded[:nv, :nu] = data
        expected = 0.25 * (padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:])
        np.testing.assert_allclose(carto.model_img, expected, rtol=1e-4, atol=1e-4)

    def test_round_trip_resolution_ratio(self) -> None:
        """The resolution ratio is median mosaic eff_resolution over center resolution."""
        nv, nu = 6, 8
        data = np.arange(1.0, nv * nu + 1.0).reshape(nv, nu)
        obs = FakeObs(data)
        bp = FakeBackplane((nv, nu))
        mosaic = _make_mosaic()
        mosaic.add(_reproject(mosaic, obs, bp))
        bounded = mosaic.to_bounded()

        with patch('oops.backplane.Backplane', new=lambda o: bp):
            carto = create_cartographic_model(
                bounded, obs, body_name='MIMAS', latlon_type='squashed'
            )

        assert carto is not None
        expected_ratio = (_DEF_CENTER_RES / math.cos(_DEF_EMISSION)) / _DEF_CENTER_RES
        assert carto.resolution_ratio == pytest.approx(expected_ratio, rel=1e-5)
