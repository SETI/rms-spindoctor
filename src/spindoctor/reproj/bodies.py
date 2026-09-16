"""Body reprojection and mosaicing utilities.

This module provides the BodyMosaic class for reprojecting planetary body
images onto latitude/longitude grids and accumulating them into mosaics.
"""

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Final, Literal, cast

import numpy as np
import numpy.ma as ma
import oops
import polymath
from scipy.ndimage import maximum_filter
from scipy.ndimage import zoom as ndimage_zoom

from spindoctor.config import IMAGE_LOGGER
from spindoctor.reproj._serialization import (
    infer_format,
    load_fits,
    load_npz,
    save_fits,
    save_npz,
    tuple_of_strings_field,
    verify_dtype,
)
from spindoctor.reproj.photometric_model import PhotometricModel
from spindoctor.support.image import array_zoom
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType, PathLike

# Slop values: must be smaller than the smallest resolution we will ever use
_LATITUDE_SLOP = 1e-6  # rad
_LONGITUDE_SLOP = 1e-6  # rad

_MIN_LATITUDE = -math.pi / 2.0
_MAX_LATITUDE = math.pi / 2.0 - _LATITUDE_SLOP * 2
_MAX_LONGITUDE = math.pi * 2.0 - _LONGITUDE_SLOP * 2

# Module-level defaults for BodyMosaic parameters
DEFAULT_LAT_RESOLUTION = 0.1 * math.pi / 180.0  # 0.1 degrees in rad
DEFAULT_LON_RESOLUTION = 0.1 * math.pi / 180.0
DEFAULT_MAX_INCIDENCE: float | None = None
DEFAULT_MAX_EMISSION: float | None = None
DEFAULT_MAX_RESOLUTION: float | None = None
DEFAULT_EDGE_MARGIN = 3
DEFAULT_ZOOM = 1

# ``BodyMosaic.__init__`` numeric validation (``bool`` subclasses ``int``).
_REAL_SCALAR_TYPES = (int, float, np.integer, np.floating)
_INTEGRAL_SCALAR_TYPES = (int, np.integer)


def _validate_navigation_uncertainty(value: object) -> float:
    """Validate ``navigation_uncertainty`` for :meth:`BodyMosaic.reproject`.

    Returns:
        ``float(value)`` when valid.

    Raises:
        TypeError: If ``value`` is not a real scalar (``bool`` is rejected).
        ValueError: If ``value`` is not finite or is negative.
    """
    if isinstance(value, bool) or not isinstance(value, _REAL_SCALAR_TYPES):
        raise TypeError(f'navigation_uncertainty must be a real number, got {type(value).__name__}')
    nu = float(value)
    if not math.isfinite(nu) or nu < 0.0:
        raise ValueError(
            'navigation_uncertainty must be finite and >= 0 (effective resolution uses '
            'resolution * (1.0 + navigation_uncertainty)).'
        )
    return nu


# Lambert is set mainly based on the depth of craters - if the incidence angle
# is too large the crater interior is too shadowed and we don't get a good
# reprojection. If the emission angle is too large we can't see over the crater
# lip.
_LAMBERT_THRESHOLD = 0.0001


class _UseMosaicLimitsSentinel:
    """Singleton marker for :meth:`BodyMosaic.add` max-* keyword defaults."""

    __slots__ = ()

    def __repr__(self) -> str:
        return 'USE_MOSAIC_LIMITS'


USE_MOSAIC_LIMITS: Final[_UseMosaicLimitsSentinel] = _UseMosaicLimitsSentinel()
"""Pass to :meth:`BodyMosaic.add` for ``max_incidence`` / ``max_emission`` / ``max_resolution``.

Uses the :class:`BodyMosaic` constructor limits instead of per-call overrides.
"""


def _create_repro_array(n_lat: int, n_lon: int, dtype: np.dtype) -> ma.MaskedArray:
    """Return an ``n_lat`` x ``n_lon`` zero array that is fully masked (reprojection grid slot)."""
    arr = ma.zeros((n_lat, n_lon), dtype=dtype)
    arr.mask = True
    return arr


class BodyMosaicMergeStrategy(enum.Enum):
    """Strategy for resolving per-pixel conflicts when adding data to a BodyMosaic.

    Members:
        BEST_RESOLUTION: Replace an existing pixel only when the new data has
            strictly better effective resolution, or if the existing pixel is
            empty (masked).
    """

    BEST_RESOLUTION = 'best_resolution'


@dataclass(frozen=True)
class BodyReprojResult:
    """Data returned by BodyMosaic.reproject().

    This is a :func:`dataclasses.dataclass`; each field is documented on its own
    member entry below.
    """

    body_name: str
    img: ma.MaskedArray
    lat_resolution: float
    lon_resolution: float
    lat_idx_range: tuple[int, int]
    lon_idx_range: tuple[int, int]
    latlon_type: Literal['centric', 'graphic', 'squashed']
    lon_direction: Literal['east', 'west']
    resolution: ma.MaskedArray
    eff_resolution: ma.MaskedArray
    phase: ma.MaskedArray
    emission: ma.MaskedArray
    incidence: ma.MaskedArray
    time: float
    photometric_model_name: str | None
    image_dtype: np.dtype
    metadata_dtype: np.dtype
    image_name: str = ''
    sub_solar_lon: float = 0.0
    sub_solar_lat: float = 0.0
    sub_observer_lon: float = 0.0
    sub_observer_lat: float = 0.0

    def save(
        self,
        path: PathLike,
        *,
        format_: str | None = None,
        compress: bool = True,
    ) -> None:
        """Save this BodyReprojResult to a file.

        Parameters:
            path: Output path (``str``, ``pathlib.Path``, or ``filecache.FCPath``).
                The format is inferred from the extension (.npz for NumPy archive,
                .fits/.fit for FITS) unless ``format_`` is given.
            format_: Explicit format: ``'npz'`` or ``'fits'``. Overrides
                extension-based inference.
            compress: If True (default), use compressed npz. Ignored for
                FITS.

        Raises:
            ValueError: If the format cannot be inferred or is not
                supported.

        Example::

            result = mosaic.reproject(obs)
            result.save('reprojection.npz')
            result.save('reprojection.fits', format_='fits')
            reloaded = BodyReprojResult.load('reprojection.npz')
        """
        fmt = infer_format(path, format_)
        payload: dict[str, Any] = {
            'body_name': self.body_name,
            'img': self.img,
            'lat_resolution': self.lat_resolution,
            'lon_resolution': self.lon_resolution,
            'lat_idx_range': self.lat_idx_range,
            'lon_idx_range': self.lon_idx_range,
            'latlon_type': self.latlon_type,
            'lon_direction': self.lon_direction,
            'resolution': self.resolution,
            'eff_resolution': self.eff_resolution,
            'phase': self.phase,
            'emission': self.emission,
            'incidence': self.incidence,
            'time': self.time,
            'photometric_model_name': self.photometric_model_name,
            'image_dtype': self.image_dtype,
            'metadata_dtype': self.metadata_dtype,
            'image_name': self.image_name,
            'sub_solar_lon': self.sub_solar_lon,
            'sub_solar_lat': self.sub_solar_lat,
            'sub_observer_lon': self.sub_observer_lon,
            'sub_observer_lat': self.sub_observer_lat,
        }
        if fmt == 'npz':
            save_npz(path, 'BodyReprojResult', 1, payload, compress=compress)
        else:
            save_fits(path, 'BodyReprojResult', 1, payload)

    @classmethod
    def load(
        cls,
        path: PathLike,
        *,
        format_: str | None = None,
    ) -> 'BodyReprojResult':
        """Load a BodyReprojResult from a file.

        Parameters:
            path: Input path (``str``, ``pathlib.Path``, or ``filecache.FCPath``).
            format_: Explicit format (``'npz'`` or ``'fits'``). If None,
                inferred from the file extension.

        Returns:
            A new BodyReprojResult with the loaded data.

        Raises:
            ValueError: If the file's kind does not match, or if any array
                dtype does not match the declared ``image_dtype`` /
                ``metadata_dtype``.

        Example::

            result = BodyReprojResult.load('reprojection.npz')
        """
        fmt = infer_format(path, format_)
        d = (
            load_npz(path, 'BodyReprojResult')
            if fmt == 'npz'
            else load_fits(path, 'BodyReprojResult')
        )

        image_dtype = np.dtype(str(d['image_dtype']))
        metadata_dtype = np.dtype(str(d['metadata_dtype']))

        verify_dtype(
            {
                k: d[k]
                for k in ('img', 'resolution', 'eff_resolution', 'phase', 'emission', 'incidence')
            },
            image_dtype=image_dtype,
            metadata_dtype=metadata_dtype,
            image_fields=['img'],
            metadata_fields=['resolution', 'eff_resolution', 'phase', 'emission', 'incidence'],
        )

        lat_idx = d['lat_idx_range']
        lat_idx_range: tuple[int, int] = (int(lat_idx[0]), int(lat_idx[1]))
        lon_idx = d['lon_idx_range']
        lon_idx_range: tuple[int, int] = (int(lon_idx[0]), int(lon_idx[1]))

        phot = d.get('photometric_model_name')
        photometric_model_name: str | None
        if phot is not None and str(phot) != 'None':
            photometric_model_name = str(phot)
        else:
            photometric_model_name = None

        latlon_type_val = str(d['latlon_type'])
        if latlon_type_val not in ('centric', 'graphic', 'squashed'):
            raise ValueError(
                f"latlon_type {latlon_type_val!r} is not one of 'centric', 'graphic', 'squashed'"
            )
        lon_direction_val = str(d['lon_direction'])
        if lon_direction_val not in ('east', 'west'):
            raise ValueError(f"lon_direction {lon_direction_val!r} is not one of 'east', 'west'")

        return cls(
            body_name=str(d['body_name']),
            img=d['img'],
            lat_resolution=float(d['lat_resolution']),
            lon_resolution=float(d['lon_resolution']),
            lat_idx_range=lat_idx_range,
            lon_idx_range=lon_idx_range,
            # latlon_type_val / lon_direction_val: validated against allowed literals above;
            # mypy cannot narrow str to the Literal field types.
            latlon_type=latlon_type_val,  # type: ignore[arg-type]
            lon_direction=lon_direction_val,  # type: ignore[arg-type]
            resolution=d['resolution'],
            eff_resolution=d['eff_resolution'],
            phase=d['phase'],
            emission=d['emission'],
            incidence=d['incidence'],
            time=float(d['time']),
            photometric_model_name=photometric_model_name,
            image_dtype=image_dtype,
            metadata_dtype=metadata_dtype,
            image_name=str(d.get('image_name', '') or ''),
            sub_solar_lon=float(d.get('sub_solar_lon', 0.0)),
            sub_solar_lat=float(d.get('sub_solar_lat', 0.0)),
            sub_observer_lon=float(d.get('sub_observer_lon', 0.0)),
            sub_observer_lat=float(d.get('sub_observer_lat', 0.0)),
        )


@dataclass(frozen=True)
class BodyMosaicData:
    """Mosaic data returned by BodyMosaic retrieval methods.

    This is a :func:`dataclasses.dataclass`; each field is documented on its own
    member entry below.
    """

    body_name: str
    img: ma.MaskedArray
    lat_resolution: float
    lon_resolution: float
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    latlon_type: Literal['centric', 'graphic', 'squashed']
    lon_direction: Literal['east', 'west']
    resolution: ma.MaskedArray
    eff_resolution: ma.MaskedArray
    phase: ma.MaskedArray
    emission: ma.MaskedArray
    incidence: ma.MaskedArray
    time: ma.MaskedArray
    image_number: ma.MaskedArray
    photometric_model_name: str | None
    image_dtype: np.dtype
    metadata_dtype: np.dtype
    contributing_image_names: tuple[str, ...] = ()
    sub_solar_lon_per_image: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    sub_solar_lat_per_image: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    sub_observer_lon_per_image: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    sub_observer_lat_per_image: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )

    def save(
        self,
        path: PathLike,
        *,
        format_: str | None = None,
        compress: bool = True,
    ) -> None:
        """Save this BodyMosaicData to a file.

        Parameters:
            path: Output path (``str``, ``pathlib.Path``, or ``filecache.FCPath``).
                The format is inferred from the extension (.npz for NumPy archive,
                .fits/.fit for FITS) unless ``format_`` is given.
            format_: Explicit format: ``'npz'`` or ``'fits'``. Overrides
                extension-based inference.
            compress: If True (default), use compressed npz. Ignored for
                FITS.

        Raises:
            ValueError: If the format cannot be inferred or is not
                supported.

        Example::

            data = mosaic.to_bounded()
            data.save('mimas.npz')
            data.save('mimas.fits', format_='fits')
            data.save('mimas.npz', compress=False)
            reloaded = BodyMosaicData.load('mimas.npz')
        """
        fmt = infer_format(path, format_)
        payload: dict[str, Any] = {
            'body_name': self.body_name,
            'img': self.img,
            'lat_resolution': self.lat_resolution,
            'lon_resolution': self.lon_resolution,
            'lat_range': self.lat_range,
            'lon_range': self.lon_range,
            'latlon_type': self.latlon_type,
            'lon_direction': self.lon_direction,
            'resolution': self.resolution,
            'eff_resolution': self.eff_resolution,
            'phase': self.phase,
            'emission': self.emission,
            'incidence': self.incidence,
            'time': self.time,
            'image_number': self.image_number,
            'photometric_model_name': self.photometric_model_name,
            'image_dtype': self.image_dtype,
            'metadata_dtype': self.metadata_dtype,
            'contributing_image_names': self.contributing_image_names,
            'sub_solar_lon_per_image': self.sub_solar_lon_per_image,
            'sub_solar_lat_per_image': self.sub_solar_lat_per_image,
            'sub_observer_lon_per_image': self.sub_observer_lon_per_image,
            'sub_observer_lat_per_image': self.sub_observer_lat_per_image,
        }
        if fmt == 'npz':
            save_npz(path, 'BodyMosaicData', 1, payload, compress=compress)
        else:
            save_fits(path, 'BodyMosaicData', 1, payload)

    @classmethod
    def load(
        cls,
        path: PathLike,
        *,
        format_: str | None = None,
    ) -> 'BodyMosaicData':
        """Load a BodyMosaicData from a file.

        Parameters:
            path: Input path (``str``, ``pathlib.Path``, or ``filecache.FCPath``).
            format_: Explicit format (``'npz'`` or ``'fits'``). If None,
                inferred from the file extension.

        Returns:
            A new BodyMosaicData with the loaded data.

        Raises:
            ValueError: If the file's kind does not match, or if any array
                dtype does not match the declared ``image_dtype`` /
                ``metadata_dtype``, or if ``image_number`` is not uint16,
                or if ``time`` is not float64.

        Example::

            data = BodyMosaicData.load('mimas.npz')
        """
        fmt = infer_format(path, format_)
        d = load_npz(path, 'BodyMosaicData') if fmt == 'npz' else load_fits(path, 'BodyMosaicData')

        image_dtype = np.dtype(str(d['image_dtype']))
        metadata_dtype = np.dtype(str(d['metadata_dtype']))

        verify_dtype(
            {
                k: d[k]
                for k in (
                    'img',
                    'resolution',
                    'eff_resolution',
                    'phase',
                    'emission',
                    'incidence',
                    'time',
                    'image_number',
                )
            },
            image_dtype=image_dtype,
            metadata_dtype=metadata_dtype,
            image_fields=['img'],
            metadata_fields=['resolution', 'eff_resolution', 'phase', 'emission', 'incidence'],
            float64_fields=['time'],
        )

        lat_r = d['lat_range']
        lat_range: tuple[float, float] = (float(lat_r[0]), float(lat_r[1]))
        lon_r = d['lon_range']
        lon_range: tuple[float, float] = (float(lon_r[0]), float(lon_r[1]))

        phot = d.get('photometric_model_name')
        photometric_model_name: str | None
        if phot is not None and str(phot) != 'None':
            photometric_model_name = str(phot)
        else:
            photometric_model_name = None

        latlon_type_val = str(d['latlon_type'])
        if latlon_type_val not in ('centric', 'graphic', 'squashed'):
            raise ValueError(
                f"latlon_type {latlon_type_val!r} is not one of 'centric', 'graphic', 'squashed'"
            )
        lon_direction_val = str(d['lon_direction'])
        if lon_direction_val not in ('east', 'west'):
            raise ValueError(f"lon_direction {lon_direction_val!r} is not one of 'east', 'west'")

        contributing_image_names = tuple_of_strings_field(d.get('contributing_image_names'))

        _raw_ssl_lon = d.get('sub_solar_lon_per_image')
        _raw_ssl_lat = d.get('sub_solar_lat_per_image')
        sub_solar_lon_per_image = (
            np.asarray(_raw_ssl_lon, dtype=np.float64)
            if _raw_ssl_lon is not None
            else np.empty((0,), dtype=np.float64)
        )
        sub_solar_lat_per_image = (
            np.asarray(_raw_ssl_lat, dtype=np.float64)
            if _raw_ssl_lat is not None
            else np.empty((0,), dtype=np.float64)
        )

        _raw_sol_lon = d.get('sub_observer_lon_per_image')
        _raw_sol_lat = d.get('sub_observer_lat_per_image')
        sub_observer_lon_per_image = (
            np.asarray(_raw_sol_lon, dtype=np.float64)
            if _raw_sol_lon is not None
            else np.empty((0,), dtype=np.float64)
        )
        sub_observer_lat_per_image = (
            np.asarray(_raw_sol_lat, dtype=np.float64)
            if _raw_sol_lat is not None
            else np.empty((0,), dtype=np.float64)
        )

        return cls(
            body_name=str(d['body_name']),
            img=d['img'],
            lat_resolution=float(d['lat_resolution']),
            lon_resolution=float(d['lon_resolution']),
            lat_range=lat_range,
            lon_range=lon_range,
            # latlon_type_val / lon_direction_val: validated against allowed literals above;
            # mypy cannot narrow str to the Literal field types.
            latlon_type=latlon_type_val,  # type: ignore[arg-type]
            lon_direction=lon_direction_val,  # type: ignore[arg-type]
            resolution=d['resolution'],
            eff_resolution=d['eff_resolution'],
            phase=d['phase'],
            emission=d['emission'],
            incidence=d['incidence'],
            time=d['time'],
            image_number=d['image_number'],
            photometric_model_name=photometric_model_name,
            image_dtype=image_dtype,
            metadata_dtype=metadata_dtype,
            contributing_image_names=contributing_image_names,
            sub_solar_lon_per_image=sub_solar_lon_per_image,
            sub_solar_lat_per_image=sub_solar_lat_per_image,
            sub_observer_lon_per_image=sub_observer_lon_per_image,
            sub_observer_lat_per_image=sub_observer_lat_per_image,
        )


class BodyMosaic:
    """Reprojects and mosaics body images in latitude/longitude space.

    Combines reprojected body images into a single mosaic, resolving conflicts
    using a configurable merge strategy.

    The coordinate grid uses the specified resolution in both latitude and
    longitude. All angular values are in radians.

    Internal storage uses a shifted circular buffer to handle longitude
    wraparound (data spanning the 0/2*pi boundary) without allocating the
    full 0..2*pi range.

    Thread safety: reproject() uses oops backplane computation which is
    not safe for concurrent use on the same observation. Do not call
    reproject() from multiple threads on the same observation.

    Example usage::

        mosaic = BodyMosaic(body_name='MIMAS')
        for obs in observations:
            result = mosaic.reproject(obs, data=obs.data)
            mosaic.add(result)
        data = mosaic.to_bounded()

    Attributes:
        body_name: The name of the body.
    """

    def __init__(
        self,
        *,
        body_name: str,
        lat_resolution: float = DEFAULT_LAT_RESOLUTION,
        lon_resolution: float = DEFAULT_LON_RESOLUTION,
        lat_range: tuple[float, float] | None = None,
        lon_range: tuple[float, float] | None = None,
        dynamic: bool = True,
        max_incidence: float | None = DEFAULT_MAX_INCIDENCE,
        max_emission: float | None = DEFAULT_MAX_EMISSION,
        max_resolution: float | None = DEFAULT_MAX_RESOLUTION,
        edge_margin: int = DEFAULT_EDGE_MARGIN,
        zoom: int = DEFAULT_ZOOM,
        latlon_type: Literal['centric', 'graphic', 'squashed'] = 'centric',
        lon_direction: Literal['east', 'west'] = 'east',
        photometric_model: PhotometricModel | None = None,
        merge_strategy: BodyMosaicMergeStrategy = BodyMosaicMergeStrategy.BEST_RESOLUTION,
        image_dtype: np.typing.DTypeLike = np.float64,
        metadata_dtype: np.typing.DTypeLike = np.float32,
    ) -> None:
        """Initialize a BodyMosaic.

        Parameters:
            body_name: The name of the body (e.g. 'MIMAS').
            lat_resolution: Latitude resolution (rad/pixel).
            lon_resolution: Longitude resolution (rad/pixel).
            lat_range: (min_lat, max_lat) extent of the mosaic in radians.
                If None, use the full valid lat range.
            lon_range: (min_lon, max_lon) longitude extent in radians.
                If None, use the full valid lon range.
            dynamic: If True, the internal arrays grow when add() receives data
                outside the current bounds. If False, out-of-bounds data is
                silently clipped to the configured lat_range / lon_range.
            max_incidence: Maximum incidence angle (rad) for valid pixels. None
                means no constraint.
            max_emission: Maximum emission angle (rad) for valid pixels. None
                means no constraint.
            max_resolution: Maximum resolution (km/pixel) for valid pixels. None
                means no constraint.
            edge_margin: Number of edge pixels to discard (ISS images have
                garbage near the edges).
            zoom: Zoom factor for sub-pixel interpolation during reprojection.
            latlon_type: Coordinate system for latitude and longitude. One of
                'centric', 'graphic', or 'squashed'.
            lon_direction: Longitude direction. One of 'east' or 'west'.
            photometric_model: Photometric correction to apply during
                reproject(). None (default) means no correction.
            merge_strategy: How to resolve conflicts when the same pixel
                appears in multiple reprojections. Defaults to
                ``BodyMosaicMergeStrategy.BEST_RESOLUTION``.
            image_dtype: NumPy dtype for the reprojected brightness ``img``
                array. Defaults to ``np.float64``.
            metadata_dtype: NumPy dtype for geometry arrays (``resolution``,
                ``eff_resolution``, ``phase``, ``emission``, ``incidence``).
                Defaults to ``np.float32``.

        Raises:
            TypeError: If ``lat_resolution``, ``lon_resolution``, ``zoom``,
                ``edge_margin``, or optional angle / resolution limits have an
                invalid type.
            ValueError: If latlon_type, lon_direction, merge_strategy, or any
                numeric parameter is out of range.

        Note:
            ``time`` is always stored as ``float64`` regardless of
            ``metadata_dtype``. ``image_number`` is always stored as
            ``uint16``, capping a single mosaic at 65,536 contributing images
            (image numbers 0..65535).  ``add()`` raises ``OverflowError`` on the
            image that would exceed that capacity.
        """
        if latlon_type not in ('centric', 'graphic', 'squashed'):
            raise ValueError(
                f"latlon_type must be 'centric', 'graphic', or 'squashed', got {latlon_type!r}"
            )
        if lon_direction not in ('east', 'west'):
            raise ValueError(f"lon_direction must be 'east' or 'west', got {lon_direction!r}")
        if not isinstance(merge_strategy, BodyMosaicMergeStrategy):
            raise ValueError(
                f'merge_strategy must be a BodyMosaicMergeStrategy member, got {merge_strategy!r}'
            )

        if isinstance(lat_resolution, bool) or not isinstance(lat_resolution, _REAL_SCALAR_TYPES):
            raise TypeError(
                f'lat_resolution must be int or float, got {type(lat_resolution).__name__}'
            )
        if isinstance(lon_resolution, bool) or not isinstance(lon_resolution, _REAL_SCALAR_TYPES):
            raise TypeError(
                f'lon_resolution must be int or float, got {type(lon_resolution).__name__}'
            )
        lat_res_f = float(lat_resolution)
        lon_res_f = float(lon_resolution)
        if not math.isfinite(lat_res_f) or lat_res_f <= 0:
            raise ValueError(f'lat_resolution must be finite and > 0, got {lat_resolution!r}')
        if not math.isfinite(lon_res_f) or lon_res_f <= 0:
            raise ValueError(f'lon_resolution must be finite and > 0, got {lon_resolution!r}')

        if isinstance(zoom, bool) or not isinstance(zoom, _INTEGRAL_SCALAR_TYPES):
            raise TypeError(f'zoom must be an int, got {type(zoom).__name__}')
        zoom_i = int(zoom)
        if zoom_i < 1:
            raise ValueError(f'zoom must be >= 1, got {zoom!r}')

        if isinstance(edge_margin, bool) or not isinstance(edge_margin, _INTEGRAL_SCALAR_TYPES):
            raise TypeError(f'edge_margin must be an int, got {type(edge_margin).__name__}')
        edge_margin_i = int(edge_margin)
        if edge_margin_i < 0:
            raise ValueError(f'edge_margin must be >= 0, got {edge_margin!r}')

        if max_incidence is not None:
            if isinstance(max_incidence, bool) or not isinstance(max_incidence, _REAL_SCALAR_TYPES):
                raise TypeError(
                    'max_incidence must be int or float or None, got '
                    f'{type(max_incidence).__name__}'
                )
            mi = float(max_incidence)
            if not math.isfinite(mi) or mi < 0.0 or mi > math.pi:
                raise ValueError(
                    f'max_incidence must be in [0, pi] radians when set, got {max_incidence!r}'
                )

        if max_emission is not None:
            if isinstance(max_emission, bool) or not isinstance(max_emission, _REAL_SCALAR_TYPES):
                raise TypeError(
                    f'max_emission must be int or float or None, got {type(max_emission).__name__}'
                )
            me = float(max_emission)
            if not math.isfinite(me) or me < 0.0 or me > math.pi:
                raise ValueError(
                    f'max_emission must be in [0, pi] radians when set, got {max_emission!r}'
                )

        if max_resolution is not None:
            if isinstance(max_resolution, bool) or not isinstance(
                max_resolution, _REAL_SCALAR_TYPES
            ):
                raise TypeError(
                    'max_resolution must be int or float or None, got '
                    f'{type(max_resolution).__name__}'
                )
            mr = float(max_resolution)
            if not math.isfinite(mr) or mr <= 0.0:
                raise ValueError(
                    'max_resolution (km/pixel) must be finite and > 0 when set, got '
                    f'{max_resolution!r}'
                )

        self.body_name = body_name
        self._lat_resolution = lat_res_f
        self._lon_resolution = lon_res_f
        self._lat_range_init = lat_range
        self._lon_range_init = lon_range
        self._dynamic = dynamic
        self._max_incidence = None if max_incidence is None else float(max_incidence)
        self._max_emission = None if max_emission is None else float(max_emission)
        self._max_resolution = None if max_resolution is None else float(max_resolution)
        self._edge_margin = edge_margin_i
        self._zoom = zoom_i
        self._latlon_type = latlon_type
        self._lon_direction = lon_direction
        self._photometric_model = photometric_model
        self._merge_strategy = merge_strategy
        self._image_dtype: np.dtype = np.dtype(image_dtype)
        self._metadata_dtype: np.dtype = np.dtype(metadata_dtype)

        # Full-grid dimensions: floor(span/res)+1 so the last bin is not dropped
        # (matches ``RingMosaic`` longitude count and avoids int() truncation).
        self._n_full_lat = math.floor(math.pi / lat_res_f) + 1
        self._n_full_lon = math.floor(2.0 * math.pi / lon_res_f) + 1

        # Internal mosaic arrays (empty = not yet initialized)
        self._img: NDArrayFloatType = np.empty((0, 0), dtype=self._image_dtype)
        self._resolution: NDArrayFloatType = np.empty((0, 0), dtype=self._metadata_dtype)
        self._eff_resolution: NDArrayFloatType = np.empty((0, 0), dtype=self._metadata_dtype)
        self._phase: NDArrayFloatType = np.empty((0, 0), dtype=self._metadata_dtype)
        self._emission: NDArrayFloatType = np.empty((0, 0), dtype=self._metadata_dtype)
        self._incidence: NDArrayFloatType = np.empty((0, 0), dtype=self._metadata_dtype)
        self._time: NDArrayFloatType = np.empty((0, 0), dtype=np.float64)
        self._image_number: NDArrayIntType = np.empty((0, 0), dtype=np.uint16)
        self._has_data: NDArrayBoolType = np.empty((0, 0), dtype=np.bool_)

        # Current buffer dimensions
        self._lat_min_bin: int = 0  # full-grid lat bin of internal row 0
        self._lon_min_bin: int = 0  # full-grid lon bin of internal column 0
        self._n_lat: int = 0
        self._n_lon: int = 0

        # Count of images added
        self._image_count: int = 0
        self._contributing_image_names: list[str] = []
        self._sub_solar_lons: list[float] = []
        self._sub_solar_lats: list[float] = []
        self._sub_observer_lons: list[float] = []
        self._sub_observer_lats: list[float] = []

        # Pre-allocate if not dynamic, or if explicit ranges provided
        if not dynamic or lat_range is not None or lon_range is not None:
            self._preallocate(lat_range, lon_range)

    # -----------------------------------------------------------------------
    # Static grid utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_latitudes(
        *,
        latitude_start: float = _MIN_LATITUDE,
        latitude_end: float = _MAX_LATITUDE,
        lat_resolution: float = DEFAULT_LAT_RESOLUTION,
    ) -> NDArrayFloatType:
        """Generate a list of latitudes on resolution boundaries.

        The returned latitudes are on lat_resolution grid boundaries and lie
        within [latitude_start, latitude_end].

        Parameters:
            latitude_start: Minimum latitude (rad). Defaults to -pi/2.
            latitude_end: Maximum latitude (rad). Defaults to pi/2 minus slop.
            lat_resolution: Latitude resolution (rad/pixel).

        Returns:
            1-D array of latitudes (rad).
        """
        start_idx = math.ceil(latitude_start / lat_resolution)
        end_idx = math.floor(latitude_end / lat_resolution)
        return np.arange(start_idx, end_idx + 1) * lat_resolution

    @staticmethod
    def generate_longitudes(
        *,
        longitude_start: float = 0.0,
        longitude_end: float = _MAX_LONGITUDE,
        lon_resolution: float = DEFAULT_LON_RESOLUTION,
    ) -> NDArrayFloatType:
        """Generate a list of longitudes on resolution boundaries.

        Parameters:
            longitude_start: Minimum longitude (rad). Defaults to 0.
            longitude_end: Maximum longitude (rad). Defaults to 2*pi minus slop.
            lon_resolution: Longitude resolution (rad/pixel).

        Returns:
            1-D array of longitudes (rad).
        """
        start_idx = math.ceil(longitude_start / lon_resolution)
        end_idx = math.floor(longitude_end / lon_resolution)
        return np.arange(start_idx, end_idx + 1) * lon_resolution

    # -----------------------------------------------------------------------
    # Coordinate conversion
    # -----------------------------------------------------------------------

    def latitude_longitude_to_pixels(
        self,
        obs: Any,
        latitude: Any,
        longitude: Any,
    ) -> Any:
        """Convert latitude/longitude arrays to (U, V) image pixel coordinates.

        Parameters:
            obs: The Observation.
            latitude: Latitude values (rad).
            longitude: Longitude values (rad).

        Returns:
            UV pairs as a polymath UV object.
        """
        latitude = polymath.Scalar.as_scalar(latitude)
        longitude = polymath.Scalar.as_scalar(longitude)

        if len(longitude) == 0:
            return polymath.Pair(np.empty((0, 2)))

        moon_surface = oops.Body.lookup(self.body_name).surface

        if self._latlon_type == 'centric':
            longitude = moon_surface.lon_from_centric(longitude)
            latitude = moon_surface.lat_from_centric(latitude, longitude)
        elif self._latlon_type == 'graphic':
            longitude = moon_surface.lon_from_graphic(longitude)
            latitude = moon_surface.lat_from_graphic(latitude, longitude)

        if self._lon_direction == 'west':
            longitude = (-longitude) % oops.TWOPI

        return obs.uv_from_coords(moon_surface, (longitude, latitude))

    def _body_reproj_result_outside_fov(
        self,
        obs: Any,
        *,
        mask_only: bool,
        image_name: str,
        navigation_uncertainty: float,
        sub_solar_lon: float = 0.0,
        sub_solar_lat: float = 0.0,
        sub_observer_lon: float = 0.0,
        sub_observer_lat: float = 0.0,
    ) -> BodyReprojResult:
        """Build a placeholder :class:`BodyReprojResult` when the body is off the detector.

        Used when no valid latitude/longitude samples fall on the detector (outside
        FOV or fully masked). Allocates only the minimal 2x2 placeholder window
        (``lat_idx_range`` / ``lon_idx_range`` ``(0, 1)``). Geometry arrays are fully
        masked; ``img`` is either a small fully-masked float grid or a boolean mask
        grid when ``mask_only`` is True. Scalar metadata (``time``, ``image_name``, sub-solar /
        sub-observer lon/lat, dtypes) is still populated so callers can merge counters
        and serialize consistently with normal reprojections.

        Parameters:
            obs: Observation (midtime read from ``obs.midtime``).
            mask_only: If True, ``img`` encodes a boolean valid-pixel mask; else a
                masked float subgrid.
            image_name: Label stored on the result.
            navigation_uncertainty: Scales ``eff_resolution`` when not ``mask_only``.
            sub_solar_lon, sub_solar_lat: Sub-solar coordinates (rad) for this obs.
            sub_observer_lon, sub_observer_lat: Sub-observer coordinates (rad).

        Returns:
            A :class:`BodyReprojResult` with minimal lat/lon index ranges and masked
            geometry fields suitable for ``BodyMosaic.add()`` bookkeeping.
        """
        min_lat_pixel, max_lat_pixel = 0, 1
        min_lon_pixel, max_lon_pixel = 0, 1
        n_sub_lat = max_lat_pixel - min_lat_pixel + 1
        n_sub_lon = max_lon_pixel - min_lon_pixel + 1

        empty_ma = ma.zeros((n_sub_lat, n_sub_lon), dtype=self._metadata_dtype)
        empty_ma.mask = True

        if mask_only:
            repro_img = np.zeros((n_sub_lat, n_sub_lon), dtype=np.bool_)
            return BodyReprojResult(
                body_name=self.body_name,
                img=ma.MaskedArray(repro_img.astype(self._image_dtype)),
                lat_resolution=self._lat_resolution,
                lon_resolution=self._lon_resolution,
                lat_idx_range=(min_lat_pixel, max_lat_pixel),
                lon_idx_range=(min_lon_pixel, max_lon_pixel),
                latlon_type=self._latlon_type,
                lon_direction=self._lon_direction,
                resolution=empty_ma.copy(),
                eff_resolution=empty_ma.copy(),
                phase=empty_ma.copy(),
                emission=empty_ma.copy(),
                incidence=empty_ma.copy(),
                time=obs.midtime,
                photometric_model_name=None,
                image_dtype=self._image_dtype,
                metadata_dtype=self._metadata_dtype,
                image_name=image_name,
                sub_solar_lon=sub_solar_lon,
                sub_solar_lat=sub_solar_lat,
                sub_observer_lon=sub_observer_lon,
                sub_observer_lat=sub_observer_lat,
            )

        repro_img = _create_repro_array(n_sub_lat, n_sub_lon, self._image_dtype)
        repro_res = _create_repro_array(n_sub_lat, n_sub_lon, self._metadata_dtype)
        repro_phase = _create_repro_array(n_sub_lat, n_sub_lon, self._metadata_dtype)
        repro_emission = _create_repro_array(n_sub_lat, n_sub_lon, self._metadata_dtype)
        repro_incidence = _create_repro_array(n_sub_lat, n_sub_lon, self._metadata_dtype)
        eff_res = repro_res.copy()
        if navigation_uncertainty != 0.0:
            eff_res = ma.MaskedArray(
                eff_res.data * (1.0 + navigation_uncertainty), mask=eff_res.mask
            )
        phot_name = self._photometric_model.name if self._photometric_model is not None else None
        return BodyReprojResult(
            body_name=self.body_name,
            img=repro_img.copy(),
            lat_resolution=self._lat_resolution,
            lon_resolution=self._lon_resolution,
            lat_idx_range=(min_lat_pixel, max_lat_pixel),
            lon_idx_range=(min_lon_pixel, max_lon_pixel),
            latlon_type=self._latlon_type,
            lon_direction=self._lon_direction,
            resolution=repro_res.copy(),
            eff_resolution=eff_res,
            phase=repro_phase.copy(),
            emission=repro_emission.copy(),
            incidence=repro_incidence.copy(),
            time=obs.midtime,
            photometric_model_name=phot_name,
            image_dtype=self._image_dtype,
            metadata_dtype=self._metadata_dtype,
            image_name=image_name,
            sub_solar_lon=sub_solar_lon,
            sub_solar_lat=sub_solar_lat,
            sub_observer_lon=sub_observer_lon,
            sub_observer_lat=sub_observer_lat,
        )

    # -----------------------------------------------------------------------
    # Reprojection
    # -----------------------------------------------------------------------

    def reproject(
        self,
        obs: Any,
        *,
        data: NDArrayFloatType | None = None,
        navigation_uncertainty: float = 0.0,
        mask_only: bool = False,
        override_backplane: Any = None,
        subimage_edges: tuple[int, int, int, int] | None = None,
        mask_bad_areas: bool = False,
        image_name: str = '',
    ) -> BodyReprojResult:
        """Reproject the body into a rectangular latitude/longitude space.

        Parameters:
            obs: The Observation (with its FOV already configured as needed).
            data: Image data to reproject. Defaults to obs.data.
            navigation_uncertainty: Maximum uncertainty in pixel pointing
                (pixels), used to compute eff_resolution.
            mask_only: If True, return only the valid-pixel boolean mask.
                Other result fields will be empty masked arrays.
            override_backplane: A Backplane to use instead of creating a new
                one from obs.
            subimage_edges: (u_min, u_max, v_min, v_max) describing the
                subimage extent if override_backplane covers only a subimage.
            mask_bad_areas: If True, expand bad-pixel regions before masking
                to handle images with transmission errors.
            image_name: Optional label stored on the result (e.g. source image stem).

        Returns:
            BodyReprojResult with the reprojected data.

        When the body has no valid latitude/longitude on the detector (fully
        outside the field of view or not projected onto the image), this returns
        immediately with an empty result without building the lat/lon grid or
        sampling image pixels.

        Note:
            The image data is taken from the zoomed, interpolated image, while
            the incidence, emission, phase, and resolution are taken from the
            original non-interpolated data and thus will be slightly more
            coarse-grained. Backplane masks and limit tests are folded into
            ``ok_body_mask`` (see below); scatter uses that same mask so ``img``
            is never valid where ``phase``, ``emission``, ``incidence``, or
            ``resolution`` would be masked or beyond configured limits.

        Raises:
            TypeError: If ``navigation_uncertainty`` is not a real number (``bool`` is not
                accepted).
            ValueError: If ``navigation_uncertainty`` is not finite or is ``< 0``.
        """
        logger = IMAGE_LOGGER

        navigation_uncertainty = _validate_navigation_uncertainty(navigation_uncertainty)

        if data is None:
            data = obs.data

        bp = override_backplane if override_backplane is not None else oops.backplane.Backplane(obs)

        sub_solar_lon = float(bp.sub_solar_longitude(self.body_name).vals)
        sub_solar_lat = float(bp.sub_solar_latitude(self.body_name).vals)
        sub_observer_lon = float(bp.sub_observer_longitude(self.body_name).vals)
        sub_observer_lat = float(bp.sub_observer_latitude(self.body_name).vals)

        if self._max_incidence is not None or not mask_only:
            bp_incidence = bp.incidence_angle(self.body_name).mvals.astype(self._metadata_dtype)

        if self._max_emission is not None or not mask_only or self._max_resolution is not None:
            bp_emission = bp.emission_angle(self.body_name).mvals.astype(self._metadata_dtype)

        if not mask_only or self._max_resolution is not None:
            center_resolution = bp.center_resolution(self.body_name).vals
            resolution = center_resolution / np.cos(bp_emission)

        if not mask_only:
            bp_phase = bp.phase_angle(self.body_name).mvals.astype(self._metadata_dtype)

        bp_latitude = bp.latitude(self.body_name, lat_type=self._latlon_type)
        body_mask_inv = ma.getmaskarray(bp_latitude.mvals).copy()
        bp_latitude = bp_latitude.mvals.astype(self._metadata_dtype)

        bp_lon_ma = bp.longitude(
            self.body_name, direction=self._lon_direction, lon_type=self._latlon_type
        )
        lon_mask_inv = ma.getmaskarray(bp_lon_ma.mvals).copy()
        bp_longitude = bp_lon_ma.mvals.astype(self._metadata_dtype)

        # No surface point on the detector: skip grid construction and image sampling.
        if np.all(np.logical_or(body_mask_inv, lon_mask_inv)):
            logger.info(
                'Body %s has no valid lat/lon on the detector (outside FOV); '
                'skipping reprojection.',
                self.body_name,
            )
            return self._body_reproj_result_outside_fov(
                obs,
                mask_only=mask_only,
                image_name=image_name,
                navigation_uncertainty=navigation_uncertainty,
                sub_solar_lon=sub_solar_lon,
                sub_solar_lat=sub_solar_lat,
                sub_observer_lon=sub_observer_lon,
                sub_observer_lat=sub_observer_lat,
            )

        if subimage_edges is not None:
            u_min, u_max, v_min, v_max = subimage_edges
            subimg = data[v_min : v_max + 1, u_min : u_max + 1]
        else:
            u_min = 0
            v_min = 0
            u_max = data.shape[1] - 1
            v_max = data.shape[0] - 1
            subimg = data

        if not mask_only:
            zero_mask = subimg == 0.0
            if mask_bad_areas:
                zero_mask = maximum_filter(zero_mask, 3)

        edge_margin = self._edge_margin
        if edge_margin > 0:
            body_mask_inv[:edge_margin, :] = True
            body_mask_inv[-edge_margin:, :] = True
            body_mask_inv[:, :edge_margin] = True
            body_mask_inv[:, -edge_margin:] = True

        ok_body_mask_inv = body_mask_inv

        # Invalid detector pixels: explicit MaskedArray mask bits, plus limit
        # tests. Plain ``bp_* > limit`` drops mask on comparisons, so use
        # ``ma.filled(..., True)`` so masked elements count as invalid.
        if self._max_incidence is not None or not mask_only:
            bad_inc = ma.getmaskarray(bp_incidence)
            if self._max_incidence is not None:
                bad_inc = np.logical_or(
                    bad_inc,
                    ma.filled(bp_incidence > self._max_incidence, True),
                )
            ok_body_mask_inv = np.logical_or(ok_body_mask_inv, bad_inc)

        if self._max_emission is not None or not mask_only or self._max_resolution is not None:
            bad_emi = ma.getmaskarray(bp_emission)
            if self._max_emission is not None:
                bad_emi = np.logical_or(
                    bad_emi,
                    ma.filled(bp_emission > self._max_emission, True),
                )
            ok_body_mask_inv = np.logical_or(ok_body_mask_inv, bad_emi)

        if not mask_only or self._max_resolution is not None:
            bad_res = ma.getmaskarray(resolution)
            if self._max_resolution is not None:
                bad_res = np.logical_or(
                    bad_res,
                    ma.filled(resolution > self._max_resolution, True),
                )
            ok_body_mask_inv = np.logical_or(ok_body_mask_inv, bad_res)

        if not mask_only:
            ok_body_mask_inv = np.logical_or(ok_body_mask_inv, ma.getmaskarray(bp_phase))
            if zero_mask.shape == ok_body_mask_inv.shape:
                ok_body_mask_inv = np.logical_or(ok_body_mask_inv, zero_mask)
            else:
                # Full-frame backplane with a subimage crop: embed the
                # subimage-shaped zero mask; everything outside the subimage
                # has no data and is invalid.
                zero_mask_full = np.ones(ok_body_mask_inv.shape, dtype=np.bool_)
                zero_mask_full[v_min : v_max + 1, u_min : u_max + 1] = zero_mask
                ok_body_mask_inv = np.logical_or(ok_body_mask_inv, zero_mask_full)

        ok_body_mask = np.logical_not(ok_body_mask_inv)

        empty_mask = not np.any(ok_body_mask)

        latitude_pixels = self._n_full_lat
        longitude_pixels = self._n_full_lon

        if empty_mask:
            min_lat_pixel, max_lat_pixel = 0, 1
            min_lon_pixel, max_lon_pixel = 0, 1
        else:
            min_latitude = np.min(bp_latitude[ok_body_mask] + math.pi / 2.0)
            max_latitude = np.max(bp_latitude[ok_body_mask] + math.pi / 2.0)
            min_lat_pixel = int(np.floor(min_latitude / self._lat_resolution))
            min_lat_pixel = np.clip(min_lat_pixel, 0, latitude_pixels - 1)
            max_lat_pixel = int(np.ceil(max_latitude / self._lat_resolution))
            max_lat_pixel = np.clip(max_lat_pixel, 0, latitude_pixels - 1)

            min_longitude = np.min(bp_longitude[ok_body_mask])
            max_longitude = np.max(bp_longitude[ok_body_mask])
            min_lon_pixel = int(np.floor(min_longitude / self._lon_resolution))
            min_lon_pixel = np.clip(min_lon_pixel, 0, longitude_pixels - 1)
            max_lon_pixel = int(np.ceil(max_longitude / self._lon_resolution))
            max_lon_pixel = np.clip(max_lon_pixel, 0, longitude_pixels - 1)

        num_lat = max_lat_pixel - min_lat_pixel + 1
        num_lon = max_lon_pixel - min_lon_pixel + 1

        lat_bins = np.repeat(np.arange(min_lat_pixel, max_lat_pixel + 1, dtype=np.int32), num_lon)
        lat_bins_act = lat_bins * self._lat_resolution - math.pi / 2.0
        lon_bins = np.tile(np.arange(min_lon_pixel, max_lon_pixel + 1, dtype=np.int32), num_lat)
        lon_bins_act = lon_bins * self._lon_resolution

        logger.info(
            'Lat Res %.3f deg / Lon Res %.3f deg / LatLon %s / LonDir %s',
            math.degrees(self._lat_resolution),
            math.degrees(self._lon_resolution),
            self._latlon_type,
            self._lon_direction,
        )
        if empty_mask:
            logger.info('Empty body mask')
        else:
            logger.info(
                'Latitude range       %8.2f %8.2f',
                math.degrees(np.min(bp_latitude[ok_body_mask])),
                math.degrees(np.max(bp_latitude[ok_body_mask])),
            )
            logger.debug(
                'Latitude bin range  %8.2f %8.2f',
                math.degrees(np.min(lat_bins_act)),
                math.degrees(np.max(lat_bins_act)),
            )
            logger.info(
                'Longitude range      %8.2f %8.2f',
                math.degrees(np.min(bp_longitude[ok_body_mask])),
                math.degrees(np.max(bp_longitude[ok_body_mask])),
            )
            logger.debug(
                'Longitude bin range %8.2f %8.2f',
                math.degrees(np.min(lon_bins_act)),
                math.degrees(np.max(lon_bins_act)),
            )
            if not mask_only:
                logger.info(
                    'Resolution range     %8.2f %8.2f',
                    np.min(resolution[ok_body_mask]),
                    np.max(resolution[ok_body_mask]),
                )
            logger.debug('Latitude pixel range %d %d', int(min_lat_pixel), int(max_lat_pixel))
            logger.debug('Longitude pixel range %d %d', int(min_lon_pixel), int(max_lon_pixel))

        uv = self.latitude_longitude_to_pixels(obs, lat_bins_act, lon_bins_act)
        u_sv, v_sv = uv.to_scalars()
        pixmask = ma.getmaskarray(u_sv.mvals)
        u_pixels = u_sv.vals
        v_pixels = v_sv.vals

        # u_pixels / v_pixels are pixel-corner coordinates, so the region running
        # from column u_min to column u_max occupies [u_min, u_max + 1) and the
        # containment test is half-open at the top. Closing it at u_max would
        # admit only the left edge of the last column and reject the rest of it.
        # Without subimage_edges these bounds are the frame's own, whose
        # outermost columns a non-zero edge_margin masks out below in any case,
        # so no product built with the default margin changes. With a subimage
        # crop the bound is interior to the frame and edge_margin does not
        # cover it, and the last column of the crop is now sampled across its
        # whole width.
        goodumask = np.logical_and(u_pixels >= u_min, u_pixels < u_max + 1)
        goodvmask = np.logical_and(v_pixels >= v_min, v_pixels < v_max + 1)
        goodmask = goodumask & goodvmask & ~pixmask

        # Keep the fractional pixel positions: integer indices serve the mask
        # and backplane lookups, while the fractions drive sub-pixel sampling
        # when zoom > 1. The mask above guarantees the region-relative
        # coordinates are non-negative, so floor is what the conversion to a
        # column number needs; it is spelled out rather than left to astype's
        # truncation, which would round the wrong way for a negative value.
        good_u_frac = u_pixels[goodmask] - u_min
        good_v_frac = v_pixels[goodmask] - v_min
        good_u = np.floor(good_u_frac).astype(int)
        good_v = np.floor(good_v_frac).astype(int)
        good_lat = lat_bins[goodmask]
        good_lon = lon_bins[goodmask]

        shape = (latitude_pixels, longitude_pixels)
        empty_ma = ma.zeros(shape, dtype=self._metadata_dtype)
        empty_ma.mask = True
        empty_sub = empty_ma[min_lat_pixel : max_lat_pixel + 1, min_lon_pixel : max_lon_pixel + 1]

        if mask_only:
            repro_img_full = np.zeros(shape, dtype=np.bool_)
            repro_img_full[good_lat, good_lon] = True
            repro_img_sub = repro_img_full[
                min_lat_pixel : max_lat_pixel + 1, min_lon_pixel : max_lon_pixel + 1
            ]
            # Return an abbreviated result; img is the bool mask on the same crop as metadata
            return BodyReprojResult(
                body_name=self.body_name,
                img=ma.MaskedArray(repro_img_sub.astype(self._image_dtype)),
                lat_resolution=self._lat_resolution,
                lon_resolution=self._lon_resolution,
                lat_idx_range=(min_lat_pixel, max_lat_pixel),
                lon_idx_range=(min_lon_pixel, max_lon_pixel),
                latlon_type=self._latlon_type,
                lon_direction=self._lon_direction,
                resolution=empty_sub.copy(),
                eff_resolution=empty_sub.copy(),
                phase=empty_sub.copy(),
                emission=empty_sub.copy(),
                incidence=empty_sub.copy(),
                time=obs.midtime,
                photometric_model_name=None,
                image_dtype=self._image_dtype,
                metadata_dtype=self._metadata_dtype,
                image_name=image_name,
                sub_solar_lon=sub_solar_lon,
                sub_solar_lat=sub_solar_lat,
                sub_observer_lon=sub_observer_lon,
                sub_observer_lat=sub_observer_lat,
            )

        # Filter additional bad data
        goodmask2 = subimg[good_v, good_u] != 0.0
        good_u = good_u[goodmask2]
        good_v = good_v[goodmask2]
        good_u_frac = good_u_frac[goodmask2]
        good_v_frac = good_v_frac[goodmask2]
        good_lat = good_lat[goodmask2]
        good_lon = good_lon[goodmask2]

        # Lat/lon bin centers can map to (u, v) that are outside ``ok_body_mask``;
        # keep scatter aligned with the same invalid mask as incidence/emission/etc.
        # ``good_u`` / ``good_v`` are subimage-relative (``subimg`` = ``data`` when
        # ``subimage_edges`` is None). ``ok_body_mask`` matches ``subimg`` when the
        # backplane (e.g. ``override_backplane``) is built on the subimage; otherwise
        # it is full-frame like ``data`` and needs ``u_min`` / ``v_min``.
        if ok_body_mask.shape == subimg.shape:
            ok_on_detector = ok_body_mask[good_v, good_u]
        else:
            ok_on_detector = ok_body_mask[good_v + v_min, good_u + u_min]
        good_u = good_u[ok_on_detector]
        good_v = good_v[ok_on_detector]
        good_u_frac = good_u_frac[ok_on_detector]
        good_v_frac = good_v_frac[ok_on_detector]
        good_lat = good_lat[ok_on_detector]
        good_lon = good_lon[ok_on_detector]

        # ``good_u`` / ``good_v`` index the cropped ``subimg``; backplanes built on
        # the full detector use full-frame shapes (same as ``ok_body_mask`` when it
        # differs from ``subimg``). Align sampling indices with ``bp_*`` / ``resolution``.
        if ok_body_mask.shape == subimg.shape:
            bp_v, bp_u = good_v, good_u
        else:
            bp_v, bp_u = good_v + v_min, good_u + u_min

        # Zoom for interpolation (always copy when zoom==1: ``subimg`` may alias
        # ``obs.data``, which can be read-only mmap, and we must not mutate the obs.)
        # The zoomed image must be interpolated, not block-replicated: sampling
        # it at the fractional pixel positions is what gives zoom > 1 its
        # documented sub-pixel meaning.
        #
        # grid_mode=True is not scipy's default and is load-bearing here. It
        # makes ndimage_zoom treat a pixel as a square of unit area rather than
        # as a point at its centre, so the zoomed array is the same region of
        # the sky divided into zoom times as many pixels: sub-pixel s of pixel p
        # covers [p + s / zoom, p + (s + 1) / zoom) in the original. That is
        # exactly the convention the sampling below relies on, where a
        # region-relative pixel-corner coordinate scaled by zoom and floored
        # names the sub-pixel containing it. With the default grid_mode=False
        # the zoomed grid's endpoints are pinned to the original pixel centres
        # instead, which shifts and rescales it by up to half a pixel.
        if self._zoom == 1:
            zoom_data = np.array(subimg, copy=True)
        else:
            zoom_data = ndimage_zoom(
                np.asarray(subimg, dtype=np.float64),
                self._zoom,
                order=1,
                mode='nearest',
                grid_mode=True,
            )
            if not zoom_data.flags.writeable:
                zoom_data = np.array(zoom_data, copy=True)

        bad_data_mask = subimg == 0.0
        if self._zoom == 1:
            bad_data_mask_zoom = bad_data_mask
        else:
            bad_data_mask_zoom = array_zoom(bad_data_mask, (self._zoom, self._zoom))
        zoom_data[bad_data_mask_zoom] = 0.0

        interp_data = zoom_data[
            np.floor(good_v_frac * self._zoom).astype('int'),
            np.floor(good_u_frac * self._zoom).astype('int'),
        ]

        # Apply photometric correction if requested
        if self._photometric_model is not None:
            inc = bp_incidence[bp_v, bp_u]
            emi = bp_emission[bp_v, bp_u]
            pha = bp_phase[bp_v, bp_u]
            interp_data = self._photometric_model.correct(
                interp_data, incidence=inc, emission=emi, phase=pha
            )

        repro_img_full = _create_repro_array(latitude_pixels, longitude_pixels, self._image_dtype)
        repro_img_full[good_lat, good_lon] = interp_data

        repro_res_full = _create_repro_array(
            latitude_pixels, longitude_pixels, self._metadata_dtype
        )
        repro_res_full[good_lat, good_lon] = resolution[bp_v, bp_u]

        repro_phase_full = _create_repro_array(
            latitude_pixels, longitude_pixels, self._metadata_dtype
        )
        repro_phase_full[good_lat, good_lon] = bp_phase[bp_v, bp_u]

        repro_emission_full = _create_repro_array(
            latitude_pixels, longitude_pixels, self._metadata_dtype
        )
        repro_emission_full[good_lat, good_lon] = bp_emission[bp_v, bp_u]

        repro_incidence_full = _create_repro_array(
            latitude_pixels, longitude_pixels, self._metadata_dtype
        )
        repro_incidence_full[good_lat, good_lon] = bp_incidence[bp_v, bp_u]

        sub = slice(min_lat_pixel, max_lat_pixel + 1), slice(min_lon_pixel, max_lon_pixel + 1)
        eff_res = repro_res_full[sub].copy()
        if navigation_uncertainty != 0.0:
            eff_res = ma.MaskedArray(
                eff_res.data * (1.0 + navigation_uncertainty), mask=eff_res.mask
            )

        phot_name = self._photometric_model.name if self._photometric_model is not None else None

        return BodyReprojResult(
            body_name=self.body_name,
            img=repro_img_full[sub].copy(),
            lat_resolution=self._lat_resolution,
            lon_resolution=self._lon_resolution,
            lat_idx_range=(min_lat_pixel, max_lat_pixel),
            lon_idx_range=(min_lon_pixel, max_lon_pixel),
            latlon_type=self._latlon_type,
            lon_direction=self._lon_direction,
            resolution=repro_res_full[sub].copy(),
            eff_resolution=eff_res,
            phase=repro_phase_full[sub].copy(),
            emission=repro_emission_full[sub].copy(),
            incidence=repro_incidence_full[sub].copy(),
            time=obs.midtime,
            photometric_model_name=phot_name,
            image_dtype=self._image_dtype,
            metadata_dtype=self._metadata_dtype,
            image_name=image_name,
            sub_solar_lon=sub_solar_lon,
            sub_solar_lat=sub_solar_lat,
            sub_observer_lon=sub_observer_lon,
            sub_observer_lat=sub_observer_lat,
        )

    # -----------------------------------------------------------------------
    # Mosaic accumulation
    # -----------------------------------------------------------------------

    def add(
        self,
        repro: BodyReprojResult,
        *,
        resolution_threshold: float = 1.0,
        copy_slop: int = 0,
        max_incidence: float | _UseMosaicLimitsSentinel | None = USE_MOSAIC_LIMITS,
        max_emission: float | _UseMosaicLimitsSentinel | None = USE_MOSAIC_LIMITS,
        max_resolution: float | _UseMosaicLimitsSentinel | None = USE_MOSAIC_LIMITS,
    ) -> None:
        """Add a reprojected image to the mosaic.

        For each valid pixel in repro, it is incorporated into the mosaic
        according to the merge strategy.

        Parameters:
            repro: Result from reproject().
            resolution_threshold: Factor by which repro's effective resolution
                must exceed the mosaic's to trigger replacement. Should be
                >= 1.0 (higher = stricter).
            copy_slop: Number of additional pixels around each copied pixel to
                also copy, reducing isolated-pixel artifacts.
            max_incidence: Maximum incidence angle (rad) for pixels that may be
                written. :data:`USE_MOSAIC_LIMITS` (default) uses the mosaic's
                constructor value; an explicit ``float`` overrides for this call
                only; ``None`` disables the incidence cutoff for this call.
            max_emission: Same pattern for emission angle (rad).
            max_resolution: Same pattern for center resolution in km/pixel
                (compared to ``repro.resolution``, as in :meth:`reproject`).

        Raises:
            OverflowError: If the number of images added would exceed the
                uint16 capacity of 65,536 (image numbers 0..65535).
            ValueError: If repro's resolution, coordinate system, or
                photometric model name does not match the mosaic's configuration.
        """
        if self._image_count > np.iinfo(np.uint16).max:
            raise OverflowError(
                f'image_count {self._image_count} exceeds uint16 max '
                f'{np.iinfo(np.uint16).max}; cannot add more images'
            )

        self._validate_repro(repro)

        lat_min = repro.lat_idx_range[0]
        lat_max = repro.lat_idx_range[1]
        lon_min = repro.lon_idx_range[0]
        lon_max = repro.lon_idx_range[1]

        # Expand internal arrays to include the reprojection range if dynamic
        if self._dynamic:
            self._expand_to_include(lat_min, lat_max, lon_min, lon_max)
        elif self._n_lat == 0:
            # dynamic=False but not yet allocated (shouldn't happen after _preallocate)
            return

        # Map repro lat/lon bins to internal row/col indices
        repro_lat_bins = np.arange(lat_min, lat_max + 1, dtype=np.int32)
        repro_lon_bins = np.arange(lon_min, lon_max + 1, dtype=np.int32)
        all_lat_2d, all_lon_2d = np.meshgrid(repro_lat_bins, repro_lon_bins, indexing='ij')
        all_lat: NDArrayIntType = all_lat_2d.ravel()
        all_lon: NDArrayIntType = all_lon_2d.ravel()

        # Valid pixels in the reprojection
        repro_valid = ~ma.getmaskarray(repro.img).ravel()
        valid_lat = all_lat[repro_valid]
        valid_lon = all_lon[repro_valid]

        if len(valid_lat) == 0:
            self._contributing_image_names.append(repro.image_name)
            self._sub_solar_lons.append(repro.sub_solar_lon)
            self._sub_solar_lats.append(repro.sub_solar_lat)
            self._sub_observer_lons.append(repro.sub_observer_lon)
            self._sub_observer_lats.append(repro.sub_observer_lat)
            self._image_count += 1
            return

        # Map to internal indices
        rows = valid_lat - self._lat_min_bin
        cols = (valid_lon - self._lon_min_bin) % self._n_full_lon

        # Filter to in-range (for dynamic=False or after clipping)
        in_range = (rows >= 0) & (rows < self._n_lat) & (cols < self._n_lon)
        rows = rows[in_range]
        cols = cols[in_range]

        if len(rows) == 0:
            self._contributing_image_names.append(repro.image_name)
            self._sub_solar_lons.append(repro.sub_solar_lon)
            self._sub_solar_lats.append(repro.sub_solar_lat)
            self._sub_observer_lons.append(repro.sub_observer_lon)
            self._sub_observer_lats.append(repro.sub_observer_lat)
            self._image_count += 1
            return

        # Build flat index into repro arrays
        repro_n_lon = lon_max - lon_min + 1
        repro_flat = (valid_lat[in_range] - lat_min) * repro_n_lon + (valid_lon[in_range] - lon_min)

        # Get repro values at valid positions
        repro_img_flat = repro.img.data.ravel()[repro_flat]
        repro_eff_res_flat = repro.eff_resolution.data.ravel()[repro_flat]
        repro_res_flat = repro.resolution.data.ravel()[repro_flat]
        repro_phase_flat = repro.phase.data.ravel()[repro_flat]
        repro_emission_flat = repro.emission.data.ravel()[repro_flat]
        repro_incidence_flat = repro.incidence.data.ravel()[repro_flat]

        # Determine which pixels to replace based on the merge strategy.
        existing_has_data = self._has_data[rows, cols]
        existing_eff_res = self._eff_resolution[rows, cols]

        if self._merge_strategy == BodyMosaicMergeStrategy.BEST_RESOLUTION:
            replace_mask = np.logical_or(
                ~existing_has_data,
                repro_eff_res_flat * resolution_threshold < existing_eff_res,
            )
        else:
            raise NotImplementedError(f'Unsupported merge strategy: {self._merge_strategy!r}')

        eff_mi = (
            self._max_incidence
            if max_incidence is USE_MOSAIC_LIMITS
            else cast(float | None, max_incidence)
        )
        eff_me = (
            self._max_emission
            if max_emission is USE_MOSAIC_LIMITS
            else cast(float | None, max_emission)
        )
        eff_mr = (
            self._max_resolution
            if max_resolution is USE_MOSAIC_LIMITS
            else cast(float | None, max_resolution)
        )

        n_cand = len(rows)
        limits_ok = np.ones(n_cand, dtype=np.bool_)
        if eff_mi is not None:
            inc_mask = ma.getmaskarray(repro.incidence).ravel()[repro_flat]
            inc_data = repro.incidence.data.ravel()[repro_flat]
            bad = np.logical_or(inc_mask, ~np.isfinite(inc_data))
            bad = np.logical_or(bad, inc_data > eff_mi)
            limits_ok &= ~bad
        if eff_me is not None:
            emi_mask = ma.getmaskarray(repro.emission).ravel()[repro_flat]
            emi_data = repro.emission.data.ravel()[repro_flat]
            bad = np.logical_or(emi_mask, ~np.isfinite(emi_data))
            bad = np.logical_or(bad, emi_data > eff_me)
            limits_ok &= ~bad
        if eff_mr is not None:
            res_mask = ma.getmaskarray(repro.resolution).ravel()[repro_flat]
            res_data = repro.resolution.data.ravel()[repro_flat]
            bad = np.logical_or(res_mask, ~np.isfinite(res_data))
            bad = np.logical_or(bad, res_data > eff_mr)
            limits_ok &= ~bad

        replace_mask = np.logical_and(replace_mask, limits_ok)

        if copy_slop > 0:
            # Expand the replace mask using the slop radius: candidate pixels
            # within the slop radius of a winning pixel are also copied, as
            # long as they pass the limits themselves.
            replace_full = np.zeros((self._n_lat, self._n_lon), dtype=np.bool_)
            replace_full[rows, cols] = replace_mask
            replace_full = maximum_filter(replace_full, copy_slop * 2 + 1)
            replace_mask = replace_full[rows, cols] & limits_ok

        r_rows = rows[replace_mask]
        r_cols = cols[replace_mask]

        self._img[r_rows, r_cols] = repro_img_flat[replace_mask]
        self._resolution[r_rows, r_cols] = repro_res_flat[replace_mask]
        self._eff_resolution[r_rows, r_cols] = repro_eff_res_flat[replace_mask]
        self._phase[r_rows, r_cols] = repro_phase_flat[replace_mask]
        self._emission[r_rows, r_cols] = repro_emission_flat[replace_mask]
        self._incidence[r_rows, r_cols] = repro_incidence_flat[replace_mask]
        self._image_number[r_rows, r_cols] = self._image_count
        self._time[r_rows, r_cols] = repro.time
        self._has_data[r_rows, r_cols] = True

        self._contributing_image_names.append(repro.image_name)
        self._sub_solar_lons.append(repro.sub_solar_lon)
        self._sub_solar_lats.append(repro.sub_solar_lat)
        self._sub_observer_lons.append(repro.sub_observer_lon)
        self._sub_observer_lats.append(repro.sub_observer_lat)
        self._image_count += 1

    # -----------------------------------------------------------------------
    # Mosaic retrieval
    # -----------------------------------------------------------------------

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Current data extent as ((lat_min, lat_max), (lon_min, lon_max)).

        Returns the bounding box of all valid data currently in the mosaic.
        Longitude bounds are in the contiguous arc order (min may be greater
        than max if the data wraps around the 0/2*pi boundary).

        Returns:
            ((lat_min, lat_max), (lon_min, lon_max)) in radians, or None if
            no data has been added yet.
        """
        if self._n_lat == 0 or not np.any(self._has_data):
            return None

        # Bounds of the valid data, not of the (possibly larger) internal
        # buffer: an all-masked contribution can grow the buffer without
        # adding any data.
        data_rows = np.flatnonzero(np.any(self._has_data, axis=1))
        data_cols = np.flatnonzero(np.any(self._has_data, axis=0))
        lat_min = (self._lat_min_bin + int(data_rows[0])) * self._lat_resolution - math.pi / 2.0
        lat_max = (self._lat_min_bin + int(data_rows[-1])) * self._lat_resolution - math.pi / 2.0
        lon_min = (
            (self._lon_min_bin + int(data_cols[0])) % self._n_full_lon
        ) * self._lon_resolution
        lon_max = (
            (self._lon_min_bin + int(data_cols[-1])) % self._n_full_lon
        ) * self._lon_resolution
        return (lat_min, lat_max), (lon_min, lon_max)

    def to_bounded(
        self,
        *,
        lat_range: tuple[float, float] | None = None,
        lon_range: tuple[float, float] | None = None,
    ) -> BodyMosaicData:
        """Return the mosaic clipped to the given lat/lon range.

        Parameters:
            lat_range: (min_lat, max_lat) in radians. Defaults to current data
                bounds.
            lon_range: (min_lon, max_lon) in radians. May wrap around 0/2*pi
                (e.g., (5.9, 0.3) means from 5.9 rad through 2*pi to 0.3 rad).
                Defaults to current data bounds.

        Returns:
            BodyMosaicData with the clipped mosaic.

        Raises:
            ValueError: If no data has been added yet and no range is specified.
        """
        if self._n_lat == 0 or not np.any(self._has_data):
            raise ValueError('No data in mosaic; cannot return bounded view')

        current_bounds = self.bounds
        assert current_bounds is not None

        if lat_range is None:
            lat_range = current_bounds[0]
        if lon_range is None:
            lon_range = current_bounds[1]

        # Convert to bin ranges
        lat_min_bin = round((lat_range[0] + math.pi / 2.0) / self._lat_resolution)
        lat_max_bin = round((lat_range[1] + math.pi / 2.0) / self._lat_resolution)
        lat_min_bin = max(0, min(lat_min_bin, self._n_full_lat - 1))
        lat_max_bin = max(0, min(lat_max_bin, self._n_full_lat - 1))

        lon_min_bin = round(lon_range[0] / self._lon_resolution)
        lon_max_bin = round(lon_range[1] / self._lon_resolution)

        # Handle wraparound
        if lon_min_bin <= lon_max_bin:
            lon_bins = np.arange(lon_min_bin, lon_max_bin + 1)
        else:
            # Wraps: from lon_min to end, then from 0 to lon_max
            lon_bins = np.concatenate(
                [
                    np.arange(lon_min_bin, self._n_full_lon),
                    np.arange(0, lon_max_bin + 1),
                ]
            )

        lat_bins = np.arange(lat_min_bin, lat_max_bin + 1)
        n_out_lat = len(lat_bins)
        n_out_lon = len(lon_bins)

        return self._extract_region(lat_bins, lon_bins, n_out_lat, n_out_lon, lat_range, lon_range)

    def to_full(self) -> BodyMosaicData:
        """Return the mosaic as a full latitude x longitude grid.

        Returns:
            BodyMosaicData covering the full -pi/2 to pi/2 x 0 to 2*pi grid.
            Pixels with no data are masked.
        """
        lat_range = (
            0 * self._lat_resolution - math.pi / 2.0,
            (self._n_full_lat - 1) * self._lat_resolution - math.pi / 2.0,
        )
        lon_range = (0.0, (self._n_full_lon - 1) * self._lon_resolution)
        lat_bins = np.arange(self._n_full_lat)
        lon_bins = np.arange(self._n_full_lon)
        return self._extract_region(
            lat_bins, lon_bins, self._n_full_lat, self._n_full_lon, lat_range, lon_range
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _preallocate(
        self,
        lat_range: tuple[float, float] | None,
        lon_range: tuple[float, float] | None,
    ) -> None:
        """Pre-allocate internal arrays based on given ranges.

        When a range is None, the full valid range is used so that
        ``dynamic=False`` with no explicit range produces a fully-allocated
        global mosaic rather than an empty one.

        Parameters:
            lat_range: (min_lat, max_lat) in radians, or None for full range.
            lon_range: (min_lon, max_lon) in radians, or None for full range.
        """
        if lat_range is not None:
            lat_min_bin = math.ceil((lat_range[0] + math.pi / 2.0) / self._lat_resolution)
            lat_max_bin = math.floor((lat_range[1] + math.pi / 2.0) / self._lat_resolution)
            lat_min_bin = max(0, lat_min_bin)
            lat_max_bin = min(self._n_full_lat - 1, lat_max_bin)
            n_lat = lat_max_bin - lat_min_bin + 1
        else:
            lat_min_bin = 0
            lat_max_bin = self._n_full_lat - 1
            n_lat = self._n_full_lat

        if lon_range is not None:
            lon_min_bin = math.ceil(lon_range[0] / self._lon_resolution)
            lon_max_bin = math.floor(lon_range[1] / self._lon_resolution)
            lon_min_bin = max(0, lon_min_bin)
            lon_max_bin = min(self._n_full_lon - 1, lon_max_bin)
            n_lon = lon_max_bin - lon_min_bin + 1
        else:
            lon_min_bin = 0
            lon_max_bin = self._n_full_lon - 1
            n_lon = self._n_full_lon

        self._lat_min_bin = lat_min_bin
        self._lon_min_bin = lon_min_bin
        self._n_lat = n_lat
        self._n_lon = n_lon

        if n_lat > 0 and n_lon > 0:
            self._allocate(n_lat, n_lon)

    def _allocate(self, n_lat: int, n_lon: int) -> None:
        """Allocate fresh internal arrays of the given shape."""
        self._img = np.zeros((n_lat, n_lon), dtype=self._image_dtype)
        self._resolution = np.zeros((n_lat, n_lon), dtype=self._metadata_dtype)
        self._eff_resolution = np.zeros((n_lat, n_lon), dtype=self._metadata_dtype)
        self._phase = np.zeros((n_lat, n_lon), dtype=self._metadata_dtype)
        self._emission = np.zeros((n_lat, n_lon), dtype=self._metadata_dtype)
        self._incidence = np.zeros((n_lat, n_lon), dtype=self._metadata_dtype)
        self._time = np.zeros((n_lat, n_lon), dtype=np.float64)
        self._image_number = np.zeros((n_lat, n_lon), dtype=np.uint16)
        self._has_data = np.zeros((n_lat, n_lon), dtype=np.bool_)

    def _expand_to_include(self, lat_min: int, lat_max: int, lon_min: int, lon_max: int) -> None:
        """Expand internal arrays to cover the given lat/lon bin range."""
        if self._n_lat == 0:
            # First allocation
            self._lat_min_bin = lat_min
            self._lon_min_bin = lon_min
            self._n_lat = lat_max - lat_min + 1
            self._n_lon = lon_max - lon_min + 1
            self._allocate(self._n_lat, self._n_lon)
            return

        self._expand_lat(lat_min, lat_max)
        self._expand_lon(lon_min, lon_max)

    def _expand_lat(self, new_lat_min: int, new_lat_max: int) -> None:
        """Expand latitude dimension to include [new_lat_min, new_lat_max]."""
        old_lat_min = self._lat_min_bin
        old_lat_max = old_lat_min + self._n_lat - 1

        new_min = min(old_lat_min, new_lat_min)
        new_max = max(old_lat_max, new_lat_max)

        if new_min == old_lat_min and new_max == old_lat_max:
            return  # No expansion needed

        n_new = new_max - new_min + 1
        row_offset = old_lat_min - new_min

        new_img = np.zeros((n_new, self._n_lon), dtype=self._image_dtype)
        new_res = np.zeros((n_new, self._n_lon), dtype=self._metadata_dtype)
        new_eff = np.zeros((n_new, self._n_lon), dtype=self._metadata_dtype)
        new_phase = np.zeros((n_new, self._n_lon), dtype=self._metadata_dtype)
        new_emi = np.zeros((n_new, self._n_lon), dtype=self._metadata_dtype)
        new_inc = np.zeros((n_new, self._n_lon), dtype=self._metadata_dtype)
        new_time = np.zeros((n_new, self._n_lon), dtype=np.float64)
        new_imgnum = np.zeros((n_new, self._n_lon), dtype=np.uint16)
        new_has = np.zeros((n_new, self._n_lon), dtype=np.bool_)

        sl = slice(row_offset, row_offset + self._n_lat)
        new_img[sl] = self._img
        new_res[sl] = self._resolution
        new_eff[sl] = self._eff_resolution
        new_phase[sl] = self._phase
        new_emi[sl] = self._emission
        new_inc[sl] = self._incidence
        new_time[sl] = self._time
        new_imgnum[sl] = self._image_number
        new_has[sl] = self._has_data

        self._img = new_img
        self._resolution = new_res
        self._eff_resolution = new_eff
        self._phase = new_phase
        self._emission = new_emi
        self._incidence = new_inc
        self._time = new_time
        self._image_number = new_imgnum
        self._has_data = new_has
        self._lat_min_bin = new_min
        self._n_lat = n_new

    def _expand_lon(self, new_lon_min: int, new_lon_max: int) -> None:
        """Expand longitude circular buffer to include [new_lon_min, new_lon_max]."""
        # Check if already fully covered
        new_min_col = (new_lon_min - self._lon_min_bin) % self._n_full_lon
        new_max_col = (new_lon_max - self._lon_min_bin) % self._n_full_lon

        if new_min_col < self._n_lon and new_max_col < self._n_lon:
            return  # Both already in range

        # Determine cost of extending right vs left
        # Right: keep _lon_min_bin, extend to cover max_col_needed
        max_col_needed = max(self._n_lon - 1, new_min_col, new_max_col)
        right_cost = max_col_needed + 1 - self._n_lon

        # Left: slide start to new_lon_min
        shift = (self._lon_min_bin - new_lon_min) % self._n_full_lon
        old_max_col_from_new_start = shift + self._n_lon - 1
        new_max_col_from_new_start = (new_lon_max - new_lon_min) % self._n_full_lon
        max_col_from_new_start = max(old_max_col_from_new_start, new_max_col_from_new_start)
        left_cost = shift

        if right_cost <= left_cost:
            # Extend right
            n_new = max_col_needed + 1
            self._expand_lon_impl(shift_left=0, n_new=n_new)
        else:
            # Extend left
            n_new = max_col_from_new_start + 1
            self._expand_lon_impl(shift_left=shift, n_new=n_new, new_lon_min=new_lon_min)

    def _expand_lon_impl(
        self,
        *,
        shift_left: int,
        n_new: int,
        new_lon_min: int | None = None,
    ) -> None:
        """Perform the actual longitude array reallocation.

        Parameters:
            shift_left: Number of new columns to insert at the left (beginning).
            n_new: Total new number of columns.
            new_lon_min: New longitude min bin (only set when shifting left).
        """
        new_img = np.zeros((self._n_lat, n_new), dtype=self._image_dtype)
        new_res = np.zeros((self._n_lat, n_new), dtype=self._metadata_dtype)
        new_eff = np.zeros((self._n_lat, n_new), dtype=self._metadata_dtype)
        new_phase = np.zeros((self._n_lat, n_new), dtype=self._metadata_dtype)
        new_emi = np.zeros((self._n_lat, n_new), dtype=self._metadata_dtype)
        new_inc = np.zeros((self._n_lat, n_new), dtype=self._metadata_dtype)
        new_time = np.zeros((self._n_lat, n_new), dtype=np.float64)
        new_imgnum = np.zeros((self._n_lat, n_new), dtype=np.uint16)
        new_has = np.zeros((self._n_lat, n_new), dtype=np.bool_)

        sl = slice(shift_left, shift_left + self._n_lon)
        new_img[:, sl] = self._img
        new_res[:, sl] = self._resolution
        new_eff[:, sl] = self._eff_resolution
        new_phase[:, sl] = self._phase
        new_emi[:, sl] = self._emission
        new_inc[:, sl] = self._incidence
        new_time[:, sl] = self._time
        new_imgnum[:, sl] = self._image_number
        new_has[:, sl] = self._has_data

        self._img = new_img
        self._resolution = new_res
        self._eff_resolution = new_eff
        self._phase = new_phase
        self._emission = new_emi
        self._incidence = new_inc
        self._time = new_time
        self._image_number = new_imgnum
        self._has_data = new_has
        self._n_lon = n_new
        if new_lon_min is not None:
            self._lon_min_bin = new_lon_min

    def _extract_region(
        self,
        lat_bins: NDArrayIntType,
        lon_bins: NDArrayIntType,
        n_out_lat: int,
        n_out_lon: int,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
    ) -> BodyMosaicData:
        """Extract a rectangular region from the internal arrays.

        Parameters:
            lat_bins: Full-grid latitude bin indices to extract.
            lon_bins: Full-grid longitude bin indices to extract.
            n_out_lat: Number of latitude rows in output.
            n_out_lon: Number of longitude columns in output.
            lat_range: (min, max) latitude extent in radians.
            lon_range: (min, max) longitude extent in radians.

        Returns:
            BodyMosaicData for the specified region.
        """
        out_img = np.zeros((n_out_lat, n_out_lon), dtype=self._image_dtype)
        out_res = np.zeros((n_out_lat, n_out_lon), dtype=self._metadata_dtype)
        out_eff = np.zeros((n_out_lat, n_out_lon), dtype=self._metadata_dtype)
        out_phase = np.zeros((n_out_lat, n_out_lon), dtype=self._metadata_dtype)
        out_emi = np.zeros((n_out_lat, n_out_lon), dtype=self._metadata_dtype)
        out_inc = np.zeros((n_out_lat, n_out_lon), dtype=self._metadata_dtype)
        out_time = np.zeros((n_out_lat, n_out_lon), dtype=np.float64)
        out_imgnum = np.zeros((n_out_lat, n_out_lon), dtype=np.uint16)
        out_has = np.zeros((n_out_lat, n_out_lon), dtype=np.bool_)

        if self._has_data.any():
            cols = (lon_bins - self._lon_min_bin) % self._n_full_lon
            rows = lat_bins - self._lat_min_bin
            col_valid = cols < self._n_lon
            row_valid = (rows >= 0) & (rows < self._n_lat)
            cols_c = np.clip(cols, 0, self._n_lon - 1)
            rows_c = np.clip(rows, 0, self._n_lat - 1)
            valid = (
                row_valid[:, np.newaxis]
                & col_valid[np.newaxis, :]
                & self._has_data[rows_c[:, np.newaxis], cols_c[np.newaxis, :]]
            )
            out_r, out_c = np.nonzero(valid)
            src_r = rows_c[out_r]
            src_c = cols_c[out_c]
            out_img[out_r, out_c] = self._img[src_r, src_c]
            out_res[out_r, out_c] = self._resolution[src_r, src_c]
            out_eff[out_r, out_c] = self._eff_resolution[src_r, src_c]
            out_phase[out_r, out_c] = self._phase[src_r, src_c]
            out_emi[out_r, out_c] = self._emission[src_r, src_c]
            out_inc[out_r, out_c] = self._incidence[src_r, src_c]
            out_time[out_r, out_c] = self._time[src_r, src_c]
            out_imgnum[out_r, out_c] = self._image_number[src_r, src_c]
            out_has[out_r, out_c] = True

        mask = ~out_has

        phot_name = self._photometric_model.name if self._photometric_model is not None else None

        return BodyMosaicData(
            body_name=self.body_name,
            img=ma.MaskedArray(out_img, mask=mask),
            lat_resolution=self._lat_resolution,
            lon_resolution=self._lon_resolution,
            lat_range=lat_range,
            lon_range=lon_range,
            latlon_type=self._latlon_type,
            lon_direction=self._lon_direction,
            resolution=ma.MaskedArray(out_res, mask=mask),
            eff_resolution=ma.MaskedArray(out_eff, mask=mask),
            phase=ma.MaskedArray(out_phase, mask=mask),
            emission=ma.MaskedArray(out_emi, mask=mask),
            incidence=ma.MaskedArray(out_inc, mask=mask),
            time=ma.MaskedArray(out_time, mask=mask),
            image_number=ma.MaskedArray(out_imgnum, mask=mask),
            photometric_model_name=phot_name,
            image_dtype=self._image_dtype,
            metadata_dtype=self._metadata_dtype,
            contributing_image_names=tuple(self._contributing_image_names),
            sub_solar_lon_per_image=np.array(self._sub_solar_lons, dtype=np.float64),
            sub_solar_lat_per_image=np.array(self._sub_solar_lats, dtype=np.float64),
            sub_observer_lon_per_image=np.array(self._sub_observer_lons, dtype=np.float64),
            sub_observer_lat_per_image=np.array(self._sub_observer_lats, dtype=np.float64),
        )

    def _validate_repro(self, repro: BodyReprojResult) -> None:
        """Validate that repro is compatible with this mosaic's configuration."""
        if repro.body_name != self.body_name:
            raise ValueError(
                f'body_name mismatch: mosaic={self.body_name!r}, repro={repro.body_name!r}'
            )
        if abs(repro.lat_resolution - self._lat_resolution) > 1e-12:
            raise ValueError(
                f'lat_resolution mismatch: mosaic={self._lat_resolution}, '
                f'repro={repro.lat_resolution}'
            )
        if abs(repro.lon_resolution - self._lon_resolution) > 1e-12:
            raise ValueError(
                f'lon_resolution mismatch: mosaic={self._lon_resolution}, '
                f'repro={repro.lon_resolution}'
            )
        if repro.latlon_type != self._latlon_type:
            raise ValueError(
                f'latlon_type mismatch: mosaic={self._latlon_type!r}, repro={repro.latlon_type!r}'
            )
        if repro.lon_direction != self._lon_direction:
            raise ValueError(
                f'lon_direction mismatch: mosaic={self._lon_direction!r}, '
                f'repro={repro.lon_direction!r}'
            )
        mosaic_phot_name = (
            self._photometric_model.name if self._photometric_model is not None else None
        )
        if repro.photometric_model_name != mosaic_phot_name:
            raise ValueError(
                f'photometric_model_name mismatch: mosaic={mosaic_phot_name!r}, '
                f'repro={repro.photometric_model_name!r}'
            )
