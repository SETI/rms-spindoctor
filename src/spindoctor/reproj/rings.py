"""Ring reprojection and mosaicing utilities.

This module provides the RingMosaic class for reprojecting planetary ring
images onto radius/longitude grids and accumulating them into sparse mosaics.

Thread safety: RingMosaic.reproject() is not thread-safe because it
temporarily reduces the oops global light-time precision (a process-global
mutation) and builds a Backplane on the shared obs. Concurrent calls from
separate threads will interfere.
"""

import enum
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.ma as ma
import oops
import scipy.ndimage as nd
from pdslogger import PdsLogger
from polymath import Scalar, Vector3

from spindoctor.config import IMAGE_LOGGER
from spindoctor.reproj._context_managers import _reduced_oops_precision
from spindoctor.reproj._serialization import (
    infer_format,
    load_fits,
    load_npz,
    orbit_model_from_dict,
    orbit_model_to_dict,
    save_fits,
    save_npz,
    tuple_of_strings_field,
    verify_dtype,
)
from spindoctor.reproj.photometric_model import PhotometricModel
from spindoctor.reproj.ring_orbit_model import RingOrbitModel
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.image import array_unzoom, array_zoom
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType, NDArrayIntType, PathLike


def _ring_plane_surface(ring_body_name: str) -> Any:
    """Return the oops ring surface used for longitude/radius <-> pixel conversion.

    Nav and oops backplanes use names like ``'saturn:ring'``. The corresponding
    ``oops.Body`` registry entry for the unbounded plane is ``SATURN_RING_PLANE``,
    not ``SATURN_RING``; the parent planet's ``ring_body`` attribute points at the
    correct body. For a two-part ``name:ring`` string we use that; otherwise we
    treat ``ring_body_name`` as a direct ``Body.lookup`` key (underscores for
    colons, uppercased), e.g. ``SATURN_MAIN_RINGS``.
    """
    parts = ring_body_name.lower().split(':')
    if len(parts) == 2 and parts[1] == 'ring':
        parent = oops.Body.lookup(parts[0].upper())
        ring_body = parent.ring_body
        if ring_body is None:
            raise ValueError(
                f'No ring plane is defined for body {parts[0]!r} '
                f'(ring_body_name was {ring_body_name!r})'
            )
        return ring_body.surface
    return oops.Body.lookup(ring_body_name.replace(':', '_').upper()).surface


class RingMosaicMergeStrategy(enum.Enum):
    """Strategy for resolving per-longitude-column conflicts when adding data to a RingMosaic.

    Members:
        BEST_RESOLUTION: Replace an existing longitude column only when the new
            data has strictly better mean radial resolution for that column.
        MOST_COVERAGE_THEN_RESOLUTION: Fill empty (missing) longitude columns
            first; for already-present columns, replace only when the new data
            has better mean radial resolution.
    """

    BEST_RESOLUTION = 'best_resolution'
    MOST_COVERAGE_THEN_RESOLUTION = 'most_coverage_then_resolution'


# Main ring radius bounds used when no explicit range is given
RINGS_MIN_RADIUS = oops.body.SATURN_MAIN_RINGS[0]
RINGS_MAX_RADIUS = oops.body.SATURN_MAIN_RINGS[1]

# Slop values: must be smaller than the smallest resolution we will ever use
_LONGITUDE_SLOP = 1e-6  # rad
_RADIUS_SLOP = 1e-6  # km

_MAX_LONGITUDE = math.pi * 2.0 - _LONGITUDE_SLOP * 2

# ``bool`` subclasses ``int``; reject for range endpoints.
_REAL_SCALAR_TYPES = (int, float, np.integer, np.floating)


def _validate_reproject_longitude_range(longitude_range: object) -> tuple[float, float]:
    """Validate ``RingMosaic.reproject(..., longitude_range=...)``; return finite floats."""
    if not isinstance(longitude_range, (tuple, list)):
        raise TypeError(
            'reproject() longitude_range must be a tuple or list of two numbers (rad), '
            f'got {type(longitude_range).__name__}'
        )
    if len(longitude_range) != 2:
        raise ValueError(
            'reproject() longitude_range must have exactly two elements (start, end) in radians, '
            f'got length {len(longitude_range)}'
        )
    lon_start, lon_end = longitude_range[0], longitude_range[1]
    for label, x in (('start', lon_start), ('end', lon_end)):
        if isinstance(x, bool) or not isinstance(x, _REAL_SCALAR_TYPES):
            raise TypeError(
                f'reproject() longitude_range {label} must be int or float, got {type(x).__name__}'
            )
    lon_start_f = float(lon_start)
    lon_end_f = float(lon_end)
    if not math.isfinite(lon_start_f) or not math.isfinite(lon_end_f):
        raise ValueError(
            'reproject() longitude_range values must be finite; '
            f'got start={longitude_range[0]!r}, end={longitude_range[1]!r}'
        )
    if lon_start_f < 0.0 or lon_start_f > _MAX_LONGITUDE:
        raise ValueError(
            'reproject() longitude_range start must satisfy 0 <= start <= _MAX_LONGITUDE (rad); '
            f'got start={lon_start_f!r}'
        )
    if lon_end_f < 0.0 or lon_end_f > _MAX_LONGITUDE:
        raise ValueError(
            'reproject() longitude_range end must satisfy 0 <= end <= _MAX_LONGITUDE (rad); '
            f'got end={lon_end_f!r}'
        )
    if lon_start_f > lon_end_f:
        raise ValueError(
            'reproject() longitude_range must have start <= end (rad); '
            f'got start={lon_start_f!r}, end={lon_end_f!r}'
        )
    return lon_start_f, lon_end_f


def _validate_reproject_radius_range(radius_range: object) -> tuple[float, float]:
    """Validate ``RingMosaic.reproject(..., radius_range=...)``; return finite floats.

    ``inner < outer`` is required in both modes: absolute radii (km) when
    ``orbit_model is None``, or signed offsets (km) from the orbital radius when
    a model is used.
    """
    if not isinstance(radius_range, (tuple, list)):
        raise TypeError(
            'reproject() radius_range must be a tuple or list of two numbers (km), '
            f'got {type(radius_range).__name__}'
        )
    if len(radius_range) != 2:
        raise ValueError(
            'reproject() radius_range must have exactly two elements (inner, outer) in km, '
            f'got length {len(radius_range)}'
        )
    ri, ro = radius_range[0], radius_range[1]
    for label, x in (('inner', ri), ('outer', ro)):
        if isinstance(x, bool) or not isinstance(x, _REAL_SCALAR_TYPES):
            raise TypeError(
                f'reproject() radius_range {label} must be int or float, got {type(x).__name__}'
            )
    ri_f = float(ri)
    ro_f = float(ro)
    if not math.isfinite(ri_f) or not math.isfinite(ro_f):
        raise ValueError(
            'reproject() radius_range values must be finite; '
            f'got inner={radius_range[0]!r}, outer={radius_range[1]!r}'
        )
    if ri_f >= ro_f:
        raise ValueError(
            'reproject() radius_range must have inner < outer (km). With orbit_model=None '
            'these are absolute radii; with a model they are signed offsets from the orbital '
            f'radius at each (longitude, time). Got inner={ri_f!r}, outer={ro_f!r}.'
        )
    return ri_f, ro_f


def _validate_reproject_zoom_amt(zoom_amt: object) -> tuple[int, int]:
    """Validate ``RingMosaic.reproject(..., zoom_amt=...)``; return ``(radial, longitude)`` ints."""
    if isinstance(zoom_amt, (list, tuple)):
        if len(zoom_amt) != 2:
            raise ValueError(
                'reproject() zoom_amt as a sequence must have exactly two elements '
                f'(radial, longitude zoom factors), got length {len(zoom_amt)}'
            )
        out: list[int] = []
        for i, z in enumerate(zoom_amt, start=1):
            if isinstance(z, (bool, np.bool_)):
                raise TypeError(
                    f'reproject() zoom_amt element {i} must be int-like (not bool), got bool'
                )
            if not isinstance(z, (int, np.integer)):
                raise TypeError(
                    f'reproject() zoom_amt element {i} must be int or numpy integer, '
                    f'got {type(z).__name__}'
                )
            out.append(int(z))
        return (out[0], out[1])

    if isinstance(zoom_amt, (bool, np.bool_)):
        raise TypeError(
            'reproject() zoom_amt must be int (not bool) or a tuple/list of two ints, got bool'
        )
    if isinstance(zoom_amt, (int, np.integer)):
        z = int(zoom_amt)
        return (z, z)

    raise TypeError(
        'reproject() zoom_amt must be int (not bool) or a tuple/list of two ints, '
        f'got {type(zoom_amt).__name__}'
    )


def _zoomed_bin_coords(n_bins: int, zoom: int) -> NDArrayFloatType:
    """Return the sub-sample coordinates of a zoomed output axis.

    Output cell ``b`` of an axis of ``n_bins`` cells is split into ``zoom``
    sub-samples, which ``array_unzoom(..., 'mean')`` later block-averages back
    into cell ``b``. For the block mean to land on cell ``b``'s own coordinate
    the sub-samples must straddle it, so sub-sample ``i`` of cell ``b`` sits at
    ``b + (i + 0.5) / zoom - 0.5``: the block runs from half a cell below ``b``
    to half a cell above it and averages to exactly ``b``. Starting the block at
    ``b`` instead would put its mean ``(zoom - 1) / (2 * zoom)`` cells too high,
    approaching half a cell as ``zoom`` grows. At ``zoom == 1`` this is the
    identity, ``np.arange(n_bins)``.

    The block mean is taken over a masked array, so sub-samples that fell
    outside the image or on a masked pixel are skipped. A cell whose surviving
    sub-samples are not symmetric about its coordinate is therefore still
    pulled towards the side that survived; only a fully sampled cell is exactly
    unbiased.

    Parameters:
        n_bins: Number of unzoomed output cells along the axis.
        zoom: Number of sub-samples per output cell. Must be >= 1.

    Returns:
        1-D float array of ``n_bins * zoom`` coordinates, in units of unzoomed
        output cells.
    """
    return (np.arange(n_bins * zoom) + 0.5) / float(zoom) - 0.5


# Module-level defaults for RingMosaic parameters
DEFAULT_LONGITUDE_RESOLUTION = 0.02 * math.pi / 180.0  # 0.02 degrees in rad
DEFAULT_RADIUS_RESOLUTION = 5.0  # km
DEFAULT_ZOOM = (1, 1)
DEFAULT_MARGIN = 3


@dataclass(frozen=True)
class RingReprojResult:
    """Data returned by RingMosaic.reproject().

    The image and per-longitude metadata arrays are always sparse: only
    longitude columns with valid data are included. Use
    ``longitude_antimask`` to reconstruct the actual longitude values.

    The interpretation of ``radius_inner`` / ``radius_outer`` and longitudes
    depends on whether ``orbit_model`` is ``None``:

    - ``orbit_model is None``: longitudes are inertial J2000 ring longitudes;
      ``radius_inner`` and ``radius_outer`` are absolute ring radii in km.
    - ``orbit_model is not None``: longitudes are co-rotating; ``radius_inner``
      and ``radius_outer`` are signed offsets in km from the orbit's actual
      radial position at each (longitude, time). For an eccentric orbit, that
      radial position varies between ``a (1 - e)`` and ``a (1 + e)``; the
      offset semantics make an eccentric ring appear as a straight line in
      the reprojection.

    This is a :func:`dataclasses.dataclass`; each field is documented on its own
    member entry below.
    """

    body_name: str
    img: ma.MaskedArray
    longitude_resolution: float
    radius_resolution: float
    radius_inner: float
    radius_outer: float
    longitude_antimask: NDArrayBoolType
    mean_radial_resolution: NDArrayFloatType
    mean_angular_resolution: NDArrayFloatType
    mean_phase: NDArrayFloatType
    mean_emission: NDArrayFloatType
    incidence: float
    time: float
    orbit_model: RingOrbitModel | None
    image_dtype: np.dtype
    metadata_dtype: np.dtype
    photometric_model_name: str | None = None
    image_name: str = ''

    def save(
        self,
        path: PathLike,
        *,
        format_: str | None = None,
        compress: bool = True,
    ) -> None:
        """Save this RingReprojResult to a file.

        Parameters:
            path: Output path (``str``, ``pathlib.Path``, or ``filecache.FCPath``).
                The format is inferred from the extension (.npz for NumPy archive,
                .fits/.fit for FITS) unless ``format_`` is given.
            format_: Explicit format: ``'npz'`` or ``'fits'``.
            compress: If True (default), use compressed npz. Ignored for
                FITS.

        Raises:
            ValueError: If the format cannot be inferred or is not
                supported.

        Example::

            result = ring_mosaic.reproject(obs)
            result.save('ring_reproj.npz')
            reloaded = RingReprojResult.load('ring_reproj.npz')
        """
        fmt = infer_format(path, format_)
        payload: dict[str, Any] = {
            'body_name': self.body_name,
            'img': self.img,
            'longitude_resolution': self.longitude_resolution,
            'radius_resolution': self.radius_resolution,
            'radius_inner': self.radius_inner,
            'radius_outer': self.radius_outer,
            'longitude_antimask': self.longitude_antimask,
            'mean_radial_resolution': self.mean_radial_resolution,
            'mean_angular_resolution': self.mean_angular_resolution,
            'mean_phase': self.mean_phase,
            'mean_emission': self.mean_emission,
            'incidence': self.incidence,
            'time': self.time,
            'orbit_model': orbit_model_to_dict(self.orbit_model),
            'image_dtype': self.image_dtype,
            'metadata_dtype': self.metadata_dtype,
            'photometric_model_name': self.photometric_model_name,
            'image_name': self.image_name,
        }
        if fmt == 'npz':
            save_npz(path, 'RingReprojResult', 1, payload, compress=compress)
        else:
            save_fits(path, 'RingReprojResult', 1, payload)

    @classmethod
    def load(
        cls,
        path: PathLike,
        *,
        format_: str | None = None,
    ) -> 'RingReprojResult':
        """Load a RingReprojResult from a file.

        Parameters:
            path: Input path (``str``, ``pathlib.Path``, or ``filecache.FCPath``).
            format_: Explicit format (``'npz'`` or ``'fits'``). If None,
                inferred from the file extension.

        Returns:
            A new RingReprojResult with the loaded data.

        Raises:
            ValueError: If the file's kind does not match, or if any array
                dtype does not match the declared ``image_dtype`` /
                ``metadata_dtype``.

        Example::

            result = RingReprojResult.load('ring_reproj.npz')
        """
        fmt = infer_format(path, format_)
        d = (
            load_npz(path, 'RingReprojResult')
            if fmt == 'npz'
            else load_fits(path, 'RingReprojResult')
        )

        _REQUIRED_KEYS_REPROJ = {
            'image_dtype',
            'metadata_dtype',
            'body_name',
            'img',
            'longitude_resolution',
            'radius_resolution',
            'radius_inner',
            'radius_outer',
            'longitude_antimask',
            'mean_radial_resolution',
            'mean_angular_resolution',
            'mean_phase',
            'mean_emission',
            'incidence',
            'time',
        }
        missing = _REQUIRED_KEYS_REPROJ - d.keys()
        if missing:
            raise ValueError(
                f'RingReprojResult file is missing required keys: {sorted(missing)}. '
                f'Found keys: {sorted(d.keys())!r}. '
                'If this path ends in .fits but the file is actually an npz archive, '
                'rename it to .npz or pass format_= explicitly; otherwise the file may be '
                'truncated or not written by RingReprojResult.save().'
            )

        image_dtype = np.dtype(str(d['image_dtype']))
        metadata_dtype = np.dtype(str(d['metadata_dtype']))

        verify_dtype(
            {
                k: d[k]
                for k in (
                    'img',
                    'mean_radial_resolution',
                    'mean_angular_resolution',
                    'mean_phase',
                    'mean_emission',
                )
            },
            image_dtype=image_dtype,
            metadata_dtype=metadata_dtype,
            image_fields=['img'],
            metadata_fields=[
                'mean_radial_resolution',
                'mean_angular_resolution',
                'mean_phase',
                'mean_emission',
            ],
        )

        # Reconstruct orbit model from flattened dict keys
        orbit_d: dict[str, Any] = {}
        prefix = 'orbit_model__'
        for k, v in d.items():
            if k.startswith(prefix):
                orbit_d[k[len(prefix) :]] = v
        orbit_model = orbit_model_from_dict(orbit_d) if orbit_d else None

        phot = d.get('photometric_model_name')
        if phot is None or (isinstance(phot, str) and phot.strip() == ''):
            photometric_model_name: str | None = None
        else:
            photometric_model_name = str(phot)

        return cls(
            body_name=str(d['body_name']),
            img=d['img'],
            longitude_resolution=float(d['longitude_resolution']),
            radius_resolution=float(d['radius_resolution']),
            radius_inner=float(d['radius_inner']),
            radius_outer=float(d['radius_outer']),
            longitude_antimask=np.asarray(d['longitude_antimask'], dtype=np.bool_),
            mean_radial_resolution=d['mean_radial_resolution'],
            mean_angular_resolution=d['mean_angular_resolution'],
            mean_phase=d['mean_phase'],
            mean_emission=d['mean_emission'],
            incidence=float(d['incidence']),
            time=float(d['time']),
            orbit_model=orbit_model,
            image_dtype=image_dtype,
            metadata_dtype=metadata_dtype,
            photometric_model_name=photometric_model_name,
            image_name=str(d.get('image_name', '') or ''),
        )


@dataclass(frozen=True)
class RingMosaicData:
    """Mosaic data returned by RingMosaic retrieval methods.

    The meaning of ``radius_inner`` / ``radius_outer`` and longitudes depends
    on whether ``orbit_model_name`` is ``None``:

    - ``orbit_model_name is None``: longitudes are inertial J2000 ring
      longitudes; ``radius_inner`` and ``radius_outer`` are absolute ring
      radii in km.
    - ``orbit_model_name is not None``: longitudes are co-rotating in the
      named orbit model's frame; ``radius_inner`` and ``radius_outer`` are
      signed offsets in km from the orbital radius at each (longitude, time)
      — i.e. from ``orbit_model.radius_at_longitude(inertial_lon, et)``.

    This is a :func:`dataclasses.dataclass`; each field is documented on its own
    member entry below.
    """

    body_name: str
    ring_body_name: str
    shadow_body_name: str
    longitude_resolution: float
    radius_resolution: float
    radius_inner: float
    radius_outer: float
    longitude_antimask: NDArrayBoolType
    img: ma.MaskedArray
    longitude_range: tuple[float, float] | None
    mean_radial_resolution: ma.MaskedArray
    mean_angular_resolution: ma.MaskedArray
    mean_phase: ma.MaskedArray
    mean_emission: ma.MaskedArray
    mean_incidence: float
    image_number: ma.MaskedArray
    time: ma.MaskedArray
    image_dtype: np.dtype
    metadata_dtype: np.dtype
    contributing_image_names: tuple[str, ...] = ()
    orbit_model_name: str | None = None
    photometric_model_name: str | None = None

    def save(
        self,
        path: PathLike,
        *,
        format_: str | None = None,
        compress: bool = True,
    ) -> None:
        """Save this RingMosaicData to a file.

        Parameters:
            path: Output path (``str``, ``pathlib.Path``, or ``filecache.FCPath``).
                The format is inferred from the extension (.npz for NumPy archive,
                .fits/.fit for FITS) unless ``format_`` is given.
            format_: Explicit format: ``'npz'`` or ``'fits'``.
            compress: If True (default), use compressed npz. Ignored for
                FITS.

        Raises:
            ValueError: If the format cannot be inferred or is not
                supported.

        Example::

            data = ring_mosaic.to_bounded(longitude_range=(0.0, math.pi))
            data.save('saturn_rings.npz')
            data.save('saturn_rings.fits', format_='fits')
            reloaded = RingMosaicData.load('saturn_rings.npz')
        """
        fmt = infer_format(path, format_)
        payload: dict[str, Any] = {
            'body_name': self.body_name,
            'ring_body_name': self.ring_body_name,
            'shadow_body_name': self.shadow_body_name,
            'longitude_resolution': self.longitude_resolution,
            'radius_resolution': self.radius_resolution,
            'radius_inner': self.radius_inner,
            'radius_outer': self.radius_outer,
            'longitude_antimask': self.longitude_antimask,
            'img': self.img,
            'longitude_range': self.longitude_range,
            'mean_radial_resolution': self.mean_radial_resolution,
            'mean_angular_resolution': self.mean_angular_resolution,
            'mean_phase': self.mean_phase,
            'mean_emission': self.mean_emission,
            'mean_incidence': self.mean_incidence,
            'image_number': self.image_number,
            'time': self.time,
            'image_dtype': self.image_dtype,
            'metadata_dtype': self.metadata_dtype,
            'contributing_image_names': self.contributing_image_names,
            'orbit_model_name': self.orbit_model_name,
            'photometric_model_name': self.photometric_model_name,
        }
        if fmt == 'npz':
            save_npz(path, 'RingMosaicData', 1, payload, compress=compress)
        else:
            save_fits(path, 'RingMosaicData', 1, payload)

    @classmethod
    def load(
        cls,
        path: PathLike,
        *,
        format_: str | None = None,
    ) -> 'RingMosaicData':
        """Load a RingMosaicData from a file.

        Parameters:
            path: Input path (``str``, ``pathlib.Path``, or ``filecache.FCPath``).
            format_: Explicit format (``'npz'`` or ``'fits'``). If None,
                inferred from the file extension.

        Returns:
            A new RingMosaicData with the loaded data.

        Raises:
            ValueError: If the file's kind does not match, or if any array
                dtype does not match the declared ``image_dtype`` /
                ``metadata_dtype``, or if ``image_number`` is not uint16,
                or if ``time`` is not float64.

        Example::

            data = RingMosaicData.load('saturn_rings.npz')
        """
        fmt = infer_format(path, format_)
        d = load_npz(path, 'RingMosaicData') if fmt == 'npz' else load_fits(path, 'RingMosaicData')

        _REQUIRED_KEYS_MOSAIC = {
            'image_dtype',
            'metadata_dtype',
            'body_name',
            'ring_body_name',
            'shadow_body_name',
            'img',
            'longitude_resolution',
            'radius_resolution',
            'radius_inner',
            'radius_outer',
            'longitude_antimask',
            'mean_radial_resolution',
            'mean_angular_resolution',
            'mean_phase',
            'mean_emission',
            'mean_incidence',
            'time',
            'image_number',
        }
        missing = _REQUIRED_KEYS_MOSAIC - d.keys()
        if missing:
            raise ValueError(
                f'RingMosaicData file is missing required keys: {sorted(missing)}. '
                f'Found keys: {sorted(d.keys())!r}. '
                'If the path ends in .fits but the payload is npz (ZIP), rename to .npz '
                'or pass format_=; otherwise the file may be corrupt or from another tool.'
            )

        image_dtype = np.dtype(str(d['image_dtype']))
        metadata_dtype = np.dtype(str(d['metadata_dtype']))

        verify_dtype(
            {
                k: d[k]
                for k in (
                    'img',
                    'mean_radial_resolution',
                    'mean_angular_resolution',
                    'mean_phase',
                    'mean_emission',
                    'time',
                    'image_number',
                )
            },
            image_dtype=image_dtype,
            metadata_dtype=metadata_dtype,
            image_fields=['img'],
            metadata_fields=[
                'mean_radial_resolution',
                'mean_angular_resolution',
                'mean_phase',
                'mean_emission',
            ],
            float64_fields=['time'],
        )

        lon_range_raw = d.get('longitude_range')
        longitude_range: tuple[float, float] | None
        if lon_range_raw is None:
            longitude_range = None
        else:
            longitude_range = (float(lon_range_raw[0]), float(lon_range_raw[1]))

        contributing_image_names = tuple_of_strings_field(d.get('contributing_image_names'))
        om_raw = d.get('orbit_model_name')
        orbit_model_name = None if om_raw is None else str(om_raw)
        pm_raw = d.get('photometric_model_name')
        if pm_raw is None or (isinstance(pm_raw, str) and pm_raw.strip() == ''):
            photometric_model_name: str | None = None
        else:
            photometric_model_name = str(pm_raw)

        return cls(
            body_name=str(d['body_name']),
            ring_body_name=str(d['ring_body_name']),
            shadow_body_name=str(d['shadow_body_name']),
            longitude_resolution=float(d['longitude_resolution']),
            radius_resolution=float(d['radius_resolution']),
            radius_inner=float(d['radius_inner']),
            radius_outer=float(d['radius_outer']),
            longitude_antimask=np.asarray(d['longitude_antimask'], dtype=np.bool_),
            img=d['img'],
            longitude_range=longitude_range,
            mean_radial_resolution=d['mean_radial_resolution'],
            mean_angular_resolution=d['mean_angular_resolution'],
            mean_phase=d['mean_phase'],
            mean_emission=d['mean_emission'],
            mean_incidence=float(d['mean_incidence']),
            image_number=d['image_number'],
            time=d['time'],
            image_dtype=image_dtype,
            metadata_dtype=metadata_dtype,
            contributing_image_names=contributing_image_names,
            orbit_model_name=orbit_model_name,
            photometric_model_name=photometric_model_name,
        )


class RingMosaic:
    """Sparse ring mosaic accumulator.

    Reprojects ring images onto a radius/longitude grid and accumulates them
    into a mosaic using true sparse longitude storage. Only longitude columns
    that actually contain reprojected data are stored; new columns are
    inserted in sorted order using batched ``np.insert`` calls.

    The interpretation of ``radius_inner`` / ``radius_outer`` and longitudes
    depends on ``orbit_model``:

    - ``orbit_model is None`` (default): longitudes are inertial J2000 ring
      longitudes; ``radius_inner`` and ``radius_outer`` are absolute ring
      radii in km.
    - ``orbit_model is not None``: longitudes are co-rotating in the model's
      frame; ``radius_inner`` and ``radius_outer`` are signed offsets in km
      from the orbital radius at each (longitude, time). For an eccentric
      orbit, that radial position varies between ``a (1 - e)`` and
      ``a (1 + e)``. The offset semantics make an eccentric ring appear as a
      straight line in the reprojection. A typical 2000 km wide window
      centred on the orbit uses ``radius_inner = -1000`` and
      ``radius_outer = +1000``.

    Parameters:
        body_name: The planet name (e.g. 'SATURN'). Used to derive the
            oops ring body name and shadow body name for backplane calls.
        radius_inner: Inner radius (km, absolute) when ``orbit_model is
            None``; signed offset (km) from the orbital radius at each
            (longitude, time) otherwise.
        radius_outer: Outer radius (km, absolute) when ``orbit_model is
            None``; signed offset (km) from the orbital radius at each
            (longitude, time) otherwise. Must be greater than ``radius_inner``.
        longitude_resolution: Longitude bin width (rad/pixel).
        radius_resolution: Radius bin height (km/pixel).
        merge_strategy: How to resolve conflicts when the same longitude
            column appears in multiple reprojections.
        orbit_model: Default RingOrbitModel for co-rotating frame
            reprojections; selects the offset-radius semantics described above.
            Can be overridden per-call in reproject(); the override must agree
            with this default.
        image_dtype: NumPy dtype for the reprojected brightness ``img``
            array. Defaults to ``np.float64``.
        metadata_dtype: NumPy dtype for geometry arrays
            (``mean_radial_resolution``, ``mean_angular_resolution``,
            ``mean_phase``, ``mean_emission``). Defaults to ``np.float32``.
        photometric_model: Optional photometric correction applied to ``img`` in
            ``reproject()`` before accumulation. ``None`` means raw brightness.

    Notes:
        reproject() is not thread-safe because it temporarily reduces the
        oops global light-time precision (a process-global mutation) and
        builds a Backplane on the shared obs.

        ``time`` is always stored as ``float64`` regardless of
        ``metadata_dtype``. ``image_number`` is always stored as
        ``uint16``; ``_image_count`` is stored as the image_number before
        incrementing, so valid image_number values span 0..65535 (a total
        of 65,536 images). ``add()`` raises ``OverflowError`` after the
        65,536th image is added.
    """

    def __init__(
        self,
        body_name: str,
        radius_inner: float,
        radius_outer: float,
        *,
        longitude_resolution: float = DEFAULT_LONGITUDE_RESOLUTION,
        radius_resolution: float = DEFAULT_RADIUS_RESOLUTION,
        merge_strategy: RingMosaicMergeStrategy = (
            RingMosaicMergeStrategy.MOST_COVERAGE_THEN_RESOLUTION
        ),
        orbit_model: RingOrbitModel | None = None,
        image_dtype: np.typing.DTypeLike = np.float64,
        metadata_dtype: np.typing.DTypeLike = np.float32,
        photometric_model: PhotometricModel | None = None,
    ) -> None:
        """Initialize an empty RingMosaic."""
        if radius_inner >= radius_outer:
            raise ValueError(
                f'radius_inner ({radius_inner}) must be less than radius_outer ({radius_outer})'
            )
        if longitude_resolution <= 0 or longitude_resolution >= 2 * math.pi:
            raise ValueError(
                f'longitude_resolution must be positive and < 2*pi, got {longitude_resolution!r}'
            )
        if radius_resolution <= 0:
            raise ValueError(f'radius_resolution must be positive, got {radius_resolution!r}')
        self._body_name = body_name
        self._ring_body_name = body_name.lower() + ':ring'
        self._shadow_body_name = body_name.lower()
        self._radius_inner = radius_inner
        self._radius_outer = radius_outer
        self._lon_resolution = longitude_resolution
        self._rad_resolution = radius_resolution
        self._merge_strategy = merge_strategy
        self._orbit_model = orbit_model
        self._image_dtype: np.dtype = np.dtype(image_dtype)
        self._metadata_dtype: np.dtype = np.dtype(metadata_dtype)
        self._photometric_model = photometric_model

        self._n_radius = math.ceil((radius_outer - radius_inner + _RADIUS_SLOP) / radius_resolution)
        # Span [0, _MAX_LONGITUDE] (slightly below 2π); matches bin indexing that clamps to
        # ``_MAX_LONGITUDE`` (e.g. ``longitude_valid_range(..., longitude_end=_MAX_LONGITUDE)``).
        self._n_full_lon = math.floor(_MAX_LONGITUDE / longitude_resolution) + 1

        # Sparse storage: only valid longitude columns are held.
        # _antimask[i] is True iff longitude bin i has data.
        self._antimask: NDArrayBoolType = np.zeros(self._n_full_lon, dtype=np.bool_)
        self._img_sparse: ma.MaskedArray = ma.MaskedArray(
            np.empty((self._n_radius, 0), dtype=self._image_dtype),
            mask=np.ones((self._n_radius, 0), dtype=np.bool_),
        )
        self._mean_radial_res: NDArrayFloatType = np.empty(0, dtype=self._metadata_dtype)
        self._mean_angular_res: NDArrayFloatType = np.empty(0, dtype=self._metadata_dtype)
        self._mean_phase: NDArrayFloatType = np.empty(0, dtype=self._metadata_dtype)
        self._mean_emission: NDArrayFloatType = np.empty(0, dtype=self._metadata_dtype)
        self._mean_incidence_sum: float = 0.0
        self._mean_incidence_count: int = 0
        self._mean_incidence: float = 0.0
        self._image_number: NDArrayIntType = np.empty(0, dtype=np.uint16)
        self._time: NDArrayFloatType = np.empty(0, dtype=np.float64)
        self._image_count: int = 0
        self._contributing_image_names: list[str] = []

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def body_name(self) -> str:
        """The planet name supplied at construction."""
        return self._body_name

    @property
    def ring_body_name(self) -> str:
        """oops body name for the ring plane (e.g. 'saturn:ring')."""
        return self._ring_body_name

    @property
    def shadow_body_name(self) -> str:
        """oops body name for the shadow-casting planet (e.g. 'saturn')."""
        return self._shadow_body_name

    @property
    def bounds(self) -> tuple[float, float] | None:
        """Longitude extent of current mosaic data as (min, max) in radians.

        Returns None if no data has been added yet.
        """
        true_indices = np.where(self._antimask)[0]
        if len(true_indices) == 0:
            return None
        return (
            float(true_indices[0]) * self._lon_resolution,
            float(true_indices[-1]) * self._lon_resolution,
        )

    # ------------------------------------------------------------------
    # Static grid generation utilities
    # ------------------------------------------------------------------

    @staticmethod
    def generate_longitudes(
        longitude_start: float = 0.0,
        longitude_end: float = _MAX_LONGITUDE,
        *,
        longitude_resolution: float = DEFAULT_LONGITUDE_RESOLUTION,
    ) -> NDArrayFloatType:
        """Generate a longitude array aligned to grid boundaries.

        Parameters:
            longitude_start: Start of the range (rad). Defaults to 0.
            longitude_end: End of the range (rad). Defaults to just under
                2*pi.
            longitude_resolution: Step size (rad/pixel).

        Returns:
            1-D array of longitudes (rad) on resolution boundaries, with no
            value less than longitude_start or greater than longitude_end.
        """
        if longitude_resolution <= 0:
            raise ValueError(f'longitude_resolution must be positive, got {longitude_resolution!r}')
        start_idx = math.ceil(longitude_start / longitude_resolution)
        end_idx = math.floor(longitude_end / longitude_resolution)
        return np.arange(start_idx, end_idx + 1) * longitude_resolution

    @staticmethod
    def generate_radii(
        radius_inner: float,
        radius_outer: float,
        *,
        radius_resolution: float = DEFAULT_RADIUS_RESOLUTION,
    ) -> NDArrayFloatType:
        """Generate a radius array from inner to outer.

        Parameters:
            radius_inner: Inner radius (km).
            radius_outer: Outer radius (km).
            radius_resolution: Step size (km/pixel).

        Returns:
            1-D array of radii (km) starting at radius_inner with step
            radius_resolution, ending at or just before radius_outer.
        """
        if radius_resolution <= 0:
            raise ValueError(f'radius_resolution must be positive, got {radius_resolution!r}')
        if radius_outer < radius_inner:
            raise ValueError(
                f'radius_outer ({radius_outer}) must be >= radius_inner ({radius_inner})'
            )
        n = math.ceil((radius_outer - radius_inner + _RADIUS_SLOP) / radius_resolution)
        return np.arange(n) * radius_resolution + radius_inner

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    @staticmethod
    def longitude_radius_to_pixels(
        obs: Any,
        longitude: NDArrayFloatType,
        radius: NDArrayFloatType,
        *,
        orbit_model: RingOrbitModel | None = None,
        ring_body_name: str = 'saturn:ring',
    ) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """Convert longitude/radius pairs to image pixel coordinates.

        If orbit_model is given, the longitude values are treated as
        co-rotating and are converted to inertial before the coordinate
        lookup.

        Parameters:
            obs: The Observation whose FOV is used.
            longitude: Longitude array (rad).
            radius: Radius array (km).
            orbit_model: RingOrbitModel for co-rotating frame conversion,
                or None for inertial longitude.
            ring_body_name: oops ring body name for the surface lookup.

        Returns:
            Tuple of (u, v) floating-point pixel-corner coordinate arrays, the
            convention the oops FOV works in: a whole number falls on the
            boundary between two pixels and the centre of column c is at
            c + 0.5.
        """
        if orbit_model is not None:
            longitude = orbit_model.corotating_to_inertial(longitude, obs.midtime)

        if len(longitude) == 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        longitude_scalar = Scalar(longitude)
        radius_scalar = Scalar(radius)

        ring_surface = _ring_plane_surface(ring_body_name)
        obs_event = oops.Event(obs.midtime, (Vector3.ZERO, Vector3.ZERO), obs.path, obs.frame)
        _, obs_event = ring_surface.photon_to_event_by_coords(
            obs_event, (radius_scalar, longitude_scalar)
        )

        uv = obs.fov.uv_from_los(obs_event.neg_arr_ap)
        u, v = uv.to_scalars()
        return u.vals, v.vals

    @staticmethod
    def orbit_pixels(
        obs: Any,
        orbit_model: RingOrbitModel,
        *,
        ring_body_name: str = 'saturn:ring',
        longitude_step: float = 0.002 * math.pi / 180.0,
    ) -> tuple[NDArrayFloatType, NDArrayFloatType]:
        """Return (u, v) pixel pairs for an orbit model feature in an image.

        Computes the full 0..2pi longitude range, converts each to image
        coordinates, and filters to those inside the FOV.

        Parameters:
            obs: The Observation (with its FOV already configured as needed).
            orbit_model: The ring orbit model to trace.
            ring_body_name: oops ring body name.
            longitude_step: Longitude step for sampling (rad).

        Returns:
            Tuple of (u_pixels, v_pixels) floating-point pixel-corner coordinate
            arrays, as returned by longitude_radius_to_pixels.
        """
        longitudes, radii = orbit_model.longitude_radius(obs.midtime, step=longitude_step)

        bp = obs.ext_bp
        # longitude_radius_to_pixels ends in obs.fov.uv_from_los, so these are
        # coordinates of the nominal frame: a position in the extended-FOV
        # margin is negative, and the padded frame runs from -margin to
        # size + margin rather than from zero to the padded width.  The bounds
        # therefore come from extfov_*_min / _max, which name the first and last
        # pixel of that frame, and containment is half-open at the top because
        # these are pixel corners: the centre of the last column is inside it
        # and the boundary past that column is not.
        u_low = float(obs.extfov_u_min)
        v_low = float(obs.extfov_v_min)
        u_limit = float(obs.extfov_u_max) + 1.0
        v_limit = float(obs.extfov_v_max) + 1.0

        bp_radius = bp.ring_radius(ring_body_name)
        bp_longitude = bp.ring_longitude(ring_body_name)
        # builtins=True: a polymath Scalar comparison result cannot index a
        # numpy array, so force plain floats here.
        min_bp_radius = bp_radius.min(builtins=True)
        max_bp_radius = bp_radius.max(builtins=True)
        min_bp_longitude = bp_longitude.min(builtins=True)
        max_bp_longitude = bp_longitude.max(builtins=True)

        goodr = (radii >= min_bp_radius) & (radii <= max_bp_radius)
        goodl = (longitudes >= min_bp_longitude) & (longitudes <= max_bp_longitude)
        good = goodr & goodl

        longitudes = longitudes[good]
        radii = radii[good]

        u_pix, v_pix = RingMosaic.longitude_radius_to_pixels(
            obs, longitudes, radii, ring_body_name=ring_body_name
        )

        in_fov = (u_pix >= u_low) & (u_pix < u_limit) & (v_pix >= v_low) & (v_pix < v_limit)
        return u_pix[in_fov], v_pix[in_fov]

    # ------------------------------------------------------------------
    # Reprojection
    # ------------------------------------------------------------------

    def reproject(
        self,
        obs: Any,
        data: ma.MaskedArray | None = None,
        *,
        longitude_range: tuple[float, float] | None = None,
        radius_range: tuple[float, float] | None = None,
        margin: int = DEFAULT_MARGIN,
        zoom_amt: int | tuple[int, int] = DEFAULT_ZOOM,
        orbit_model: RingOrbitModel | None = None,
        uv_range: tuple[int, int, int, int] | None = None,
        omit_shadow: bool = True,
        image_name: str = '',
    ) -> RingReprojResult:
        """Reproject the ring in an image to a sparse radius/longitude grid.

        Always returns a sparse RingReprojResult: only longitude columns
        with valid data are stored in the result. The returned
        longitude_antimask indicates which of the ``n_full_lon`` bins are
        present.

        Parameters:
            obs: The Observation (with its FOV already configured as needed).
            data: Image data to reproject. If None, uses obs.data.
            longitude_range: (start, end) longitude limits (rad). Defaults
                to the full 0..2pi range.
            radius_range: (inner, outer) radius limits (km). Defaults to
                the mosaic's own radius_inner/outer.
            margin: Number of edge pixels to exclude. Must be >= 1.
            zoom_amt: Positive integer or (radial, longitude) tuple giving how
                many sub-samples to take per output cell along the radial and
                longitude axes. The image itself is never interpolated: every
                sub-sample reads the one image pixel containing it, and the
                sub-samples of a cell are averaged to produce that cell. Raising
                it smooths the output and fills cells whose single centre sample
                would miss the ring, at a cost in run time of the product of the
                two factors. Negative values select spline interpolation order
                (not yet supported).
            orbit_model: RingOrbitModel for co-rotating frame conversion.
                Overrides the default set at construction. None uses inertial
                longitude.
            uv_range: (start_u, end_u, start_v, end_v) to restrict the
                image region reprojected.
            omit_shadow: If True, mask pixels inside the planet's shadow.
            image_name: Optional label stored on the result (e.g. source image stem).

        Returns:
            RingReprojResult with sparse image and metadata arrays.

        Raises:
            NotImplementedError: If negative zoom_amt (spline order) is
                requested.
            TypeError: If ``longitude_range`` or ``radius_range`` is not a
                tuple/list of numeric endpoints, or an endpoint has the wrong type.
            ValueError: If ``longitude_range`` or ``radius_range`` has wrong
                length, non-finite values, longitudes outside
                ``[0, _MAX_LONGITUDE]``, ``start > end``, ``radius_inner >= radius_outer``,
                if ``zoom_amt`` is a sequence with length other than 2,
                if ``n_longitude_bins_zoom`` is not a multiple of ``l_zoom_amt``, or
                for other argument errors.
            TypeError: If ``zoom_amt`` is not an int (excluding ``bool``) or a
                length-2 tuple/list of ints, or an element is not int-like.
        """
        logger = IMAGE_LOGGER
        logger.debug(
            'longitude_range=%s radius_range=%s zoom=%s',
            longitude_range,
            radius_range,
            zoom_amt,
        )

        if margin < 1:
            raise ValueError(f'margin must be >= 1, got {margin!r}')

        if orbit_model is None:
            orbit_model = self._orbit_model
        elif orbit_model != self._orbit_model:
            raise ValueError(
                'reproject() orbit_model must match the orbit_model passed to '
                "RingMosaic's constructor (radius_inner / radius_outer semantics "
                'are tied to that choice).'
            )

        if data is None:
            data = obs.data
        data = data.view(ma.MaskedArray)

        if radius_range is None:
            radius_inner = self._radius_inner
            radius_outer = self._radius_outer
        else:
            # radius_range uses the same convention as the constructor's
            # radius_inner / radius_outer: absolute km if orbit_model is None,
            # signed offsets from orbit_model.a otherwise.
            radius_inner, radius_outer = _validate_reproject_radius_range(radius_range)

        if longitude_range is None:
            longitude_start = 0.0
            longitude_end = _MAX_LONGITUDE
        else:
            longitude_start, longitude_end = _validate_reproject_longitude_range(longitude_range)

        zoom_radial, zoom_longitude = _validate_reproject_zoom_amt(zoom_amt)

        if zoom_radial > 0:
            r_zoom_amt: int = zoom_radial
            r_spline_order = 0
        else:
            r_zoom_amt = 1
            r_spline_order = -zoom_radial

        if zoom_longitude > 0:
            l_zoom_amt: int = zoom_longitude
            l_spline_order = 0
        else:
            l_zoom_amt = 1
            l_spline_order = -zoom_longitude

        with _reduced_oops_precision():
            return self._reproject_inner(
                obs=obs,
                data=data,
                longitude_start=longitude_start,
                longitude_end=longitude_end,
                radius_inner=radius_inner,
                radius_outer=radius_outer,
                margin=margin,
                r_zoom_amt=r_zoom_amt,
                l_zoom_amt=l_zoom_amt,
                r_spline_order=r_spline_order,
                l_spline_order=l_spline_order,
                orbit_model=orbit_model,
                uv_range=uv_range,
                omit_shadow=omit_shadow,
                logger=logger,
                image_name=image_name,
            )

    def _reproject_inner(
        self,
        *,
        obs: Any,
        data: ma.MaskedArray,
        longitude_start: float,
        longitude_end: float,
        radius_inner: float,
        radius_outer: float,
        margin: int,
        r_zoom_amt: int,
        l_zoom_amt: int,
        r_spline_order: int,
        l_spline_order: int,
        orbit_model: RingOrbitModel | None,
        uv_range: tuple[int, int, int, int] | None,
        omit_shadow: bool,
        logger: PdsLogger,
        image_name: str,
    ) -> RingReprojResult:
        """Inner reprojection logic, runs with reduced oops precision."""
        if r_spline_order != 0 or l_spline_order != 0:
            raise NotImplementedError(
                'Spline interpolation (negative zoom_amt) is not yet supported'
            )

        meshgrid = None
        start_u = 0
        end_u = obs.data_shape_uv[0] - 1
        start_v = 0
        end_v = obs.data_shape_uv[1] - 1
        if uv_range is not None:
            start_u, end_u, start_v, end_v = uv_range
            # uv_range names whole columns and rows; Meshgrid works in the
            # geometry layer's pixel-corner coordinates, so each bound is
            # converted from the column's own number to the column's centre.
            meshgrid = oops.Meshgrid.for_fov(
                obs.fov,
                origin=(
                    start_u + PIXEL_CENTER_TO_CORNER_PX,
                    start_v + PIXEL_CENTER_TO_CORNER_PX,
                ),
                limit=(end_u + PIXEL_CENTER_TO_CORNER_PX, end_v + PIXEL_CENTER_TO_CORNER_PX),
                swap=True,
            )
            # Every backplane product below is region-shaped; crop the data to
            # match so masks and the region-relative pixel-corner coordinates
            # (which np.floor turns into region-relative columns and rows below)
            # stay aligned.
            data = data[start_v : end_v + 1, start_u : end_u + 1]

        bp = oops.backplane.Backplane(obs, meshgrid)

        logger.debug('Computing radius backplane')
        bp_radius = bp.ring_radius(self._ring_body_name)
        logger.debug('Computing longitude backplane')
        bp_longitude = bp.ring_longitude(self._ring_body_name)

        if ma.is_masked(data):
            bp_radius = bp_radius.remask_or(data.mask)
            bp_longitude = bp_longitude.remask_or(data.mask)

        logger.debug('Computing geometry backplanes')
        bp_radial_res = bp.ring_radial_resolution(self._ring_body_name)
        bp_angular_res = bp.ring_angular_resolution(self._ring_body_name)
        bp_phase = bp.phase_angle(self._ring_body_name)
        bp_emission = bp.emission_angle(self._ring_body_name)
        bp_incidence = bp.incidence_angle(self._ring_body_name)
        if ma.is_masked(data):
            bp_radial_res = bp_radial_res.remask_or(data.mask)
            bp_angular_res = bp_angular_res.remask_or(data.mask)
            bp_phase = bp_phase.remask_or(data.mask)
            bp_emission = bp_emission.remask_or(data.mask)
            bp_incidence = bp_incidence.remask_or(data.mask)

        if omit_shadow:
            logger.debug('Computing shadow mask')
            shadow = bp.where_inside_shadow(self._ring_body_name, self._shadow_body_name).vals
            data = ma.masked_where(shadow, data)
            # Match image masking so per-column geometry means exclude shadowed pixels.
            bp_radius = bp_radius.remask_or(shadow)
            bp_longitude = bp_longitude.remask_or(shadow)
            bp_radial_res = bp_radial_res.remask_or(shadow)
            bp_angular_res = bp_angular_res.remask_or(shadow)
            bp_phase = bp_phase.remask_or(shadow)
            bp_emission = bp_emission.remask_or(shadow)
            bp_incidence = bp_incidence.remask_or(shadow)

        if orbit_model is not None:
            # Per-pixel signed offset from the orbit model's radius at each
            # pixel's inertial longitude. The radius filter uses these offsets
            # because radius_inner / radius_outer are themselves offsets from
            # the orbital radius at each (longitude, time) in the orbit-model
            # semantics. ``radius_at_longitude`` calls ``np.cos`` internally,
            # which does not handle polymath Scalars, so we feed it the raw
            # numpy values; subtracting the resulting numpy array from the
            # ``bp_radius`` Scalar preserves the existing mask.
            model_radii = orbit_model.radius_at_longitude(bp_longitude.vals, obs.midtime)
            bp_radius_filter = bp_radius - model_radii
            bp_longitude = orbit_model.inertial_to_corotating(bp_longitude, obs.midtime)
        else:
            bp_radius_filter = bp_radius

        n_radius_bins = math.ceil(
            (radius_outer - radius_inner + _RADIUS_SLOP) / self._rad_resolution
        )
        n_radius_bins_zoom = n_radius_bins * r_zoom_amt

        full_min_lon_bin = int(np.floor(longitude_start / self._lon_resolution))
        full_max_lon_bin = int(np.floor(longitude_end / self._lon_resolution))
        n_full_lon_bins = full_max_lon_bin - full_min_lon_bin + 1
        # Bin relative to a *grid-aligned* origin (the floor of longitude_start
        # snapped to the global longitude grid), NOT to longitude_start itself.
        # The global mosaic's bin -> longitude convention is ``bin * res`` (see
        # ``bounds`` / ``to_bounded``), so placing relative bin ``b`` at global
        # bin ``b + full_min_lon_bin`` is only correct when the binning origin
        # is grid-aligned.  Using longitude_start directly would offset every
        # column by the sub-bin remainder ``longitude_start - full_min_lon_bin *
        # res`` whenever a custom ``--longitude-range`` start is not a multiple
        # of the longitude resolution.  The requested range is still enforced by
        # the mask below, which uses the true longitude_start / longitude_end.
        lon_bin_origin = full_min_lon_bin * self._lon_resolution

        restr_bp_lon = bp_longitude[
            (bp_longitude >= longitude_start)
            & (bp_longitude <= longitude_end)
            & (bp_radius_filter >= radius_inner)
            & (bp_radius_filter <= radius_outer)
        ]

        bp_lon_binned = np.floor(
            (restr_bp_lon.vals - lon_bin_origin) / self._lon_resolution
        ).astype('int')
        full_good_antimask = np.zeros(n_full_lon_bins, dtype=np.bool_)
        full_good_antimask[bp_lon_binned] = True

        filter_width = int(1.0 / np.degrees(self._lon_resolution))
        if filter_width > 1:
            full_good_antimask = nd.maximum_filter1d(full_good_antimask, filter_width, mode='wrap')

        full_lon_bins: NDArrayIntType = np.arange(n_full_lon_bins, dtype=np.int32)
        if l_zoom_amt == 1:
            full_lon_bins_zoom: NDArrayFloatType = full_lon_bins.astype(np.float64)
        else:
            full_lon_bins_zoom = _zoomed_bin_coords(n_full_lon_bins, l_zoom_amt)

        full_good_antimask_zoom = array_zoom(full_good_antimask, (l_zoom_amt,))

        lon_bins_restr: NDArrayIntType = full_lon_bins[full_good_antimask]
        lon_bins_restr_zoom: NDArrayFloatType
        if l_zoom_amt == 1:
            lon_bins_restr_zoom = lon_bins_restr.astype(np.float64)
        else:
            lon_bins_restr_zoom = full_lon_bins_zoom[full_good_antimask_zoom]

        n_lon_bins = len(lon_bins_restr)
        n_lon_bins_zoom = len(lon_bins_restr_zoom)
        if (n_lon_bins_zoom % l_zoom_amt) != 0:
            raise ValueError(
                f'n_lon_bins_zoom ({n_lon_bins_zoom}) is not a multiple of '
                f'l_zoom_amt ({l_zoom_amt})'
            )

        long_bins = np.tile(np.arange(n_lon_bins), n_radius_bins)
        long_bins_act = np.tile(
            lon_bins_restr * self._lon_resolution + lon_bin_origin, n_radius_bins
        )

        if r_zoom_amt == 1 and l_zoom_amt == 1:
            long_bins_zoom = long_bins
            long_bins_act_zoom = long_bins_act
        else:
            long_bins_zoom = np.tile(np.arange(n_lon_bins_zoom), n_radius_bins_zoom)
            long_bins_act_zoom = np.tile(
                lon_bins_restr_zoom * self._lon_resolution + lon_bin_origin,
                n_radius_bins_zoom,
            )

        rad_bins = np.repeat(np.arange(n_radius_bins), n_lon_bins)
        # Only the zoomed path needs a fractional radial coordinate; the
        # unzoomed one reads rad_bins directly, so leave this empty there
        # rather than building a second full-size array for nothing.
        rad_bins_frac_zoom: NDArrayFloatType = np.zeros(0, dtype=np.float64)
        if r_zoom_amt == 1 and l_zoom_amt == 1:
            rad_bins_zoom = rad_bins
        else:
            # rad_bins_zoom numbers the rows of the zoomed array (it scatters
            # samples into it); rad_bins_frac_zoom is the same rows' radial
            # coordinate in units of unzoomed rows, which straddles each
            # unzoomed row so the block mean lands on the row itself.
            rad_bins_zoom = np.repeat(np.arange(n_radius_bins_zoom), n_lon_bins_zoom)
            rad_bins_frac_zoom = np.repeat(
                _zoomed_bin_coords(n_radius_bins, r_zoom_amt), n_lon_bins_zoom
            )

        rad_bins_act: NDArrayFloatType
        rad_bins_act_zoom: NDArrayFloatType
        if orbit_model is not None:
            # radius_inner is a signed offset from the orbital radius at each
            # (longitude, time); the absolute radius at column c, row r is
            # (offset_at_row_r) + model_r(c). This makes an eccentric ring
            # appear as a straight line in the reprojection.
            inertial_lons_act = orbit_model.corotating_to_inertial(long_bins_act, obs.midtime)
            rad_bins_act = (
                rad_bins * self._rad_resolution
                + radius_inner
                + orbit_model.radius_at_longitude(inertial_lons_act, obs.midtime)
            )
            if r_zoom_amt == 1 and l_zoom_amt == 1:
                rad_bins_act_zoom = rad_bins_act
            else:
                inertial_lons_zoom = orbit_model.corotating_to_inertial(
                    long_bins_act_zoom, obs.midtime
                )
                rad_offset_zoom = orbit_model.radius_at_longitude(inertial_lons_zoom, obs.midtime)
                rad_bins_act_zoom = (
                    rad_bins_frac_zoom * self._rad_resolution + radius_inner + rad_offset_zoom
                )
        else:
            rad_bins_act = rad_bins * self._rad_resolution + radius_inner
            if r_zoom_amt == 1 and l_zoom_amt == 1:
                rad_bins_act_zoom = rad_bins_act
            else:
                raise NotImplementedError(
                    'Zoom with non-corotating inertial radius is not yet implemented'
                )

        u_frac_zoom, v_frac_zoom = self.longitude_radius_to_pixels(
            obs,
            long_bins_act_zoom,
            rad_bins_act_zoom,
            orbit_model=orbit_model,
            ring_body_name=self._ring_body_name,
        )

        zoomed = r_zoom_amt != 1 or l_zoom_amt != 1
        # The per-column metadata means are read at each unzoomed cell's own
        # coordinate.  Those coordinates are `long_bins_act` and `rad_bins_act`,
        # so the pixel positions come from them directly rather than out of the
        # zoomed ones: the sub-samples straddle a cell and none of them sits on
        # it, so any one of them would answer for a point up to half a cell away
        # and the means would move with the zoom.  The extra call costs the
        # reciprocal of the zoom product against the one above it.
        if zoomed:
            u_frac, v_frac = self.longitude_radius_to_pixels(
                obs,
                long_bins_act,
                rad_bins_act,
                orbit_model=orbit_model,
                ring_body_name=self._ring_body_name,
            )
        else:
            u_frac = u_frac_zoom
            v_frac = v_frac_zoom
        u_frac = u_frac.reshape(long_bins_act.shape).copy()
        v_frac = v_frac.reshape(long_bins_act.shape).copy()

        u_frac -= start_u
        v_frac -= start_v
        u_pix = np.floor(u_frac).astype('int')
        v_pix = np.floor(v_frac).astype('int')

        good = (
            (u_pix >= margin)
            & (u_pix <= end_u - start_u - margin)
            & (v_pix >= margin)
            & (v_pix <= end_v - start_v - margin)
        )
        u_frac = u_frac[good]
        v_frac = v_frac[good]
        u_pix = u_pix[good]
        v_pix = v_pix[good]

        if r_zoom_amt == 1 and l_zoom_amt == 1:
            u_pix_zoom, v_pix_zoom = u_pix, v_pix
            good_zoom = good
        else:
            u_frac_zoom -= start_u
            v_frac_zoom -= start_v
            u_pix_zoom = np.floor(u_frac_zoom).astype('int')
            v_pix_zoom = np.floor(v_frac_zoom).astype('int')
            good_zoom = (
                (u_pix_zoom >= margin)
                & (u_pix_zoom <= end_u - start_u - margin)
                & (v_pix_zoom >= margin)
                & (v_pix_zoom <= end_v - start_v - margin)
            )
            u_pix_zoom = u_pix_zoom[good_zoom]
            v_pix_zoom = v_pix_zoom[good_zoom]

        good_rad = rad_bins[good]
        good_lon = long_bins[good]
        good_rad_zoom = rad_bins_zoom[good_zoom] if zoomed else good_rad
        good_lon_zoom = long_bins_zoom[good_zoom] if zoomed else good_lon

        restr_data = data[v_pix_zoom, u_pix_zoom]
        repro_img: ma.MaskedArray = ma.zeros(
            (n_radius_bins_zoom, n_lon_bins_zoom), dtype=self._image_dtype
        )
        repro_img[:, :] = ma.masked
        repro_img[good_rad_zoom, good_lon_zoom] = restr_data
        # Block-average each cell's sub-samples back down. This is a masked mean:
        # sub-samples that fell outside the image or on a masked pixel are
        # skipped, and a cell with no surviving sub-sample stays masked. The
        # sub-samples straddle the cell (see _zoomed_bin_coords), so a fully
        # sampled cell averages to exactly its own radius and longitude; a
        # partly sampled one leans towards the side that survived.
        repro_img = ma.MaskedArray(array_unzoom(repro_img, (r_zoom_amt, l_zoom_amt)))

        good_lon_antimask = ~ma.getmaskarray(ma.sum(repro_img, axis=0))

        repro_radial_res = ma.zeros((n_radius_bins, n_lon_bins), dtype=self._metadata_dtype)
        repro_radial_res[:] = ma.masked
        repro_radial_res[good_rad, good_lon] = bp_radial_res.mvals[v_pix, u_pix]
        radial_res_antimask = ~ma.getmaskarray(ma.mean(repro_radial_res, axis=0))
        good_lon_antimask &= radial_res_antimask

        repro_angular_res = ma.zeros((n_radius_bins, n_lon_bins), dtype=self._metadata_dtype)
        repro_angular_res[:] = ma.masked
        repro_angular_res[good_rad, good_lon] = bp_angular_res.mvals[v_pix, u_pix]

        repro_phase = ma.zeros((n_radius_bins, n_lon_bins), dtype=self._metadata_dtype)
        repro_phase[:] = ma.masked
        repro_phase[good_rad, good_lon] = bp_phase.mvals[v_pix, u_pix]

        repro_emission = ma.zeros((n_radius_bins, n_lon_bins), dtype=self._metadata_dtype)
        repro_emission[:] = ma.masked
        repro_emission[good_rad, good_lon] = bp_emission.mvals[v_pix, u_pix]

        mean_incidence = ma.mean(bp_incidence.mvals[v_pix, u_pix])
        repro_incidence = float(mean_incidence) if mean_incidence is not ma.masked else float('nan')

        # Compress to sparse representation
        repro_img = repro_img[:, good_lon_antimask]
        repro_mean_radial_res = ma.mean(repro_radial_res[:, good_lon_antimask], axis=0).filled(0.0)
        repro_mean_angular_res = ma.mean(repro_angular_res[:, good_lon_antimask], axis=0).filled(
            0.0
        )
        repro_mean_phase = ma.mean(repro_phase[:, good_lon_antimask], axis=0).filled(0.0)
        repro_mean_emission = ma.mean(repro_emission[:, good_lon_antimask], axis=0).filled(0.0)

        # Full antimask for the longitude range covered
        new_antimask = np.zeros(self._n_full_lon, dtype=np.bool_)
        new_antimask[lon_bins_restr[good_lon_antimask] + full_min_lon_bin] = True

        logger.debug('Reprojection complete: %d valid longitudes', int(good_lon_antimask.sum()))

        photometric_model_name: str | None = None
        if self._photometric_model is not None and math.isfinite(float(repro_incidence)):
            photometric_model_name = self._photometric_model.name
            n_rad_sparse, n_lon_sparse = repro_img.shape
            phase_2d = np.broadcast_to(
                np.asarray(repro_mean_phase, dtype=np.float64)[np.newaxis, :],
                (n_rad_sparse, n_lon_sparse),
            )
            emission_2d = np.broadcast_to(
                np.asarray(repro_mean_emission, dtype=np.float64)[np.newaxis, :],
                (n_rad_sparse, n_lon_sparse),
            )
            incidence_2d = np.broadcast_to(
                np.asarray(repro_incidence, dtype=np.float64),
                (n_rad_sparse, n_lon_sparse),
            )
            mask_img = ma.getmaskarray(repro_img)
            d_work = np.asarray(repro_img.filled(np.nan), dtype=np.float64)
            d_work = np.nan_to_num(d_work, nan=0.0)
            inc_w = np.nan_to_num(incidence_2d, nan=0.0)
            emi_w = np.nan_to_num(emission_2d, nan=0.0)
            pha_w = np.nan_to_num(phase_2d, nan=0.0)
            corr = self._photometric_model.correct(
                d_work, incidence=inc_w, emission=emi_w, phase=pha_w
            )
            out_dtype = repro_img.dtype
            corr = np.asarray(corr, dtype=out_dtype)
            corr = np.where(mask_img, np.nan, corr)
            repro_img = ma.masked_array(corr, mask=mask_img)

        return RingReprojResult(
            body_name=self._body_name,
            longitude_resolution=self._lon_resolution,
            radius_resolution=self._rad_resolution,
            radius_inner=radius_inner,
            radius_outer=radius_outer,
            longitude_antimask=new_antimask,
            img=repro_img,
            mean_radial_resolution=np.asarray(repro_mean_radial_res, dtype=self._metadata_dtype),
            mean_angular_resolution=np.asarray(repro_mean_angular_res, dtype=self._metadata_dtype),
            mean_phase=np.asarray(repro_mean_phase, dtype=self._metadata_dtype),
            mean_emission=np.asarray(repro_mean_emission, dtype=self._metadata_dtype),
            incidence=repro_incidence,
            time=obs.midtime,
            orbit_model=orbit_model,
            image_dtype=self._image_dtype,
            metadata_dtype=self._metadata_dtype,
            photometric_model_name=photometric_model_name,
            image_name=image_name,
        )

    # ------------------------------------------------------------------
    # Mosaic accumulation
    # ------------------------------------------------------------------

    def _validate_repro_compatible(self, repro: RingReprojResult) -> None:
        """Raise ``ValueError`` if ``repro`` does not match this mosaic's grid and dtypes."""
        if repro.body_name != self._body_name:
            raise ValueError(
                'RingMosaic.add(): body mismatch (field body): '
                f'mosaic={self._body_name!r}, repro={repro.body_name!r}'
            )
        if abs(repro.longitude_resolution - self._lon_resolution) > 1e-12:
            raise ValueError(
                'RingMosaic.add(): grid resolution mismatch (field longitude_resolution): '
                f'mosaic={self._lon_resolution!r}, repro={repro.longitude_resolution!r}'
            )
        if abs(repro.radius_resolution - self._rad_resolution) > 1e-9:
            raise ValueError(
                'RingMosaic.add(): grid resolution mismatch (field radius_resolution): '
                f'mosaic={self._rad_resolution!r}, repro={repro.radius_resolution!r}'
            )
        if not math.isclose(repro.radius_inner, self._radius_inner, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                'RingMosaic.add(): radius bounds mismatch (field radius_inner): '
                f'mosaic={self._radius_inner!r}, repro={repro.radius_inner!r}'
            )
        if not math.isclose(repro.radius_outer, self._radius_outer, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                'RingMosaic.add(): radius bounds mismatch (field radius_outer): '
                f'mosaic={self._radius_outer!r}, repro={repro.radius_outer!r}'
            )
        if np.dtype(repro.image_dtype) != np.dtype(self._image_dtype):
            raise ValueError(
                'RingMosaic.add(): dtype mismatch (field image_dtype): '
                f'mosaic={self._image_dtype!r}, repro={repro.image_dtype!r}'
            )
        if np.dtype(repro.metadata_dtype) != np.dtype(self._metadata_dtype):
            raise ValueError(
                'RingMosaic.add(): dtype mismatch (field metadata_dtype): '
                f'mosaic={self._metadata_dtype!r}, repro={repro.metadata_dtype!r}'
            )
        if repro.longitude_antimask.shape != (self._n_full_lon,):
            raise ValueError(
                'RingMosaic.add(): grid mismatch (field longitude_antimask length): '
                f'mosaic expects {self._n_full_lon} longitude bins, repro has '
                f'{int(np.size(repro.longitude_antimask))}'
            )
        if repro.img.ndim != 2:
            raise ValueError(
                'RingMosaic.add(): repro.img must be 2-D '
                f'(field sparse img), got shape {repro.img.shape!r}'
            )
        if repro.img.shape[0] != self._n_radius:
            raise ValueError(
                'RingMosaic.add(): grid mismatch (field radius bin count): '
                f'mosaic has {self._n_radius} radius bins, repro.img.shape[0]={repro.img.shape[0]}'
            )
        n_valid = int(repro.img.shape[1])
        for field_name, arr in (
            ('mean_radial_resolution', repro.mean_radial_resolution),
            ('mean_angular_resolution', repro.mean_angular_resolution),
            ('mean_phase', repro.mean_phase),
            ('mean_emission', repro.mean_emission),
        ):
            if arr.ndim != 1 or int(arr.shape[0]) != n_valid:
                raise ValueError(
                    'RingMosaic.add(): sparse column count mismatch '
                    f'(field {field_name} vs repro.img): '
                    f'img has {n_valid} columns, {field_name}.shape={arr.shape!r}'
                )

    def add(self, repro: RingReprojResult) -> None:
        """Add a reprojected image to the mosaic.

        New longitude columns are inserted into the sparse arrays using a
        single batched ``np.insert`` call. Existing columns are updated
        according to the merge strategy.

        The reprojection's orbit model and photometric model must match the
        mosaic's. ``orbit_model`` must be the same instance (or both ``None``);
        ``photometric_model_name`` must equal the mosaic's photometric model
        name (or both ``None``). Mixing different settings would silently
        corrupt the mosaic because radii and longitudes have different
        meanings under different orbit models.

        Parameters:
            repro: Sparse reprojection result from reproject().

        Raises:
            OverflowError: If the number of images added would exceed the
                uint16 maximum of 65 535.
            ValueError: If the reprojection's body name, grid resolution, radius
                bounds, dtypes, sparse shapes, ``longitude_antimask`` true count vs.
                sparse column count, orbit model, or photometric model does not match
                the mosaic's.
        """
        self._validate_repro_compatible(repro)
        n_lon_true = int(np.count_nonzero(repro.longitude_antimask))
        n_sparse_cols = int(repro.img.shape[1])
        if n_lon_true != n_sparse_cols:
            raise ValueError(
                'RingMosaic.add(): RingReprojResult.longitude_antimask must have one True '
                'entry per sparse column in repro.img; '
                f'count_nonzero(longitude_antimask)={n_lon_true}, '
                f'repro.img.shape[1]={n_sparse_cols}'
            )
        # RingOrbitModel is a frozen dataclass with value-based equality
        # (auto-generated __eq__), so a model loaded from disk is equal to
        # but not identical to the in-memory mosaic instance. Compare by
        # value, not identity.
        if repro.orbit_model != self._orbit_model:
            mosaic_name = self._orbit_model.name if self._orbit_model is not None else None
            repro_name = repro.orbit_model.name if repro.orbit_model is not None else None
            raise ValueError(
                'RingMosaic.add(): reprojection orbit model does not match '
                f'mosaic orbit model (mosaic={mosaic_name!r}, repro={repro_name!r}). '
                'All reprojections added to a mosaic must use the same orbit '
                'model so that radius and longitude semantics agree.'
            )

        mosaic_phot_name = (
            self._photometric_model.name if self._photometric_model is not None else None
        )
        if repro.photometric_model_name != mosaic_phot_name:
            raise ValueError(
                'RingMosaic.add(): reprojection photometric model does not match '
                f'mosaic photometric model (mosaic={mosaic_phot_name!r}, '
                f'repro={repro.photometric_model_name!r}).'
            )

        if self._image_count > np.iinfo(np.uint16).max:
            raise OverflowError(
                f'image_count {self._image_count} exceeds uint16 max '
                f'{np.iinfo(np.uint16).max}; cannot add more images'
            )

        valid_bins = np.where(repro.longitude_antimask)[0]
        if len(valid_bins) == 0:
            return

        existing_bins = np.where(self._antimask)[0]

        new_mask = ~self._antimask[valid_bins]
        new_bins = valid_bins[new_mask]
        old_bins = valid_bins[~new_mask]

        if len(new_bins) > 0:
            self._insert_new_columns(new_bins, valid_bins, repro)
            # Update antimask immediately so _update_existing_columns
            # sees the correct sparse column offsets.
            self._antimask[new_bins] = True

        if len(old_bins) > 0:
            self._update_existing_columns(old_bins, valid_bins, existing_bins, repro)

        self._antimask[valid_bins] = True
        self._contributing_image_names.append(repro.image_name)
        self._image_count += 1
        if math.isfinite(repro.incidence):
            self._mean_incidence_sum += float(repro.incidence)
            self._mean_incidence_count += 1
        self._mean_incidence = (
            self._mean_incidence_sum / float(self._mean_incidence_count)
            if self._mean_incidence_count > 0
            else 0.0
        )

    def _insert_new_columns(
        self,
        new_bins: NDArrayIntType,
        valid_bins: NDArrayIntType,
        repro: RingReprojResult,
    ) -> None:
        """Insert columns for brand-new longitude bins via batched np.insert."""
        # For each new bin, find where it sits among the currently-stored
        # columns (which correspond to sorted existing_bins).
        existing_bins = np.where(self._antimask)[0]
        insert_positions = np.searchsorted(existing_bins, new_bins)

        # Indices of new_bins in the repro valid_bins array
        new_repro_idx = np.searchsorted(valid_bins, new_bins)

        # Expand all sparse arrays by inserting the new columns
        new_img_cols = repro.img[:, new_repro_idx]  # [n_radius, n_new]
        new_img_data = ma.getdata(new_img_cols)
        new_img_mask = ma.getmaskarray(new_img_cols)

        img_data = np.insert(ma.getdata(self._img_sparse), insert_positions, new_img_data, axis=1)
        img_mask = np.insert(
            ma.getmaskarray(self._img_sparse), insert_positions, new_img_mask, axis=1
        )
        self._img_sparse = ma.MaskedArray(img_data, mask=img_mask)

        self._mean_radial_res = np.insert(
            self._mean_radial_res, insert_positions, repro.mean_radial_resolution[new_repro_idx]
        )
        self._mean_angular_res = np.insert(
            self._mean_angular_res, insert_positions, repro.mean_angular_resolution[new_repro_idx]
        )
        self._mean_phase = np.insert(
            self._mean_phase, insert_positions, repro.mean_phase[new_repro_idx]
        )
        self._mean_emission = np.insert(
            self._mean_emission, insert_positions, repro.mean_emission[new_repro_idx]
        )
        self._image_number = np.insert(
            self._image_number,
            insert_positions,
            np.full(len(new_bins), self._image_count, dtype=np.uint16),
        )
        self._time = np.insert(
            self._time,
            insert_positions,
            np.full(len(new_bins), repro.time, dtype=np.float64),
        )

    def _update_existing_columns(
        self,
        old_bins: NDArrayIntType,
        valid_bins: NDArrayIntType,
        existing_bins: NDArrayIntType,
        repro: RingReprojResult,
    ) -> None:
        """Update columns that already exist in the mosaic per merge strategy."""
        # sparse index of each old bin in current storage
        existing_bins_current = np.where(self._antimask)[0]
        sparse_idx = np.searchsorted(existing_bins_current, old_bins)
        # index of each old bin in repro.valid_bins
        repro_idx = np.searchsorted(valid_bins, old_bins)

        for k in range(len(old_bins)):
            si = sparse_idx[k]
            ri = repro_idx[k]
            if self._should_replace(si, ri, repro):
                self._img_sparse[:, si] = repro.img[:, ri]
                self._mean_radial_res[si] = repro.mean_radial_resolution[ri]
                self._mean_angular_res[si] = repro.mean_angular_resolution[ri]
                self._mean_phase[si] = repro.mean_phase[ri]
                self._mean_emission[si] = repro.mean_emission[ri]
                self._image_number[si] = self._image_count
                self._time[si] = repro.time

    def _should_replace(
        self,
        sparse_idx: int,
        repro_idx: int,
        repro: RingReprojResult,
    ) -> bool:
        """Return True if the repro column should replace the existing sparse column."""
        if self._merge_strategy == RingMosaicMergeStrategy.BEST_RESOLUTION:
            return bool(repro.mean_radial_resolution[repro_idx] < self._mean_radial_res[sparse_idx])

        # MOST_COVERAGE_THEN_RESOLUTION
        existing_valid = int((~ma.getmaskarray(self._img_sparse[:, sparse_idx])).sum())
        new_valid = int((~ma.getmaskarray(repro.img[:, repro_idx])).sum())
        if new_valid != existing_valid:
            return new_valid > existing_valid
        return bool(repro.mean_radial_resolution[repro_idx] < self._mean_radial_res[sparse_idx])

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def to_sparse(self) -> RingMosaicData:
        """Return the mosaic in its native sparse representation.

        The returned img has shape [n_radius, n_valid_longitudes] and the
        longitude_antimask has length n_full_lon with True at each stored
        column.

        Returns:
            RingMosaicData with the sparse internal arrays.
        """
        return self._build_result(
            img=self._img_sparse,
            antimask=self._antimask.copy(),
            longitude_range=None,
            per_lon_1d=True,
        )

    def to_full(self) -> RingMosaicData:
        """Return the mosaic as a dense full-circle array.

        The returned img has shape [n_radius, n_full_lon]. Longitude columns
        with no data are fully masked.

        Returns:
            RingMosaicData with a full-circle dense array.
        """
        full_img_data = np.zeros((self._n_radius, self._n_full_lon), dtype=self._image_dtype)
        full_img_mask = np.ones((self._n_radius, self._n_full_lon), dtype=np.bool_)
        full_mean_rad_res = np.zeros(self._n_full_lon, dtype=self._metadata_dtype)
        full_mean_ang_res = np.zeros(self._n_full_lon, dtype=self._metadata_dtype)
        full_mean_phase = np.zeros(self._n_full_lon, dtype=self._metadata_dtype)
        full_mean_emission = np.zeros(self._n_full_lon, dtype=self._metadata_dtype)
        full_img_number = np.zeros(self._n_full_lon, dtype=np.uint16)
        full_time = np.zeros(self._n_full_lon, dtype=np.float64)

        full_mask_1d = np.ones(self._n_full_lon, dtype=np.bool_)

        valid_bins = np.where(self._antimask)[0]
        if len(valid_bins) > 0:
            full_img_data[:, valid_bins] = ma.getdata(self._img_sparse)
            full_img_mask[:, valid_bins] = ma.getmaskarray(self._img_sparse)
            full_mean_rad_res[valid_bins] = self._mean_radial_res
            full_mean_ang_res[valid_bins] = self._mean_angular_res
            full_mean_phase[valid_bins] = self._mean_phase
            full_mean_emission[valid_bins] = self._mean_emission
            full_img_number[valid_bins] = self._image_number
            full_time[valid_bins] = self._time
            full_mask_1d[valid_bins] = False

        return RingMosaicData(
            body_name=self._body_name,
            ring_body_name=self._ring_body_name,
            shadow_body_name=self._shadow_body_name,
            longitude_resolution=self._lon_resolution,
            radius_resolution=self._rad_resolution,
            radius_inner=self._radius_inner,
            radius_outer=self._radius_outer,
            longitude_antimask=self._antimask.copy(),
            img=ma.MaskedArray(full_img_data, mask=full_img_mask),
            longitude_range=None,
            mean_radial_resolution=ma.MaskedArray(full_mean_rad_res, mask=full_mask_1d),
            mean_angular_resolution=ma.MaskedArray(full_mean_ang_res, mask=full_mask_1d),
            mean_phase=ma.MaskedArray(full_mean_phase, mask=full_mask_1d),
            mean_emission=ma.MaskedArray(full_mean_emission, mask=full_mask_1d),
            mean_incidence=self._mean_incidence,
            image_number=ma.MaskedArray(full_img_number, mask=full_mask_1d),
            time=ma.MaskedArray(full_time, mask=full_mask_1d),
            image_dtype=self._image_dtype,
            metadata_dtype=self._metadata_dtype,
            contributing_image_names=tuple(self._contributing_image_names),
            orbit_model_name=self._orbit_model.name if self._orbit_model else None,
            photometric_model_name=(
                self._photometric_model.name if self._photometric_model is not None else None
            ),
        )

    def to_bounded(
        self,
        *,
        longitude_range: tuple[float, float],
    ) -> RingMosaicData:
        """Return the mosaic restricted to the given longitude range.

        The returned img has shape [n_radius, n_bins_in_range] where every
        bin from the start to end of the range is included. Bins without
        data are fully masked.

        Parameters:
            longitude_range: (start, end) in radians.

        Returns:
            RingMosaicData covering exactly the requested longitude range.

        Raises:
            TypeError: If ``longitude_range`` is not a tuple/list of two numbers, or an
                endpoint has the wrong type (same rules as :meth:`reproject`).
            ValueError: If endpoints are non-finite, outside ``[0, _MAX_LONGITUDE]``,
                or ``start > end`` (same rules as :meth:`reproject`).
        """
        lon_start_f, lon_end_f = _validate_reproject_longitude_range(longitude_range)
        start_bin = round(lon_start_f / self._lon_resolution)
        end_bin = round(lon_end_f / self._lon_resolution)
        n_bins = end_bin - start_bin + 1

        bounded_img_data = np.zeros((self._n_radius, n_bins), dtype=self._image_dtype)
        bounded_img_mask = np.ones((self._n_radius, n_bins), dtype=np.bool_)
        bounded_mean_rad_res = np.zeros(n_bins, dtype=self._metadata_dtype)
        bounded_mean_ang_res = np.zeros(n_bins, dtype=self._metadata_dtype)
        bounded_mean_phase = np.zeros(n_bins, dtype=self._metadata_dtype)
        bounded_mean_emission = np.zeros(n_bins, dtype=self._metadata_dtype)
        bounded_img_number = np.zeros(n_bins, dtype=np.uint16)
        bounded_time = np.zeros(n_bins, dtype=np.float64)
        bounded_mask_1d = np.ones(n_bins, dtype=np.bool_)

        bounded_antimask = np.zeros(self._n_full_lon, dtype=np.bool_)

        valid_global_bins = np.where(self._antimask)[0]
        in_range = (valid_global_bins >= start_bin) & (valid_global_bins <= end_bin)
        range_global_bins = valid_global_bins[in_range]

        if len(range_global_bins) > 0:
            lbs = range_global_bins - start_bin
            sparse_idxs = np.searchsorted(valid_global_bins, range_global_bins)
            bounded_img_data[:, lbs] = ma.getdata(self._img_sparse)[:, sparse_idxs]
            bounded_img_mask[:, lbs] = ma.getmaskarray(self._img_sparse)[:, sparse_idxs]
            bounded_mean_rad_res[lbs] = self._mean_radial_res[sparse_idxs]
            bounded_mean_ang_res[lbs] = self._mean_angular_res[sparse_idxs]
            bounded_mean_phase[lbs] = self._mean_phase[sparse_idxs]
            bounded_mean_emission[lbs] = self._mean_emission[sparse_idxs]
            bounded_img_number[lbs] = self._image_number[sparse_idxs]
            bounded_time[lbs] = self._time[sparse_idxs]
            bounded_mask_1d[lbs] = False
            bounded_antimask[range_global_bins] = True

        return RingMosaicData(
            body_name=self._body_name,
            ring_body_name=self._ring_body_name,
            shadow_body_name=self._shadow_body_name,
            longitude_resolution=self._lon_resolution,
            radius_resolution=self._rad_resolution,
            radius_inner=self._radius_inner,
            radius_outer=self._radius_outer,
            longitude_antimask=bounded_antimask,
            img=ma.MaskedArray(bounded_img_data, mask=bounded_img_mask),
            longitude_range=(lon_start_f, lon_end_f),
            mean_radial_resolution=ma.MaskedArray(bounded_mean_rad_res, mask=bounded_mask_1d),
            mean_angular_resolution=ma.MaskedArray(bounded_mean_ang_res, mask=bounded_mask_1d),
            mean_phase=ma.MaskedArray(bounded_mean_phase, mask=bounded_mask_1d),
            mean_emission=ma.MaskedArray(bounded_mean_emission, mask=bounded_mask_1d),
            mean_incidence=self._mean_incidence,
            image_number=ma.MaskedArray(bounded_img_number, mask=bounded_mask_1d),
            time=ma.MaskedArray(bounded_time, mask=bounded_mask_1d),
            image_dtype=self._image_dtype,
            metadata_dtype=self._metadata_dtype,
            contributing_image_names=tuple(self._contributing_image_names),
            orbit_model_name=self._orbit_model.name if self._orbit_model else None,
            photometric_model_name=(
                self._photometric_model.name if self._photometric_model is not None else None
            ),
        )

    def _build_result(
        self,
        *,
        img: ma.MaskedArray,
        antimask: NDArrayBoolType,
        longitude_range: tuple[float, float] | None,
        per_lon_1d: bool,
    ) -> RingMosaicData:
        """Build a RingMosaicData for to_sparse()."""
        if per_lon_1d:
            mask_1d = np.zeros(img.shape[1], dtype=np.bool_)
        else:
            mask_1d = np.ones(img.shape[1], dtype=np.bool_)

        return RingMosaicData(
            body_name=self._body_name,
            ring_body_name=self._ring_body_name,
            shadow_body_name=self._shadow_body_name,
            longitude_resolution=self._lon_resolution,
            radius_resolution=self._rad_resolution,
            radius_inner=self._radius_inner,
            radius_outer=self._radius_outer,
            longitude_antimask=antimask,
            img=img.copy(),
            longitude_range=longitude_range,
            mean_radial_resolution=ma.MaskedArray(self._mean_radial_res.copy(), mask=mask_1d),
            mean_angular_resolution=ma.MaskedArray(self._mean_angular_res.copy(), mask=mask_1d),
            mean_phase=ma.MaskedArray(self._mean_phase.copy(), mask=mask_1d),
            mean_emission=ma.MaskedArray(self._mean_emission.copy(), mask=mask_1d),
            mean_incidence=self._mean_incidence,
            image_number=ma.MaskedArray(self._image_number.copy(), mask=mask_1d),
            time=ma.MaskedArray(self._time.copy(), mask=mask_1d),
            image_dtype=self._image_dtype,
            metadata_dtype=self._metadata_dtype,
            contributing_image_names=tuple(self._contributing_image_names),
            orbit_model_name=self._orbit_model.name if self._orbit_model else None,
            photometric_model_name=(
                self._photometric_model.name if self._photometric_model is not None else None
            ),
        )
