"""Data-loading helpers for spindoctor.ui.mosaic_viewer.

Provides ``load_ring_file`` and ``load_body_file`` that read any of the four
reprojection / mosaic dataclasses (``RingReprojResult``, ``RingMosaicData``,
``BodyReprojResult``, ``BodyMosaicData``) and return a normalised
``DisplayData`` object ready for use in the ring or body window.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import astropy.io.fits as pyfits
import numpy as np
import numpy.ma as ma
from filecache import FCPath
from PyQt6.QtWidgets import QLineEdit, QSlider

from spindoctor.reproj._serialization import infer_format
from spindoctor.reproj.bodies import BodyMosaicData, BodyReprojResult
from spindoctor.reproj.ring_orbit_model import RingOrbitModel, get_orbit_model_by_name
from spindoctor.reproj.rings import RingMosaicData, RingReprojResult
from spindoctor.ui.mosaic_viewer.tiled_image_widget import slider_to_zoom, zoom_to_slider


class _SyncedSlider:
    """Keeps a QLineEdit and QSlider in sync for a single numeric parameter."""

    def __init__(
        self,
        line_edit: QLineEdit,
        slider: QSlider,
        lo: float,
        hi: float,
        fmt: str = '%.4f',
        on_change: Callable[[float], None] | None = None,
    ) -> None:
        """Wire ``line_edit`` and ``slider`` over ``[lo, hi]`` with optional ``on_change``.

        Parameters:
            line_edit: Numeric text field.
            slider: 0..1000 slider mapped linearly to ``[lo, hi]``.
            lo: Lower bound of the mapped range.
            hi: Upper bound of the mapped range.
            fmt: ``printf`` format for the line edit.
            on_change: Called with the new float after slider or edit updates.
        """
        self._le = line_edit
        self._sl = slider
        self._lo = lo
        self._hi = hi
        self._fmt = fmt
        self._on_change = on_change
        self._updating = False
        self._sl.valueChanged.connect(self._slider_moved)
        self._le.editingFinished.connect(self._edit_done)

    def _to_slider(self, val: float) -> int:
        """Map ``val`` in ``[lo, hi]`` to a slider position in ``[0, 1000]``."""
        if self._hi <= self._lo:
            return 0
        pos = (val - self._lo) / (self._hi - self._lo) * 1000.0
        return round(float(np.clip(pos, 0, 1000)))

    def _from_slider(self, pos: int) -> float:
        """Map slider position ``pos`` (0..1000) back to a float in ``[lo, hi]``."""
        return self._lo + (self._hi - self._lo) * pos / 1000.0

    def _slider_moved(self, pos: int) -> None:
        """Mirror slider motion into the line edit and invoke ``on_change``."""
        if self._updating:
            return
        val = self._from_slider(pos)
        self._updating = True
        self._le.setText(self._fmt % val)
        self._updating = False
        if self._on_change:
            self._on_change(val)

    def _edit_done(self) -> None:
        """Parse the line edit, clamp to ``[lo, hi]``, sync the slider, call ``on_change``."""
        if self._updating:
            return
        try:
            val = float(self._le.text())
        except ValueError:
            return
        val = max(self._lo, min(self._hi, val))
        self._updating = True
        self._sl.setValue(self._to_slider(val))
        self._le.setText(self._fmt % val)
        self._updating = False
        if self._on_change:
            self._on_change(val)

    def set_range(self, lo: float, hi: float) -> None:
        """Update ``lo``/``hi`` without changing the displayed value."""
        self._lo = lo
        self._hi = hi

    def set_value(self, val: float) -> None:
        """Programmatically set both widgets to ``val`` (clamped by current range)."""
        self._updating = True
        self._sl.setValue(self._to_slider(val))
        self._le.setText(self._fmt % val)
        self._updating = False

    def get_value(self) -> float:
        """Return the line-edit float if valid, otherwise infer from the slider."""
        try:
            return float(self._le.text())
        except ValueError:
            return self._from_slider(self._sl.value())


class _ZoomSync(_SyncedSlider):
    """A :class:`_SyncedSlider` that uses logarithmic zoom mapping.

    Converts between the zoom float value and a 0-1000 slider integer using
    :func:`~spindoctor.ui.mosaic_viewer.tiled_image_widget.zoom_to_slider` and
    :func:`~spindoctor.ui.mosaic_viewer.tiled_image_widget.slider_to_zoom` so that
    zooming feels perceptually uniform.
    """

    def _to_slider(self, val: float) -> int:
        """Convert zoom value to slider integer position via ``zoom_to_slider``."""
        return zoom_to_slider(val)

    def _from_slider(self, pos: int) -> float:
        """Convert slider integer position to zoom value via ``slider_to_zoom``."""
        return slider_to_zoom(pos)


_RAD_TO_DEG: float = 180.0 / math.pi


def _rad_to_deg_ma(arr: ma.MaskedArray) -> ma.MaskedArray:
    """Convert masked array values from radians to degrees (mask preserved)."""
    return ma.MaskedArray(np.rad2deg(arr.data), mask=ma.getmaskarray(arr))


def _compute_vmin_vmax(image_ma: ma.MaskedArray) -> tuple[float, float]:
    """Return ``(vmin, vmax)`` from valid (unmasked) pixels of ``image_ma``.

    Parameters:
        image_ma: 2-D masked image; masked entries are ignored.

    Returns:
        ``(min, max)`` of compressed valid data, or ``(0.0, 1.0)`` when empty.
    """
    valid = image_ma.compressed()
    if valid.size > 0:
        return float(np.nanmin(valid)), float(np.nanmax(valid))
    return 0.0, 1.0


def _ring_longitude_column_origin_and_extent_hi_deg(
    *,
    n_cols: int,
    lon_res_rad: float,
    longitude_antimask: np.ndarray,
    longitude_range: tuple[float, float] | None,
) -> tuple[float, float, np.ndarray | None]:
    """Longitude (deg) at column 0 and upper extent for ring sparse/dense grids.

    Sparse ring columns map to global bins ``np.flatnonzero(longitude_antimask)``
    in sorted order.  Dense full-circle mosaics have one column per bin starting
    at longitude 0.  Bounded mosaics use ``longitude_range`` to fix the bin grid.

    Parameters:
        n_cols: Number of longitude columns in the stored image.
        lon_res_rad: Longitude bin width in radians.
        longitude_antimask: Boolean array marking which global longitude bins exist.
        longitude_range: Optional ``(start, end)`` mosaic longitude bounds (rad).

    Returns:
        ``(origin_deg, extent_deg, global_bins)`` where ``origin_deg`` is the
        longitude in degrees column 0 samples and ``extent_deg`` is one
        resolution step past the last column's longitude.  ``global_bins`` is
        ``None`` when columns map contiguously from ``origin_deg`` (i.e. origin
        + ix * resolution is correct); when the populated bins have gaps,
        ``global_bins`` is the ``np.flatnonzero(longitude_antimask)`` array and
        callers must use ``global_bins[ix] * lon_res_rad * _RAD_TO_DEG`` for
        per-column longitude instead of the linear formula.
    """
    lon_res_deg = lon_res_rad * _RAD_TO_DEG
    n_full = int(longitude_antimask.shape[0])
    if longitude_range is not None:
        lon_start, _lon_end = longitude_range
        start_bin = round(lon_start / lon_res_rad)
        origin_deg = float(start_bin * lon_res_rad * _RAD_TO_DEG)
        return origin_deg, origin_deg + float(n_cols * lon_res_deg), None
    if n_cols == n_full:
        return 0.0, float(n_cols * lon_res_deg), None
    global_bins = np.flatnonzero(longitude_antimask)
    if global_bins.size != n_cols:
        print(
            f'Ring longitude antimask length {int(global_bins.size)} does not match '
            f'image columns {int(n_cols)}; assuming longitudes start at 0 deg for '
            f'display.',
            file=sys.stderr,
        )
        return 0.0, float(n_cols * lon_res_deg), None
    # Check for contiguous bins: diff should be all-ones.
    if global_bins.size > 1 and not bool(np.all(np.diff(global_bins) == 1)):
        # Non-contiguous: return sentinel so callers use per-column bin lookup.
        origin_deg = float(global_bins[0] * lon_res_rad * _RAD_TO_DEG)
        extent_deg = float((global_bins[-1] + 1) * lon_res_rad * _RAD_TO_DEG)
        return origin_deg, extent_deg, global_bins
    origin_deg = float(global_bins[0] * lon_res_rad * _RAD_TO_DEG)
    return origin_deg, origin_deg + float(n_cols * lon_res_deg), None


def _peek_kind(path: str | FCPath) -> str:
    """Read the dataclass kind from a reproj/mosaic file without loading full arrays.

    For ``.npz`` archives this is the ``__kind__`` array entry; for FITS it is the
    primary header keyword ``KIND`` (see :func:`spindoctor.reproj._serialization.save_fits`).
    Uses :func:`infer_format` and resolves ``path`` via :class:`filecache.FCPath`.

    Parameters:
        path: Local or remote path to ``.npz`` / ``.fits`` (``str`` or ``FCPath``).

    Returns:
        Kind string (e.g. ``'BodyMosaicData'``) read from the file header.

    Raises:
        ValueError: If no ``KIND`` / ``__kind__`` entry is present for this format.
        OSError: If the file cannot be opened (propagated from NumPy or Astropy).
    """
    fmt = infer_format(path, None)
    _missing_kind_msg = (
        f'No KIND header found in {path!r}; expected a spindoctor.reproj FITS export.'
    )
    if fmt == 'npz':
        local = cast(Path, FCPath(path).get_local_path())
        with np.load(local, allow_pickle=False) as raw:
            if '__kind__' not in raw:
                raise ValueError(_missing_kind_msg)
            kind = str(raw['__kind__'])
    else:
        local = cast(Path, FCPath(path).get_local_path())
        with pyfits.open(local) as hdul:
            hdr = hdul[0].header
            kind = str(hdr.get('KIND', '')).strip()
    if not kind:
        raise ValueError(_missing_kind_msg)
    return kind


# ---------------------------------------------------------------------------
# Ring DisplayData
# ---------------------------------------------------------------------------


@dataclass
class RingDisplayData:
    """All display data extracted from a ring reprojection or mosaic file.

    Parameters:
        title: Short title for the window bar (filename stem).
        image_ma: 2-D masked array (n_radius, n_longitude), row 0 = inner.
        longitude_resolution_deg: Column pitch in degrees.
        radius_resolution_km: Row pitch in km.
        radius_inner: Inner radius (km, absolute) when ``orbit_model_name`` is
            ``None``; signed offset (km) from the orbital radius at each
            (longitude, time) otherwise.
        radius_outer: Outer radius (km, absolute) when ``orbit_model_name`` is
            ``None``; signed offset (km) from the orbital radius at each
            (longitude, time) otherwise.
        n_radii: Number of radius rows.
        n_longitude: Number of longitude columns stored in ``image_ma``.
        mean_radial_resolution: Per-column masked array (km/pixel).
        mean_angular_resolution: Per-column masked array (deg/pixel).
        mean_phase: Per-column masked array (deg).
        mean_emission: Per-column masked array (deg).
        image_number: Per-column masked uint16 (mosaic only; None for reproj).
        orbit_model_name: Name of the orbit model when longitudes are
            co-rotating and ``radius_inner`` / ``radius_outer`` are signed
            offsets from the orbital radius at each (longitude, time);
            ``None`` for inertial longitudes and absolute ring radii (also
            ``None`` in older mosaic files that omit this field).
        orbit_model: The :class:`~spindoctor.reproj.ring_orbit_model.RingOrbitModel` object
            when available (loaded from reproj file or looked up by name); ``None``
            otherwise.
        vmin: Minimum valid image value.
        vmax: Maximum valid image value.
        is_mosaic: True if loaded from a RingMosaicData.
        longitude_antimask: For reproj, full-circle antimask; for mosaic,
            antimask of populated columns.
        body_name: Planet / host body name when stored in the file.
        mean_incidence_deg: Mean incidence angle (deg), when available.
        photometric_model_name: Model applied when the file was written, if any.
        observation_time_tdb: Per-column TDB seconds past J2000, when present.
        longitude_column_origin_deg: Longitude (deg) image column 0 samples, for
            sparse reprojections and mosaics whose first column is not at 0 deg.
            Each column is one point sample, so this is the column's own
            longitude and not a boundary around it.
        longitude_extent_hi_deg: Upper cap for longitude (deg) for EW x-axis sync
            and cursor clipping: one resolution step past the last column's
            longitude.
        contributing_image_names: Names in ``image_number`` order (mosaic); for a single
            reproj, at most one entry when ``image_name`` was stored on save.
        longitude_global_bins: For sparse mosaics with non-contiguous populated bins,
            the sorted global bin indices (``np.flatnonzero(longitude_antimask)``).
            When set, per-column longitude must be computed as
            ``longitude_global_bins[col] * longitude_resolution_rad * RAD_TO_DEG``
            rather than ``longitude_column_origin_deg + col * longitude_resolution_deg``.
            ``None`` when bins are contiguous (the linear formula is correct).
    """

    title: str
    image_ma: ma.MaskedArray
    longitude_resolution_deg: float
    radius_resolution_km: float
    radius_inner: float
    radius_outer: float
    n_radii: int
    n_longitude: int
    mean_radial_resolution: ma.MaskedArray
    mean_angular_resolution: ma.MaskedArray
    mean_phase: ma.MaskedArray
    mean_emission: ma.MaskedArray
    image_number: ma.MaskedArray | None
    orbit_model_name: str | None
    orbit_model: RingOrbitModel | None
    vmin: float
    vmax: float
    is_mosaic: bool
    longitude_antimask: np.ndarray
    body_name: str | None = None
    mean_incidence_deg: float | None = None
    photometric_model_name: str | None = None
    observation_time_tdb: ma.MaskedArray | None = None
    longitude_column_origin_deg: float = 0.0
    longitude_extent_hi_deg: float | None = None
    contributing_image_names: tuple[str, ...] = ()
    longitude_global_bins: np.ndarray | None = None


def load_ring_file(path: str) -> RingDisplayData:
    """Load a ring reprojection or mosaic file and return ``RingDisplayData``.

    Accepts all four ring result types: ``RingReprojResult`` and
    ``RingMosaicData``.

    Parameters:
        path: Local file path (str or anything :class:`filecache.FCPath`-like).

    Returns:
        ``RingDisplayData`` ready to be passed to ``RingMosaicWindow``.

    Raises:
        ValueError: If the file kind is not a supported ring type.
    """
    title = Path(str(path)).stem
    kind = _peek_kind(path)

    if kind == 'RingReprojResult':
        result = RingReprojResult.load(path)
        image_ma = result.img  # (n_radius, n_valid_lon)
        lon_res_deg = result.longitude_resolution * _RAD_TO_DEG
        rad_res_km = result.radius_resolution
        orbit_model_name = result.orbit_model.name if result.orbit_model is not None else None
        orbit_model: RingOrbitModel | None = result.orbit_model
        n_radii, n_lon = image_ma.shape
        # Build 1-D per-column metadata masked arrays (same length as sparse img cols)
        mrr = ma.MaskedArray(result.mean_radial_resolution)
        mar = ma.MaskedArray(result.mean_angular_resolution * _RAD_TO_DEG)
        mphase = ma.MaskedArray(result.mean_phase * _RAD_TO_DEG)
        memission = ma.MaskedArray(result.mean_emission * _RAD_TO_DEG)
        vmin, vmax = _compute_vmin_vmax(image_ma)
        inc_deg = float(result.incidence * _RAD_TO_DEG) if np.isfinite(result.incidence) else None
        if np.isfinite(result.time):
            obs_tdb = ma.MaskedArray(
                np.full(n_lon, float(result.time), dtype=np.float64),
                mask=np.zeros(n_lon, dtype=bool),
            )
        else:
            obs_tdb = None
        lon_origin_deg, lon_hi, lon_global_bins = _ring_longitude_column_origin_and_extent_hi_deg(
            n_cols=n_lon,
            lon_res_rad=result.longitude_resolution,
            longitude_antimask=result.longitude_antimask,
            longitude_range=None,
        )
        return RingDisplayData(
            title=title,
            image_ma=image_ma,
            longitude_resolution_deg=lon_res_deg,
            radius_resolution_km=rad_res_km,
            radius_inner=result.radius_inner,
            radius_outer=result.radius_outer,
            n_radii=n_radii,
            n_longitude=n_lon,
            mean_radial_resolution=mrr,
            mean_angular_resolution=mar,
            mean_phase=mphase,
            mean_emission=memission,
            image_number=None,
            orbit_model_name=orbit_model_name,
            orbit_model=orbit_model,
            vmin=vmin,
            vmax=vmax,
            is_mosaic=False,
            longitude_antimask=result.longitude_antimask,
            body_name=result.body_name,
            mean_incidence_deg=inc_deg,
            photometric_model_name=result.photometric_model_name,
            observation_time_tdb=obs_tdb,
            longitude_column_origin_deg=lon_origin_deg,
            longitude_extent_hi_deg=lon_hi,
            contributing_image_names=(result.image_name,) if result.image_name else (),
            longitude_global_bins=lon_global_bins,
        )

    if kind == 'RingMosaicData':
        result_m = RingMosaicData.load(path)
        image_ma = result_m.img  # (n_radius, n_sparse_lon)
        lon_res_deg = result_m.longitude_resolution * _RAD_TO_DEG
        rad_res_km = result_m.radius_resolution
        orbit_model_name = result_m.orbit_model_name
        if orbit_model_name is None or not orbit_model_name.strip():
            orbit_model = None
        else:
            orbit_model = get_orbit_model_by_name(orbit_model_name)
        n_radii, n_lon = image_ma.shape
        mrr = result_m.mean_radial_resolution
        mar = ma.MaskedArray(result_m.mean_angular_resolution * _RAD_TO_DEG)
        mphase = ma.MaskedArray(result_m.mean_phase * _RAD_TO_DEG)
        memission = ma.MaskedArray(result_m.mean_emission * _RAD_TO_DEG)
        vmin, vmax = _compute_vmin_vmax(image_ma)
        inc_deg = (
            float(result_m.mean_incidence * _RAD_TO_DEG)
            if np.isfinite(result_m.mean_incidence)
            else None
        )
        lon_origin_deg, lon_hi, lon_global_bins = _ring_longitude_column_origin_and_extent_hi_deg(
            n_cols=n_lon,
            lon_res_rad=result_m.longitude_resolution,
            longitude_antimask=result_m.longitude_antimask,
            longitude_range=result_m.longitude_range,
        )
        return RingDisplayData(
            title=title,
            image_ma=image_ma,
            longitude_resolution_deg=lon_res_deg,
            radius_resolution_km=rad_res_km,
            radius_inner=result_m.radius_inner,
            radius_outer=result_m.radius_outer,
            n_radii=n_radii,
            n_longitude=n_lon,
            mean_radial_resolution=mrr,
            mean_angular_resolution=mar,
            mean_phase=mphase,
            mean_emission=memission,
            image_number=result_m.image_number,
            orbit_model_name=orbit_model_name,
            orbit_model=orbit_model,
            vmin=vmin,
            vmax=vmax,
            is_mosaic=True,
            longitude_antimask=result_m.longitude_antimask,
            body_name=result_m.body_name,
            mean_incidence_deg=inc_deg,
            photometric_model_name=result_m.photometric_model_name,
            observation_time_tdb=result_m.time,
            longitude_column_origin_deg=lon_origin_deg,
            longitude_extent_hi_deg=lon_hi,
            contributing_image_names=result_m.contributing_image_names,
            longitude_global_bins=lon_global_bins,
        )

    raise ValueError(f'Expected RingReprojResult or RingMosaicData in {path!r}, got kind={kind!r}')


# ---------------------------------------------------------------------------
# Body DisplayData
# ---------------------------------------------------------------------------


@dataclass
class BodyDisplayData:
    """All display data extracted from a body reprojection or mosaic file.

    Parameters:
        title: Short title for the window bar (filename stem).
        image_ma: 2-D masked array (n_lat, n_lon), row 0 = first latitude bin.
        lat_resolution_deg: Row pitch in degrees.
        lon_resolution_deg: Column pitch in degrees.
        lat_range_deg: (min_lat, max_lat) in degrees.
        lon_range_deg: (min_lon, max_lon) in degrees.
        latlon_type: Coordinate type ('centric', 'graphic', 'squashed').
        lon_direction: Longitude direction ('east', 'west').
        resolution: Per-pixel resolution (km/pixel) as masked array.
        eff_resolution: Per-pixel effective resolution as masked array.
        phase: Per-pixel phase angle (deg) as masked array.
        emission: Per-pixel emission angle (deg) as masked array.
        incidence: Per-pixel incidence angle (deg) as masked array.
        image_number: Per-pixel uint16 (mosaic only; None for reproj).
        body_name: The body name (upper-case).
        vmin: Minimum valid image value.
        vmax: Maximum valid image value.
        is_mosaic: True if loaded from a BodyMosaicData.
        contributing_image_names: Names in ``image_number`` order (mosaic); for reproj,
            optional single entry when ``image_name`` was stored on save.
        observation_time_tdb: Per-pixel TDB seconds past J2000 (same shape as ``image_ma``),
            or ``None`` if unavailable. Built from ``BodyReprojResult.time`` for reproj files.
        photometric_model_name: Model applied when the file was written, if any.
        sub_solar_lon_per_image_deg: Sub-solar longitude (deg) indexed by image number.
            Single-element for reproj; one entry per contributing image for mosaics.
            Empty array if not available.
        sub_solar_lat_per_image_deg: Sub-solar latitude (deg), same indexing.
        sub_observer_lon_per_image_deg: Sub-observer longitude (deg), same indexing as
            the sub-solar arrays.
        sub_observer_lat_per_image_deg: Sub-observer latitude (deg), same indexing.
    """

    title: str
    image_ma: ma.MaskedArray
    lat_resolution_deg: float
    lon_resolution_deg: float
    lat_range_deg: tuple[float, float]
    lon_range_deg: tuple[float, float]
    latlon_type: str
    lon_direction: str
    resolution: ma.MaskedArray
    eff_resolution: ma.MaskedArray
    phase: ma.MaskedArray
    emission: ma.MaskedArray
    incidence: ma.MaskedArray
    image_number: ma.MaskedArray | None
    body_name: str
    vmin: float
    vmax: float
    is_mosaic: bool
    photometric_model_name: str | None = None
    contributing_image_names: tuple[str, ...] = ()
    observation_time_tdb: ma.MaskedArray | None = None
    sub_solar_lon_per_image_deg: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    sub_solar_lat_per_image_deg: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    sub_observer_lon_per_image_deg: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )
    sub_observer_lat_per_image_deg: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float64)
    )


def load_body_file(path: str) -> BodyDisplayData:
    """Load a body reprojection or mosaic file and return ``BodyDisplayData``.

    Parameters:
        path: Local file path.

    Returns:
        ``BodyDisplayData`` ready to be passed to ``BodyMosaicWindow``.

    Raises:
        ValueError: If the file kind is not a supported body type.
    """
    title = Path(str(path)).stem
    kind = _peek_kind(path)

    if kind == 'BodyReprojResult':
        result = BodyReprojResult.load(path)
        image_ma = result.img
        lat_res_deg = result.lat_resolution * _RAD_TO_DEG
        lon_res_deg = result.lon_resolution * _RAD_TO_DEG
        # Reconstruct spatial extent from idx_range (same convention as ``BodyMosaic.bounds``).
        lat_min = (result.lat_idx_range[0] * result.lat_resolution - math.pi / 2.0) * _RAD_TO_DEG
        lat_max = (result.lat_idx_range[1] * result.lat_resolution - math.pi / 2.0) * _RAD_TO_DEG
        lon_min = result.lon_idx_range[0] * result.lon_resolution * _RAD_TO_DEG
        lon_max = result.lon_idx_range[1] * result.lon_resolution * _RAD_TO_DEG
        vmin, vmax = _compute_vmin_vmax(image_ma)
        obs_time_tdb = ma.MaskedArray(
            np.full(image_ma.shape, float(result.time), dtype=np.float64),
            mask=ma.getmaskarray(image_ma),
        )
        return BodyDisplayData(
            title=title,
            image_ma=image_ma,
            lat_resolution_deg=lat_res_deg,
            lon_resolution_deg=lon_res_deg,
            lat_range_deg=(lat_min, lat_max),
            lon_range_deg=(lon_min, lon_max),
            latlon_type=result.latlon_type,
            lon_direction=result.lon_direction,
            resolution=result.resolution,
            eff_resolution=result.eff_resolution,
            phase=_rad_to_deg_ma(result.phase),
            emission=_rad_to_deg_ma(result.emission),
            incidence=_rad_to_deg_ma(result.incidence),
            image_number=None,
            body_name=result.body_name,
            vmin=vmin,
            vmax=vmax,
            is_mosaic=False,
            photometric_model_name=result.photometric_model_name,
            contributing_image_names=(result.image_name,) if result.image_name else (),
            observation_time_tdb=obs_time_tdb,
            sub_solar_lon_per_image_deg=np.array(
                [result.sub_solar_lon * _RAD_TO_DEG], dtype=np.float64
            ),
            sub_solar_lat_per_image_deg=np.array(
                [result.sub_solar_lat * _RAD_TO_DEG], dtype=np.float64
            ),
            sub_observer_lon_per_image_deg=np.array(
                [result.sub_observer_lon * _RAD_TO_DEG], dtype=np.float64
            ),
            sub_observer_lat_per_image_deg=np.array(
                [result.sub_observer_lat * _RAD_TO_DEG], dtype=np.float64
            ),
        )

    if kind == 'BodyMosaicData':
        result_m = BodyMosaicData.load(path)
        image_ma = result_m.img
        lat_res_deg = result_m.lat_resolution * _RAD_TO_DEG
        lon_res_deg = result_m.lon_resolution * _RAD_TO_DEG
        lat_range_deg = (
            result_m.lat_range[0] * _RAD_TO_DEG,
            result_m.lat_range[1] * _RAD_TO_DEG,
        )
        lon_range_deg = (
            result_m.lon_range[0] * _RAD_TO_DEG,
            result_m.lon_range[1] * _RAD_TO_DEG,
        )
        vmin, vmax = _compute_vmin_vmax(image_ma)
        return BodyDisplayData(
            title=title,
            image_ma=image_ma,
            lat_resolution_deg=lat_res_deg,
            lon_resolution_deg=lon_res_deg,
            lat_range_deg=lat_range_deg,
            lon_range_deg=lon_range_deg,
            latlon_type=result_m.latlon_type,
            lon_direction=result_m.lon_direction,
            resolution=result_m.resolution,
            eff_resolution=result_m.eff_resolution,
            phase=_rad_to_deg_ma(result_m.phase),
            emission=_rad_to_deg_ma(result_m.emission),
            incidence=_rad_to_deg_ma(result_m.incidence),
            image_number=result_m.image_number,
            body_name=result_m.body_name,
            vmin=vmin,
            vmax=vmax,
            is_mosaic=True,
            photometric_model_name=result_m.photometric_model_name,
            contributing_image_names=result_m.contributing_image_names,
            observation_time_tdb=result_m.time,
            sub_solar_lon_per_image_deg=result_m.sub_solar_lon_per_image * _RAD_TO_DEG,
            sub_solar_lat_per_image_deg=result_m.sub_solar_lat_per_image * _RAD_TO_DEG,
            sub_observer_lon_per_image_deg=result_m.sub_observer_lon_per_image * _RAD_TO_DEG,
            sub_observer_lat_per_image_deg=result_m.sub_observer_lat_per_image * _RAD_TO_DEG,
        )

    raise ValueError(f'Expected BodyReprojResult or BodyMosaicData in {path!r}, got kind={kind!r}')
