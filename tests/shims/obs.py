"""Fake ``ObsSnapshot`` for unit-testing nav-pipeline code paths.

Combines the backplane shim (:class:`tests.shims.backplane.FakeBackplane`)
with a numpy-array image-side surface so the navigation pipeline can be
driven end-to-end without a real ``oops.Observation``, SPICE kernels,
or PDS3 holdings.

Tests construct a :class:`FakeObs` with any subset of:

- A sensor-area image plus an extfov margin (the shim derives the
  extfov-shaped ``extdata`` from the two).
- A ``FakeBackplane`` for ``ext_bp``.
- A per-body inventory dict (mapping body name to the ``unclipped``
  bounding-box record :meth:`oops.Observation.inventory` returns).
- A PSF object exposing ``.sigma`` / ``fwhm()`` for ``star_psf``.
- A magnitude window and RA/DEC limits.

Methods that the code under test does not call do not need to be
populated.  Methods that are called but that have no plausible
fallback raise :class:`NotImplementedError` naming the missing wiring
so the test failure is informative.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import polymath

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from tests.shims.backplane import FakeBackplane

__all__ = [
    'FakeFOV',
    'FakeMeshgrid',
    'FakeObs',
    'FakePSF',
    'FakeUV',
]


class FakePSF:
    """PSF stand-in exposing the ``sigma`` attribute the predictor reads.

    Parameters:
        sigma: Per-pixel Gaussian sigma in pixels.
    """

    def __init__(self, sigma: float = 1.0) -> None:
        self.sigma = sigma


class FakeFOV:
    """Stand-in for ``oops.FOV`` accepted by ``Meshgrid.for_fov``."""


class FakeMeshgrid:
    """Stand-in for ``oops.Meshgrid``.

    Two construction forms mirror the real class: the rectangular
    ``for_fov`` factory (origin / limit / oversample), and the scattered
    ``FakeMeshgrid(fov, uv_pair)`` form the silhouette probe uses, where
    ``uv_pair`` carries explicit ``(u, v)`` probe positions.  Fake
    backplanes read ``origin`` / ``limit`` / ``oversample`` for the grid
    form and ``uv`` (via :func:`probe_grid_vu`) for the scattered form.
    """

    def __init__(
        self,
        fov: FakeFOV | None = None,
        uv_pair: Any | None = None,
        *,
        origin: tuple[float, float] | None = None,
        limit: tuple[float, float] | None = None,
        oversample: tuple[int, int] | None = None,
        swap: bool = False,
    ) -> None:
        del fov
        self.uv = uv_pair
        self.origin = origin
        self.limit = limit
        self.oversample = oversample
        self.swap = swap

    @classmethod
    def for_fov(
        cls,
        fov: FakeFOV,
        *,
        origin: tuple[float, float],
        limit: tuple[float, float],
        oversample: tuple[int, int] = (1, 1),
        swap: bool = False,
    ) -> FakeMeshgrid:
        """Mirror the ``oops.Meshgrid.for_fov`` factory signature."""
        del fov
        return cls(origin=origin, limit=limit, oversample=oversample, swap=swap)


def probe_grid_vu(
    meshgrid: FakeMeshgrid,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(vv, uu)`` for a scattered probe meshgrid, or ``None`` for a grid.

    The silhouette probe builds its meshgrid from explicit ``(u, v)`` pairs;
    fake backplanes call this first and fall back to their rectangular-grid
    coordinates when it returns ``None``.

    Parameters:
        meshgrid: The fake meshgrid handed to the backplane.

    Returns:
        ``(vv, uu)`` coordinate arrays of the probe positions, or ``None``.
    """
    if meshgrid.uv is None:
        return None
    vals = np.asarray(meshgrid.uv.vals, np.float64)
    return vals[..., 1], vals[..., 0]


@dataclass
class FakeUV:
    """Stand-in for the result of ``Observation.uv_from_ra_and_dec``.

    Parameters:
        u_vals: Array (or scalar) of U-pixel coordinates.
        v_vals: Array (or scalar) of V-pixel coordinates.
    """

    u_vals: np.ndarray
    v_vals: np.ndarray

    def to_scalars(self) -> tuple[polymath.Scalar, polymath.Scalar]:
        """Return ``(u_scalar, v_scalar)`` mirroring ``polymath`` UV results."""
        return polymath.Scalar(self.u_vals), polymath.Scalar(self.v_vals)


@dataclass
class FakeObs:
    """Numpy-array stand-in for an ``ObsSnapshotInst``.

    Parameters:
        data: Sensor-area image.  Required.
        extfov_margin_vu: ``(margin_v, margin_u)`` extfov-padding margin.
            Defaults to ``(0, 0)``.
        midtime: Observation midtime in TDB seconds.
        closest_planet: Upper-case planet name returned by
            ``obs.closest_planet``.
        ext_bp: :class:`FakeBackplane` for ``obs.ext_bp``.
        inventory_records: Per-body inventory dict; keyed by upper-case
            body name with each value mirroring the
            ``return_type='full'`` shape (``u_min_unclipped``,
            ``u_max_unclipped``, ``v_min_unclipped``,
            ``v_max_unclipped``, ``u_pixel_size``, ``v_pixel_size``,
            ``range``, ``center_uv``).
        psf: :class:`FakePSF` returned by ``star_psf``.
        psf_size: Tuple returned by ``star_psf_size``.
        star_min_vmag: Lower magnitude floor.
        star_max_vmag: Upper magnitude floor.
        ra_dec_limits_ext_rad: ``(ra_min, ra_max, dec_min, dec_max)``
            tuple returned by ``ra_dec_limits_ext()``.
        radec_to_uv: Optional callable mapping ``(ra, dec, tfrac)`` to
            ``(u, v)`` for ``uv_from_ra_and_dec``.  When unset every
            star projects onto the FOV centre.
        boresight_ra_rad: RA returned by ``boresight_ra()``.
        boresight_dec_rad: DEC returned by ``boresight_dec()``.
    """

    data: np.ndarray
    extfov_margin_vu: tuple[int, int] = (0, 0)
    midtime: float = 0.0
    closest_planet: str | None = None
    ext_bp: FakeBackplane | None = None
    inventory_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    psf: FakePSF = field(default_factory=FakePSF)
    psf_size: tuple[int, int] = (5, 5)
    star_min_vmag: float = 0.0
    star_max_vmag: float = 17.0
    ra_dec_limits_ext_rad: tuple[float, float, float, float] = (0.0, 0.1, 0.0, 0.1)
    radec_to_uv: Callable[[float, float, float], tuple[float, float]] | None = None
    boresight_ra_rad: float = 0.0
    boresight_dec_rad: float = 0.0
    fov: FakeFOV = field(default_factory=FakeFOV)

    def __post_init__(self) -> None:
        """Build the extfov-padded ``extdata`` from ``data`` plus the margin."""
        margin_v, margin_u = self.extfov_margin_vu
        if margin_v < 0 or margin_u < 0:
            raise ValueError(
                f'extfov_margin_vu must be non-negative; got {self.extfov_margin_vu!r}'
            )
        ext_shape = (
            self.data.shape[0] + 2 * margin_v,
            self.data.shape[1] + 2 * margin_u,
        )
        ext = np.zeros(ext_shape, dtype=self.data.dtype)
        ext[
            margin_v : margin_v + self.data.shape[0],
            margin_u : margin_u + self.data.shape[1],
        ] = self.data
        self._extdata = ext

    # ------------------------------------------------------------------
    # Image-side surface
    # ------------------------------------------------------------------

    @property
    def extdata(self) -> np.ndarray:
        """Return the extfov-padded image."""
        return self._extdata

    @property
    def data_shape_v(self) -> int:
        """Sensor-area V (rows)."""
        return int(self.data.shape[0])

    @property
    def data_shape_u(self) -> int:
        """Sensor-area U (cols)."""
        return int(self.data.shape[1])

    @property
    def data_shape_vu(self) -> tuple[int, int]:
        """Sensor-area ``(rows, cols)`` shape."""
        return (self.data.shape[0], self.data.shape[1])

    @property
    def data_shape_uv(self) -> tuple[int, int]:
        """Sensor-area ``(cols, rows)`` shape."""
        return (self.data.shape[1], self.data.shape[0])

    @property
    def extdata_shape_vu(self) -> tuple[int, int]:
        """Extfov ``(rows, cols)`` shape."""
        return (self._extdata.shape[0], self._extdata.shape[1])

    @property
    def extdata_shape_uv(self) -> tuple[int, int]:
        """Extfov ``(cols, rows)`` shape."""
        return (self._extdata.shape[1], self._extdata.shape[0])

    @property
    def extfov_margin_v(self) -> int:
        """Extfov margin along V."""
        return self.extfov_margin_vu[0]

    @property
    def extfov_margin_u(self) -> int:
        """Extfov margin along U."""
        return self.extfov_margin_vu[1]

    @property
    def extfov_v_min(self) -> int:
        """Minimum V coordinate in extfov coordinates (negative when margin > 0)."""
        return -self.extfov_margin_vu[0]

    @property
    def extfov_u_min(self) -> int:
        """Minimum U coordinate in extfov coordinates."""
        return -self.extfov_margin_vu[1]

    @property
    def extfov_v_max(self) -> int:
        """Maximum V coordinate in extfov coordinates."""
        return int(self.data.shape[0]) + self.extfov_margin_vu[0] - 1

    @property
    def extfov_u_max(self) -> int:
        """Maximum U coordinate in extfov coordinates."""
        return int(self.data.shape[1]) + self.extfov_margin_vu[1] - 1

    def make_extfov_zeros(self, dtype: Any = np.float64) -> np.ndarray:
        """Return a zero-filled array of the extfov shape."""
        return np.zeros(self.extdata_shape_vu, dtype=dtype)

    def make_extfov_false(self) -> np.ndarray:
        """Return a False-filled boolean array of the extfov shape."""
        return np.zeros(self.extdata_shape_vu, dtype=bool)

    def extract_offset_array(
        self, array: np.ndarray, offset: tuple[float, float] | tuple[int, int] | None
    ) -> np.ndarray:
        """Mirror ``ObsSnapshot.extract_offset_array``.

        Extracts the sensor-area window of an extfov-shaped array at the
        given ``(dv, du)`` offset; the portion of the window that falls
        outside the extfov is left zero-filled (``False`` for a boolean
        array), which is what the overlay compositor expects.

        Parameters:
            array: Extfov-shaped array to extract from.
            offset: ``(dv, du)`` offset; ``None`` means no offset.

        Returns:
            The sensor-area-shaped extraction.

        Raises:
            ValueError: If ``array`` is not exactly extfov-shaped, which
                includes a 3-D array whose leading two axes match -- the
                real method rejects those too.
        """
        if array.shape != self.extdata_shape_vu:
            raise ValueError(
                f'array shape {array.shape} must equal extdata shape {self.extdata_shape_vu}'
            )
        if offset is None:
            offset = (0, 0)
        v_size, u_size = self.extdata_shape_vu
        v0 = self.extfov_margin_v - int(np.round(offset[0]))
        u0 = self.extfov_margin_u - int(np.round(offset[1]))
        v1 = v0 + self.data_shape_v
        u1 = u0 + self.data_shape_u
        out = np.zeros((self.data_shape_v, self.data_shape_u, *array.shape[2:]), dtype=array.dtype)
        src_v_lo = max(0, v0)
        src_u_lo = max(0, u0)
        src_v_hi = min(v_size, v1)
        src_u_hi = min(u_size, u1)
        if src_v_hi <= src_v_lo or src_u_hi <= src_u_lo:
            return out
        out[
            src_v_lo - v0 : src_v_lo - v0 + (src_v_hi - src_v_lo),
            src_u_lo - u0 : src_u_lo - u0 + (src_u_hi - src_u_lo),
        ] = array[src_v_lo:src_v_hi, src_u_lo:src_u_hi]
        return out

    def extfov_data_sensor_mask(self) -> np.ndarray:
        """Return a boolean mask True over the sensor area inside extfov."""
        mask = np.zeros(self.extdata_shape_vu, dtype=bool)
        margin_v, margin_u = self.extfov_margin_vu
        mask[
            margin_v : margin_v + self.data.shape[0],
            margin_u : margin_u + self.data.shape[1],
        ] = True
        return mask

    def clip_extfov(self, u: int, v: int) -> tuple[int, int]:
        """Clip ``(u, v)`` to the extfov rectangle."""
        return (
            int(np.clip(u, self.extfov_u_min, self.extfov_u_max)),
            int(np.clip(v, self.extfov_v_min, self.extfov_v_max)),
        )

    def inventory(
        self, body_list: list[str], *, return_type: str = 'full'
    ) -> dict[str, dict[str, Any]]:
        """Return the configured inventory records for each requested body.

        Parameters:
            body_list: Bodies to look up.
            return_type: Accepted for API parity; the shim always
                returns the ``'full'`` shape.
        """
        del return_type
        out: dict[str, dict[str, Any]] = {}
        for body in body_list:
            key = body.upper()
            if key in self.inventory_records:
                out[key] = self.inventory_records[key]
        return out

    def inventory_body_in_extfov(self, inv: dict[str, Any]) -> bool:
        """Mirror ``ObsSnapshot.inventory_body_in_extfov``."""
        return bool(
            inv['u_max_unclipped'] >= self.extfov_u_min
            and inv['u_min_unclipped'] <= self.extfov_u_max
            and inv['v_max_unclipped'] >= self.extfov_v_min
            and inv['v_min_unclipped'] <= self.extfov_v_max
        )

    def inventory_body_in_fov(self, inv: dict[str, Any]) -> bool:
        """Mirror ``ObsSnapshot.inventory_body_in_fov``."""
        return bool(
            inv['u_max_unclipped'] >= 0
            and inv['u_min_unclipped'] < self.data.shape[1]
            and inv['v_max_unclipped'] >= 0
            and inv['v_min_unclipped'] < self.data.shape[0]
        )

    # ------------------------------------------------------------------
    # Star surface
    # ------------------------------------------------------------------

    def star_psf(self) -> FakePSF:
        """Return the PSF used for star rendering / SNR estimation."""
        return self.psf

    def star_psf_size(self, _star: object) -> tuple[int, int]:
        """Return the PSF size for any star (no per-star variation)."""
        return self.psf_size

    def star_min_usable_vmag(self) -> float:
        """Return the catalog-search lower magnitude floor."""
        return self.star_min_vmag

    def star_max_usable_vmag(self) -> float:
        """Return the catalog-search upper magnitude floor."""
        return self.star_max_vmag

    def ra_dec_limits_ext(self, apparent: bool = True) -> tuple[float, float, float, float]:
        """Return the configured RA / DEC limits in radians."""
        del apparent
        return self.ra_dec_limits_ext_rad

    # ------------------------------------------------------------------
    # SPICE-like surface
    # ------------------------------------------------------------------

    def boresight_ra(self) -> float:
        """Return the configured boresight RA in radians."""
        return self.boresight_ra_rad

    def boresight_dec(self) -> float:
        """Return the configured boresight DEC in radians."""
        return self.boresight_dec_rad

    def uv_from_ra_and_dec(
        self,
        ra: Any,
        dec: Any,
        *,
        tfrac: float = 0.5,
        apparent: bool = True,
    ) -> FakeUV:
        """Project RA/DEC into pixel coordinates.

        When ``self.radec_to_uv`` is set, the callable is invoked once per
        ``(ra, dec)`` point with ``(ra, dec, tfrac)`` and must return
        ``(u, v)``.  Otherwise the shim places every requested point at
        the FOV centre with a small ``tfrac``-driven shift so the
        smear-bracket calculation produces a deterministic non-zero
        displacement when desired.

        Parameters:
            ra: Scalar or polymath-Scalar RA.
            dec: Scalar or polymath-Scalar DEC.
            tfrac: Fraction along the exposure window.
            apparent: Accepted for API parity; ignored by the shim.
        """
        del apparent
        ra_arr = np.atleast_1d(np.asarray(_extract_vals(ra), dtype=np.float64))
        dec_arr = np.atleast_1d(np.asarray(_extract_vals(dec), dtype=np.float64))
        n = max(ra_arr.size, dec_arr.size)
        ra_b = np.broadcast_to(ra_arr, (n,))
        dec_b = np.broadcast_to(dec_arr, (n,))
        if self.radec_to_uv is not None:
            uv = [
                self.radec_to_uv(float(r), float(d), tfrac)
                for r, d in zip(ra_b, dec_b, strict=True)
            ]
            u = np.asarray([p[0] for p in uv], dtype=np.float64)
            v = np.asarray([p[1] for p in uv], dtype=np.float64)
            return FakeUV(u_vals=u, v_vals=v)
        v_centre = self.data.shape[0] / 2.0 + self.extfov_margin_vu[0]
        u_centre = self.data.shape[1] / 2.0 + self.extfov_margin_vu[1]
        u = np.full(n, u_centre)
        v = np.full(n, v_centre)
        # Apply a per-point shift driven by tfrac so the bracket
        # difference is non-zero when the test wants smear.
        u = u + (tfrac - 0.5) * 2.0  # -1 at tfrac=0, +1 at tfrac=1
        v = v + (tfrac - 0.5) * 1.0
        return FakeUV(u_vals=u, v_vals=v)


def _extract_vals(obj: Any) -> Any:
    """Return ``obj.vals`` when present, else ``obj`` unchanged."""
    return obj.vals if hasattr(obj, 'vals') else obj
