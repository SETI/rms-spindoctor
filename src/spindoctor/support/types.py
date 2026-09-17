"""Shared type aliases and protocols used across the nav package.

This module defines the numpy-array aliases (``NDArrayAnyType``,
``NDArrayBoolType``,
``NDArrayFloatType``, ``NDArrayIntType``, ``NDArrayUint8Type``,
``NDArrayUint32Type``, ``NDArrayType``), the generic ``NPType`` type
variable, the ``PathLike`` union accepted by I/O helpers, and the
``MutableStar`` protocol describing the in-memory star-record shape
used by the star-catalog reduction code.

Centralizing these aliases keeps every import site aligned on a single
spelling for the heavily-used numpy types and lets a downstream module
narrow them in one place.
"""

from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
from filecache import FCPath

NDArrayLike = npt.ArrayLike
DTypeLike = npt.DTypeLike
NDArrayAnyType = npt.NDArray[Any]
NDArrayBoolType = npt.NDArray[np.bool_]
NDArrayComplexType = npt.NDArray[np.complexfloating[Any, Any]]
NDArrayFloatType = npt.NDArray[np.floating[Any]]
NDArrayIntType = npt.NDArray[np.integer[Any]]
NDArrayUint8Type = npt.NDArray[np.uint8]
NDArrayUint32Type = npt.NDArray[np.uint32]
NPType = TypeVar('NPType', bound=np.generic, covariant=True)
NDArrayType = npt.NDArray[NPType]

PathLike = str | Path | FCPath


class MutableStar(Protocol):
    """In-memory star record shared by the catalog reduction and the simulator.

    ``v`` and ``u`` are the star's position in **pixel corner coordinates** in
    the nominal (unpadded) field of view -- an integer names a boundary between
    two pixels, not a pixel.  The same point in pixel-centric coordinates is
    ``v - PIXEL_CENTER_TO_CORNER_PX``,
    ``u - PIXEL_CENTER_TO_CORNER_PX`` (see
    :data:`~spindoctor.support.constants.PIXEL_CENTER_TO_CORNER_PX`).  It is the
    convention the geometry layer speaks, which is why the record holds it
    rather than converting at the producer.  ``move_v`` / ``move_u`` are
    per-exposure displacements, so they need no conversion.
    """

    unique_number: int | None
    catalog_name: str
    pretty_name: str
    name: str

    # Image-space location and motion (see the class docstring: ``v``/``u``
    # are pixel-corner, not pixel-centric).
    v: float
    u: float
    move_v: float
    move_u: float

    # Photometry and spectral info
    vmag: float | None
    b_v: float | None
    johnson_mag_v: float | None
    johnson_mag_b: float | None
    johnson_mag_faked: bool
    # Bright-end saturation provenance (see nav_model.stars.saturation):
    # ``photometry_corrected`` is set when ``vmag`` was replaced by a YBSC or
    # Tycho-2 reference value; ``photometry_saturated`` marks a bright record
    # whose catalog magnitude is unreliable and potentially too faint (no
    # reference in either catalog).
    photometry_corrected: bool
    photometry_saturated: bool
    spectral_class: str | None
    temperature: float | None
    temperature_faked: bool

    # Proper motion and current RA/DEC
    ra: float | None
    dec: float | None
    ra_pm: float
    dec_pm: float

    # Additional fields used during processing
    psf_size: tuple[int, int]
    dn: float
    conflicts: str
    diff_u: float
    diff_v: float

    def ra_dec_with_pm(self, tdb: float) -> tuple[float, float] | tuple[None, None]: ...
