"""Symmetry-breaking structure of a rendered haze layer.

The base haze of :mod:`spindoctor.sim.forward.atmosphere` is exactly
mirror-symmetric about the image-plane line through the body center and the
sub-solar direction: one exponential column, one illumination weight, no
azimuthal or hemispheric structure.  A navigator that measures pointing FROM
that symmetry would therefore be graded against a scene built from its own
assumption.  This module supplies the optional structure that breaks it --
each field a truth key of the body's ``atmosphere`` block, none of it visible
through ``nav_params``:

- ``axis_tilt_deg`` rotates the haze's own illumination axis away from the
  geometric sun direction the navigator predicts, so the true mirror plane
  is not the predicted one;
- ``ns_falloff_ratio`` scales the haze falloff length on one hemisphere
  only, a genuinely NON-affine difference between the two halves the mirror
  maps onto each other (a Pearson score is blind to an affine one, so this
  is the axis that actually probes it);
- ``sector_sharpness_gradient`` scales the falloff length with azimuth
  around the limb, which walks the detected limb ridge radially as a
  function of ray angle -- the sector-asymmetric edge-localization bias;
- ``ns_asymmetry_amplitude`` scales one hemisphere's brightness, the affine
  counterpart the Pearson score is supposed to absorb;
- ``interior_ramp_amplitude`` adds a linear brightness ramp along the haze
  axis inside the disc, the seasonal north-south gradient;
- ``cloud_blobs`` add Gaussian clouds on the disc.

Every field is optional and the whole module is bypassed when a body's
``atmosphere`` block names none of them: :func:`haze_structure_from_params`
returns None, and the haze renders through exactly the arithmetic it did
before this structure existed.  That gating is a performance contract, not
only a tidiness one -- the per-pixel scale-height field costs an array where
the base haze costs a scalar.

Geometry conventions.  The hemispheres are the two sides of the body's first
semi-axis in the body frame (``v_rot < 0`` and ``v_rot > 0``), i.e. the
rotation of the disc by ``rotation_z``, which is the sim's stand-in for a
polar axis.  A scene that wants the hemispheric split to be the one the
mirror maps onto itself orients its illumination perpendicular to that axis
(``illumination_angle`` 90 with ``rotation_z`` 0, as the committed haze
scenes do).  Azimuth ``phi`` is measured in the image plane about the disc
center and compared against the haze illumination direction, so the
sharpness gradient is symmetric about the sunward axis.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'CLOUD_BLOB_KEYS',
    'HAZE_STRUCTURE_KEYS',
    'SECTOR_REFERENCE_HALF_ANGLE_DEG',
    'CloudBlob',
    'HazeStructure',
    'apply_disc_structure',
    'apply_hemispheric_scaling',
    'haze_structure_from_params',
    'scale_height_field',
    'tilted_illumination_2d',
]


SECTOR_REFERENCE_HALF_ANGLE_DEG: float = 60.0
"""Azimuthal half-width the sharpness gradient is expressed against.

``sector_sharpness_gradient`` states the fractional change in falloff length
between the sunward point of the limb and the edge of a sector this wide, so
a gradient of 1.0 doubles the falloff length there.  A fixed sim-side
reference, deliberately not read from the navigator's sector configuration:
the rendered scene is nature, and it must not move when a navigation
threshold is retuned.
"""


HAZE_STRUCTURE_KEYS: frozenset[str] = frozenset(
    {
        'interior_ramp_amplitude',
        'ns_asymmetry_amplitude',
        'ns_falloff_ratio',
        'axis_tilt_deg',
        'sector_sharpness_gradient',
        'cloud_blobs',
    }
)
"""The ``atmosphere`` keys this module consumes.

A body's atmosphere block naming none of these renders through the base
haze path with no per-pixel structure cost at all.
"""


CLOUD_BLOB_KEYS: frozenset[str] = frozenset({'center_vu', 'sigma_px', 'amplitude'})
"""Keys of one ``cloud_blobs`` entry.

Public and defined here, beside the :class:`CloudBlob` that consumes them,
so the scene validator and the renderer cannot disagree about what a cloud
entry may carry.
"""


# Largest azimuth the sector term is evaluated at, as a multiple of the
# reference half-angle: the anti-sunward point, half a turn from the axis.
# Both falloff-factor extremes live there, not at the sector edge the
# gradient is quoted against.
_MAX_SECTOR_AZIMUTH_RATIO: float = 180.0 / SECTOR_REFERENCE_HALF_ANGLE_DEG


# Floor on the falloff-length multiplier.  A hemispheric ratio near zero, or a
# sharpness gradient near the negative bound the validator allows, drives the
# multiplier toward zero at the far side of the disc, and a falloff length of
# a thousandth of a pixel makes every exponent in the column arithmetic
# enormous.  A twentieth of the nominal length is far shorter than any limb
# the renderer resolves -- it renders as a hard edge, which is the intended
# limit of a vanishing scale height -- while keeping the exponents finite.
_MIN_SCALE_FACTOR: float = 0.05

# Below this radius a cloud blob is a single-pixel spike rather than a
# resolved cloud; the Gaussian is evaluated over a box this many sigma wide,
# so the floor also bounds the box to at least a pixel.
_MIN_BLOB_SIGMA_PX: float = 1.0e-3
_BLOB_BOX_SIGMAS: float = 4.0


@dataclass(frozen=True)
class CloudBlob:
    """One Gaussian cloud painted on the haze disc.

    Parameters:
        center_v: Cloud center offset from the body center along v, in grid
            pixels.
        center_u: Cloud center offset from the body center along u, in grid
            pixels.
        sigma_px: Gaussian standard deviation in grid pixels (> 0).
        amplitude: Peak brightness added at the cloud center, in the [0, 1]
            signal plane; negative darkens.
    """

    center_v: float
    center_u: float
    sigma_px: float
    amplitude: float


@dataclass(frozen=True)
class HazeStructure:
    """The symmetry-breaking structure of one body's haze.

    All pixel quantities are on the render grid the haze is composited on
    (the oversampled grid when the scene oversamples), matching the body's
    already-scaled semi-axes.

    Parameters:
        interior_ramp_amplitude: Peak brightness of a linear ramp along the
            haze axis inside the disc, added at the sunward limb and
            subtracted at the anti-sunward one, in the [0, 1] signal plane.
        ns_asymmetry_amplitude: Fractional brightness scaling applied to the
            positive-``v_rot`` hemisphere (an affine difference between the
            two halves).
        ns_falloff_ratio: Multiplier on the haze falloff length over that
            same hemisphere (a non-affine difference).
        axis_tilt_deg: Rotation of the haze's illumination axis away from
            the body's geometric sun direction, in degrees.
        sector_sharpness_gradient: Fractional change in falloff length at
            :data:`SECTOR_REFERENCE_HALF_ANGLE_DEG` from the sunward axis,
            growing linearly with azimuth away from it.
        cloud_blobs: Gaussian clouds painted on the disc.
    """

    interior_ramp_amplitude: float = 0.0
    ns_asymmetry_amplitude: float = 0.0
    ns_falloff_ratio: float = 1.0
    axis_tilt_deg: float = 0.0
    sector_sharpness_gradient: float = 0.0
    cloud_blobs: tuple[CloudBlob, ...] = ()

    @property
    def scales_falloff(self) -> bool:
        """Whether any field makes the falloff length vary across the disc."""
        return self.ns_falloff_ratio != 1.0 or self.sector_sharpness_gradient != 0.0

    @property
    def max_falloff_factor(self) -> float:
        """The largest falloff-length multiplier this structure can produce.

        The haze band the renderer evaluates over is sized from the falloff
        length, so it must be sized from the LONGEST one anywhere on the
        disc; a band sized from the nominal length would clip the extended
        hemisphere's glow at an artificial edge.

        The sector term peaks at the anti-sunward point, where the azimuth
        reaches ``pi``, so a positive gradient is evaluated there rather
        than at the reference sector edge.
        """
        hemispheric = max(1.0, self.ns_falloff_ratio)
        sector = max(1.0, 1.0 + self.sector_sharpness_gradient * _MAX_SECTOR_AZIMUTH_RATIO)
        return max(_MIN_SCALE_FACTOR, hemispheric * sector)

    @property
    def min_falloff_factor(self) -> float:
        """The smallest falloff-length multiplier this structure can produce.

        The on-disc band's inner edge is sized from the DEEPEST vertical
        column, and a shorter falloff length means a deeper one, so that
        edge needs the smallest multiplier rather than the largest.  Both
        structure terms can shorten the length: a hemispheric ratio below
        one, and a negative sharpness gradient (deepest at the anti-sunward
        point, where the azimuth reaches ``pi``).
        """
        hemispheric = min(1.0, self.ns_falloff_ratio)
        sector = min(1.0, 1.0 + self.sector_sharpness_gradient * _MAX_SECTOR_AZIMUTH_RATIO)
        return max(_MIN_SCALE_FACTOR, hemispheric * sector)


def _blob_from_params(entry: dict[str, Any], *, os_factor: int) -> CloudBlob:
    """Build one :class:`CloudBlob` from a scene ``cloud_blobs`` entry."""
    center = entry.get('center_vu', (0.0, 0.0))
    return CloudBlob(
        center_v=float(center[0]) * os_factor,
        center_u=float(center[1]) * os_factor,
        sigma_px=max(float(entry['sigma_px']) * os_factor, _MIN_BLOB_SIGMA_PX),
        amplitude=float(entry['amplitude']),
    )


def haze_structure_from_params(
    atmosphere: dict[str, Any], *, oversample: int
) -> HazeStructure | None:
    """Build a :class:`HazeStructure` from a body's ``atmosphere`` block.

    Returns None when the block names none of :data:`HAZE_STRUCTURE_KEYS`,
    which is what keeps a plain haze on exactly the arithmetic it had before
    this structure existed.  Pixel lengths are scaled to the render grid by
    ``oversample`` exactly as the body's axes are; the amplitudes, ratios,
    and angles are dimensionless and are not.

    Parameters:
        atmosphere: One body's ``atmosphere`` mapping.
        oversample: The render grid's oversampling factor.

    Returns:
        The scaled structure, or None when the block carries no structure
        keys.
    """
    if not HAZE_STRUCTURE_KEYS & set(atmosphere):
        return None
    os_factor = max(1, int(oversample))
    blobs = atmosphere.get('cloud_blobs') or ()
    return HazeStructure(
        interior_ramp_amplitude=float(atmosphere.get('interior_ramp_amplitude', 0.0)),
        ns_asymmetry_amplitude=float(atmosphere.get('ns_asymmetry_amplitude', 0.0)),
        ns_falloff_ratio=float(atmosphere.get('ns_falloff_ratio', 1.0)),
        axis_tilt_deg=float(atmosphere.get('axis_tilt_deg', 0.0)),
        sector_sharpness_gradient=float(atmosphere.get('sector_sharpness_gradient', 0.0)),
        cloud_blobs=tuple(_blob_from_params(dict(b), os_factor=os_factor) for b in blobs),
    )


def scale_height_field(
    structure: HazeStructure | None,
    scale_height_px: float,
    *,
    v_rot: NDArrayFloatType,
    v_ctr: NDArrayFloatType,
    u_ctr: NDArrayFloatType,
    illum_v: float,
    illum_u: float,
) -> float | NDArrayFloatType:
    """Return the per-pixel haze falloff length over a set of pixels.

    A plain scalar when the structure varies nothing (the base haze, and the
    common case), so the caller's exponential stays scalar arithmetic; an
    array of the same shape as ``v_rot`` when a hemispheric ratio or a
    sharpness gradient applies.

    The hemispheric term keys on the sign of ``v_rot`` (the body-frame
    first-axis coordinate), and the sector term on the image-plane azimuth
    between each pixel and the haze illumination direction, so a pixel on
    the sunward axis keeps the nominal length and one
    :data:`SECTOR_REFERENCE_HALF_ANGLE_DEG` away carries the full gradient.

    Parameters:
        structure: The haze structure, or None for the base haze.
        scale_height_px: Nominal falloff length in grid pixels.
        v_rot: Body-frame first-axis coordinate of each pixel.
        v_ctr: Image-frame v offset of each pixel from the body center.
        u_ctr: Image-frame u offset of each pixel from the body center.
        illum_v: V component of the (tilted) in-plane illumination direction.
        illum_u: U component of the same direction.

    Returns:
        The nominal scale height, or the per-pixel field.
    """
    if structure is None or not structure.scales_falloff:
        return scale_height_px
    factor: NDArrayFloatType = np.ones_like(v_rot)
    if structure.ns_falloff_ratio != 1.0:
        factor = np.where(v_rot > 0.0, structure.ns_falloff_ratio, 1.0)
    if structure.sector_sharpness_gradient != 0.0:
        rho = np.maximum(np.hypot(v_ctr, u_ctr), 1e-9)
        cos_phi = np.clip((v_ctr * illum_v + u_ctr * illum_u) / rho, -1.0, 1.0)
        angle = np.arccos(cos_phi)
        reference = math.radians(SECTOR_REFERENCE_HALF_ANGLE_DEG)
        factor = factor * (1.0 + structure.sector_sharpness_gradient * angle / reference)
    field: NDArrayFloatType = scale_height_px * np.maximum(factor, _MIN_SCALE_FACTOR)
    return field


def apply_hemispheric_scaling(
    values: NDArrayFloatType, structure: HazeStructure | None, v_rot: NDArrayFloatType
) -> NDArrayFloatType:
    """Scale one hemisphere's brightness by the affine asymmetry amplitude.

    Multiplies the positive-``v_rot`` half by ``1 + ns_asymmetry_amplitude``
    and leaves the other half alone, so the two halves the mirror maps onto
    each other differ by a pure scaling -- the relation a Pearson symmetry
    score is invariant to, which is exactly what makes this axis the control
    against the non-affine ones.

    Parameters:
        values: Brightness values to scale, modified in place and returned.
        structure: The haze structure, or None for the base haze.
        v_rot: Body-frame first-axis coordinate of the same pixels.

    Returns:
        The scaled values (the input array).
    """
    if structure is None or structure.ns_asymmetry_amplitude == 0.0:
        return values
    values[v_rot > 0.0] *= 1.0 + structure.ns_asymmetry_amplitude
    return values


def apply_disc_structure(
    out_box: NDArrayFloatType,
    structure: HazeStructure | None,
    *,
    disc: NDArrayBoolType,
    v_ctr: NDArrayFloatType,
    u_ctr: NDArrayFloatType,
    r_mean: float,
    illum_v: float,
    illum_u: float,
) -> None:
    """Add the interior ramp and the cloud blobs to a body's disc radiance.

    Both are painted only where the disc render already painted, so neither
    can create light outside the silhouette (a cloud is on the body, and the
    seasonal ramp is a property of the illuminated disc).  The result is
    clipped back into the [0, 1] signal plane.

    Parameters:
        out_box: Body radiance over the haze bounding box, modified in place.
        structure: The haze structure, or None for the base haze.
        disc: Boolean mask of the box pixels the body silhouette painted.
        v_ctr: Image-frame v offset of each box pixel from the body center.
        u_ctr: Image-frame u offset of each box pixel from the body center.
        r_mean: Mean apparent body radius in grid pixels.
        illum_v: V component of the (tilted) in-plane illumination direction.
        illum_u: U component of the same direction.
    """
    if structure is None:
        return
    if structure.interior_ramp_amplitude == 0.0 and not structure.cloud_blobs:
        return
    extra: NDArrayFloatType = np.zeros(disc.shape, dtype=np.float64)
    if structure.interior_ramp_amplitude != 0.0:
        # The ramp runs along the haze axis: +amplitude at the sunward limb,
        # -amplitude at the anti-sunward one, linear in between.
        axial = (v_ctr * illum_v + u_ctr * illum_u) / max(r_mean, 1e-6)
        extra = extra + structure.interior_ramp_amplitude * np.clip(axial, -1.0, 1.0)
    for blob in structure.cloud_blobs:
        _add_blob(extra, blob, v_ctr=v_ctr, u_ctr=u_ctr)
    out_box[disc] = np.clip(out_box[disc] + extra[disc], 0.0, 1.0)


def _add_blob(
    extra: NDArrayFloatType,
    blob: CloudBlob,
    *,
    v_ctr: NDArrayFloatType,
    u_ctr: NDArrayFloatType,
) -> None:
    """Add one Gaussian cloud to a box-shaped additive brightness field.

    Evaluated over the blob's own few-sigma sub-box, so a small cloud on a
    large disc costs its own footprint rather than the whole box.

    Parameters:
        extra: Box-shaped additive field, modified in place.
        blob: The cloud to add.
        v_ctr: Image-frame v offsets of the box rows from the body center,
            shaped ``(rows, 1)``.
        u_ctr: Image-frame u offsets of the box columns from the body
            center, shaped ``(1, cols)``.
    """
    v_axis = v_ctr[:, 0]
    u_axis = u_ctr[0, :]
    reach = _BLOB_BOX_SIGMAS * blob.sigma_px
    v_hits = np.flatnonzero(np.abs(v_axis - blob.center_v) <= reach)
    u_hits = np.flatnonzero(np.abs(u_axis - blob.center_u) <= reach)
    if v_hits.size == 0 or u_hits.size == 0:
        return
    v_lo, v_hi = int(v_hits[0]), int(v_hits[-1]) + 1
    u_lo, u_hi = int(u_hits[0]), int(u_hits[-1]) + 1
    dv = (v_axis[v_lo:v_hi] - blob.center_v)[:, None]
    du = (u_axis[u_lo:u_hi] - blob.center_u)[None, :]
    inv_two_sigma2 = 0.5 / (blob.sigma_px * blob.sigma_px)
    extra[v_lo:v_hi, u_lo:u_hi] += blob.amplitude * np.exp(-(dv * dv + du * du) * inv_two_sigma2)


def tilted_illumination_2d(
    structure: HazeStructure | None, illumination_angle: float
) -> tuple[float, float]:
    """Return the in-plane haze illumination direction, tilt included.

    The base haze lights from the body's own geometric sun direction; a
    structure carrying ``axis_tilt_deg`` rotates that direction, so the
    haze's mirror plane no longer coincides with the one a navigator derives
    from the predicted sub-solar point.

    Parameters:
        structure: The haze structure, or None for the base haze.
        illumination_angle: The body's in-plane light direction in radians
            (0 = from the top of the image).

    Returns:
        The ``(v, u)`` components of the unit in-plane direction toward the
        light, using the renderer's convention that v increases downward.
    """
    tilt = 0.0 if structure is None else math.radians(structure.axis_tilt_deg)
    angle = illumination_angle + tilt
    return -math.cos(angle), math.sin(angle)
