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
  counterpart the Pearson score is supposed to absorb -- affine only poleward
  of the equatorial blend band described below, and only where the scaled
  half stays below saturation;
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

The two hemispheric terms do not switch at the equator.  Each is weighted by
:func:`hemisphere_weight`, which rises smoothly from 0 to 1 across a band
:data:`HEMISPHERE_BLEND_HALF_WIDTH_DEG` either side of it, keyed on the
latitude of the haze column a pixel sees (:func:`latitude_sine`).  Poleward
of that band each term applies at exactly its stated value, so
``ns_asymmetry_amplitude`` and ``ns_falloff_ratio`` keep their documented
meaning there; inside it the haze changes continuously, as a real haze does,
where a switch would draw a seam across the halo along one image row.

The blend costs ``ns_asymmetry_amplitude`` its affine character inside the
band, and no continuous blend can avoid that.  The mirror maps latitude
``s`` onto ``-s``, so a pure scaling of one half needs the brightness ratio
of each mirror pair to be the same ``1 + amplitude`` everywhere.  On the
mirror line itself the two members of the pair are the same point, so a
continuous term gives them the same multiplier and their ratio is 1 there;
it climbs to ``1 + amplitude`` only at the band edge.  Only a switch keeps
the ratio constant, and a switch is the seam.  Where a scene puts the
hemispheric split along the sun axis, the band covers the sunward limb a
haze navigator scores on.

Poleward of the band the amplitude is a pure scaling only where the scaled
half stays below saturation.  The renderer clips the signal at 1, so a
positive amplitude flattens the bright crescent wherever the base brightness
times ``1 + ns_asymmetry_amplitude`` would pass 1.  Over 26 of the 40 clean
Titan truth scenes that navigate both with and without an amplitude of +0.35, the
amplitude shifts the recovered cross-track position by 0.063 pixels on
average with a switch at the equator and 0.065 pixels with a 10 degree band,
about four times what -0.35 costs with the band.  A switch carries nearly
all of that cost too, so the clip sets it and the band does not.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'CLOUD_BLOB_KEYS',
    'HAZE_STRUCTURE_KEYS',
    'HEMISPHERE_BLEND_HALF_WIDTH_DEG',
    'SECTOR_REFERENCE_HALF_ANGLE_DEG',
    'CloudBlob',
    'HazeStructure',
    'apply_disc_structure',
    'apply_hemispheric_scaling',
    'haze_structure_from_params',
    'hemisphere_weight',
    'latitude_sine',
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


HEMISPHERE_BLEND_HALF_WIDTH_DEG: float = 10.0
"""Latitude either side of the equator over which the hemispheric terms blend.

``ns_asymmetry_amplitude`` and ``ns_falloff_ratio`` apply in full poleward of
this latitude in the positive-``v_rot`` hemisphere and not at all poleward of
it in the other, rising smoothly in between (see :func:`hemisphere_weight`).

A fixed sim-side modeling choice, set by what the band costs.  Inside it
``ns_asymmetry_amplitude`` stops being a pure scaling of one half (see the
module docstring), and that cost grows with the width.  Over the 27 of 40
clean Titan truth scenes (hemispheric split on the sun axis) that navigate
both with and without an amplitude of -0.35, the amplitude shifts the
recovered cross-track position by 0.004 pixels on average with a switch,
0.016 pixels (largest 0.086) with a 10 degree band, and 0.031 pixels
(largest 0.176) with a 20 degree band.  The band must still be wide enough
to remove the seam.  At 10 degrees the brightness change across the
equator row of a 64 pixel radius haze is about the same as across the rows
beside it, a ratio of 0.95 to 1.03 where a switch gives 3 to 26; at 5
degrees the ratio for the brightness term rises to 1.24.  The band's pixel
width scales with the body, ``sin(10 deg)``, about 0.17, times the radius
each side of the equator: 10 pixels at a radius of 55, and 87 at a radius
of 500.

The band is not small.  For a sphere seen with its axis in the image plane
it covers 22% of the projected disc and 11% of the limb, together with the
halo above that part of the limb.
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
            two halves poleward of the equatorial blend band, where the
            scaled half stays below saturation).
        ns_falloff_ratio: Multiplier on the haze falloff length over that
            same hemisphere (a non-affine difference), blended across the
            equator the same way.
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
    def splits_hemispheres(self) -> bool:
        """Whether either hemispheric term differs from its identity value."""
        return self.ns_asymmetry_amplitude != 0.0 or self.ns_falloff_ratio != 1.0

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


def latitude_sine(
    v_rot: NDArrayFloatType, e2: NDArrayFloatType, *, semi_a: float
) -> NDArrayFloatType:
    """Return the sine of the latitude of the haze column each pixel sees.

    On the disc that is the pixel's own normalized first-axis coordinate,
    ``v_rot / semi_a``.  Above the limb the line of sight grazes the column
    standing over the limb point at the pixel's azimuth, whose coordinate is
    ``v_rot / (semi_a * sqrt(e2))``.  The two expressions agree on the limb
    (``e2`` = 1), so the coordinate is continuous across the silhouette, and
    for a sphere seen with its axis in the image plane it is exactly the
    sine of latitude.  It lies in [-1, 1].

    Parameters:
        v_rot: Body-frame first-axis coordinate of each pixel, in grid pixels.
        e2: Squared normalized ellipse radius of each pixel (1 on the limb).
        semi_a: The first semi-axis in grid pixels, the one ``v_rot`` is
            measured along.

    Returns:
        The sine of latitude, shaped like ``v_rot`` broadcast with ``e2``.
    """
    sin_lat: NDArrayFloatType = v_rot / (semi_a * np.sqrt(np.maximum(e2, 1.0)))
    return sin_lat


def hemisphere_weight(sin_latitude: NDArrayFloatType) -> NDArrayFloatType:
    """Return how fully each pixel belongs to the positive-``v_rot`` hemisphere.

    0 poleward of :data:`HEMISPHERE_BLEND_HALF_WIDTH_DEG` in the negative
    hemisphere, 1 poleward of it in the positive one, and a cubic smoothstep
    of the sine of latitude in between: exactly 0.5 on the equator, and
    continuous with a continuous slope everywhere, so a hemispheric term
    weighted by it can draw no seam.  The weight is point-symmetric about
    ``(0, 0.5)``: ``w(-s) = 1 - w(s)``.

    Parameters:
        sin_latitude: Sine of the latitude of each pixel's haze column (see
            :func:`latitude_sine`).

    Returns:
        The weight in [0, 1], shaped like ``sin_latitude``.
    """
    half_width = math.sin(math.radians(HEMISPHERE_BLEND_HALF_WIDTH_DEG))
    t = np.clip((sin_latitude + half_width) / (2.0 * half_width), 0.0, 1.0)
    weight: NDArrayFloatType = t * t * (3.0 - 2.0 * t)
    return weight


def scale_height_field(
    structure: HazeStructure | None,
    scale_height_px: float,
    *,
    weight: NDArrayFloatType | None,
    v_ctr: NDArrayFloatType,
    u_ctr: NDArrayFloatType,
    illum_v: float,
    illum_u: float,
) -> float | NDArrayFloatType:
    """Return the per-pixel haze falloff length over a set of pixels.

    A plain scalar when the structure varies nothing (the base haze, and the
    common case), so the caller's exponential stays scalar arithmetic; an
    array of the same shape as ``v_ctr`` when a hemispheric ratio or a
    sharpness gradient applies.

    The hemispheric term multiplies the length by
    ``1 + (ns_falloff_ratio - 1) * w``, ``w`` the :func:`hemisphere_weight`
    of the pixel's latitude, so the full ratio applies poleward of the blend
    band in the positive-``v_rot`` hemisphere, the nominal length poleward of
    it in the other, and the length changes continuously across the equator.
    The sector term keys on the image-plane azimuth between each pixel and
    the haze illumination direction, so a pixel on the sunward axis keeps
    the nominal length and one :data:`SECTOR_REFERENCE_HALF_ANGLE_DEG` away
    carries the full gradient.

    Parameters:
        structure: The haze structure, or None for the base haze.
        scale_height_px: Nominal falloff length in grid pixels.
        weight: The :func:`hemisphere_weight` of each pixel, or None when the
            structure carries neither hemispheric term.
        v_ctr: Image-frame v offset of each pixel from the body center.
        u_ctr: Image-frame u offset of each pixel from the body center.
        illum_v: V component of the (tilted) in-plane illumination direction.
        illum_u: U component of the same direction.

    Returns:
        The nominal scale height, or the per-pixel field.

    Raises:
        ValueError: If the structure carries a falloff ratio and ``weight``
            is None.
    """
    if structure is None or not structure.scales_falloff:
        return scale_height_px
    factor: NDArrayFloatType = np.ones_like(v_ctr, dtype=np.float64)
    if structure.ns_falloff_ratio != 1.0:
        if weight is None:
            raise ValueError('ns_falloff_ratio needs the hemisphere weight')
        factor = 1.0 + (structure.ns_falloff_ratio - 1.0) * weight
    if structure.sector_sharpness_gradient != 0.0:
        rho = np.maximum(np.hypot(v_ctr, u_ctr), 1e-9)
        cos_phi = np.clip((v_ctr * illum_v + u_ctr * illum_u) / rho, -1.0, 1.0)
        angle = np.arccos(cos_phi)
        reference = math.radians(SECTOR_REFERENCE_HALF_ANGLE_DEG)
        factor = factor * (1.0 + structure.sector_sharpness_gradient * angle / reference)
    field: NDArrayFloatType = scale_height_px * np.maximum(factor, _MIN_SCALE_FACTOR)
    return field


def apply_hemispheric_scaling(
    values: NDArrayFloatType,
    structure: HazeStructure | None,
    weight: NDArrayFloatType | None,
) -> NDArrayFloatType:
    """Scale one hemisphere's brightness by the asymmetry amplitude.

    Multiplies each pixel by ``1 + ns_asymmetry_amplitude * w``, ``w`` the
    :func:`hemisphere_weight` of its latitude.  Poleward of the blend band
    that scales the positive-``v_rot`` half by ``1 + ns_asymmetry_amplitude``
    and leaves the other half alone, so there the two halves the mirror maps
    onto each other differ by a pure scaling -- the relation a Pearson
    symmetry score is invariant to, which is what makes this axis the
    control against the non-affine ones -- wherever the scaled half stays
    below saturation.  The caller clips the result to [0, 1], so a positive
    amplitude flattens the brightest pixels and the difference there is the
    clip (see the module docstring for what that costs).  Inside the band the scaling
    changes continuously with latitude, so the brightness carries no seam
    at the equator, and the control is not affine there: a mirror pair's
    ratio falls from ``1 + ns_asymmetry_amplitude`` at the band edge to 1 on
    the mirror line (see the module docstring for why every continuous blend
    does this).

    Parameters:
        values: Brightness values to scale, modified in place and returned.
        structure: The haze structure, or None for the base haze.
        weight: The :func:`hemisphere_weight` of each pixel, shaped like
            ``values``, or None when the structure carries neither
            hemispheric term.

    Returns:
        The scaled values (the input array).

    Raises:
        ValueError: If the structure carries an amplitude and ``weight`` is
            None.
    """
    if structure is None or structure.ns_asymmetry_amplitude == 0.0:
        return values
    if weight is None:
        raise ValueError('ns_asymmetry_amplitude needs the hemisphere weight')
    values *= 1.0 + structure.ns_asymmetry_amplitude * weight
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
