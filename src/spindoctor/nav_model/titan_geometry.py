"""Observation-side geometry extraction for the Titan haze model.

Every ``oops`` and star-catalog query the haze feature depends on lives here,
behind one entry point, :func:`geometry_from_obs`.  The result is a frozen
:class:`TitanGeometryInputs`, on which every downstream decision --
reliability, the hard-zero conditions, the emitted feature payload -- is a pure
function.

Which failures are answered and which propagate
-----------------------------------------------

Three conditions are answered with a degenerate geometry rather than an
exception, because the model's always-emit invariant needs a feature for them:
an inventory field that is not finite, an image scale or body radius that is
not finite and positive, and an envelope box with no surface-intercept pixel
in it around a body narrower than the sampling stride.  Each yields zero radii
or a degenerate axis, so the reliability hard-zero path fires and the emitted
feature is gated out with its cause recorded.  The first is the only exception
this module absorbs, and it has a type of its own,
:class:`NonFiniteInventoryError`, so the ``except`` can match nothing else.
The inventory's ``range`` follows the same rule when present; an absent one is
an unknown distance and reads as infinite, and a negative one is a defect.

Every other exception propagates.  A stage that fails logs one line naming
what it could not evaluate and re-raises; the orchestrator logs the traceback
once and fails the image with ``status_reason=internal_error``, recording
which component raised and what it raised.  Two conditions this module detects
itself are raised the same way, because they are defects and never frame
conditions: an envelope box that must contain the body yet holds no
surface-intercept pixel, and a backplane whose shape differs from its
meshgrid's.

No stage catches an exception and carries on with a default, for two reasons.
An exception's type says nothing about its cause: ``oops`` raises ``ValueError``
and ``LookupError`` when it cannot answer a query, and so does a defect in
``oops``, in ``numpy`` or in this module.  And a stage that carried on would
still produce an offset.  If the mask stage swallowed a fault, the fit would run
against a mask that was never built, and the image's document would record a
navigation that finished, which no selection flag can tell from one that ran on
all its evidence.

Coordinate conventions.  Positions -- the predicted center and the sunward
pixel that sets the symmetry axis -- are field-of-view coordinates plus the
extfov margin, the same convention every predicted position in the pipeline
uses (a catalog star's extfov position is ``star.v + extfov_margin_v``).
Holding to it is what lets a haze offset and a star offset on the same frame
be compared directly.  Bounding boxes are a separate matter: they are
integer pixel indices and they only bound where backplanes are evaluated.

A box grows with the body's apparent size, which is unbounded: Titan at
0.754 km/pixel has an envelope 4343 pixels in radius inside a 1024-pixel
frame, an 8688-square box covering seventy times the frame's area.  The two
boxes are bounded differently because they need different things.

The mask box is CLIPPED to the extended field of view, which costs nothing:
every pixel it computes outside the extfov is discarded downstream, once
where the box-local mask is embedded and again where the mask is
intersected with the box region, so clipping first changes no result.

The envelope box the symmetry axis reads stays UNCLIPPED, because ``oops``
evaluates fine at off-detector pixel coordinates while a clipped box would
leave zero surface-intercept pixels on exactly the off-edge frames the
visibility condition exists for.  It is bounded by UNDERSAMPLING instead.
Striping would bound its memory too, and would cost no accuracy, but it
would not bound its TIME: the work is one evaluation per sample however the
samples are grouped, and seventy times the frame's area is seventy frames'
work.  Undersampling is the bound that reaches the cost itself, and what it
spends is angular resolution.

The axis is one angle, ``atan2`` of the offset from the disc centre to the
minimum-incidence pixel, which is the sub-solar point on the SOLID body and
projects ``r_solid * sin(phase)`` from the centre (``r_solid`` itself, at
the limb, above 90 degrees of phase).  Sampling every k-th pixel locates
that pixel to within the stride, so the angle moves by about
``k / (r_solid * sin(phase))`` radians.  The stride is the smallest INTEGER
that fits the box inside the sample cap, so it is the ceiling of
``2 * r_env / sqrt(cap)``, the box side over the square root of the cap;
just above a threshold that ceiling is up to twice the continuous estimate,
and the cap is then only a quarter used.  Taking the estimate, the
quantization is about ``2 * (r_env / r_solid) / (sqrt(cap) * sin(phase))``
radians, and up to twice that in the band above a threshold.  It depends on
the phase and on the envelope-to-solid ratio, which is 1.27 for Titan's
2575 km radius under the 700 km atmosphere, and not on the body's apparent
size, because the stride and the arm scale together; it is bounded, not
vanishing.  At a million samples the estimate is 0.15 degrees at 90 degrees
of phase, 0.29 at 30 and 0.84 at 10, and the worst case in the band above a
threshold is 0.30, 0.58 and 1.68.  The worst case crosses the 0.5-degree
step of the angle refinement below about 35 degrees of phase.  That
refinement searches plus or minus 5 degrees around the initial axis, so the
initial axis need only land inside that window, which it does above about
3.5 degrees of phase; a disc at lower phase is nearly rotationally
symmetric, where the axis matters least.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import oops
from oops import Meshgrid, Observation
from oops.backplane import Backplane

from spindoctor.config import IMAGE_LOGGER, Config
from spindoctor.nav_model.nav_model_body import (
    TITAN_BODY_NAME,
    bodies_in_extfov,
    occluder_mask_for_body,
)
from spindoctor.nav_model.stars.catalog import stars_in_extfov
from spindoctor.support.memory import release_transient_memory
from spindoctor.support.types import NDArrayBoolType

__all__ = [
    'OCCLUDER_STRIP_ROWS',
    'STAR_MASK_PHOTOMETRY_SPLIT_VMAG',
    'STAR_MASK_YBSC_MIN_VMAG',
    'NonFiniteInventoryError',
    'TitanGeometryInputs',
    'geometry_from_obs',
    'occluded_disc_fraction',
    'paint_disc',
]


STAR_MASK_YBSC_MIN_VMAG: float = -2.0
"""Bright end of the YBSC query used to build the star contaminant mask.

Brighter than any catalog star, so the query is bounded rather than
truncating the brightest sources the mask most needs.
"""


STAR_MASK_PHOTOMETRY_SPLIT_VMAG: float = 6.5
"""Magnitude where the star-mask query switches from YBSC to Tycho-2.

Both are photometry-reference catalogs whose bright-end magnitudes are
trusted; the mask query never touches UCAC4, whose merged magnitudes
saturate above V ~ 8 and can read several magnitudes too faint exactly in
the range the mask covers.
"""


_RING_TARGET_SUFFIX: str = ':ring'
"""Suffix appended to a lowercase planet name to name its ring surface."""


OCCLUDER_STRIP_ROWS: int = 128
"""How many rows one sibling-occlusion evaluation covers at a time.

See ``_striped_occlusion``. The same bound, for the same reason, as
:data:`~spindoctor.nav_model.nav_model_rings.BACKPLANE_STRIP_ROWS`, whose
docstring carries the reasoning and the measured agreement between a striped
pass and a whole-frame one.
"""

_MASK_BOX_SLOP_PX: float = 0.5
"""Slop added when converting a field-of-view centre to a bounding-box index.

Field-of-view coordinates run half a pixel ahead of the pixel indices a
bounding box is expressed in, so a box derived from a centre plus a radius
is widened by this much to compensate that frame shift. The inventory
midpoint feeding the centre may itself be quantized by up to half a pixel,
so the low edge of the box can still fall one pixel short; every consumer
carries pad far in excess of that residual.
"""


@dataclass(frozen=True)
class TitanGeometryInputs:
    """Everything the haze feature needs, with all observation access done.

    Produced by :func:`geometry_from_obs`, which owns every ``oops`` query;
    every downstream decision (reliability, hard-zero conditions, feature
    payload) is a pure function of this dataclass, so it is testable without
    an observation or a SPICE kernel.

    Parameters:
        predicted_center_vu: Geometric disc center in extfov coordinates --
            the body's field-of-view center plus the extfov margin.
        r_solid_px: Apparent solid-body radius in pixels; ``0.0`` when the
            image scale or body radius is not finite and positive.
        r_env_px: Apparent haze-envelope radius in pixels; ``0.0`` under the
            same condition.
        km_per_px: Image scale at the body center in kilometers per pixel;
            ``0.0`` when the image scale or body radius is not finite and
            positive.
        phase_deg: Phase angle at the body center in degrees.
        theta_rad: Symmetry-axis angle; ``atan2`` of the offset from the
            disc center to the minimum-incidence surface pixel.
        axis_degenerate: True when that offset was too short to define a
            direction (a near-zero-phase, rotationally symmetric disc), when
            the envelope box holds no surface-intercept pixel around a body
            narrower than the sampling stride, or when the inventory or
            image scale is degenerate.
        occluded_fraction: Fraction of the envelope disc, clipped to the
            extended frame, that a nearer body or the rings hide.
        contaminant_mask: Undilated boolean array of the extfov image shape
            marking pixels the fits must ignore, or ``None`` when nothing is
            masked.
        extfov_shape_vu: ``(rows, columns)`` of the extended-FOV frame.
        window_px: Pointing search half-window in pixels -- the larger of
            the two extfov margins.
        extfov_margin_vu: The two extfov margins themselves, ``(rows,
            columns)``.  The search runs in the rotated symmetry frame,
            where only the scalar ``window_px`` is meaningful; the
            visibility test runs in image axes, where the per-axis margins
            are what say whether the envelope clears the detector.
        bbox_extfov_vu: Half-open envelope bounding box in extfov
            coordinates.
        subject_range_km: Observer-to-Titan center range in kilometers.
        filters: Instrument filter names for this image.
    """

    predicted_center_vu: tuple[float, float]
    r_solid_px: float
    r_env_px: float
    km_per_px: float
    phase_deg: float
    theta_rad: float
    axis_degenerate: bool
    occluded_fraction: float
    contaminant_mask: NDArrayBoolType | None
    extfov_shape_vu: tuple[int, int]
    window_px: float
    extfov_margin_vu: tuple[float, float]
    bbox_extfov_vu: tuple[int, int, int, int]
    subject_range_km: float
    filters: tuple[str, ...]


class NonFiniteInventoryError(ValueError):
    """A Titan inventory field that is not a finite number.

    Its own type because it is the one condition this module expects and
    answers: the inventory reports a bounding box, and a NaN or an infinity in
    it is a frame whose geometry cannot be built, which the always-emit
    invariant answers with a degenerate geometry and a zero-reliability
    feature.  Every other exception propagates and fails the image, and a bare
    ``ValueError`` could not be told from one, since ``oops`` raises
    ``ValueError`` when it cannot answer a query.
    """


def _finite(value: Any, name: str) -> float:
    """Return ``value`` as a finite float, raising when it is not one.

    Inventory coordinates come from a SPICE-driven projection that can
    return NaN or infinity for an unresolvable geometry.  Converting through
    this turns such a value into an exception, which :func:`geometry_from_obs`
    answers with the degenerate geometry, instead of into a NaN that silently
    poisons every quantity derived from it.

    Parameters:
        value: The raw inventory quantity.
        name: Field name used in the error message.

    Returns:
        The value as a float.

    Raises:
        NonFiniteInventoryError: If the value is not finite.
    """
    out = float(value)
    if not math.isfinite(out):
        raise NonFiniteInventoryError(f'Titan inventory field {name} is not finite; got {out!r}')
    return out


def _frame_bounds(obs: Observation) -> tuple[tuple[int, int], float, tuple[float, float]]:
    """Return the extfov shape, search half-window, and per-axis margins.

    Parameters:
        obs: The observation to read.

    Returns:
        ``((height, width), window_px, (margin_v, margin_u))``.

    Raises:
        Exception: Whatever ``obs`` raises when it cannot report its
            extended-FOV geometry, after one log line naming the stage.
    """
    try:
        shape = obs.extdata_shape_vu
        margin = obs.extfov_margin_vu
        bounds = (int(shape[0]), int(shape[1]))
        margin_vu = (float(margin[0]), float(margin[1]))
        window_px = max(margin_vu)
    except Exception as exc:
        IMAGE_LOGGER.error('Titan: extended-FOV geometry could not be evaluated: %s', exc)
        raise
    return bounds, window_px, margin_vu


def _filter_names(obs: Observation) -> tuple[str, ...]:
    """Return the image's filter names, or an empty tuple when it has none.

    Each instrument's observation names its filters differently -- Cassini
    ISS carries ``filter1`` and ``filter2``, Voyager ISS and Galileo SSI carry
    ``filter`` -- so the lookup is by presence.
    """
    names = [getattr(obs, attr, None) for attr in ('filter1', 'filter2', 'filter')]
    return tuple(str(name) for name in names if name)


def _degenerate_geometry(
    *,
    extfov_shape_vu: tuple[int, int],
    window_px: float,
    extfov_margin_vu: tuple[float, float],
    filters: tuple[str, ...],
    predicted_center_vu: tuple[float, float] = (0.0, 0.0),
    subject_range_km: float = float('inf'),
) -> TitanGeometryInputs:
    """Return defensible defaults for a frame whose inventory or scale is degenerate.

    Zero radii put the envelope diameter below any positive floor, so the
    reliability hard-zero path fires and the emitted feature is gated out
    with its cause recorded.
    """
    return TitanGeometryInputs(
        predicted_center_vu=predicted_center_vu,
        r_solid_px=0.0,
        r_env_px=0.0,
        km_per_px=0.0,
        phase_deg=0.0,
        theta_rad=0.0,
        axis_degenerate=True,
        occluded_fraction=0.0,
        contaminant_mask=None,
        extfov_shape_vu=extfov_shape_vu,
        window_px=window_px,
        extfov_margin_vu=extfov_margin_vu,
        bbox_extfov_vu=(0, 0, 0, 0),
        subject_range_km=subject_range_km,
        filters=filters,
    )


@dataclass(frozen=True)
class _BodyScale:
    """Image scale, apparent radii, and phase at the body center."""

    km_per_px: float
    r_solid_px: float
    r_env_px: float
    phase_deg: float


def _body_radius_km(body_name: str) -> float:
    """Return a body's registered equatorial radius in kilometers.

    No modified body is ever registered with ``oops`` for the haze
    envelope: nothing here needs an inflated body in the SPICE inventory,
    and registering one would mutate process-wide registry state.  The
    envelope is a plain number derived from this radius plus the configured
    atmosphere height.
    """
    return float(oops.Body.lookup(body_name).radius)


def _body_scale(obs: Observation, config: Config) -> _BodyScale | None:
    """Return the image scale, apparent radii, and phase.

    ``km_per_px`` averages the per-axis center resolutions; the radii come
    from the body's registered equatorial radius plus the configured
    atmosphere height.

    Returns:
        The scale, or ``None`` when the scale or radius is not finite and
        positive.

    Raises:
        Exception: Whatever the backplane or the body registry raises when the
            center resolution, phase or radius cannot be evaluated, after one
            log line naming the stage.
    """
    try:
        ext_bp = obs.ext_bp
        res_u = float(ext_bp.center_resolution(TITAN_BODY_NAME, axis='u').vals)
        res_v = float(ext_bp.center_resolution(TITAN_BODY_NAME, axis='v').vals)
        phase_deg = float(np.degrees(ext_bp.center_phase_angle(TITAN_BODY_NAME).vals))
        radius_km = _body_radius_km(TITAN_BODY_NAME)
    except Exception as exc:
        IMAGE_LOGGER.error(
            'Titan: image scale, phase or body radius could not be evaluated: %s', exc
        )
        raise
    km_per_px = 0.5 * (res_u + res_v)
    # NaN fails every comparison, so it must be rejected by an explicit
    # finiteness test rather than by the positivity bounds: a NaN radius
    # would otherwise propagate all the way to a NaN reliability, which
    # NavFeature rejects at construction -- turning a marginal frame into an
    # unattributable failure instead of a gated one.
    if (
        not math.isfinite(km_per_px)
        or km_per_px <= 0.0
        or not math.isfinite(radius_km)
        or radius_km <= 0.0
    ):
        IMAGE_LOGGER.warning(
            'Titan: degenerate image scale (km/px = %r, radius = %r km)', km_per_px, radius_km
        )
        return None
    atmosphere_height_km = float(config.titan['atmosphere_height'])
    return _BodyScale(
        km_per_px=km_per_px,
        r_solid_px=radius_km / km_per_px,
        r_env_px=(radius_km + atmosphere_height_km) / km_per_px,
        phase_deg=phase_deg,
    )


def _bbox_extent(bbox_nominal: tuple[int, int, int, int]) -> tuple[int, int]:
    """Return the ``(width, height)`` of a nominal-frame bbox, in pixels."""
    u_min, u_max, v_min, v_max = bbox_nominal
    return max(0, u_max - u_min + 1), max(0, v_max - v_min + 1)


def _bbox_undersample(bbox_nominal: tuple[int, int, int, int], max_samples: int) -> int:
    """Return the stride that keeps a box within ``max_samples`` samples.

    A body's apparent size sets the box, and nothing bounds it: a close
    approach puts a body of thousands of pixels' radius inside a
    thousand-pixel frame, and a backplane evaluated at one sample per pixel
    over that box costs tens of times what the frame itself costs -- seventy,
    for the 8688-square box of the module docstring over a 1024-square
    frame.  Striding the grid bounds that cost by the sample count rather
    than by the geometry, which striping the box would not: striping moves
    the same work into smaller pieces, and it is the work that is unbounded
    here.

    The stride is an integer, so it steps rather than slides.  A box just
    over a threshold takes the next stride up and lands at a quarter of the
    cap: a 1001-square box under a cap of a million strides by 2 and samples
    251001, where the continuous estimate ``side / sqrt(cap)`` says 1.001.
    The stride is therefore up to twice that estimate, and whatever the
    stride costs is up to twice what the estimate predicts.

    Parameters:
        bbox_nominal: ``(u_min, u_max, v_min, v_max)`` in pixel indices.
        max_samples: Largest number of samples the grid may hold; a
            non-positive value imposes no bound.

    Returns:
        The stride to sample each axis by, never less than 1.  A box already
        within the bound samples every pixel.
    """
    width, height = _bbox_extent(bbox_nominal)
    samples = width * height
    if max_samples <= 0 or samples <= max_samples:
        return 1
    # Each axis rounds up, so the count a stride actually produces can exceed
    # the count the exact division promised -- a 101x101 box under a cap of
    # 2600 divides to a stride of 2 and then samples 51x51 = 2601.  Step until
    # what it produces is inside the bound.
    stride = max(1, math.ceil(math.sqrt(samples / max_samples)))
    while _sampled_count(width, height, stride) > max_samples:
        stride += 1
    return stride


def _sampled_count(width: int, height: int, stride: int) -> int:
    """How many samples a stride actually takes over a box.

    Parameters:
        width: The box width in pixels.
        height: The box height in pixels.
        stride: The sampling stride along each axis.

    Returns:
        The number of grid points, both axes rounding up.
    """
    return math.ceil(width / stride) * math.ceil(height / stride)


def _clip_bbox_to_extfov(
    bbox_nominal: tuple[int, int, int, int],
    extfov_shape_vu: tuple[int, int],
    margin_vu: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Intersect a nominal-frame bbox with the extended field of view.

    Parameters:
        bbox_nominal: ``(u_min, u_max, v_min, v_max)`` in pixel indices.
        extfov_shape_vu: ``(rows, cols)`` of the extended frame.
        margin_vu: ``(margin_v, margin_u)`` extfov margins, so nominal index
            ``v`` sits at extfov row ``v + margin_v``.

    Returns:
        The clipped box, or ``None`` when it does not reach the extended
        frame at all.
    """
    u_min, u_max, v_min, v_max = bbox_nominal
    rows, cols = extfov_shape_vu
    v_lo = max(v_min, -margin_vu[0])
    v_hi = min(v_max, rows - margin_vu[0] - 1)
    u_lo = max(u_min, -margin_vu[1])
    u_hi = min(u_max, cols - margin_vu[1] - 1)
    if v_hi < v_lo or u_hi < u_lo:
        return None
    return u_lo, u_hi, v_lo, v_hi


def _restricted_backplane(
    obs: Observation, bbox_nominal: tuple[int, int, int, int], *, undersample: int = 1
) -> tuple[Backplane, Meshgrid]:
    """Build a backplane over a nominal-frame bbox.

    ``bbox_nominal`` is ``(u_min, u_max, v_min, v_max)`` in nominal-frame
    pixel indices.  It may run negative inside the extfov margin or past the
    detector: ``oops`` backplanes evaluate fine at off-detector pixel
    coordinates, so nothing here clips the box.  Whether a box is clipped is
    the caller's decision.  The symmetry axis keeps its envelope box
    unclipped, because the sunward pixel it wants may lie off-frame; the
    mask box arrives clipped, because its off-frame pixels are discarded
    downstream.

    Parameters:
        obs: Observation snapshot.
        bbox_nominal: The box to evaluate over.
        undersample: Sample every ``undersample``-th pixel along each axis;
            ``1`` samples every pixel.  The meshgrid reports the pixel
            coordinate of every sample either way, so a caller reading
            positions off it needs no correction for the stride, only the
            knowledge that its answer is quantized by it.

    Returns:
        The backplane and the meshgrid it was built over.
    """
    u_min, u_max, v_min, v_max = bbox_nominal
    meshgrid = Meshgrid.for_fov(
        obs.fov,
        origin=(u_min + 0.5, v_min + 0.5),
        limit=(u_max + 0.5, v_max + 0.5),
        undersample=max(1, undersample),
        swap=True,
    )
    return Backplane(obs, meshgrid=meshgrid), meshgrid


def _symmetry_axis(
    obs: Observation,
    bbox_nominal: tuple[int, int, int, int],
    center_vu: tuple[float, float],
    margin_vu: tuple[int, int],
    *,
    axis_min_offset_px: float,
    r_solid_px: float,
    max_samples: int,
) -> tuple[float, bool]:
    """Return ``(theta_rad, axis_degenerate)`` from the incidence backplane.

    The visible pixel of MINIMUM incidence always projects in the sunward
    image direction, at every phase.  (The maximum-incidence pixel is the
    anti-solar surface point, which becomes visible past 90 degrees phase
    and points the wrong way, so the choice is deliberately phase-free.)
    The axis angle is ``atan2`` of that pixel's offset from the disc center;
    an offset shorter than ``axis_min_offset_px`` sampling strides means a
    near-zero-phase disc that is rotationally symmetric, where any axis is
    equally valid.

    Both ends of that difference are expressed in the same frame -- the
    field-of-view coordinate plus the extfov margin, which is what the
    meshgrid reports and what :func:`geometry_from_obs` builds the predicted
    center in.  Only consistency matters here, because the angle is a
    difference; converting one end to pixel indices and not the other would
    tilt the axis by a half pixel over the disc radius.

    Parameters:
        obs: Observation snapshot.
        bbox_nominal: Unclipped envelope bbox ``(u_min, u_max, v_min,
            v_max)`` in nominal-frame pixel indices.
        center_vu: Predicted disc center in extfov coordinates.
        margin_vu: ``(margin_v, margin_u)`` extfov margins.
        axis_min_offset_px: Offset below which the axis is degenerate, at a
            sampling quantum of one pixel: with the sunward pixel located to
            one pixel and an arm this long, the axis is known to about
            ``1 / axis_min_offset_px`` radians.  Under a stride the located
            pixel can sit up to ``stride / sqrt(2)`` from the true one, so
            the same angular floor is applied at the sampling quantum in
            use, an arm of ``axis_min_offset_px * stride``.  A strided box
            implies an envelope over about 500 pixels in radius, so the
            stride reaches this guard only on frames of very low phase.
        r_solid_px: Apparent solid-body radius in pixels, which decides
            whether a box with no surface-intercept pixel is a frame
            condition or a defect.
        max_samples: Largest grid the incidence backplane may be evaluated
            over.  A box wider than this is strided rather than clipped,
            because the pixel wanted is the sunward one and it can lie
            outside the frame.  Striping it would bound the memory but not
            the work, which is what is unbounded.  The stride is the
            smallest integer fitting the box inside this, so it is the
            ceiling of the box side over the square root of this, and the
            side is twice the envelope radius: ``k`` is the ceiling of
            ``2 * r_env / sqrt(max_samples)``, up to twice that estimate
            just above a threshold.  The pixel it locates is the sub-solar
            point on the solid body, ``r_solid * sin(phase)`` from the
            centre (``r_solid``, at the limb, above 90 degrees of phase), so
            the angle quantizes by about ``2 * (r_env / r_solid) /
            (sqrt(max_samples) * sin(phase))`` radians and up to twice that:
            for a million samples under the shipped 700 km atmosphere, 0.15
            degrees at 90 degrees of phase, 0.29 at 30 and 0.84 at 10, or
            0.30, 0.58 and 1.68 at worst.  The worst case crosses the
            0.5-degree step of the angle refinement below about 35 degrees
            of phase and its 5-degree search window only below about 3.5.

    Returns:
        ``(theta_rad, axis_degenerate)``.  The axis is degenerate when the
        minimum-incidence pixel lies within ``axis_min_offset_px`` sampling
        strides of the disc center -- the same angular floor, applied at the
        sampling quantum in use -- and when the box holds no
        surface-intercept pixel around a body narrower than the sampling
        stride.  The backplane mask marks pixels with no surface intercept at
        all; an unlit surface is still intercepted, so lighting never empties
        the box.

    Raises:
        RuntimeError: If the incidence backplane's shape differs from its
            meshgrid's, or if the box holds no surface-intercept pixel around
            a body at least as wide as the sampling stride, which the box
            must then contain.  Both are defects, never frame conditions.
        Exception: Whatever the backplane raises when the incidence angle
            cannot be evaluated, after one log line naming the stage.
    """
    # The pixel this search locates is fixed by the geometry, so the search,
    # its cap and the stride-scaled degeneracy floor could all give way to
    # projecting the sub-solar direction and clamping to the limb above 90
    # degrees of phase (#594).
    undersample = _bbox_undersample(bbox_nominal, max_samples)
    if undersample > 1:
        width, height = _bbox_extent(bbox_nominal)
        IMAGE_LOGGER.info(
            'Titan: envelope box is %d x %d px (%d samples); sampling every %d px '
            'to stay within %d',
            width,
            height,
            width * height,
            undersample,
            max_samples,
        )
    try:
        bp, meshgrid = _restricted_backplane(obs, bbox_nominal, undersample=undersample)
        incidence = bp.incidence_angle(TITAN_BODY_NAME)
        invalid = np.asarray(incidence.expand_mask().mask, dtype=bool)
        values = np.asarray(incidence.vals, dtype=np.float64)
        uv = np.asarray(meshgrid.uv.vals, dtype=np.float64)
    except Exception as exc:
        IMAGE_LOGGER.error('Titan: incidence backplane could not be evaluated: %s', exc)
        raise
    # oops evaluates every backplane over the meshgrid it was given, so a
    # shape mismatch is a defect, never a frame condition.
    if uv.shape[:-1] != values.shape:
        raise RuntimeError(
            f'Titan: incidence backplane {values.shape} does not match its meshgrid {uv.shape[:-1]}'
        )
    valid = ~invalid
    if not valid.any():
        # A square lattice with stride k has covering radius k / sqrt(2), so a
        # disc of radius at least k always holds a sample: an empty box around
        # such a body is a defect, while a body narrower than the stride can
        # fall between samples.
        if r_solid_px >= undersample:
            raise RuntimeError(
                'Titan: no surface-intercept pixel in an envelope box that must contain '
                f'the body (solid radius {r_solid_px:.3f} px, sampling stride {undersample} px)'
            )
        IMAGE_LOGGER.warning(
            'Titan: body narrower than the sampling stride (solid radius %.3f px, stride '
            '%d px) shows no surface-intercept pixel; axis is degenerate',
            r_solid_px,
            undersample,
        )
        return 0.0, True
    index = np.unravel_index(int(np.argmin(np.where(valid, values, np.inf))), values.shape)
    sun_u = float(uv[index][0]) + margin_vu[1]
    sun_v = float(uv[index][1]) + margin_vu[0]
    d_v = sun_v - center_vu[0]
    d_u = sun_u - center_vu[1]
    # The floor was set for a one-pixel sampling quantum; a strided box
    # locates the sunward pixel only to within the stride, so the same
    # angular floor needs an arm that many times longer.
    if math.hypot(d_v, d_u) < axis_min_offset_px * undersample:
        return 0.0, True
    return math.atan2(d_v, d_u), False


def _ring_occlusion_local(
    bp: Backplane,
    planet: str,
    subject_range_km: float,
    radii_km: tuple[float, float],
) -> NDArrayBoolType | None:
    """Pixels where the main rings pass in front of the subject body.

    A pixel is occluded when its ring-plane intercept radius falls inside
    the configured annulus AND the intercept is nearer than the body center.
    The rings are treated as opaque; translucency is out of scope, so a
    frame with Titan behind them gates out rather than fitting through ring
    stripes.

    Returns:
        The bbox-local boolean mask, or ``None`` when no ring-plane intercept
        inside the annulus lies nearer than the body center.

    Raises:
        Exception: Whatever the backplane raises when the ring radius or
            distance cannot be evaluated, after one log line naming the stage.
    """
    ring_target = f'{planet.lower()}{_RING_TARGET_SUFFIX}'
    try:
        radius = np.asarray(bp.ring_radius(ring_target).mvals.filled(np.nan), dtype=np.float64)
        distance = np.asarray(bp.distance(ring_target).mvals.filled(np.inf), dtype=np.float64)
    except Exception as exc:
        IMAGE_LOGGER.error('Titan: ring backplanes could not be evaluated: %s', exc)
        raise
    with np.errstate(invalid='ignore'):
        in_annulus = (radius >= radii_km[0]) & (radius <= radii_km[1])
        nearer = distance < subject_range_km
    occluded: NDArrayBoolType = in_annulus & nearer
    if not occluded.any():
        return None
    return occluded


def paint_disc(mask: NDArrayBoolType, center_vu: tuple[float, float], radius_px: float) -> None:
    """Set every pixel within ``radius_px`` of ``center_vu`` in place.

    Public because the simulated haze model paints the same star discs into
    the same kind of mask from operator parameters, and the two masks must
    be built by one piece of code if a sim frame is to exercise what a real
    one does.

    Parameters:
        mask: Extfov-shaped boolean mask, modified in place.
        center_vu: ``(v, u)`` disc centre in extfov coordinates.
        radius_px: Disc radius in pixels.
    """
    rows, cols = mask.shape
    v_lo = max(0, math.floor(center_vu[0] - radius_px))
    v_hi = min(rows, math.ceil(center_vu[0] + radius_px) + 1)
    u_lo = max(0, math.floor(center_vu[1] - radius_px))
    u_hi = min(cols, math.ceil(center_vu[1] + radius_px) + 1)
    if v_hi <= v_lo or u_hi <= u_lo:
        return
    vs = np.arange(v_lo, v_hi, dtype=np.float64)[:, np.newaxis]
    us = np.arange(u_lo, u_hi, dtype=np.float64)[np.newaxis, :]
    inside = (vs - center_vu[0]) ** 2 + (us - center_vu[1]) ** 2 <= radius_px * radius_px
    mask[v_lo:v_hi, u_lo:u_hi] |= inside


def _paint_sibling_bboxes(
    mask: NDArrayBoolType,
    siblings: list[tuple[str, dict[str, Any]]],
    margin_vu: tuple[int, int],
) -> None:
    """Paint every other in-FOV body's inventory bounding box in place.

    Range order is deliberately ignored: a moon behind Titan occludes
    nothing, but its visible sliver beside the limb sits squarely in the
    symmetry annulus and in the arc rays.  Box masking is deliberately
    conservative -- a moon entirely hidden behind Titan costs a box-sized
    patch of valid pairs, which the fit's coverage gates then meter.
    """
    rows, cols = mask.shape
    for _name, entry in siblings:
        v_min = int(entry['v_min_unclipped']) + margin_vu[0]
        v_max = int(entry['v_max_unclipped']) + margin_vu[0]
        u_min = int(entry['u_min_unclipped']) + margin_vu[1]
        u_max = int(entry['u_max_unclipped']) + margin_vu[1]
        v_lo = max(0, v_min)
        v_hi = min(rows, v_max + 1)
        u_lo = max(0, u_min)
        u_hi = min(cols, u_max + 1)
        if v_hi > v_lo and u_hi > u_lo:
            mask[v_lo:v_hi, u_lo:u_hi] = True


def _paint_bright_stars(
    mask: NDArrayBoolType,
    obs: Observation,
    config: Config,
    *,
    margin_vu: tuple[int, int],
    vmag_limit: float,
    radius_px: float,
) -> None:
    """Paint a disc over every catalog star brighter than ``vmag_limit``.

    Queries the two photometry-reference catalogs and never the bright end
    of UCAC4, whose merged magnitudes saturate inside the mask's range.
    Duplicates between the two queries are harmless: they paint overlapping
    discs.  Predicted star positions are nominal-frame, so the extfov
    margins are added before painting.
    """
    queries = (
        ('ybsc', STAR_MASK_YBSC_MIN_VMAG, STAR_MASK_PHOTOMETRY_SPLIT_VMAG),
        ('tycho2', STAR_MASK_PHOTOMETRY_SPLIT_VMAG, vmag_limit),
    )
    for catalog_name, mag_min, mag_max in queries:
        if mag_max <= mag_min:
            continue
        try:
            stars = stars_in_extfov(
                obs, config, catalog_name=catalog_name, mag_min=mag_min, mag_max=mag_max
            )
        except Exception as exc:
            IMAGE_LOGGER.error('Titan: %s star query could not be evaluated: %s', catalog_name, exc)
            raise
        for star in stars:
            paint_disc(mask, (star.v + margin_vu[0], star.u + margin_vu[1]), radius_px)


def occluded_disc_fraction(
    occluder_ext: NDArrayBoolType,
    center_vu: tuple[float, float],
    r_env_px: float,
) -> float:
    """Fraction of the framed envelope disc that occluding matter hides.

    Only true occlusion counts -- nearer bodies and the rings.  The sibling
    footprints and star discs of the contaminant mask are search-robustness
    devices, not evidence that Titan is hidden, so they are excluded here.

    Public for the same reason as :func:`paint_disc`: the simulated haze
    model reports the same quantity from operator parameters, and the
    reliability formula both feed must not be able to disagree with itself.

    Parameters:
        occluder_ext: Extfov-shaped boolean mask of occluding pixels.
        center_vu: ``(v, u)`` envelope centre in extfov coordinates.
        r_env_px: Envelope radius in pixels.

    Returns:
        The hidden fraction in ``[0, 1]``; ``0.0`` for a degenerate or
        entirely off-frame envelope.
    """
    if r_env_px <= 0.0:
        return 0.0
    rows, cols = occluder_ext.shape
    v_lo = max(0, math.floor(center_vu[0] - r_env_px))
    v_hi = min(rows, math.ceil(center_vu[0] + r_env_px) + 1)
    u_lo = max(0, math.floor(center_vu[1] - r_env_px))
    u_hi = min(cols, math.ceil(center_vu[1] + r_env_px) + 1)
    if v_hi <= v_lo or u_hi <= u_lo:
        return 0.0
    vs = np.arange(v_lo, v_hi, dtype=np.float64)[:, np.newaxis]
    us = np.arange(u_lo, u_hi, dtype=np.float64)[np.newaxis, :]
    disc = (vs - center_vu[0]) ** 2 + (us - center_vu[1]) ** 2 <= r_env_px * r_env_px
    disc_count = int(np.count_nonzero(disc))
    if disc_count == 0:
        return 0.0
    hidden = int(np.count_nonzero(disc & occluder_ext[v_lo:v_hi, u_lo:u_hi]))
    return hidden / disc_count


@dataclass(frozen=True)
class _ContaminantMask:
    """The undilated contaminant mask plus the occlusion it implies."""

    mask: NDArrayBoolType | None
    occluded_fraction: float


def _striped_occlusion(
    *,
    obs: Observation,
    bbox_nominal: tuple[int, int, int, int],
    sibling_ranges: list[tuple[str, float]],
    subject_range_km: float,
    planet: str | None,
    ring_radii_km: tuple[float, float],
) -> tuple[NDArrayBoolType | None, NDArrayBoolType | None]:
    """Both occlusion masks over a box, a strip of rows at a time.

    The bodies and the rings are asked about together because they are asked
    about over the same box: one set of strips, and each strip's backplane
    answers both questions before it is discarded.

    Parameters:
        obs: Observation snapshot.
        bbox_nominal: ``(u_min, u_max, v_min, v_max)`` of the mask box.
        sibling_ranges: ``(body name, range_km)`` for the other bodies in the
            field of view.
        subject_range_km: Center range of the subject body; only strictly
            nearer siblings and ring intercepts can hide it.
        planet: The planet whose rings might occlude, or None for a frame with
            no planet to speak of, which asks the rings nothing.
        ring_radii_km: Inner and outer radius of the annulus treated as opaque.

    Returns:
        ``(body mask, ring mask)`` box-local, each None when nothing is hidden
        -- the same answers the whole-box calls give.
    """
    u_min, u_max, v_min, v_max = bbox_nominal
    height = v_max - v_min + 1
    width = u_max - u_min + 1
    if height <= 0 or width <= 0:
        return None, None
    body_parts: list[NDArrayBoolType] = []
    ring_parts: list[NDArrayBoolType] = []
    any_body = False
    any_ring = False
    for start in range(0, height, OCCLUDER_STRIP_ROWS):
        stop = min(start + OCCLUDER_STRIP_ROWS, height)
        try:
            strip_bp, strip_meshgrid = _restricted_backplane(
                obs, (u_min, u_max, v_min + start, v_min + stop - 1)
            )
        except Exception as exc:
            IMAGE_LOGGER.error('Titan: mask-box backplane could not be evaluated: %s', exc)
            raise
        empty = np.zeros((stop - start, width), dtype=bool)
        body = occluder_mask_for_body(
            strip_bp,
            TITAN_BODY_NAME,
            sibling_ranges,
            subject_range_km,
            oversample_v=1,
            oversample_u=1,
        )
        body_parts.append(empty if body is None else body)
        any_body = any_body or body is not None
        ring = (
            None
            if planet is None
            else _ring_occlusion_local(strip_bp, planet, subject_range_km, ring_radii_km)
        )
        ring_parts.append(empty if ring is None else ring)
        any_ring = any_ring or ring is not None
        # The meshgrid is the other half of the strip and has to go with it:
        # bound to a throwaway name it outlives the release and is only
        # replaced once the next strip has been built beside it.
        del strip_bp, strip_meshgrid, body, ring
        release_transient_memory()
    return (
        np.vstack(body_parts) if any_body else None,
        np.vstack(ring_parts) if any_ring else None,
    )


def _contaminant_mask(
    obs: Observation,
    config: Config,
    *,
    siblings: list[tuple[str, dict[str, Any]]],
    center_vu: tuple[float, float],
    r_env_px: float,
    subject_range_km: float,
    bbox_nominal: tuple[int, int, int, int],
    extfov_shape_vu: tuple[int, int],
    margin_vu: tuple[int, int],
    window_px: float,
) -> _ContaminantMask:
    """Build the four-component contaminant mask and the occluded fraction.

    The mask covers the *mask box*: the envelope box dilated by the annulus
    outer pad plus twice the search window, because the fits sample out to
    ``r_env + pad + W`` from centers hypothesized up to ``W`` away.  Its
    components are nearer-body occlusion, ring occlusion, the inventory
    boxes of the other in-FOV bodies, and discs over bright catalog stars.
    Cosmic rays, hot pixels, and faint stars are deliberately unmasked: they
    are a handful of pixels against thousands of mirror pairs, and the first
    two have no predicted position to ride the offset hypothesis with.

    The result is embedded in a full extfov-shaped array so the fitting
    signatures need no box-origin parameter, and is shipped UNDILATED --
    hypothesis alignment and along-axis dilation belong to the fit, which
    knows its own current center.

    Returns:
        The mask (``None`` when nothing is masked) and the fraction of the
        envelope disc that the occlusion components alone cover.
    """
    nav_config = config.titan['navigation']
    pad = float(nav_config['symmetry']['annulus_outer_pad_px']) + 2.0 * window_px
    pad_int = math.ceil(pad)
    u_min, u_max, v_min, v_max = bbox_nominal
    mask_bbox = (u_min - pad_int, u_max + pad_int, v_min - pad_int, v_max + pad_int)
    occluder_ext: NDArrayBoolType = np.zeros(extfov_shape_vu, dtype=bool)
    contaminant_ext: NDArrayBoolType = np.zeros(extfov_shape_vu, dtype=bool)
    # Clipping here costs nothing and bounds the box by the frame instead of by
    # the body's apparent size: a mask pixel outside the extended frame is
    # dropped twice downstream, once by the embed and once by the box region,
    # so the only thing evaluating it ever bought was the memory it took.
    clipped_bbox = _clip_bbox_to_extfov(mask_bbox, extfov_shape_vu, margin_vu)
    if clipped_bbox is None:
        IMAGE_LOGGER.info('Titan: mask box lies outside the extended frame; occlusion not masked')
    else:
        sibling_ranges = [
            (name.upper(), float(entry.get('range', float('inf')))) for name, entry in siblings
        ]
        # Evaluated a strip of rows at a time. Each sibling costs one whole-box
        # occlusion backplane, and the box is the envelope plus twice the search
        # window -- which is the extfov margin, 400 pixels on Voyager, so the box
        # is most of the frame however small the body in it. Sixteen siblings
        # over that box measured 12.3 GB (2026-09-05). oops materializes its
        # intermediates over whatever meshgrid it is handed, so handing it a
        # strip costs a strip's worth; the mask assembled from the strips is the
        # mask the whole-box call returns.
        planet = obs.closest_planet
        radii = nav_config['ring_occlusion_radii_km']
        body_local, ring_local = _striped_occlusion(
            obs=obs,
            bbox_nominal=clipped_bbox,
            sibling_ranges=sibling_ranges,
            subject_range_km=subject_range_km,
            planet=None if planet is None else str(planet),
            ring_radii_km=(float(radii[0]), float(radii[1])),
        )
        for local in (body_local, ring_local):
            if local is not None:
                _embed_local(occluder_ext, local, clipped_bbox, margin_vu)
    contaminant_ext |= occluder_ext
    _paint_sibling_bboxes(contaminant_ext, siblings, margin_vu)
    _paint_bright_stars(
        contaminant_ext,
        obs,
        config,
        margin_vu=margin_vu,
        vmag_limit=float(nav_config['star_mask_vmag_limit']),
        radius_px=float(nav_config['star_mask_radius_px']),
    )
    contaminant_ext &= _mask_box_region(extfov_shape_vu, mask_bbox, margin_vu)
    fraction = occluded_disc_fraction(occluder_ext, center_vu, r_env_px)
    return _ContaminantMask(
        mask=contaminant_ext if contaminant_ext.any() else None,
        occluded_fraction=fraction,
    )


def _mask_box_region(
    extfov_shape_vu: tuple[int, int],
    bbox_nominal: tuple[int, int, int, int],
    margin_vu: tuple[int, int],
) -> NDArrayBoolType:
    """Return the extfov-shaped indicator of a nominal-frame bounding box."""
    u_min, u_max, v_min, v_max = bbox_nominal
    region: NDArrayBoolType = np.zeros(extfov_shape_vu, dtype=bool)
    rows, cols = extfov_shape_vu
    v_lo = max(0, v_min + margin_vu[0])
    v_hi = min(rows, v_max + margin_vu[0] + 1)
    u_lo = max(0, u_min + margin_vu[1])
    u_hi = min(cols, u_max + margin_vu[1] + 1)
    if v_hi > v_lo and u_hi > u_lo:
        region[v_lo:v_hi, u_lo:u_hi] = True
    return region


def _embed_local(
    target: NDArrayBoolType,
    local: NDArrayBoolType,
    bbox_nominal: tuple[int, int, int, int],
    margin_vu: tuple[int, int],
) -> None:
    """OR a box-local mask into an extfov-shaped array at the box origin."""
    u_min, _u_max, v_min, _v_max = bbox_nominal
    rows, cols = target.shape
    v0 = v_min + margin_vu[0]
    u0 = u_min + margin_vu[1]
    v_lo = max(0, v0)
    u_lo = max(0, u0)
    v_hi = min(rows, v0 + local.shape[0])
    u_hi = min(cols, u0 + local.shape[1])
    if v_hi <= v_lo or u_hi <= u_lo:
        return
    target[v_lo:v_hi, u_lo:u_hi] |= local[v_lo - v0 : v_hi - v0, u_lo - u0 : u_hi - u0]


def geometry_from_obs(
    obs: Observation,
    config: Config,
    *,
    inventory: dict[str, Any] | None = None,
    siblings: list[tuple[str, dict[str, Any]]] | None = None,
) -> TitanGeometryInputs:
    """Compute the haze geometry from an observation.

    Every ``oops`` and catalog query the haze feature depends on happens here.
    A non-finite inventory entry is a frame condition, so it is answered with
    the degenerate geometry and reaches the reliability gate as a zero-scored
    feature, which keeps the always-emit invariant.  Every other exception
    propagates and fails the image, because a stage that could not be
    evaluated is not evidence that the frame is degenerate.

    Parameters:
        obs: Observation snapshot.
        config: Configuration supplying the ``titan`` section.
        inventory: Pre-computed Titan inventory entry; looked up from
            ``obs.inventory`` when omitted.
        siblings: ``(body_name, inventory_entry)`` for the other in-FOV
            bodies; enumerated from ``obs`` when omitted.

    Returns:
        A fully-populated :class:`TitanGeometryInputs`.  A frame whose
        inventory entry is not finite, or whose image scale or body radius is
        not finite and positive, gets zero radii and ``axis_degenerate=True``,
        which forces the reliability hard-zero path.

    Raises:
        ValueError: If the inventory's ``range`` is negative.
        RuntimeError: If the envelope box holds no surface-intercept pixel
            although it must contain the body, or if a backplane's shape
            differs from its meshgrid's.
        Exception: Whatever an ``oops`` or catalog query raises when it
            cannot be evaluated; the stage logs one line naming what it could
            not evaluate before the exception propagates.
    """
    extfov_shape_vu, window_px, extfov_margin_vu = _frame_bounds(obs)
    filters = _filter_names(obs)
    margin_vu = (int(obs.extfov_margin_vu[0]), int(obs.extfov_margin_vu[1]))
    if inventory is None:
        inventory = obs.inventory([TITAN_BODY_NAME], return_type='full')[TITAN_BODY_NAME]
    if siblings is None:
        siblings = [
            (name, entry)
            for name, entry in bodies_in_extfov(obs, config=config)
            if name.upper() != TITAN_BODY_NAME
        ]
    try:
        u_min_unc = _finite(inventory['u_min_unclipped'], 'u_min_unclipped')
        u_max_unc = _finite(inventory['u_max_unclipped'], 'u_max_unclipped')
        v_min_unc = _finite(inventory['v_min_unclipped'], 'v_min_unclipped')
        v_max_unc = _finite(inventory['v_max_unclipped'], 'v_max_unclipped')
        center_uv = inventory['center_uv']
        # The predicted center is the body's exact field-of-view position,
        # not the midpoint of the integer bounding box: that midpoint is
        # quantized by up to half a pixel per axis, a third of the method's
        # whole cross-track budget on a real frame.  It is converted the way
        # every other predicted position in the pipeline is -- field-of-view
        # coordinate plus the extfov margin -- so the uniform half-pixel
        # convention cancels between this technique and the star techniques
        # it is cross-checked against.
        center_vu = (
            _finite(center_uv[1], 'center_uv[v]') + margin_vu[0],
            _finite(center_uv[0], 'center_uv[u]') + margin_vu[1],
        )
        # An absent range is an unknown distance, which is infinite.  A
        # present one is converted like every other inventory field, so NaN
        # and infinity answer with the degenerate geometry rather than
        # reaching NavFeature, which rejects a NaN outright.
        subject_range_km = (
            _finite(inventory['range'], 'range') if 'range' in inventory else float('inf')
        )
    except NonFiniteInventoryError as exc:
        # The one exception answered rather than propagated; see "Which
        # failures are answered and which propagate" in the module docstring
        # for why it is the only one, and why it has a type of its own.
        # The message already starts with the body name.
        IMAGE_LOGGER.warning('%s; geometry is degenerate', exc)
        return _degenerate_geometry(
            extfov_shape_vu=extfov_shape_vu,
            window_px=window_px,
            extfov_margin_vu=extfov_margin_vu,
            filters=filters,
        )
    # A negative range is a defect: it would make every sibling and every
    # ring intercept count as nearer than the body.
    if subject_range_km < 0.0:
        raise ValueError(f'Titan inventory field range is negative; got {subject_range_km!r}')
    scale = _body_scale(obs, config)
    if scale is None:
        return _degenerate_geometry(
            extfov_shape_vu=extfov_shape_vu,
            window_px=window_px,
            extfov_margin_vu=extfov_margin_vu,
            filters=filters,
            predicted_center_vu=center_vu,
            subject_range_km=subject_range_km,
        )
    # The envelope box stays anchored on the integer inventory bounding box:
    # it only bounds where backplanes are evaluated, so a whole-pixel box
    # widened by the field-of-view slop is exactly right, and quantization
    # that would matter for the center is irrelevant for a box.
    u_center_nominal = 0.5 * (u_min_unc + u_max_unc)
    v_center_nominal = 0.5 * (v_min_unc + v_max_unc)
    reach_px = scale.r_env_px + _MASK_BOX_SLOP_PX
    env_bbox = (
        math.floor(u_center_nominal - reach_px),
        math.ceil(u_center_nominal + reach_px),
        math.floor(v_center_nominal - reach_px),
        math.ceil(v_center_nominal + reach_px),
    )
    theta_rad, axis_degenerate = _symmetry_axis(
        obs,
        env_bbox,
        center_vu,
        margin_vu,
        axis_min_offset_px=float(config.titan['navigation']['axis_min_offset_px']),
        r_solid_px=scale.r_solid_px,
        max_samples=int(config.titan['navigation']['backplane_max_samples']),
    )
    contaminant = _contaminant_mask(
        obs,
        config,
        siblings=siblings,
        center_vu=center_vu,
        r_env_px=scale.r_env_px,
        subject_range_km=subject_range_km,
        bbox_nominal=env_bbox,
        extfov_shape_vu=extfov_shape_vu,
        margin_vu=margin_vu,
        window_px=window_px,
    )
    return TitanGeometryInputs(
        predicted_center_vu=center_vu,
        r_solid_px=scale.r_solid_px,
        r_env_px=scale.r_env_px,
        km_per_px=scale.km_per_px,
        phase_deg=scale.phase_deg,
        theta_rad=theta_rad,
        axis_degenerate=axis_degenerate,
        occluded_fraction=contaminant.occluded_fraction,
        contaminant_mask=contaminant.mask,
        extfov_shape_vu=extfov_shape_vu,
        window_px=window_px,
        extfov_margin_vu=extfov_margin_vu,
        bbox_extfov_vu=(
            env_bbox[2] + margin_vu[0],
            env_bbox[0] + margin_vu[1],
            env_bbox[3] + margin_vu[0] + 1,
            env_bbox[1] + margin_vu[1] + 1,
        ),
        subject_range_km=subject_range_km,
        filters=filters,
    )
