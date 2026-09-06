"""Observation-side geometry extraction for the Titan haze model.

Every ``oops`` and star-catalog query the haze feature depends on lives here,
behind one entry point.  It answers a frame whose geometry is genuinely
degenerate -- an inventory entry that is not finite, an envelope box with no
surface intercept in it -- with a defensible geometry rather than an exception,
because the model's always-emit invariant needs one.  It answers a frame it
could not evaluate with the exception, for the reasons below.  The result is a
frozen :class:`TitanGeometryInputs`, on which every downstream decision --
reliability, the hard-zero conditions, the emitted feature payload -- is a pure
function.

Nothing here absorbs an exception
---------------------------------

Each stage asks ``oops`` about the frame, and a stage that raises says what it
could not compute and re-raises.  Nothing degrades, and no stage decides on its
own that the navigation can go on without it.

The temptation is real, because some frames genuinely have no answer -- no
SPICE for this body at this epoch, a frame the kernels do not define -- and
carrying on with less would navigate them.  It was resisted because no
``except`` can tell that frame from a defect.  An exception type is not
evidence: ``oops`` declines with ``ValueError`` and ``LookupError``, and so does
a bug inside ``oops``, inside ``numpy``, or here.  Any rule narrow enough to
name the honest declines admits the dishonest ones wearing the same type.

And the cost of admitting one is not a lost image.  A degraded stage still
produces an offset: swallow a fault in the mask and the fit runs on against a
mask that was never built.  The document written for it then says a navigation
ran and concluded something, ``failed`` being the status for a run that finished
and concluded -- which no error filter selects, which ``--has-no-offset-file``
passes over because the document exists, and which no later pass corrects.  One
image that fails loudly costs a rerun.  One image that succeeds quietly on half
its evidence is a wrong answer kept forever, and nothing downstream can tell.

So every exception reaches the orchestrator, which fails that image with
``status_reason=internal_error`` and records the component that raised and the
type it raised.  A frame that no kernel set can answer for fails that way too,
and is meant to: it is a fact worth seeing rather than one worth working
around.

Coordinate conventions.  Positions -- the predicted center and the sunward
pixel that sets the symmetry axis -- are field-of-view coordinates plus the
extfov margin, the same convention every predicted position in the pipeline
uses (a catalog star's extfov position is ``star.v + extfov_margin_v``).
Holding to it is what lets a haze offset and a star offset on the same frame
be compared directly.  Bounding boxes are a separate matter: they are
integer pixel indices, they only bound where backplanes are evaluated, and
the ones handed to the backplane are deliberately UNCLIPPED because ``oops``
evaluates fine at off-detector pixel coordinates while a clipped box would
leave zero surface-intercept pixels on exactly the off-edge frames the
visibility condition exists for.
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
from spindoctor.support.types import NDArrayBoolType

__all__ = [
    'STAR_MASK_PHOTOMETRY_SPLIT_VMAG',
    'STAR_MASK_YBSC_MIN_VMAG',
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
            image scale could not be evaluated.
        r_env_px: Apparent haze-envelope radius in pixels; ``0.0`` under the
            same condition.
        km_per_px: Image scale at the body center in kilometers per pixel;
            ``0.0`` when it could not be evaluated.
        phase_deg: Phase angle at the body center in degrees.
        theta_rad: Symmetry-axis angle; ``atan2`` of the offset from the
            disc center to the minimum-incidence surface pixel.
        axis_degenerate: True when that offset was too short to define a
            direction (a near-zero-phase, rotationally symmetric disc) or
            when the geometry could not be evaluated at all.
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
    feature.  Every other exception is unexpected and fatal, and a bare
    ``ValueError`` would be indistinguishable from one -- ``oops`` raises those
    by the hundred, and so does a defect.
    """


def _finite(value: Any, name: str) -> float:
    """Return ``value`` as a finite float, raising when it is not one.

    Inventory coordinates come from a SPICE-driven projection that can
    return NaN or infinity for an unresolvable geometry.  Converting through
    this turns such a value into an exception inside the caller's guarded
    block, where it lands on the degenerate-geometry path, instead of into a
    NaN that silently poisons every quantity derived from it.

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

    An observation that cannot report those quantities is reported and the
    failure re-raised: a zero-sized frame would put every later measurement on
    a frame that does not exist.

    Parameters:
        obs: The observation to read.

    Returns:
        ``((height, width), window_px, (margin_v, margin_u))``.
    """
    try:
        shape = obs.extdata_shape_vu
        margin = obs.extfov_margin_vu
        bounds = (int(shape[0]), int(shape[1]))
        margin_vu = (float(margin[0]), float(margin[1]))
        window_px = max(margin_vu)
    except Exception:
        IMAGE_LOGGER.exception('Titan: observation exposes no extended-FOV geometry')
        raise
    return bounds, window_px, margin_vu


def _filter_names(obs: Observation) -> tuple[str, ...]:
    """Return the image's filter names, or an empty tuple when it has none.

    Single-filter instruments and observation stand-ins carry no filter
    attributes at all, so the lookup is by presence rather than by
    assumption.
    """
    names = [getattr(obs, attr, None) for attr in ('filter1', 'filter2')]
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
    """Return defensible defaults for a frame whose geometry did not evaluate.

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
    """Return the image scale, apparent radii, and phase, or None on failure.

    ``km_per_px`` averages the per-axis center resolutions; the radii come
    from the body's registered equatorial radius plus the configured
    atmosphere height.
    """
    try:
        ext_bp = obs.ext_bp
        res_u = float(ext_bp.center_resolution(TITAN_BODY_NAME, axis='u').vals)
        res_v = float(ext_bp.center_resolution(TITAN_BODY_NAME, axis='v').vals)
        phase_deg = float(np.degrees(ext_bp.center_phase_angle(TITAN_BODY_NAME).vals))
        radius_km = _body_radius_km(TITAN_BODY_NAME)
    except Exception:
        IMAGE_LOGGER.exception('Titan: image scale / phase unavailable')
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


def _restricted_backplane(
    obs: Observation, bbox_nominal: tuple[int, int, int, int]
) -> tuple[Backplane, Meshgrid]:
    """Build a one-sample-per-pixel backplane over a nominal-frame bbox.

    ``bbox_nominal`` is ``(u_min, u_max, v_min, v_max)`` in nominal-frame
    pixel indices, which may run negative inside the extfov margin or past
    the detector: ``oops`` backplanes evaluate at off-detector pixel
    coordinates, and clipping the box would leave zero surface-intercept
    pixels on exactly the off-edge frames the visibility condition targets.
    """
    u_min, u_max, v_min, v_max = bbox_nominal
    meshgrid = Meshgrid.for_fov(
        obs.fov,
        origin=(u_min + 0.5, v_min + 0.5),
        limit=(u_max + 0.5, v_max + 0.5),
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
) -> tuple[float, bool]:
    """Return ``(theta_rad, axis_degenerate)`` from the incidence backplane.

    The visible pixel of MINIMUM incidence always projects in the sunward
    image direction, at every phase.  (The maximum-incidence pixel is the
    anti-solar surface point, which becomes visible past 90 degrees phase
    and points the wrong way, so the choice is deliberately phase-free.)
    The axis angle is ``atan2`` of that pixel's offset from the disc center;
    an offset shorter than ``axis_min_offset_px`` means a near-zero-phase
    disc that is rotationally symmetric, where any axis is equally valid.

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
        axis_min_offset_px: Offset below which the axis is degenerate.

    Returns:
        ``(theta_rad, axis_degenerate)``.  A box with no surface-intercept
        pixel in it yields ``(0.0, True)``, which is the honest answer for a
        frame that shows no lit surface.  A backplane that could not be
        evaluated is reported and re-raised.
    """
    try:
        bp, meshgrid = _restricted_backplane(obs, bbox_nominal)
        incidence = bp.incidence_angle(TITAN_BODY_NAME)
        invalid = np.asarray(incidence.expand_mask().mask, dtype=bool)
        values = np.asarray(incidence.vals, dtype=np.float64)
        uv = np.asarray(meshgrid.uv.vals, dtype=np.float64)
    except Exception:
        IMAGE_LOGGER.exception('Titan: incidence backplane could not be evaluated')
        raise
    valid = ~invalid
    if not valid.any():
        IMAGE_LOGGER.info('Titan: no surface-intercept pixels in the envelope box')
        return 0.0, True
    if uv.shape[:-1] != values.shape:
        IMAGE_LOGGER.warning(
            'Titan: meshgrid shape %r does not match the incidence backplane %r',
            uv.shape[:-1],
            values.shape,
        )
        return 0.0, True
    index = np.unravel_index(int(np.argmin(np.where(valid, values, np.inf))), values.shape)
    sun_u = float(uv[index][0]) + margin_vu[1]
    sun_v = float(uv[index][1]) + margin_vu[0]
    d_v = sun_v - center_vu[0]
    d_u = sun_u - center_vu[1]
    if math.hypot(d_v, d_u) < axis_min_offset_px:
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
        The bbox-local boolean mask, or ``None`` when no ring pixel qualifies.
        Ring backplanes that could not be evaluated are reported and the
        failure re-raised, because a frame behind the rings and a frame whose
        rings could not be rendered look identical from an empty mask.
    """
    ring_target = f'{planet.lower()}{_RING_TARGET_SUFFIX}'
    try:
        radius = np.asarray(bp.ring_radius(ring_target).mvals.filled(np.nan), dtype=np.float64)
        distance = np.asarray(bp.distance(ring_target).mvals.filled(np.inf), dtype=np.float64)
    except Exception:
        IMAGE_LOGGER.exception('Titan: ring backplanes could not be evaluated')
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
        except Exception:
            IMAGE_LOGGER.exception('Titan: %s star query failed', catalog_name)
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
    try:
        bp, _ = _restricted_backplane(obs, mask_bbox)
    except Exception:
        IMAGE_LOGGER.exception('Titan: mask-box backplane could not be evaluated')
        raise
    sibling_ranges = [
        (name.upper(), float(entry.get('range', float('inf')))) for name, entry in siblings
    ]
    body_local = occluder_mask_for_body(
        bp,
        TITAN_BODY_NAME,
        sibling_ranges,
        subject_range_km,
        oversample_v=1,
        oversample_u=1,
    )
    planet = getattr(obs, 'closest_planet', None)
    radii = nav_config['ring_occlusion_radii_km']
    ring_local = (
        None
        if planet is None
        else _ring_occlusion_local(
            bp, str(planet), subject_range_km, (float(radii[0]), float(radii[1]))
        )
    )
    for local in (body_local, ring_local):
        if local is not None:
            _embed_local(occluder_ext, local, mask_bbox, margin_vu)
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
    A frame whose inventory entry is not finite gets the degenerate geometry,
    which forces the reliability hard-zero path and keeps the always-emit
    invariant: the orchestrator reads a raising ``to_features`` as zero
    features, so a frame that is merely off the edge has to reach the gate
    with a record rather than vanish.  Every other failure is reported and
    re-raised, because a stage that could not be evaluated is not evidence
    that the frame is degenerate.

    Parameters:
        obs: Observation snapshot.
        config: Configuration supplying the ``titan`` section.
        inventory: Pre-computed Titan inventory entry; looked up from
            ``obs.inventory`` when omitted.
        siblings: ``(body_name, inventory_entry)`` for the other in-FOV
            bodies; enumerated from ``obs`` when omitted.

    Returns:
        A fully-populated :class:`TitanGeometryInputs`.  A frame whose
        inventory entry is not finite gets zero radii and
        ``axis_degenerate=True``, which forces the reliability hard-zero
        path.
    """
    extfov_shape_vu, window_px, extfov_margin_vu = _frame_bounds(obs)
    filters = _filter_names(obs)
    try:
        margin_vu = (int(obs.extfov_margin_vu[0]), int(obs.extfov_margin_vu[1]))
        if inventory is None:
            inventory = obs.inventory([TITAN_BODY_NAME], return_type='full')[TITAN_BODY_NAME]
        if siblings is None:
            siblings = [
                (name, entry)
                for name, entry in bodies_in_extfov(obs, config=config)
                if name.upper() != TITAN_BODY_NAME
            ]
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
        raw_range_km = float(inventory.get('range', float('inf')))
        # A NaN range would reach NavFeature, which rejects it outright; an
        # unknown distance is honestly infinite, not zero.
        subject_range_km = (
            raw_range_km if not math.isnan(raw_range_km) and raw_range_km >= 0.0 else float('inf')
        )
    except NonFiniteInventoryError:
        # The one expected condition, answered rather than propagated; see
        # "Nothing here absorbs an exception" in the module docstring for why
        # it is the only one, and why it has a type of its own.
        IMAGE_LOGGER.exception('Titan: inventory entry is not finite')
        return _degenerate_geometry(
            extfov_shape_vu=extfov_shape_vu,
            window_px=window_px,
            extfov_margin_vu=extfov_margin_vu,
            filters=filters,
        )
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
