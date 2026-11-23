"""Limb and terminator arcs of a simulated body.

The simulated body model hands these functions what it predicted for one body (the
rendered image, its masks, its idealized parameters, and a way to render it again) as a
:class:`BodyArcInputs`, and they return the body's ``LIMB_ARC`` and ``TERMINATOR_ARC``
features.  Both arcs follow the SPICE-backed ``NavModelBody``'s geometry and gating
semantics: the same polyline conventions, emission gates, and reliability inputs, so a
simulated scene exercises the same techniques a real frame would.  The per-vertex sigma
model is the deliberate exception.  The real model derives per-vertex sigmas from the
PSF, limb softness, and albedo terms, while the noise-free simulated render carries
fixed per-vertex values (see ``_LIMB_SIGMA_NORMAL_PX`` and
``_TERMINATOR_SIGMA_NORMAL_PX``).

Every value read here comes from the body's filtered navigator view, never from the
scene's truth.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from spindoctor.config import Config
from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import LimbArcFlags, TerminatorArcFlags
from spindoctor.feature.geometry import LimbPolyline, TerminatorPolyline
from spindoctor.nav_model.body_shape import load_body_shape
from spindoctor.nav_model.nav_model_body import (
    LIMB_ARC_MIN_VERTICES,
    TERMINATOR_MIN_PHASE_FACTOR,
    TERMINATOR_MIN_VERTICES,
    limb_reliability,
    shape_features_suppressed,
    terminator_reliability,
)
from spindoctor.nav_model.simulated.predicted_body import PredictedBody, predicted_semi_axes_px
from spindoctor.obs import ObsSnapshotInst
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.image import shift_array
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'BodyArcInputs',
    'BodyRenderer',
    'build_limb_arc_feature',
    'build_terminator_arc_feature',
]

# Minimum silhouette diameter to emit a LIMB_ARC.  The catalog body model gates
# the limb on its ellipsoid-fit uncertainty (``<= 3 px``) plus the shared
# vertex-count floor; the sim has no km scale, so it gates on resolution
# directly.  Below this the limb fit is imprecise and -- because it is an
# LM-refined distance-transform fit -- it injects cross-process jitter into the
# fused offset of any body scene, so only a well-resolved body emits a limb.
_MIN_LIMB_DIAMETER_PX: float = 100.0
# Per-vertex limb uncertainty for a simulated body.  The rendered silhouette edge
# is sharp and noise-free, so the predicted limb sits within ~1 px of the image
# edge; the tangent sigma reflects the one-pixel polyline sampling resolution.
_LIMB_SIGMA_NORMAL_PX: float = 1.0
_LIMB_SIGMA_TANGENT_PX: float = 0.5
# Above this phase the lit limb is a thin crescent and most of the silhouette
# boundary is terminator (a soft brightness gradient, not a sharp image edge), so
# a limb fit is unreliable -- the body is then navigated by the blob centroid.
# Gating emission here keeps LIMB_ARC off the high-phase scenes and matches the
# catalog body model's limb/blob handoff.
_LIMB_MAX_PHASE_DEG: float = 60.0
# Per-vertex terminator uncertainty for a simulated body.  The terminator is a
# soft photometric boundary (a shading ramp crossing zero) rather than a sharp
# silhouette edge, so its normal sigma is one pixel wider than the limb's; the
# tangent sigma reflects the one-pixel polyline sampling resolution.  The
# catalog body model derives these from the ellipsoid residual and albedo; the
# noise-free sim render has neither, so fixed values stand in.
_TERMINATOR_SIGMA_NORMAL_PX: float = 2.0
_TERMINATOR_SIGMA_TANGENT_PX: float = 0.5


class BodyRenderer(Protocol):
    """Renders the simulated body's own predicted image and silhouette.

    The simulated body model's renderer satisfies this: it draws the body from its
    idealized parameters at the given phase, on the observation's data canvas unless a
    canvas and center are given.
    """

    def __call__(
        self,
        phase_angle_rad: float,
        *,
        size: tuple[int, int] | None = None,
        center: tuple[float, float] | None = None,
    ) -> PredictedBody:
        """Render the body at ``phase_angle_rad`` on an optional canvas and center."""
        ...  # pragma: no cover - protocol stub


# eq=False: the ndarray fields make a generated __eq__ ambiguous (an elementwise array
# comparison has no single truth value), and nothing compares two of these records.
@dataclass(frozen=True, eq=False)
class BodyArcInputs:
    """What the arc builders read from one simulated body model.

    The simulated body model builds this once each time it renders its body, and it keeps
    ownership of the state; the arc builders only read it.

    Parameters:
        obs: The observation the body was rendered for; supplies the data shape
            and extfov margins.
        nav_body: The body's entry in the navigator view.  The visible-arc fraction
            reads ``semi_a_px``, ``semi_b_px``, and ``semi_c_px`` from it.
        config: Configuration the body's shape catalog entry is loaded from.
        body_name: Upper-case body name, used in feature IDs and flags.
        source_model: Name of the model instance emitting the features.
        model_img: Extfov-shape rendered body image at the model's phase.  It is 0 on
            the night side, so brightness above 0 is the lit region.
        body_mask: Extfov-shape boolean silhouette of the body: the pixels its
            outline touches, lit or not.
        lit_mask: Extfov-shape boolean lit region: the silhouette pixels whose mean
            angle of incidence is below 90 deg, the SPICE-backed body model's rule.
            The limb keeps only these pixels.
        limb_mask: Extfov-shape boolean silhouette boundary of ``body_mask``.
        occluder_mask: Extfov-shape union silhouette of the explicitly nearer sibling
            bodies, or ``None`` when no sibling occludes this one.
        bbox_extfov_vu: The body's half-open extfov bounding box, carried on each
            polyline.
        subject_range_km: Subject distance in km.
        predicted_diameter_px: Longer pixel extent of the rendered silhouette.
        phase_angle_deg: The model's phase angle in degrees.
        render: Renders the body's own predicted image and silhouette again, for the
            unclipped whole-body and phase-0 renders.
        silhouette_boundary: Takes a boolean silhouette and returns its boundary mask:
            the silhouette pixels that have at least one 4-neighbor outside the
            silhouette.  It reads nothing but its argument.
    """

    obs: ObsSnapshotInst
    nav_body: dict[str, Any]
    config: Config
    body_name: str
    source_model: str
    model_img: NDArrayFloatType
    body_mask: NDArrayBoolType
    lit_mask: NDArrayBoolType
    limb_mask: NDArrayBoolType
    occluder_mask: NDArrayBoolType | None
    bbox_extfov_vu: tuple[int, int, int, int]
    subject_range_km: float
    predicted_diameter_px: float
    phase_angle_deg: float
    render: BodyRenderer
    silhouette_boundary: Callable[[NDArrayBoolType], NDArrayBoolType]


@dataclass(frozen=True)
class _ArcKind:
    """The fixed emission choices that tell one arc family from the other.

    Parameters:
        feature_type: The feature type emitted.
        id_prefix: Feature-ID prefix; the body name follows a colon.
        geometry: The polyline geometry class the feature carries.
        sigma_normal_px: Per-vertex sigma along the normal, in pixels.
        sigma_tangent_px: Per-vertex sigma along the tangent, in pixels.
    """

    feature_type: NavFeatureType
    id_prefix: str
    geometry: type[LimbPolyline] | type[TerminatorPolyline]
    sigma_normal_px: float
    sigma_tangent_px: float


_LIMB_ARC = _ArcKind(
    feature_type=NavFeatureType.LIMB_ARC,
    id_prefix='limb_arc',
    geometry=LimbPolyline,
    sigma_normal_px=_LIMB_SIGMA_NORMAL_PX,
    sigma_tangent_px=_LIMB_SIGMA_TANGENT_PX,
)
_TERMINATOR_ARC = _ArcKind(
    feature_type=NavFeatureType.TERMINATOR_ARC,
    id_prefix='terminator_arc',
    geometry=TerminatorPolyline,
    sigma_normal_px=_TERMINATOR_SIGMA_NORMAL_PX,
    sigma_tangent_px=_TERMINATOR_SIGMA_TANGENT_PX,
)


def _polyline_from_mask(
    edge_mask: NDArrayBoolType,
    region_mask: NDArrayBoolType,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Extract polyline vertices and outward normals from a 1-pixel edge mask.

    Each ``True`` pixel of ``edge_mask`` (a region pixel adjacent to a non-region
    pixel) becomes one vertex.  The outward normal points away from the region
    interior: it is the normalized sum of the unit directions toward each non-region
    4-neighbor, so a vertex on the sunward limb of a body silhouette gets a normal
    pointing into the sky.

    Parameters:
        edge_mask: Extfov-shape boolean edge mask (the region's boundary).
        region_mask: Extfov-shape boolean region the edge bounds.

    Returns:
        ``(vertices_vu, normals_vu)`` each shaped ``(N, 2)``; empty when the mask
        has no pixels.
    """
    if not edge_mask.any():
        empty: NDArrayFloatType = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    vs, us = np.where(edge_mask)
    vertices_vu = np.stack([vs.astype(np.float64), us.astype(np.float64)], axis=1)
    rows, cols = region_mask.shape
    normals_vu = np.zeros_like(vertices_vu)
    for i, (v, u) in enumerate(zip(vs, us, strict=True)):
        dv = 0.0
        du = 0.0
        if v > 0 and not region_mask[v - 1, u]:
            dv -= 1.0
        if v < rows - 1 and not region_mask[v + 1, u]:
            dv += 1.0
        if u > 0 and not region_mask[v, u - 1]:
            du -= 1.0
        if u < cols - 1 and not region_mask[v, u + 1]:
            du += 1.0
        norm = float(np.hypot(dv, du)) or 1.0
        normals_vu[i, 0] = dv / norm
        normals_vu[i, 1] = du / norm
    return vertices_vu, normals_vu


def _terminator_ridge_mask(
    model_img: NDArrayFloatType, disc: NDArrayBoolType
) -> tuple[NDArrayBoolType, NDArrayBoolType]:
    """Interior lit/unlit boundary ridge of a rendered body image.

    The night side renders 0, so brightness above 0 is the lit region, and the
    unlit disc is the silhouette minus the lit region.  A ridge pixel is a lit
    pixel with an unlit *interior* disc pixel as a 4-neighbor.  The interior
    restriction drops the night side's limb ring, which keeps the ridge off the
    silhouette everywhere the lit and unlit regions are separated by more than a
    pixel (the cusps of a very thin crescent are the exception; see
    ``_terminator_polyline``).

    Parameters:
        model_img: A rendered body image (any canvas).
        disc: The body's silhouette on the same canvas.

    Returns:
        ``(ridge_mask, lit_mask)`` boolean arrays of the same shape.
    """
    lit = model_img > 0.0
    dark_disc = disc & ~lit
    touches_sky = (
        shift_array(~disc, (-1, 0))
        | shift_array(~disc, (1, 0))
        | shift_array(~disc, (0, -1))
        | shift_array(~disc, (0, 1))
    )
    dark_disc_interior = dark_disc & ~touches_sky
    neighbor_dark = (
        shift_array(dark_disc_interior, (-1, 0))
        | shift_array(dark_disc_interior, (1, 0))
        | shift_array(dark_disc_interior, (0, -1))
        | shift_array(dark_disc_interior, (0, 1))
    )
    ridge = lit & neighbor_dark
    return ridge, lit


def _drop_occluded_vertices(
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    occluder_mask: NDArrayBoolType | None,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Drop polyline vertices hidden behind a nearer sibling body.

    A hidden arc has no counterpart edge in the image (the occluder's
    disc covers it), so fitting it could only anchor on the occluder's
    own limb; dropping it keeps the DT fit on arcs that exist and lets
    the visible-arc fraction count the hidden arc as lost.

    Parameters:
        vertices_vu: Polyline vertices, extfov coords, shape ``(N, 2)``.
        normals_vu: Matching outward normals, shape ``(N, 2)``.
        occluder_mask: Extfov-shape union silhouette of the nearer siblings, or
            ``None`` when no sibling occludes.

    Returns:
        The surviving ``(vertices_vu, normals_vu)``; unchanged when no
        sibling occludes.
    """
    if occluder_mask is None or vertices_vu.shape[0] == 0:
        return vertices_vu, normals_vu
    vs = vertices_vu[:, 0].astype(np.intp)
    us = vertices_vu[:, 1].astype(np.intp)
    keep = ~occluder_mask[vs, us]
    return vertices_vu[keep], normals_vu[keep]


def _visible_arc_fraction(
    inputs: BodyArcInputs,
    visible_vertex_count: int,
    *,
    edge_mask: Callable[[PredictedBody], NDArrayBoolType],
) -> float:
    """Fraction of an arc's full predicted length that survives in the frame.

    The sim analog of the SPICE-backed model's ``_visible_arc_fraction``
    (survivors over the predicted ridge): the denominator is the edge length
    of an *unclipped* render of the same body at the same phase and lighting --
    a canvas sized to the whole silhouette, independent of the frame -- and the
    numerator is the vertex count that survived frame clipping and sibling-body
    occlusion.  A fully framed, unoccluded body scores ~1.0; a body whose arc runs
    off the frame edge or behind a nearer body scores the surviving fraction.  The
    render reads only the body's own idealized params.

    Parameters:
        inputs: The body's arc inputs.
        visible_vertex_count: Vertex count of the surviving polyline.
        edge_mask: Maps the unclipped render to the boolean mask of the arc's edge
            (the silhouette boundary for the limb, the lit/unlit ridge for the
            terminator).

    Returns:
        The [0, 1] visible-arc fraction.  1.0 when the unclipped render has no
        edge to compare against (a degenerate geometry).
    """
    max_semi = max(predicted_semi_axes_px(inputs.nav_body))
    # Canvas comfortably larger than the silhouette: half again the
    # longest diameter (mesh relief can push past the ellipsoid bound)
    # plus a fixed margin.
    canvas = math.ceil(max_semi * 3.0) + 16
    full = inputs.render(
        float(np.radians(inputs.phase_angle_deg)),
        size=(canvas, canvas),
        center=(canvas / 2.0, canvas / 2.0),
    )
    total = int(np.count_nonzero(edge_mask(full)))
    if total <= 0:
        return 1.0
    return min(1.0, float(visible_vertex_count) / float(total))


def _arc_feature(
    inputs: BodyArcInputs,
    kind: _ArcKind,
    *,
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    reliability: float,
    reliability_reasons: NavReliabilityBreakdown,
    flags: LimbArcFlags | TerminatorArcFlags,
) -> NavFeature:
    """Build one arc feature from its surviving polyline and scores.

    Parameters:
        inputs: The body's arc inputs.
        kind: Which arc family to emit.
        vertices_vu: Surviving polyline vertices, extfov coords, shape ``(N, 2)``.
        normals_vu: Matching outward normals, shape ``(N, 2)``.
        reliability: The feature's reliability score.
        reliability_reasons: The inputs behind ``reliability``.
        flags: The arc family's flags.

    Returns:
        The feature, carrying the family's fixed per-vertex sigmas.
    """
    n = vertices_vu.shape[0]
    sigma_normal = np.full(n, kind.sigma_normal_px, dtype=np.float64)
    sigma_tangent = np.full(n, kind.sigma_tangent_px, dtype=np.float64)
    return NavFeature(
        feature_id=f'{kind.id_prefix}:{inputs.body_name}',
        feature_type=kind.feature_type,
        source_model=inputs.source_model,
        geometry=kind.geometry(
            vertices_vu=vertices_vu,
            normals_vu=normals_vu,
            sigma_normal_per_vertex_px=sigma_normal,
            sigma_tangent_per_vertex_px=sigma_tangent,
            bbox_extfov_vu=inputs.bbox_extfov_vu,
        ),
        subject_range_km=inputs.subject_range_km,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=reliability,
        reliability_reasons=reliability_reasons,
        usable_types=frozenset({kind.feature_type}),
        flags=flags,
    )


def _lit_geometric_limb_polyline(
    inputs: BodyArcInputs,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Geometric-limb polyline for the gates-off measurement path.

    The polyline is the silhouette boundary of a phase-0 render of the same body, with
    outward normals taken from that silhouette.  The boundary is then intersected with
    the lit region at the model's phase, the pixels whose mean angle of incidence is
    below 90 deg.  A body's silhouette does not change with phase, so the result is the
    lit part of the silhouette boundary, the limb the SPICE-backed body model emits.

    Parameters:
        inputs: The body's arc inputs.

    Returns:
        ``(vertices_vu, normals_vu)`` in extfov coordinates; empty when the phase-0
        render has no silhouette-boundary pixels in the frame.
    """
    obs = inputs.obs
    full_mask = inputs.render(0.0).silhouette
    boundary = inputs.silhouette_boundary(full_mask)
    ext_margin_v = int(obs.extfov_margin_v)
    ext_margin_u = int(obs.extfov_margin_u)
    slice_v = slice(ext_margin_v, ext_margin_v + int(obs.data_shape_v))
    slice_u = slice(ext_margin_u, ext_margin_u + int(obs.data_shape_u))
    # The lit region follows the SPICE-backed model's rule (#1056).
    lit = np.asarray(inputs.lit_mask[slice_v, slice_u], dtype=bool)
    full_mask_ext = obs.make_extfov_false()
    lit_boundary_ext = obs.make_extfov_false()
    full_mask_ext[slice_v, slice_u] = full_mask
    lit_boundary_ext[slice_v, slice_u] = boundary & lit
    return _polyline_from_mask(lit_boundary_ext, full_mask_ext)


def build_limb_arc_feature(
    inputs: BodyArcInputs,
    *,
    apply_emission_gates: bool,
) -> NavFeature | None:
    """Emit a LIMB_ARC from the lit part of the rendered silhouette boundary, or ``None``.

    The predicted limb is the silhouette boundary of the rendered body, kept only on its
    lit region, the pixels whose mean angle of incidence is below 90 deg.  That is the
    SPICE-backed body model's lit limb, by the same rule.  The unlit limb merges into
    the dark sky and carries no image edge, so a vertex there could only pull the fit
    toward some other edge.

    With the emission gates on, the feature is emitted only for a body whose predicted
    diameter is at least ``_MIN_LIMB_DIAMETER_PX`` and whose phase is at most
    ``_LIMB_MAX_PHASE_DEG``.  Its boundary must also keep at least
    ``LIMB_ARC_MIN_VERTICES`` vertices once those hidden behind a nearer sibling are
    dropped, because a tiny or barely-resolved body yields too short an arc.  The
    feature's visible-arc fraction compares the surviving polyline against the lit
    silhouette boundary of an unclipped whole-body render, and its reliability is the
    shared limb formula.

    With the gates off (the measurement path) the resolution, phase, and arc-length
    gates are skipped, no vertices are dropped for occlusion, and the polyline is the
    silhouette boundary of a phase-0 render, kept where the body is lit at its own
    phase (see ``_lit_geometric_limb_polyline``).
    Its visible-arc fraction and reliability are both 1.0, because no technique
    consumes the score there.

    Parameters:
        inputs: The body's arc inputs.
        apply_emission_gates: Whether to apply the navigation-policy gates; ``False``
            takes the measurement path instead.

    Returns:
        The LIMB_ARC feature, or ``None`` when a gate fails or the polyline is
        empty.
    """
    if apply_emission_gates:
        if inputs.predicted_diameter_px < _MIN_LIMB_DIAMETER_PX:
            return None
        if inputs.phase_angle_deg > _LIMB_MAX_PHASE_DEG:
            return None
        lit_limb = inputs.limb_mask & inputs.lit_mask
        vertices_vu, normals_vu = _polyline_from_mask(lit_limb, inputs.body_mask)
        vertices_vu, normals_vu = _drop_occluded_vertices(
            vertices_vu, normals_vu, inputs.occluder_mask
        )
        # The vertex floor matches BodyLimbNav's ``min_arc_vertices`` feasibility
        # floor: a shorter arc cannot constrain the fit.  It is the catalog body
        # model's own constant, so the two emission gates cannot desync.
        if vertices_vu.shape[0] < LIMB_ARC_MIN_VERTICES:
            return None
        # Reliability inputs computed as the SPICE-backed model does:
        # the visible-arc fraction compares the surviving polyline (net
        # of frame clipping and sibling-body occlusion) against the
        # unclipped whole-body lit silhouette boundary, and the shared
        # reliability formula scores it.  These feed BodyLimbNav's
        # visible_limb_arc_fraction confidence term, so a clipped or
        # occluded sim limb scores like a real one instead of pinning
        # every input at 1.0.
        visible_arc_fraction = _visible_arc_fraction(
            inputs,
            vertices_vu.shape[0],
            edge_mask=lambda full: inputs.silhouette_boundary(full.silhouette) & full.lit,
        )
        reliability = limb_reliability(
            visible_arc_fraction=visible_arc_fraction,
            visible_arc_px=float(vertices_vu.shape[0]),
        )
    else:
        vertices_vu, normals_vu = _lit_geometric_limb_polyline(inputs)
        if vertices_vu.shape[0] == 0:
            return None
        # Measurement path (realism match): the geometry is the product
        # and no technique consumes the score, so the reliability inputs
        # stay at their vacuous 1.0.
        visible_arc_fraction = 1.0
        reliability = 1.0
    return _arc_feature(
        inputs,
        _LIMB_ARC,
        vertices_vu=vertices_vu,
        normals_vu=normals_vu,
        reliability=reliability,
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=visible_arc_fraction),
        flags=LimbArcFlags(
            body_name=inputs.body_name,
            visible_arc_fraction=visible_arc_fraction,
        ),
    )


def _terminator_polyline(
    inputs: BodyArcInputs,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Interior lit/unlit boundary polyline for the terminator arc.

    The terminator is the boundary *inside* the disc where the lit
    hemisphere meets the unlit one -- distinct from the geometric limb,
    which is the disc's silhouette edge against sky.  The body mask is the whole
    visible disc, and the night side renders 0, so ``model_img > 0`` is the lit
    region and the unlit disc is their difference.  A terminator vertex is a lit
    pixel with an unlit *interior* disc pixel as a 4-neighbor -- the interior
    restriction drops the night side's limb ring, keeping the polyline interior to
    the disc everywhere except the cusp-adjacent vertices of a very thin crescent,
    where terminator and limb meet within a pixel and a few vertices land
    on the silhouette (at phase 150 on a sphere of radius 60 px, 13 of 152
    vertices; the SPICE-backed model's sampler shares the behavior).  The outward
    normal is the gradient of the lit mask, so it points from the lit side
    toward the unlit side, the convention the SPICE-backed body model's
    terminator sampler uses.

    Parameters:
        inputs: The body's arc inputs.

    Returns:
        ``(vertices_vu, normals_vu)`` in extfov coordinates; empty arrays
        when the body renders no interior lit/unlit boundary (a fully lit
        or fully unlit visible disc).
    """
    terminator_ridge, lit = _terminator_ridge_mask(
        np.asarray(inputs.model_img, dtype=np.float64), inputs.body_mask
    )
    return _polyline_from_mask(terminator_ridge, lit)


def build_terminator_arc_feature(inputs: BodyArcInputs) -> NavFeature | None:
    """Emit a TERMINATOR_ARC from the interior lit/unlit boundary, or ``None``.

    Mirrors the SPICE-backed body model's terminator gates: a resolved
    ``highly_irregular`` body suppresses the feature (the shared
    :func:`~spindoctor.nav_model.nav_model_body.shape_features_suppressed`
    policy -- a chaotic rotator's rendered terminator does not match the
    real body's), the phase must be far enough from zero for a terminator
    to separate from the limb (``sin(phase) >=
    TERMINATOR_MIN_PHASE_FACTOR``), and the boundary polyline must be long
    enough to constrain a fit (``>= TERMINATOR_MIN_VERTICES`` vertices once those
    hidden behind a nearer sibling are dropped).  Emission is ungated by the
    limb-path resolution / phase policy gates -- the terminator is exactly the
    high-phase feature those gates hand the body off to.  The feature's visible-arc
    fraction compares the surviving ridge against the ridge of an unclipped
    whole-body render, and its reliability is the shared terminator formula.  The
    recovered offset it produces stays ``confidence_provisional`` like every
    sim-anchored technique result, because ``BodyTerminatorNav`` is not yet
    calibrated against the simulated renderer.

    Parameters:
        inputs: The body's arc inputs.

    Returns:
        The TERMINATOR_ARC feature, or ``None`` when a gate fails.
    """
    shape = load_body_shape(inputs.body_name, config=inputs.config)
    if shape_features_suppressed(shape, inputs.predicted_diameter_px, config=inputs.config):
        return None
    phase_angle_factor = abs(math.sin(math.radians(inputs.phase_angle_deg)))
    if phase_angle_factor < TERMINATOR_MIN_PHASE_FACTOR:
        return None
    vertices_vu, normals_vu = _terminator_polyline(inputs)
    vertices_vu, normals_vu = _drop_occluded_vertices(vertices_vu, normals_vu, inputs.occluder_mask)
    if vertices_vu.shape[0] < TERMINATOR_MIN_VERTICES:
        return None
    # Reliability inputs computed as the SPICE-backed model does: the
    # visible-arc fraction compares the surviving ridge (net of frame
    # clipping and sibling-body occlusion) against the unclipped
    # whole-body ridge, and the shared reliability formula applies the
    # catalog albedo-variation and sin(phase) penalties.  These feed
    # BodyTerminatorNav's confidence terms, so a sim terminator scores
    # like a real one instead of pinning every input at 1.0.
    visible_arc_fraction = _visible_arc_fraction(
        inputs,
        vertices_vu.shape[0],
        edge_mask=lambda full: _terminator_ridge_mask(full.image, full.silhouette)[0],
    )
    albedo_penalty = min(1.0, shape.albedo_variation)
    reliability = terminator_reliability(
        visible_arc_fraction=visible_arc_fraction,
        albedo_variation=shape.albedo_variation,
        phase_factor=phase_angle_factor,
    )
    return _arc_feature(
        inputs,
        _TERMINATOR_ARC,
        vertices_vu=vertices_vu,
        normals_vu=normals_vu,
        reliability=reliability,
        reliability_reasons=NavReliabilityBreakdown(
            visible_arc_fraction=visible_arc_fraction,
            albedo_penalty=albedo_penalty,
        ),
        flags=TerminatorArcFlags(
            body_name=inputs.body_name,
            visible_arc_fraction=visible_arc_fraction,
            phase_angle_factor=min(1.0, phase_angle_factor),
        ),
    )
