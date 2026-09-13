"""Sum-type variants describing the geometry of a NavFeature.

Each ``NavFeature`` carries a ``geometry`` payload whose concrete dataclass
type matches the ``feature_type``.  The payload holds whatever the consuming
technique needs to know about *where in the image* the feature lives —
image-side operations remain global, so no payload describes a per-feature
image crop.

Coordinates are in extended-FOV (extfov) image coordinates (v, u).  Bounding
boxes are half-open in the numpy slicing sense: ``v_min, u_min, v_max,
u_max`` with ``arr[v_min:v_max, u_min:u_max]`` covering the box.
"""

from dataclasses import dataclass, field

import numpy as np

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'BodyBlobGeometry',
    'BodyDiscGeometry',
    'CartographicModelGeometry',
    'LimbPolyline',
    'NavFeatureGeometry',
    'RingAnnulusGeometry',
    'RingEdgePolyline',
    'StarGeometry',
    'TerminatorPolyline',
    'TitanHazeGeometry',
]


@dataclass(frozen=True, eq=False)
class StarGeometry:
    """Geometry payload for a STAR feature.

    Single (v, u) point in extended-FOV coordinates plus the catalog-derived
    prediction from which it was generated.  The two are equal at extraction
    time and may differ after a refinement step records the matched
    detection.

    Both are pixel indices in the padded frame, which is the convention a
    technique measures a centroid in.  The star record they come from holds
    oops uv instead, half a pixel higher on each axis; the conversion happens
    once, where the model emits the feature.

    Parameters:
        predicted_vu: Predicted star (v, u) as an extfov pixel index.
        catalog_vu: Catalog-aberrated star (v, u) as an extfov pixel index.
            Equal to ``predicted_vu`` at extraction; may differ after
            refinement.
        bbox_extfov_vu: Half-open bounding box ``(v_min, u_min, v_max,
            u_max)`` covering the postage stamp around the predicted
            position.
    """

    predicted_vu: tuple[float, float]
    catalog_vu: tuple[float, float]
    bbox_extfov_vu: tuple[int, int, int, int]


@dataclass(frozen=True, eq=False)
class LimbPolyline:
    """Geometry payload for a LIMB_ARC feature.

    A polyline of vertices along a body's predicted limb, after
    extraction-time cropping for occlusion / off-FOV / shadow.  Each vertex
    carries its own normal direction and per-vertex anisotropic uncertainty.

    Parameters:
        vertices_vu: ``(N, 2)`` array of (v, u) per surviving vertex.
        normals_vu: ``(N, 2)`` array of outward limb normal per vertex.
        sigma_normal_per_vertex_px: ``(N,)`` per-vertex sigma along normal.
        sigma_tangent_per_vertex_px: ``(N,)`` per-vertex sigma along tangent;
            typically a small constant (~0.5 px) reflecting polyline
            sampling resolution.
        bbox_extfov_vu: Half-open bounding box of the polyline.
    """

    vertices_vu: NDArrayFloatType
    normals_vu: NDArrayFloatType
    sigma_normal_per_vertex_px: NDArrayFloatType
    sigma_tangent_per_vertex_px: NDArrayFloatType
    bbox_extfov_vu: tuple[int, int, int, int]


@dataclass(frozen=True, eq=False)
class TerminatorPolyline:
    """Geometry payload for a TERMINATOR_ARC feature.

    Mirrors ``LimbPolyline`` with terminator-specific semantics — the vertices
    lie along the terminator (where ``cos(incidence) == 0``) rather than the
    silhouette.  Per-vertex sigma_normal is generally larger than the
    matching limb because albedo variation softens the photometric edge.

    Parameters: see ``LimbPolyline`` (identical field set).
    """

    vertices_vu: NDArrayFloatType
    normals_vu: NDArrayFloatType
    sigma_normal_per_vertex_px: NDArrayFloatType
    sigma_tangent_per_vertex_px: NDArrayFloatType
    bbox_extfov_vu: tuple[int, int, int, int]


@dataclass(frozen=True, eq=False)
class RingEdgePolyline:
    """Geometry payload for a RING_EDGE feature.

    Polyline of vertices along one named ring edge.  Each vertex's per-axis
    uncertainty is along the radial direction (across the edge) and along
    the edge tangent.  The straight-line flag is set when the projected
    polyline's deviation from a best-fit straight line is below threshold;
    in that case its rank-1 covariance must be combined with another feature
    to resolve a 2-D offset.

    Parameters:
        vertices_vu: ``(N, 2)`` (v, u) per vertex.
        normals_vu: ``(N, 2)`` radially outward per vertex.
        sigma_radial_per_vertex_px: ``(N,)`` sigma across the edge (radial).
        sigma_along_edge_per_vertex_px: ``(N,)`` sigma along the edge.
        is_straight_line: ``True`` if the polyline's max-deviation from a
            best-fit straight line is below the curvature threshold.
        bbox_extfov_vu: Half-open bounding box of the polyline.
        sigma_orbit_radial_px: Fully-correlated 1-sigma radial displacement
            of the whole predicted edge (the catalog orbit-solution
            uncertainty), in pixels at the feature.  Distinct from
            ``sigma_radial_per_vertex_px``: the per-vertex sigma is the
            statistical scale of one vertex's residual and averages down as
            ``1/sqrt(N)`` in the fit covariance, while an orbit error
            displaces every vertex coherently and does not average down.
            ``RingEdgeNav`` adds this term in quadrature to its reported
            covariance along the fit's radial direction so a tight lock on
            an uncertain orbit is not reported as a tight pointing fix.
    """

    vertices_vu: NDArrayFloatType
    normals_vu: NDArrayFloatType
    sigma_radial_per_vertex_px: NDArrayFloatType
    sigma_along_edge_per_vertex_px: NDArrayFloatType
    is_straight_line: bool
    bbox_extfov_vu: tuple[int, int, int, int]
    sigma_orbit_radial_px: float = 0.0


@dataclass(frozen=True, eq=False)
class BodyDiscGeometry:
    """Geometry payload for a BODY_DISC feature.

    The body's full-disc rendering is carried on ``NavFeature.template_img``;
    the geometry payload only records the position of that template within
    the extfov image, the predicted body-center pixel, and the fraction of
    the predicted disc area that falls outside the sensor.

    Parameters:
        bbox_extfov_vu: Half-open bounding box where the template sits.
        predicted_center_vu: Predicted body center in extfov coordinates.
        overflow_fraction: Fraction of the disc area outside the sensor
            ``[0, 1]``; ``0`` means fully in-FOV.
    """

    bbox_extfov_vu: tuple[int, int, int, int]
    predicted_center_vu: tuple[float, float]
    overflow_fraction: float


@dataclass(frozen=True, eq=False)
class BodyBlobGeometry:
    """Geometry payload for a BODY_BLOB feature.

    Carries only the predicted centroid and bounding extent of an under-
    resolved or irregular body.  No template is rendered.

    Parameters:
        predicted_center_vu: Predicted body center in extfov coordinates.
        bbox_extfov_vu: Half-open bounding box around the predicted body.
        predicted_diameter_px: Predicted disc diameter in pixels (longer
            axis of the predicted ellipse silhouette).
    """

    predicted_center_vu: tuple[float, float]
    bbox_extfov_vu: tuple[int, int, int, int]
    predicted_diameter_px: float


@dataclass(frozen=True, eq=False)
class RingAnnulusGeometry:
    """Geometry payload for a RING_ANNULUS feature.

    Multi-ring composite template carried on ``NavFeature.template_img``;
    this payload records only the template's location in the extfov image
    and the predicted ring-system center.

    Parameters:
        bbox_extfov_vu: Half-open bounding box where the template sits.
        predicted_center_vu: Predicted planet center for the ring system.
        orbit_normals_vu: ``(M, 2)`` outward radial normals of the constituent
            ring edges painted into the composite template, concatenated across
            edges with their signs intact.  This is the annulus fit's own
            radial geometry: a coherent catalog-orbit error moves every one of
            these vertices along its own normal, and ``RingAnnulusNav`` uses
            the aggregate to derive how much of that displacement its
            translation-only NCC absorbs (the same absorbed-sensitivity solve
            ``RingEdgeNav`` runs on its fit vertices).  Empty when the emitting
            model tracks no per-edge geometry for the composite.
        sigma_orbit_radial_px: Effective fully-correlated 1-sigma radial
            displacement of the predicted annulus (the catalog orbit-solution
            uncertainty), in pixels, aggregated over the constituent edges by
            their vertex share.  Distinct from any per-vertex statistical
            sigma: an orbit error displaces the whole annulus coherently and
            does not average down over vertices, so ``RingAnnulusNav`` adds it
            to its reported covariance through the absorbed translation
            direction rather than letting a tight NCC lock on a misplaced
            annulus be reported as a tight pointing fix.  ``0.0`` when no
            constituent edge declares an orbit uncertainty.
    """

    bbox_extfov_vu: tuple[int, int, int, int]
    predicted_center_vu: tuple[float, float]
    orbit_normals_vu: NDArrayFloatType = field(default_factory=lambda: np.zeros((0, 2)))
    sigma_orbit_radial_px: float = 0.0


@dataclass(frozen=True, eq=False)
class CartographicModelGeometry:
    """Geometry payload for a CARTOGRAPHIC_MODEL feature.

    A pre-built cartographic mosaic of a body, reprojected into the predicted
    body silhouette and stored on ``NavFeature.template_img``.  Same shape
    as ``BodyDiscGeometry`` — the difference is that the template carries
    surface detail rather than smooth Lambert shading.

    Parameters:
        bbox_extfov_vu: Half-open bounding box where the template sits.
        predicted_center_vu: Predicted body center in extfov coordinates.
        overflow_fraction: Fraction of the disc area outside the sensor.
    """

    bbox_extfov_vu: tuple[int, int, int, int]
    predicted_center_vu: tuple[float, float]
    overflow_fraction: float


@dataclass(frozen=True, eq=False)
class TitanHazeGeometry:
    """Geometry payload for a TITAN_LIMB feature.

    Describes a hazy body at its predicted pointing: where the geometric
    disc center is, which way the sub-solar direction points, how big the
    solid body and its haze envelope are, and which pixels the fit must
    ignore.  The consuming technique needs no other scene knowledge.

    Parameters:
        predicted_center_vu: Geometric disc center in extfov coordinates --
            the body's projected field-of-view center plus the extfov
            margin.  NOT a brightness-weighted centroid, which phase biases
            along the very axis the haze fit measures.
        sun_angle_rad: Symmetry-axis angle ``theta``; the unit vector
            ``(sin theta, cos theta)`` in ``(v, u)`` points from the disc
            center toward the sub-solar side.
        axis_degenerate: True when the sub-solar direction could not be
            localized -- a near-zero-phase disc that is rotationally
            symmetric, or a frame whose geometry could not be evaluated.
            ``sun_angle_rad`` is then ``0.0`` and any axis is equally
            valid, so the consuming technique skips angle refinement.
        phase_deg: Phase angle (Sun -> body -> observer) at the disc
            center, in degrees.
        r_solid_px: Apparent radius of the solid body in pixels.
        r_env_px: Apparent radius of the haze envelope (solid radius plus
            the configured atmosphere height) in pixels.
        km_per_px: Image scale at the body center, in kilometers per
            pixel.
        contaminant_mask: Boolean array of the extfov image shape marking
            pixels the fits must ignore -- nearer bodies, ring occlusion,
            in-frame sibling bodies, and bright catalog stars -- or
            ``None`` when nothing is masked.  Supplied UNDILATED at
            predicted geometry: because a pointing error translates the
            whole scene identically, the consuming technique shifts the
            mask by its current center hypothesis and dilates it along the
            symmetry axis, rather than applying it statically.
        filters: Instrument filter names for this image, recorded so
            filter-dependent haze behavior is analyzable from production
            output.
        bbox_extfov_vu: Half-open bounding box ``(v_min, u_min, v_max,
            u_max)`` covering the haze envelope in extfov coordinates.
    """

    predicted_center_vu: tuple[float, float]
    sun_angle_rad: float
    axis_degenerate: bool
    phase_deg: float
    r_solid_px: float
    r_env_px: float
    km_per_px: float
    contaminant_mask: NDArrayBoolType | None
    filters: tuple[str, ...]
    bbox_extfov_vu: tuple[int, int, int, int]


NavFeatureGeometry = (
    StarGeometry
    | LimbPolyline
    | TerminatorPolyline
    | RingEdgePolyline
    | BodyDiscGeometry
    | BodyBlobGeometry
    | RingAnnulusGeometry
    | CartographicModelGeometry
    | TitanHazeGeometry
)
"""Sum type spanning every NavFeatureType's geometry payload."""
