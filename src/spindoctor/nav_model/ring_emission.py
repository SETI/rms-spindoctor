"""NavFeature construction for the catalog ring model.

Builds the ``RING_EDGE`` and ``RING_ANNULUS`` features from rendered
polylines and templates, including the per-vertex sigmas, the coherent
orbit-uncertainty term, and the per-feature reliability sigmoids.  Split out
of :mod:`spindoctor.nav_model.nav_model_rings` to keep both modules under the
size cap.
"""

from __future__ import annotations

import math

import numpy as np

from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import RingAnnulusFlags, RingEdgeFlags
from spindoctor.feature.geometry import RingAnnulusGeometry, RingEdgePolyline
from spindoctor.nav_model.rings import RingFeature
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__: list[str] = []


RING_EDGE_DEFAULT_RELIABILITY: float = 0.7
"""Catalog default reliability for ring edges before per-image scaling.

Mirrors the design's "catalog_default_reliability" term in the RING_EDGE
sigmoid.  Phase-5 calibration may override it per planet ring.
"""


RING_EDGE_SIGMA_ALONG_PX: float = 0.5
"""Per-vertex sigma_along_edge in pixels.

Reflects polyline-sampling resolution; the design specifies ~0.5 px.
"""


def _build_edge_feature(
    *,
    ring_feat: RingFeature,
    edge_label: str,
    label_text: str,
    planet: str,
    vertices_vu: NDArrayFloatType,
    normals_vu: NDArrayFloatType,
    uncertainty_km: float,
    orbit_sigma_km: float,
    km_per_pixel_radial: float,
    orbit_km_per_pixel_radial: float,
    is_straight_line: bool,
    bbox: tuple[int, int, int, int],
    subject_range_km: float,
    source_model: str,
) -> NavFeature:
    """Construct a RING_EDGE NavFeature from polyline data.

    ``uncertainty_km`` scales the per-vertex radial sigma (the statistical
    residual scale of the robust fit), converted at the floored whole-image
    ``km_per_pixel_radial`` because that sigma is a rasterization-resolution
    scale.  ``orbit_sigma_km`` is this edge's own orbit-solution uncertainty,
    carried on ``RingEdgePolyline.sigma_orbit_radial_px`` so ``RingEdgeNav``
    can widen its reported covariance -- a coherent orbit error does not
    average down over vertices the way the per-vertex sigma does.  It converts
    at ``orbit_km_per_pixel_radial``, the edge's OWN unfloored radial scale: a
    physical km displacement grows in pixels as resolution improves, so the
    per-vertex floor would understate it on the sub-km/px frames where it
    matters most.
    """
    n = vertices_vu.shape[0]
    sigma_radial_px = np.full(n, uncertainty_km / km_per_pixel_radial, dtype=np.float64)
    sigma_along_px = np.full(n, RING_EDGE_SIGMA_ALONG_PX, dtype=np.float64)
    sigma_orbit_px = orbit_sigma_km / orbit_km_per_pixel_radial
    visible_arc_fraction = 1.0
    feature_id = f'ring_edge:{planet}:{ring_feat.key}:{edge_label}'
    reliability = _ring_edge_reliability(
        catalog_default=RING_EDGE_DEFAULT_RELIABILITY,
        visible_arc_fraction=visible_arc_fraction,
        shadow_occluded_fraction=0.0,
        is_straight_line=is_straight_line,
    )
    return NavFeature(
        feature_id=feature_id,
        feature_type=NavFeatureType.RING_EDGE,
        source_model=source_model,
        geometry=RingEdgePolyline(
            vertices_vu=vertices_vu,
            normals_vu=normals_vu,
            sigma_radial_per_vertex_px=sigma_radial_px,
            sigma_along_edge_per_vertex_px=sigma_along_px,
            is_straight_line=is_straight_line,
            bbox_extfov_vu=bbox,
            sigma_orbit_radial_px=sigma_orbit_px,
        ),
        subject_range_km=subject_range_km,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=reliability,
        reliability_reasons=NavReliabilityBreakdown(
            visible_arc_fraction=visible_arc_fraction,
            shadow_occluded_fraction=0.0,
        ),
        usable_types=frozenset({NavFeatureType.RING_EDGE}),
        flags=RingEdgeFlags(
            is_straight_line=is_straight_line,
            polarity_predictable=False,
            edge_name=f'{ring_feat.key}:{edge_label}',
            planet_name=planet,
        ),
    )


def _aggregate_annulus_orbit_terms(
    orbit_terms: list[tuple[NDArrayFloatType, int, float]],
) -> tuple[NDArrayFloatType, float]:
    """Collapse per-edge annulus orbit terms into the composite's channel inputs.

    Each entry is ``(normals_vu, vertex_count, orbit_sigma_px)`` for one edge
    painted into the composite annulus.  Returns the concatenated outward
    normals (signs intact, so opposing radial sides cancel in the downstream
    absorbed-translation solve exactly as they do for RingEdgeNav) and the
    vertex-weighted effective orbit sigma.

    The per-edge sigmas are combined as a vertex-weighted mean, not in
    quadrature: the constituent edges of one ringlet share the same orbit
    solution, so their radial errors are treated as fully correlated -- the
    same conservative same-sense reading RingEdgeNav's
    ``_effective_orbit_sigma_px`` uses.  Longer visible arcs carry proportionally
    more weight because they dominate the composite NCC.

    Parameters:
        orbit_terms: Per-edge ``(normals_vu, vertex_count, orbit_sigma_px)``.

    Returns:
        ``(orbit_normals_vu, sigma_orbit_radial_px)``.  The normals are
        ``(0, 2)`` and the sigma ``0.0`` when no edge declares an orbit
        uncertainty or the list is empty.
    """
    normals_list = [normals for normals, _count, _sigma in orbit_terms if normals.shape[0] > 0]
    if not normals_list:
        return np.zeros((0, 2), dtype=np.float64), 0.0
    orbit_normals_vu = np.concatenate(normals_list, axis=0)
    weighted_sum = 0.0
    weight_total = 0.0
    for _normals, count, sigma in orbit_terms:
        weighted_sum += float(count) * sigma
        weight_total += float(count)
    sigma_orbit_radial_px = weighted_sum / weight_total if weight_total > 0.0 else 0.0
    return orbit_normals_vu, sigma_orbit_radial_px


def _build_annulus_feature(
    *,
    ring_name: str,
    planet: str,
    model_img: NDArrayFloatType,
    model_mask: NDArrayBoolType,
    bbox: tuple[int, int, int, int],
    predicted_center_vu: tuple[float, float],
    orbit_normals_vu: NDArrayFloatType,
    sigma_orbit_radial_px: float,
    subject_range_km: float,
    constituent_count: int,
    source_model: str,
) -> NavFeature:
    """Construct a RING_ANNULUS NavFeature when edges compress radially.

    ``orbit_normals_vu`` are the outward radial normals of the constituent
    edges concatenated into the composite, and ``sigma_orbit_radial_px`` is
    the vertex-weighted effective orbit-solution uncertainty over those edges.
    Both are carried so ``RingAnnulusNav`` can widen its reported covariance by
    the coherent radial orbit error along the translation its NCC would absorb
    it into -- the correlation-side analog of the ``RingEdgeNav`` channel.
    """
    feature_id = f'ring_annulus:{planet}:{ring_name}'
    return NavFeature(
        feature_id=feature_id,
        feature_type=NavFeatureType.RING_ANNULUS,
        source_model=source_model,
        geometry=RingAnnulusGeometry(
            bbox_extfov_vu=bbox,
            predicted_center_vu=predicted_center_vu,
            orbit_normals_vu=orbit_normals_vu,
            sigma_orbit_radial_px=sigma_orbit_radial_px,
        ),
        subject_range_km=subject_range_km,
        position_cov_px=None,
        intensity_sigma_rel=0.0,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=_ring_annulus_reliability(
            constituent_count=constituent_count,
            radial_extent_px=float(bbox[2] - bbox[0]),
        ),
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
        usable_types=frozenset({NavFeatureType.RING_ANNULUS}),
        flags=RingAnnulusFlags(
            planet_name=planet,
            constituent_edge_count=constituent_count,
        ),
        template_img=model_img,
        template_mask=model_mask,
    )


def _ring_edge_reliability(
    *,
    catalog_default: float,
    visible_arc_fraction: float,
    shadow_occluded_fraction: float,
    is_straight_line: bool,
) -> float:
    """Reliability of RING_EDGE per the design's formula.

    ``catalog_default * visible_arc_fraction * (1 - shadow_occluded_fraction)``
    times a sigmoid of the (yet-uncalibrated) emission-angle factor; we
    approximate the sigmoid by a constant in this implementation pending
    Phase-5 calibration.  Straight-line edges get a 70% multiplier
    because they contribute rank-1 constraints only.
    """
    base = catalog_default * visible_arc_fraction * (1.0 - shadow_occluded_fraction)
    if is_straight_line:
        base = base * 0.7
    return float(np.clip(base, 0.0, 1.0))


def _ring_annulus_reliability(*, constituent_count: int, radial_extent_px: float) -> float:
    """Reliability of RING_ANNULUS per the design's formula.

    The plan specifies
    ``mean(constituent_reliabilities) * sigmoid(radial_extent_px / 50 - 1)``.
    Per-edge constituent reliabilities are not tracked yet (the catalog
    surfaces edge ``rms`` but no per-edge reliability scalar), so we
    substitute the catalog-default reliability scaled by the number of
    constituent edges (more edges -> more confident annulus, capped at 1).

    Parameters:
        constituent_count: Number of catalog edges fused into this annulus.
        radial_extent_px: Width of the annulus in the radial direction.

    Returns:
        Reliability in ``[0, 1]``.
    """
    if constituent_count <= 0:
        return 0.0
    sigmoid_term = 1.0 / (1.0 + math.exp(-(radial_extent_px / 50.0 - 1.0)))
    constituent_term = min(1.0, constituent_count / 5.0) * RING_EDGE_DEFAULT_RELIABILITY
    return float(np.clip(constituent_term * sigmoid_term, 0.0, 1.0))
