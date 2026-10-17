"""Per-edge fit statistics for the ring-edge technique.

The residual and inlier summaries the spurious gates and the absent-edge
waiver consume, split out of
:mod:`spindoctor.nav_technique.nav_technique_ring_edge` to keep both modules
under the size cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.geometry import RingEdgePolyline
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__: list[str] = []

# Distance (px) within which a model vertex counts as on a detected edge pixel when
# measuring matched edge strength.
_EDGE_STRENGTH_MATCH_RADIUS_PX = 1.5
# Percentile of the in-frame gradient magnitude taken as the frame's strongest structure.
_EDGE_STRENGTH_REFERENCE_PERCENTILE = 99.9


def _per_edge_rms_summed(features: list[NavFeature], residuals: NDArrayFloatType) -> float:
    """Sum each consumed edge's UNWEIGHTED RMS DT residual.

    Each edge contributes ``sqrt(mean(r**2))`` over its own slice of the
    per-vertex residual array, and the per-edge values are summed.  No robust
    weighting is applied, which is the point: the LM's own ``rms_px`` is
    Tukey-WEIGHTED and collapses to near zero on exactly the mis-convergence
    this statistic exists to surface (the fit locks onto one edge, Tukey
    rejects the wholly mis-aligned others, and the weighted RMS reports a
    clean fit).  Passing the raw ``LMRefineResult.residuals_px`` keeps those
    rejected vertices in the number.

    The sum grows with the number of fused edges, so the confidence formula
    consumes the edge-count-independent mean derived from it rather than this
    value directly.

    Parameters:
        features: The consumed features, in the order their vertices were
            concatenated for the LM fit.
        residuals: Per-vertex raw DT residuals from the fit, in the same
            concatenation order.

    Returns:
        The summed per-edge RMS in pixels; ``0.0`` when no consumed edge has
        vertices.
    """
    total = 0.0
    cursor = 0
    for feat in features:
        if not isinstance(feat.geometry, RingEdgePolyline):
            continue
        n = feat.geometry.vertices_vu.shape[0]
        if n == 0:
            continue
        slice_residuals = residuals[cursor : cursor + n]
        cursor += n
        if slice_residuals.size == 0:
            continue
        rms = float(np.sqrt(np.mean(slice_residuals * slice_residuals)))
        total += rms
    return total


@dataclass(frozen=True)
class _EdgeFitStat:
    """Per-edge fit statistics for the spurious-veto waiver.

    Parameters:
        inlier_count: Vertices of this edge with a strictly positive final
            weight (prior precision times Tukey biweight) -- the same
            criterion ``LMRefineResult.inlier_count`` applies to the
            aggregated vertex set.
        vertex_count: Total vertices of this edge.
        median_abs_residual_px: Median absolute DT residual of this edge's
            vertices at the converged offset.  Small when the edge lies
            along a detected image edge (whether or not the robust fit
            kept it); tens of pixels when no image edge exists near it.
        is_straight: The edge's ``is_straight_line`` flag; a straight edge
            constrains only its normal axis (rank-1).
    """

    inlier_count: int
    vertex_count: int
    median_abs_residual_px: float
    is_straight: bool

    @property
    def inlier_fraction(self) -> float:
        """Fraction of this edge's vertices retained as Tukey inliers."""
        return float(self.inlier_count) / float(self.vertex_count)


def _per_edge_fit_stats(
    features: list[NavFeature],
    *,
    residuals: NDArrayFloatType,
    weights: NDArrayFloatType,
) -> list[_EdgeFitStat]:
    """Return per-edge fit statistics from the final LM residuals and weights.

    Splitting the aggregate fit per edge lets the spurious gate
    distinguish a fusion whose vertices are rejected because some edges
    are undetectable in the image (large per-edge median residuals --
    nothing is there) from a wrong-ring lock that leaves a rejected edge
    sitting on a detected image edge it disagrees with.

    Parameters:
        features: The consumed features, in the order their vertices were
            concatenated for the LM fit.
        residuals: Per-vertex raw DT residuals from the fit, in the same
            concatenation order.
        weights: Per-vertex final weights from the fit, in the same
            concatenation order.

    Returns:
        One :class:`_EdgeFitStat` per consumed non-empty edge, in
        concatenation order.
    """
    stats: list[_EdgeFitStat] = []
    cursor = 0
    for feat in features:
        if not isinstance(feat.geometry, RingEdgePolyline):
            continue
        n = feat.geometry.vertices_vu.shape[0]
        if n == 0:
            continue
        slice_weights = weights[cursor : cursor + n]
        slice_residuals = residuals[cursor : cursor + n]
        cursor += n
        stats.append(
            _EdgeFitStat(
                inlier_count=int(np.count_nonzero(slice_weights > 0.0)),
                vertex_count=n,
                median_abs_residual_px=float(np.median(np.abs(slice_residuals))),
                is_straight=bool(feat.geometry.is_straight_line),
            )
        )
    return stats


def _per_edge_median_max(features: list[NavFeature], residuals: NDArrayFloatType) -> float:
    """Return the largest per-edge median absolute DT residual across edges.

    A mis-convergence *diagnostic* (surfaced through
    :class:`RingEdgeDiagnostics`, not a spurious gate): a wholly misaligned
    or undetected edge puts most of its vertices roughly a ringlet spacing
    from the nearest image edge, driving its median to that spacing, while
    a well-matched edge's median sits at the fit residual.  Taking the max
    over edges keeps one bad edge visible among any number of clean ones.
    Residuals alone cannot separate "misaligned" from "undetectable in the
    image", which is why the spurious decision uses the inlier fraction
    instead.

    Parameters:
        features: The consumed features, in the order their vertices were
            concatenated for the LM fit.
        residuals: Per-vertex signed DT residuals from the fit, in the same
            concatenation order.

    Returns:
        ``max_e median(|residuals_e|)`` over the consumed edges, or ``0.0``
        when no edge has vertices.
    """
    worst = 0.0
    cursor = 0
    for feat in features:
        if not isinstance(feat.geometry, RingEdgePolyline):
            continue
        n = feat.geometry.vertices_vu.shape[0]
        if n == 0:
            continue
        slice_residuals = residuals[cursor : cursor + n]
        cursor += n
        if slice_residuals.size == 0:
            continue
        worst = max(worst, float(np.median(np.abs(slice_residuals))))
    return worst


def _matched_edge_strength_fraction(
    edge_dt: NDArrayFloatType,
    gradient_vu: NDArrayFloatType,
    vertices: NDArrayFloatType,
    normals: NDArrayFloatType,
    offset_vu: tuple[float, float],
    *,
    valid_mask: NDArrayBoolType,
    match_radius_px: float = _EDGE_STRENGTH_MATCH_RADIUS_PX,
) -> float:
    """Return how strong the edges under a fit are, relative to the frame's strongest.

    The model is laid at ``offset_vu`` and each vertex rounded to its pixel.
    Over the vertices within ``match_radius_px`` of a detected edge pixel and
    inside ``valid_mask``, the image gradient projected onto each vertex's
    normal gives the strength of the edge it matched (the gradient is smoothed
    over more than a pixel, so it still reads the edge from a pixel away); the
    median of those is divided by the 99.9th
    percentile of the gradient magnitude over ``valid_mask``, the strength of
    the frame's own strongest structure.

    A cataloged ring edge is a large brightness step, so a fit that found one
    scores a sizable fraction of the frame's strongest edges.  A fit that found
    only the faint contours a thresholded smooth glow leaves behind scores a
    fraction of a percent.

    Parameters:
        edge_dt: Distance transform of the image edge mask, in pixels.
        gradient_vu: ``(H, W, 2)`` image gradient.
        vertices: ``(N, 2)`` model vertices.
        normals: ``(N, 2)`` model normals, either sign.
        offset_vu: The fitted ``(dv, du)`` offset.
        valid_mask: Pixels whose edges may count; the sensor interior.
        match_radius_px: Largest distance to a detected edge at which a vertex
            counts as having matched it.

    Returns:
        The fraction, ``0.0`` when no vertex lands on a counted edge pixel or
        the frame carries no gradient at all.
    """
    height, width = edge_dt.shape
    vs = np.rint(vertices[:, 0] + offset_vu[0]).astype(np.int64)
    us = np.rint(vertices[:, 1] + offset_vu[1]).astype(np.int64)
    inside = (vs >= 0) & (vs < height) & (us >= 0) & (us < width)
    vs, us = vs[inside], us[inside]
    unit = (
        normals[inside]
        / np.maximum(np.hypot(normals[inside, 0], normals[inside, 1]), 1e-300)[:, None]
    )
    matched = (edge_dt[vs, us] <= match_radius_px) & valid_mask[vs, us]
    magnitude = np.hypot(gradient_vu[:, :, 0], gradient_vu[:, :, 1])[valid_mask]
    if not matched.any() or magnitude.size == 0:
        return 0.0
    reference = float(np.percentile(magnitude, _EDGE_STRENGTH_REFERENCE_PERCENTILE))
    if reference <= 0.0:
        return 0.0
    projected = np.abs(
        gradient_vu[vs[matched], us[matched], 0] * unit[matched, 0]
        + gradient_vu[vs[matched], us[matched], 1] * unit[matched, 1]
    )
    return float(np.median(projected)) / reference
