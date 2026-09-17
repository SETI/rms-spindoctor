"""Polyline extraction and geometry helpers for the catalog ring model.

Turns a rendered one-pixel edge mask into a vertex polyline with
radially-signed normals, and the small geometry predicates the emission gate
reads (radial extent, straightness, bbox, composite rendering).  Split out of
:mod:`spindoctor.nav_model.nav_model_rings` to keep both modules under the
size cap.
"""

from __future__ import annotations

import math

import numpy as np

from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__: list[str] = []


_MEAN_NORMAL_DEGENERATE_TOLERANCE: float = 1.0e-6
"""Mean-normal magnitude below which the radial axis is taken as undefined.

Per-vertex normals are unit length, so a well-defined mean radial direction
has a magnitude of order one; a closed ring's radially-signed normals cancel
to ~1e-18.  Anything below this tolerance is rounding noise whose direction
carries no geometry.
"""


FLAT_CURVATURE_THRESHOLD_PX: float = 1.0
"""Pixel-deviation threshold below which a polyline is flagged straight.

When the maximum deviation of the polyline from its best-fit straight
line is below this value, the corresponding ``RING_EDGE`` feature is
emitted with ``is_straight_line=True``.  The technique-level fitter
then handles its rank-1 covariance.
"""


def _median_radial_scale(
    resolutions: NDArrayFloatType | None, mask: NDArrayBoolType, fallback_km_per_px: float
) -> float:
    """Median ring-radial resolution (km/px) over an edge's own pixels.

    Parameters:
        resolutions: Ext-FOV ring-radial-resolution array, or ``None`` when the
            backplane was never evaluated (a stubbed model in unit tests).
        mask: Ext-FOV boolean mask of the edge's pixels.
        fallback_km_per_px: Whole-image scale to use when no per-pixel array is
            available or no masked pixel carries a finite resolution.

    Returns:
        The edge-local scale in km per pixel; strictly positive.

    Raises:
        ValueError: If neither a per-pixel resolution nor a usable fallback is
            available.  A sentinel-small scale here would divide a km sigma
            into a covariance of order 1e12 px^2 -- a finite but garbage
            covariance that silently poisons the tier gates, which is a worse
            failure than refusing to emit.
    """
    if resolutions is not None and resolutions.shape == mask.shape:
        local = resolutions[mask]
        local = local[np.isfinite(local) & (local > 0.0)]
        if local.size:
            return float(np.median(local))
    if math.isfinite(fallback_km_per_px) and fallback_km_per_px > 0.0:
        return float(fallback_km_per_px)
    raise ValueError(
        'ring radial scale is unavailable: no finite per-pixel resolution over the '
        f'edge mask and no usable whole-image fallback (got {fallback_km_per_px!r})'
    )


def _polyline_from_edge_mask(
    mask: NDArrayBoolType,
    ring_radius: NDArrayFloatType | None = None,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Extract a polyline + per-vertex normal from a 1-pixel-wide edge mask.

    Each True pixel becomes one polyline vertex.  The normal AXIS at each
    vertex comes from the local mask-neighbour test: whichever side has no
    mask pixel is the off-edge side.  That test alone fixes only the axis,
    never a consistent sense -- it scans ``v - 1`` before ``v + 1`` and
    ``u - 1`` before ``u + 1``, so on a closed ring the emitted signs follow
    scan order and rasterization rather than geometry, and the eight
    quantized directions come out biased toward one quadrant.

    ``ring_radius`` fixes that: with the ring-radius backplane in hand each
    normal is flipped to point toward INCREASING ring radius, which is the
    outward-radial sense ``RingEdgePolyline.normals_vu`` documents and that
    ``RingEdgeNav``'s orbit-uncertainty channel depends on (it sums the
    normals, so a random sign per vertex fabricates coherence on geometry
    that should cancel).  Vertices whose two radius probes are unusable
    (off-image or masked) keep the unsigned axis.

    When ``ring_radius`` is None the historical unsigned behavior applies;
    only stubbed callers in unit tests take that path.

    Parameters:
        mask: 2-D boolean ``(rows, cols)`` array of edge pixels in
            extfov coordinates.
        ring_radius: Optional ext-FOV ring-radius array (km) matching
            ``mask``; non-finite entries mark pixels off the ring plane.

    Returns:
        ``(vertices_vu, normals_vu)`` arrays each of shape ``(N, 2)``.
        Normals are unit length, radially outward when ``ring_radius``
        allowed the sense to be determined.
    """
    if not mask.any():
        empty: NDArrayFloatType = np.empty((0, 2), dtype=np.float64)
        return empty, empty
    vs, us = np.where(mask)
    vertices_vu: NDArrayFloatType = np.stack([vs.astype(np.float64), us.astype(np.float64)], axis=1)
    rows, cols = mask.shape
    normals_vu = np.zeros_like(vertices_vu)
    for i, (v, u) in enumerate(zip(vs, us, strict=True)):
        v_dir = 0.0
        u_dir = 0.0
        if v > 0 and not mask[v - 1, u]:
            v_dir = -1.0
        elif v < rows - 1 and not mask[v + 1, u]:
            v_dir = 1.0
        if u > 0 and not mask[v, u - 1]:
            u_dir = -1.0
        elif u < cols - 1 and not mask[v, u + 1]:
            u_dir = 1.0
        norm = math.hypot(v_dir, u_dir) or 1.0
        nv = v_dir / norm
        nu = u_dir / norm
        if ring_radius is not None:
            # Probe the ring radius one pixel to either side along the normal
            # axis and orient the normal toward the larger radius.
            sign = _radial_outward_sign(ring_radius, int(v), int(u), nv, nu)
            nv *= sign
            nu *= sign
        normals_vu[i, 0] = nv
        normals_vu[i, 1] = nu
    return vertices_vu, normals_vu


def _radial_outward_sign(
    ring_radius: NDArrayFloatType, v: int, u: int, nv: float, nu: float
) -> float:
    """Return ``+1`` if ``(nv, nu)`` already points toward increasing radius.

    Samples the ring radius one pixel either side of ``(v, u)`` along the
    normal axis.  Returns ``-1`` when the negative side has the larger radius,
    and ``+1`` when the comparison is unusable (either probe off-image or off
    the ring plane), leaving the unsigned axis untouched.

    Parameters:
        ring_radius: Ext-FOV ring-radius array (km); non-finite off the plane.
        v, u: Integer pixel position of the vertex.
        nv, nu: Unit normal-axis components at the vertex.

    Returns:
        ``+1.0`` or ``-1.0``.
    """
    rows, cols = ring_radius.shape
    v_plus, u_plus = round(v + nv), round(u + nu)
    v_minus, u_minus = round(v - nv), round(u - nu)
    if not (0 <= v_plus < rows and 0 <= u_plus < cols):
        return 1.0
    if not (0 <= v_minus < rows and 0 <= u_minus < cols):
        return 1.0
    r_plus = float(ring_radius[v_plus, u_plus])
    r_minus = float(ring_radius[v_minus, u_minus])
    if not (math.isfinite(r_plus) and math.isfinite(r_minus)):
        return 1.0
    return -1.0 if r_minus > r_plus else 1.0


def _radial_extent_px(vertices_vu: NDArrayFloatType, normals_vu: NDArrayFloatType) -> float:
    """Return the polyline's radial extent (max - min projection on mean normal).

    The projection axis is the MEAN normal, so it is sign-sensitive.  Once the
    normals are radially signed (see :func:`_polyline_from_edge_mask`) this
    measures the extent along the polyline's genuine mean radial direction; a
    half turn of arc then returns the ring radius, which is geometrically
    right, where scan-order signs previously returned a larger number set by
    the rasterizer's quadrant bias.  The values feed the annulus emission gate
    (``radial_extent_px <= max_radial_px``), so a short curved edge can measure
    roughly half what it used to.

    A closed or near-closed ring cancels the mean normal to numerical noise
    rather than exactly zero, so the degenerate case is caught by a relative
    tolerance rather than an ``== 0`` test: normalizing noise would otherwise
    project onto an arbitrary axis, which for a projected (elliptical) ring
    reports an extent unrelated to the true radial span.  The fallback is the
    dominant axis of the normals' outer-product sum, which is well defined
    whatever the signs do.

    Parameters:
        vertices_vu: ``(N, 2)`` polyline vertices.
        normals_vu: ``(N, 2)`` per-vertex normals.

    Returns:
        The extent in pixels; ``0.0`` for an empty polyline.
    """
    if vertices_vu.shape[0] == 0:
        return 0.0
    mean_normal = normals_vu.mean(axis=0)
    norm = float(np.linalg.norm(mean_normal))
    if norm < _MEAN_NORMAL_DEGENERATE_TOLERANCE:
        # Signs cancel (a closed ring): fall back to the sign-independent
        # dominant axis of the normal distribution.
        outer_sum = normals_vu.T @ normals_vu
        _eigvals, eigvecs = np.linalg.eigh(outer_sum)
        axis = eigvecs[:, -1]
        if not np.isfinite(axis).all():
            return 0.0
        projections = vertices_vu @ axis
        return float(projections.max() - projections.min())
    mean_normal = mean_normal / norm
    projections = vertices_vu @ mean_normal
    return float(projections.max() - projections.min())


def _is_straight_line(vertices_vu: NDArrayFloatType) -> bool:
    """Return True when the polyline's max-deviation from a best-fit line is tiny.

    The polyline is straight when its maximum perpendicular deviation
    from the best-fit straight line is below
    ``FLAT_CURVATURE_THRESHOLD_PX``.  Computed by SVD of the centered
    point cloud (the smallest singular vector is the normal direction).
    """
    if vertices_vu.shape[0] < 3:
        return True
    centered = vertices_vu - vertices_vu.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    deviations = centered @ normal
    return bool(float(np.max(np.abs(deviations))) <= FLAT_CURVATURE_THRESHOLD_PX)


def _mask_bbox(mask: NDArrayBoolType) -> tuple[int, int, int, int]:
    """Return ``(v_min, u_min, v_max, u_max)`` half-open bbox of True pixels."""
    if not mask.any():
        return (0, 0, 0, 0)
    vs, us = np.where(mask)
    return (
        int(vs.min()),
        int(us.min()),
        int(vs.max()) + 1,
        int(us.max()) + 1,
    )


def _composite_ring_renderings(
    renderings: list[tuple[NDArrayFloatType, NDArrayBoolType, str, float]],
    *,
    extfov_shape: tuple[int, int],
) -> tuple[NDArrayFloatType, NDArrayBoolType]:
    """Union per-ring rendered images + masks into one composite annulus.

    Each per-ring rendering carries an ext-FOV-shaped ``model_img`` /
    ``model_mask`` pair from :class:`RingFeature.render`.  The composite
    is the OR of the masks and the per-pixel maximum of the images so
    overlapping rings keep their brightest contribution.

    Parameters:
        renderings: List of ``(model_img, model_mask, label, radial_extent)``
            tuples (radial_extent is unused by this helper but kept on
            the input row so the caller can read it without a second
            iteration).
        extfov_shape: ``(v, u)`` shape of the ext-FOV array — every
            input rendering shares this shape.

    Returns:
        ``(composite_img, composite_mask)`` both shaped ``extfov_shape``.
    """
    composite_img: NDArrayFloatType = np.zeros(extfov_shape, dtype=np.float64)
    composite_mask: NDArrayBoolType = np.zeros(extfov_shape, dtype=bool)
    for img, mask, _label, _extent in renderings:
        composite_img = np.maximum(composite_img, img)
        composite_mask = composite_mask | mask
    return composite_img, composite_mask
