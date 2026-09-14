"""Compose template-bearing features into a single ext-FOV image and mask.

Some downstream consumers (the manual-navigation dialog, the summary-PNG
overlay) want a single 2-D representation of "what the predicted scene
looks like" rather than a per-feature collection.  This module builds that
composite by Z-buffer painting every feature that carries a
``template_img`` / ``template_mask`` payload, ordered by subject range so
nearer features paint over farther ones.

:func:`compose_template_features` is the bitmap-only composer (limbs,
terminators, ring edges are skipped; their geometry belongs to the
annotation pipeline).  :func:`compose_dialog_overlay` extends it by
rasterizing every polyline-bearing feature on top — that is the composite
the manual-navigation dialog overlays on the source image so an operator
sees limbs / terminators / ring edges even when the body emits no
full-disc template.
"""

from __future__ import annotations

import numpy as np

from spindoctor.feature.feature import NavFeature
from spindoctor.feature.geometry import BodyBlobGeometry, StarGeometry, TitanHazeGeometry
from spindoctor.support.image import draw_circle, draw_rect
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['compose_dialog_overlay', 'compose_template_features']


_STAR_MARKER_HALFWIDTH_PX_FLOOR: int = 3
"""Minimum half-width of the rectangle marker drawn around a STAR feature.

The dialog overlay paints a rectangle outline at each STAR's predicted
position so the operator can see where the catalog says the star sits
before they click on the actual bright pixel.  The rectangle's
half-width is the larger of the feature's per-axis bbox half-extent
and this floor, so a tightly-bbox'd star (e.g. PSF sigma < 1 px) still
gets a visible 7x7 outline rather than collapsing to a single pixel.
"""


def compose_template_features(
    features: list[NavFeature], extfov_shape_vu: tuple[int, int]
) -> tuple[NDArrayFloatType, NDArrayBoolType]:
    """Z-buffer paint template features into a single ext-FOV image+mask.

    Features are sorted by ``subject_range_km`` ascending (nearer features
    last), so the closest body's pixels overwrite farther bodies' on
    overlap.  Each feature's template is placed at its
    ``geometry.bbox_extfov_vu`` location; pixels marked True in the
    feature's ``template_mask`` carry the template's value into the
    composite, and the composite mask becomes the OR of every painted
    feature's mask.

    Features without a ``template_img`` or ``template_mask`` are skipped.

    Parameters:
        features: The feature list (may include non-template features).
        extfov_shape_vu: Shape ``(v, u)`` of the ext-FOV array to build.

    Returns:
        Tuple ``(image, mask)`` where ``image`` is float64 in ext-FOV
        coordinates and ``mask`` is a boolean array of the same shape.
    """
    image: NDArrayFloatType = np.zeros(extfov_shape_vu, dtype=np.float64)
    mask: NDArrayBoolType = np.zeros(extfov_shape_vu, dtype=bool)
    template_features = [
        f for f in features if f.template_img is not None and f.template_mask is not None
    ]
    # Nearer features paint last so they overwrite farther ones on overlap.
    ordered = sorted(template_features, key=lambda f: f.subject_range_km, reverse=True)
    for feature in ordered:
        v_min, u_min, v_max, u_max = _bbox_clamped(feature.geometry.bbox_extfov_vu, extfov_shape_vu)
        if v_max <= v_min or u_max <= u_min:
            continue
        assert feature.template_img is not None
        assert feature.template_mask is not None
        template_img = feature.template_img
        template_mask = feature.template_mask
        bbox = feature.geometry.bbox_extfov_vu
        # The payload must be a postage stamp exactly the size of its
        # declared bbox.  The slice below anchors the template's (0, 0) at
        # the bbox origin, so an oversized template (e.g. a full ext-FOV
        # render paired with an interior bbox) silently paints its content
        # displaced by the padding amount -- two ring-model emitters
        # shipped exactly that defect.  Fail loudly instead.
        expected_shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        if template_img.shape != expected_shape or template_mask.shape != expected_shape:
            raise ValueError(
                f'feature {feature.feature_id!r}: template shape '
                f'{template_img.shape!r} / mask shape {template_mask.shape!r} '
                f'must equal the declared bbox {bbox!r} extents '
                f'{expected_shape!r}; template payloads are bbox-local '
                f'postage stamps'
            )
        # Slice the part of the template that fits inside ext-FOV.
        t_v_lo = v_min - bbox[0]
        t_u_lo = u_min - bbox[1]
        t_v_hi = t_v_lo + (v_max - v_min)
        t_u_hi = t_u_lo + (u_max - u_min)
        sub_img = template_img[t_v_lo:t_v_hi, t_u_lo:t_u_hi]
        sub_mask = template_mask[t_v_lo:t_v_hi, t_u_lo:t_u_hi]
        target_image_slice = image[v_min:v_max, u_min:u_max]
        target_mask_slice = mask[v_min:v_max, u_min:u_max]
        np.copyto(target_image_slice, sub_img, where=sub_mask)
        np.logical_or(target_mask_slice, sub_mask, out=target_mask_slice)
    return image, mask


def compose_dialog_overlay(
    features: list[NavFeature], extfov_shape_vu: tuple[int, int]
) -> tuple[NDArrayFloatType, NDArrayBoolType]:
    """Compose the manual-navigation dialog's full-scene overlay.

    Starts from :func:`compose_template_features` (BODY_DISC,
    RING_ANNULUS, CARTOGRAPHIC_MODEL templates) and additionally
    rasterizes:

    - every polyline-bearing feature's ``vertices_vu`` as single-pixel
      marks (LIMB_ARC, TERMINATOR_ARC, RING_EDGE);
    - every BODY_BLOB's predicted-diameter circle outline at the
      predicted centroid;
    - every TITAN_LIMB's haze-envelope circle outline at the predicted
      disc center;
    - every STAR feature's bbox as a rectangle outline at the
      predicted-vu position so the operator can manually align
      catalog stars with the observed bright pixels.

    All rasterization is intentionally minimal — it is a visibility
    aid, not a precise renderer; the autonomous DT / RANSAC pipeline
    owns the quantitative fit.

    Vertices and markers that fall outside the ext-FOV bounds are
    silently dropped (the polyline samplers can hand back
    partially-clipped polylines near the FOV edge; star markers near
    the FOV edge are clipped to the visible portion).

    Parameters:
        features: The feature list (templated + polyline + plain).
        extfov_shape_vu: Shape ``(v, u)`` of the ext-FOV array to build.

    Returns:
        Tuple ``(image, mask)`` where ``image`` is float64 in ext-FOV
        coordinates and ``mask`` is a boolean array of the same shape.
        Every painted pixel (template *or* polyline *or* marker) is
        True in the mask.
    """
    image, mask = compose_template_features(features, extfov_shape_vu)
    h, w = extfov_shape_vu
    for feature in features:
        # 1) Polyline-bearing geometries (LIMB_ARC, TERMINATOR_ARC, RING_EDGE)
        #    rasterize as single-pixel marks at every vertex.
        vertices_vu = getattr(feature.geometry, 'vertices_vu', None)
        if vertices_vu is not None:
            verts = np.asarray(vertices_vu, dtype=np.float64)
            if verts.size > 0:
                v_idx = np.rint(verts[:, 0]).astype(np.int64)
                u_idx = np.rint(verts[:, 1]).astype(np.int64)
                in_bounds = (v_idx >= 0) & (v_idx < h) & (u_idx >= 0) & (u_idx < w)
                v_idx = v_idx[in_bounds]
                u_idx = u_idx[in_bounds]
                if v_idx.size > 0:
                    # Polyline pixels paint a value of 1.0 — the dialog
                    # re-stretches the model channel for display, so any
                    # positive value is visible against the zero background.
                    image[v_idx, u_idx] = 1.0
                    mask[v_idx, u_idx] = True
        # 2) BodyBlobGeometry has no template and no polyline; render the
        #    predicted silhouette as a 1-pixel circle outline so the
        #    operator can still align the centroid by eye.
        if isinstance(feature.geometry, BodyBlobGeometry):
            _paint_circle_outline(
                image,
                mask,
                feature.geometry.predicted_center_vu,
                feature.geometry.predicted_diameter_px / 2.0,
            )
        # 3) TitanHazeGeometry likewise has neither template nor polyline;
        #    render the predicted haze envelope as a circle outline, the
        #    same silhouette the haze technique fits, so a Titan-only frame
        #    is manually navigable.
        if isinstance(feature.geometry, TitanHazeGeometry):
            _paint_circle_outline(
                image,
                mask,
                feature.geometry.predicted_center_vu,
                feature.geometry.r_env_px,
            )
        # 4) StarGeometry carries only a predicted (v, u) point and a
        #    PSF-sized bbox.  Render a rectangle outline around the
        #    bbox so the operator can see where the catalog says the
        #    star sits before clicking on the actual peak.
        if isinstance(feature.geometry, StarGeometry):
            _paint_star_marker(image, mask, feature.geometry, extfov_shape_vu)
    return image, mask


def _paint_circle_outline(
    image: NDArrayFloatType,
    mask: NDArrayBoolType,
    center_vu: tuple[float, float],
    radius_px: float,
) -> None:
    """Paint a 1-pixel circle outline into the overlay image and mask.

    ``draw_circle`` uses ``(x=u, y=v)`` and clips internally to the array
    bounds, so a partially-visible circle renders the arc that lands inside
    the ext-FOV.  The radius is floored at one pixel so a body smaller than
    a pixel still marks its predicted position.

    The center is rounded to the cell it falls in, which is what a pixel
    centric position means: a whole number there is a pixel's center, so the
    nearest whole number is the pixel that holds the position.  Every payload
    reaching here states its center that way.

    Parameters:
        image: Overlay image, modified in place.
        mask: Overlay mask, modified in place.
        center_vu: Predicted ``(v, u)`` center of the circle, extfov pixel
            centric.
        radius_px: Circle radius in pixels.
    """
    v_int = round(center_vu[0])
    u_int = round(center_vu[1])
    radius_int = max(1, round(radius_px))
    draw_circle(image, 1.0, u_int, v_int, radius_int)
    draw_circle(mask, True, u_int, v_int, radius_int)


def _paint_star_marker(
    image: NDArrayFloatType,
    mask: NDArrayBoolType,
    geometry: StarGeometry,
    extfov_shape_vu: tuple[int, int],
) -> None:
    """Paint a rectangle outline around a STAR feature's predicted bbox.

    ``draw_rect`` does not clip on the array bounds — partial-FOV
    rectangles get truncated by the slice operations, but a center
    that is far enough off-image that the rectangle is entirely
    out-of-bounds is silently a no-op.  A center close enough to an
    edge that part of the rectangle would index negatively is the
    risky case; the explicit half-width clamp below keeps every drawn
    pixel inside the array.
    """
    h, w = extfov_shape_vu
    v_center, u_center = geometry.predicted_vu
    v_int = round(v_center)
    u_int = round(u_center)
    if v_int < 0 or v_int >= h or u_int < 0 or u_int >= w:
        return  # marker centre off-image; no painting
    v_min, u_min, v_max, u_max = geometry.bbox_extfov_vu
    # The bbox is half-open, so its half-extent is one less than the
    # diff.  Clamp to a visible-marker floor and also to the largest
    # half-width that keeps every rectangle pixel inside the FOV.
    v_half = max(_STAR_MARKER_HALFWIDTH_PX_FLOOR, ((v_max - v_min) // 2) - 1)
    u_half = max(_STAR_MARKER_HALFWIDTH_PX_FLOOR, ((u_max - u_min) // 2) - 1)
    v_half = min(v_half, v_int, h - 1 - v_int)
    u_half = min(u_half, u_int, w - 1 - u_int)
    if v_half <= 0 or u_half <= 0:
        # Edge-tight: fall back to a single pixel at the predicted
        # position rather than emit no marker at all.
        image[v_int, u_int] = 1.0
        mask[v_int, u_int] = True
        return
    draw_rect(image, 1.0, u_int, v_int, u_half, v_half)
    draw_rect(mask, True, u_int, v_int, u_half, v_half)


def _bbox_clamped(
    bbox_extfov_vu: tuple[int, int, int, int], extfov_shape_vu: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Clamp a feature's bbox to lie inside ``extfov_shape_vu``."""
    v_min, u_min, v_max, u_max = bbox_extfov_vu
    h, w = extfov_shape_vu
    return (
        max(0, int(v_min)),
        max(0, int(u_min)),
        min(int(h), int(v_max)),
        min(int(w), int(u_max)),
    )
