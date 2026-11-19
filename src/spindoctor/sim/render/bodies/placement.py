"""Placement of a rendered body: translate it into the frame and paint it.

A body renderer draws its body once at a reference center, the frame center, so a
shape cache can serve the same body at any position.  :func:`finish_single_body`
then translates that drawing to the body's center, paints its opaque disc into the
frame, positions its translucent halo, and builds the body's inventory entry.  The
ellipsoid path in :mod:`spindoctor.sim.render.bodies.render` and the mesh renderer
(:mod:`spindoctor.sim.render.bodies.mesh`) both finish here.
"""

from typing import Any

from spindoctor.sim.render.bodies.atmosphere import HaloScreen
from spindoctor.sim.render.bodies.translate import nonzero_box, shifted_box, translate_box
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = ['finish_single_body']


def finish_single_body(
    img: NDArrayFloatType,
    body_shape: NDArrayFloatType,
    *,
    body_name: str,
    center_v: float,
    center_u: float,
    half_extent_v: float,
    half_extent_u: float,
    ref_center_v: float,
    ref_center_u: float,
    range_km: float | None,
    halo: HaloScreen | None = None,
    silhouette: NDArrayFloatType | None = None,
) -> tuple[NDArrayBoolType, dict[str, Any]]:
    """Translate a reference-centered body shape into place and composite it.

    Only the opaque body paints here (last writer wins); an atmospheric
    body's translucent halo is translated alongside it and returned on the
    info dict (key ``'halo'``) for the radiance stage to composite as a
    transmission screen, so the returned mask and the painted pixels are the
    solid silhouette only.  The mask is every pixel the silhouette touches,
    so the body paints, occludes stars and farther bodies, and claims its
    pixels in the truth maps there even where it renders black.

    Parameters:
        img: Image array to modify in-place.
        body_shape: Body intensity rendered at the reference center.
        body_name: Upper-cased body name for the inventory keys.
        center_v: Body center V in the image (offset already applied).
        center_u: Body center U in the image (offset already applied).
        half_extent_v: Projected half-extent of the silhouette along v.
        half_extent_u: Projected half-extent of the silhouette along u.
        ref_center_v: Reference center V the shape was rendered at.
        ref_center_u: Reference center U the shape was rendered at.
        range_km: The body's distance from the observer, recorded in the
            inventory.  None records 1.0.
        halo: The body's translucent halo screen at the reference center, or
            None for a body without an atmosphere.
        silhouette: The body's solid-outline pixel coverage at the reference
            center, or None when every pixel inside the body renders above
            zero, so ``body_shape > 0`` is the outline.

    Returns:
        Tuple of (body_mask, body_info_dict) where body_info_dict contains
        the name and the inventory item (plus the positioned halo screen under
        ``'halo'`` when the body carries one).
    """
    dv = center_v - ref_center_v
    du = center_u - ref_center_u
    positioned_body, positioned_outline = _position_body(body_shape, silhouette, dv=dv, du=du)
    mask = positioned_outline > 0
    img[mask] = positioned_body[mask]

    body_info: dict[str, Any] = {
        'name': body_name,
        'inventory': _inventory_item(
            center_v,
            center_u,
            half_extent_v=half_extent_v,
            half_extent_u=half_extent_u,
            range_km=range_km,
        ),
    }
    if halo is not None:
        body_info['halo'] = _position_halo(halo, mask, dv=dv, du=du)
    return mask, body_info


def _position_body(
    body_shape: NDArrayFloatType,
    silhouette: NDArrayFloatType | None,
    *,
    dv: float,
    du: float,
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Translate a body's intensity and its outline by ``(dv, du)``.

    Parameters:
        body_shape: Body intensity rendered at the reference center.
        silhouette: The body's solid-outline coverage at the reference center, or
            None when ``body_shape > 0`` is the outline.
        dv: The translation along v, in pixels.
        du: The translation along u, in pixels.

    Returns:
        The translated intensity and the translated outline.  At a zero translation
        they are the input arrays themselves, which the caller only reads.
    """
    # An exactly zero translation is the bitwise identity for the order-1
    # spline, so a body already at the reference center skips the full-frame
    # interpolation (the shape is only read, never written).
    if dv == 0.0 and du == 0.0:
        return body_shape, body_shape if silhouette is None else silhouette
    # Only the box the body covers is resampled; see the translate module.
    lit_planes = (body_shape,) if silhouette is None else (body_shape, silhouette)
    box_v, box_u = nonzero_box(*lit_planes)
    positioned_body = translate_box(body_shape, dv, du, box_v, box_u)
    positioned_outline = (
        positioned_body if silhouette is None else translate_box(silhouette, dv, du, box_v, box_u)
    )
    return positioned_body, positioned_outline


def _position_halo(halo: HaloScreen, mask: NDArrayBoolType, *, dv: float, du: float) -> HaloScreen:
    """Translate a body's halo screen by ``(dv, du)``, clear of the body's opaque paint.

    Parameters:
        halo: The halo screen at the reference center.
        mask: The positioned body's opaque mask on the frame.
        dv: The translation along v, in pixels.
        du: The translation along u, in pixels.

    Returns:
        The positioned halo screen (``halo`` itself at a zero translation).
    """
    if dv == 0.0 and du == 0.0:
        # The screen already excludes every pixel the (unshifted) disc
        # painted, so it is positioned as-is.
        return halo
    size_v, size_u = mask.shape
    emission = translate_box(halo.emission, dv, du, halo.box_v, halo.box_u)
    transmission = translate_box(halo.transmission, dv, du, halo.box_v, halo.box_u, cval=1.0)
    box_v = shifted_box(halo.box_v, dv, size_v)
    box_u = shifted_box(halo.box_u, du, size_u)
    # The translation's linear interpolation bleeds the halo one pixel
    # into the painted rim; a screen never overlaps its own opaque
    # paint.  Outside the box the screen is identity already.
    mask_box = mask[box_v, box_u]
    emission_view = emission[box_v, box_u]
    emission_view[mask_box] = 0.0
    transmission_view = transmission[box_v, box_u]
    transmission_view[mask_box] = 1.0
    halo_mask = (emission_view > 0.0) | (transmission_view < 1.0)
    return HaloScreen(
        emission=emission,
        transmission=transmission,
        box_v=box_v,
        box_u=box_u,
        mask=halo_mask,
    )


def _inventory_item(
    center_v: float,
    center_u: float,
    *,
    half_extent_v: float,
    half_extent_u: float,
    range_km: float | None,
) -> dict[str, float]:
    """The body's inventory entry: its unclipped bounding box, size, and range.

    Parameters:
        center_v: Body center V in the image.
        center_u: Body center U in the image.
        half_extent_v: Projected half-extent of the silhouette along v.
        half_extent_u: Projected half-extent of the silhouette along u.
        range_km: The body's distance from the observer, or None to record 1.0.

    Returns:
        The inventory entry.
    """
    return {
        'v_min_unclipped': center_v - half_extent_v,
        'v_max_unclipped': center_v + half_extent_v,
        'u_min_unclipped': center_u - half_extent_u,
        'u_max_unclipped': center_u + half_extent_u,
        'v_pixel_size': 2 * half_extent_v,
        'u_pixel_size': 2 * half_extent_u,
        # The inventory mirrors the oops inventory contract for real frames,
        # whose distance key is named 'range'; the scene schema key is the
        # per-body 'range_km'.
        'range': 1.0 if range_km is None else range_km,
    }
