"""Ring-screen compositing and depth-ambiguity checks for the radiance stage.

The radiance stage (:mod:`spindoctor.sim.render.radiance.compose`) composites the
scene far to near.  Each body goes over the image by its coverage, after its own
translucent halo, and the optical-depth ring system goes in pixel by pixel at its
own depth between them.  This module holds the ring's part of that machinery:
:func:`ring_screen_op` builds the application of the ring screen to a set of
pixels, and :func:`apply_screen_op` applies it.  The ``check_*`` functions enforce
the depth contract behind every ordering decision -- overlapping objects must
ALL carry an explicit scene ``range_km``, or their stacking would be a silent
guess and the scene fails loudly.

Everything here operates on rendered geometry the radiance stage hands in
(painted masks, halo screens, ring maps); no scene mapping is read directly.
"""

from dataclasses import dataclass

import numpy as np

from spindoctor.sim.render.bodies.atmosphere import HaloScreen
from spindoctor.sim.render.rings.render import RingSystemMaps
from spindoctor.sim.scene.errors import SimSceneValidationError
from spindoctor.support.types import NDArrayBoolType, NDArrayFloatType

__all__ = [
    'ScreenOp',
    'apply_screen_op',
    'check_depth_ambiguity',
    'check_halo_ambiguity',
    'check_ring_system_ambiguity',
    'ring_screen_op',
]


@dataclass(frozen=True)
class ScreenOp:
    """One application of the ring screen over the composed image.

    Applied to the ``(box_v, box_u)`` view of the frame as ``view[mask] =
    intensity[mask] + transmission[mask] * view[mask]`` (the screen is
    identity outside its box).

    Parameters:
        box_v: Frame rows the op is restricted to.
        box_u: Frame columns the op is restricted to.
        mask: Box-sized mask of the pixels the screen composites over.
        intensity: Box-sized emission map.
        transmission: Box-sized per-pixel background transmission.
    """

    box_v: slice
    box_u: slice
    mask: NDArrayBoolType
    intensity: NDArrayFloatType
    transmission: NDArrayFloatType


def ring_screen_op(ring_maps: RingSystemMaps, pixels: NDArrayBoolType) -> ScreenOp:
    """The application of the ring screen to some of the frame's pixels.

    Parameters:
        ring_maps: The rendered ring-system maps.
        pixels: The frame pixels the ring composites over.

    Returns:
        The screen op over the whole frame, restricted to ``pixels``.
    """
    full = slice(None)
    return ScreenOp(
        box_v=full,
        box_u=full,
        mask=pixels,
        intensity=ring_maps.intensity,
        transmission=ring_maps.transmission,
    )


def apply_screen_op(img: NDArrayFloatType, op: ScreenOp) -> None:
    """Composite a screen op over the image in place.

    Parameters:
        img: The image.
        op: The screen op.
    """
    view = img[op.box_v, op.box_u]
    view[op.mask] = op.intensity[op.mask] + op.transmission[op.mask] * view[op.mask]


def check_depth_ambiguity(
    painted_items: list[tuple[str, NDArrayBoolType, bool]],
    item_label: str,
    item_mask: NDArrayBoolType,
    explicit_depth: bool,
) -> None:
    """Fail when overlapping bodies are stacked without explicit depths.

    ``range_km`` is the compositing depth.  Two bodies whose painted pixels
    overlap must both carry an explicit scene ``range_km``, or their stacking
    order would be a silent guess; a scene that leaves it off an overlapping
    body is malformed and fails loudly here.  Non-overlapping bodies need
    no ranges: their paint order is unobservable.

    Parameters:
        painted_items: ``(label, painted mask, explicit range_km?)`` for
            every body already painted, in render order.
        item_label: Name of the body just painted.
        item_mask: Pixels the body just painted.
        explicit_depth: Whether the body carries an explicit ``range_km``.

    Raises:
        spindoctor.sim.scene.errors.SimSceneValidationError: If the new
            object overlaps a painted one and either of the pair lacks an
            explicit ``range_km``.
    """
    if not item_mask.any():
        return
    for other_label, other_mask, other_explicit in painted_items:
        if explicit_depth and other_explicit:
            continue
        if np.any(item_mask & other_mask):
            raise SimSceneValidationError(
                f'objects {other_label!r} and {item_label!r} overlap but do not both '
                f'carry an explicit range_km; set range_km on both to order them'
            )


def _halo_screens_overlap(screen_a: HaloScreen, screen_b: HaloScreen) -> bool:
    """Whether two halo screens carry haze on any common pixel.

    Each screen's mask is box-sized, so the test intersects the two bounding
    boxes and compares the corresponding mask windows.

    Parameters:
        screen_a: One halo screen.
        screen_b: The other halo screen.

    Returns:
        True when the screens' masked pixels intersect.
    """
    v_lo = max(screen_a.box_v.start, screen_b.box_v.start)
    v_hi = min(screen_a.box_v.stop, screen_b.box_v.stop)
    u_lo = max(screen_a.box_u.start, screen_b.box_u.start)
    u_hi = min(screen_a.box_u.stop, screen_b.box_u.stop)
    if v_lo >= v_hi or u_lo >= u_hi:
        return False
    window_a = screen_a.mask[
        v_lo - screen_a.box_v.start : v_hi - screen_a.box_v.start,
        u_lo - screen_a.box_u.start : u_hi - screen_a.box_u.start,
    ]
    window_b = screen_b.mask[
        v_lo - screen_b.box_v.start : v_hi - screen_b.box_v.start,
        u_lo - screen_b.box_u.start : u_hi - screen_b.box_u.start,
    ]
    return bool(np.any(window_a & window_b))


def check_halo_ambiguity(
    painted_items: list[tuple[str, NDArrayBoolType, bool]],
    halo_check_items: list[tuple[str, bool, HaloScreen]],
) -> None:
    """Fail when a halo overlaps another body or halo without explicit depths.

    A halo composites against everything it covers by its body's
    ``range_km``, so an overlap ordered only by the positional default
    ranges would be a silent guess exactly like an opaque overlap: a halo
    that reaches another body's painted silhouette, or another body's halo,
    requires an explicit scene ``range_km`` on both bodies.  A halo over
    only the empty sky needs no range: its order is unobservable.

    Parameters:
        painted_items: ``(label, painted mask, explicit range_km?)`` for
            every painted body.
        halo_check_items: ``(body label, explicit range_km?, halo screen)``
            for every atmospheric body, in render order.

    Raises:
        spindoctor.sim.scene.errors.SimSceneValidationError: If a halo
            overlaps another body's paint or halo and either body lacks an
            explicit ``range_km``.
    """
    for halo_label, halo_explicit, screen in halo_check_items:
        if not screen.mask.any():
            continue
        box = (screen.box_v, screen.box_u)
        for other_label, other_mask, other_explicit in painted_items:
            if other_label == halo_label:
                continue
            if halo_explicit and other_explicit:
                continue
            if np.any(other_mask[box] & screen.mask):
                raise SimSceneValidationError(
                    f'the halo of body {halo_label!r} overlaps body {other_label!r} '
                    f'but they do not both carry an explicit range_km; set range_km '
                    f'on both to order them'
                )
    for index, (label_a, explicit_a, screen_a) in enumerate(halo_check_items):
        for label_b, explicit_b, screen_b in halo_check_items[index + 1 :]:
            if explicit_a and explicit_b:
                continue
            if _halo_screens_overlap(screen_a, screen_b):
                raise SimSceneValidationError(
                    f'the halos of bodies {label_a!r} and {label_b!r} overlap but '
                    f'they do not both carry an explicit range_km; set range_km on '
                    f'both to order them'
                )


def check_ring_system_ambiguity(
    painted_items: list[tuple[str, NDArrayBoolType, bool]],
    halo_check_items: list[tuple[str, bool, HaloScreen]],
    ring_mask: NDArrayBoolType,
    *,
    ring_explicit: bool,
) -> None:
    """Fail when the ring system overlaps bodies or halos without depths.

    Per-pixel depth ordering against a body needs an explicit ``range_km`` on
    both the ring system and the body.  A body's translucent halo takes the
    same rule: the ring interleaves with a halo by the two ranges, so a ring
    over a halo ordered only by positional defaults (or a depth-less ring
    that would silently screen the halo) is the same ambiguity.

    Parameters:
        painted_items: ``(label, painted mask, explicit range_km?)`` for
            every painted body.
        halo_check_items: ``(body label, explicit range_km?, halo screen)``
            for every atmospheric body, in render order.
        ring_mask: Pixels where the ring system carries optical depth.
        ring_explicit: Whether the ring system carries an explicit
            ``range_km``.

    Raises:
        spindoctor.sim.scene.errors.SimSceneValidationError: On any overlap
            without a defined depth relation.
    """
    if not ring_mask.any():
        return
    for label, mask, explicit in painted_items:
        if not np.any(mask & ring_mask):
            continue
        if not (ring_explicit and explicit):
            raise SimSceneValidationError(
                f'ring_system and body {label!r} overlap but do not both carry an '
                f'explicit range_km; set range_km on both to order them'
            )
    for halo_label, halo_explicit, screen in halo_check_items:
        if ring_explicit and halo_explicit:
            continue
        if not screen.mask.any():
            continue
        if np.any(ring_mask[screen.box_v, screen.box_u] & screen.mask):
            raise SimSceneValidationError(
                f'ring_system and the halo of body {halo_label!r} overlap but do '
                f'not both carry an explicit range_km; set range_km on both to '
                f'order them'
            )
