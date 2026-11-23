"""Ghost reflections for the optics stage.

A ghost is a faint internal reflection of the focal-plane image: a displaced,
defocused, low-amplitude copy of the scene added back onto it.  A bright star
casts a ghost the same way an extended source does, so each ghost reflects both
the intensive signal (bodies, rings, sky) and the point-source plane (stars).
Each ghost copies the pre-ghost planes (so ghosts do not reflect one another),
shifts them by its offset, blurs them by its defocus, scales them by its
amplitude, and adds them back in.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

from scipy import ndimage

from spindoctor.sim.render.stages import SimFrame
from spindoctor.sim.scene.fields import scene_value
from spindoctor.support.types import NDArrayFloatType

__all__ = ['apply_ghosts', 'ghost_reach_px']

# How many sigmas the defocus blur reaches: scipy's gaussian_filter default.
_DEFOCUS_TRUNCATE_SIGMA = 4.0


def ghost_reach_px(ghosts: Sequence[Mapping[str, Any]], *, oversample: int) -> int:
    """How far the scene's ghosts carry light, in render-grid pixels.

    A ghost moves light by its offset, rounded up to a whole pixel because the shift
    interpolates between neighbors, and its defocus blur spreads it farther by the
    blur's truncated half-width.

    Parameters:
        ghosts: The scene ``optics.ghosts`` list of ghost specifications.
        oversample: The render-grid oversampling factor.

    Returns:
        The reach of the farthest-reaching ghost; 0 when no ghost carries light.
    """
    reach = 0
    for ghost in ghosts:
        if float(scene_value(ghost, 'amplitude', 0.0)) == 0.0:
            continue
        dv = abs(float(scene_value(ghost, 'dv_px', 0.0))) * oversample
        du = abs(float(scene_value(ghost, 'du_px', 0.0))) * oversample
        sigma = float(scene_value(ghost, 'defocus_sigma', 0.0)) * oversample
        blur = int(_DEFOCUS_TRUNCATE_SIGMA * sigma + 0.5) if sigma > 0.0 else 0
        reach = max(reach, math.ceil(max(dv, du)) + blur)
    return reach


def _add_ghosts_to_plane(
    plane: NDArrayFloatType,
    ghosts: Sequence[Mapping[str, Any]],
    oversample: int,
) -> None:
    """Add every ghost's displaced, defocused, scaled copy to one plane in place."""
    if not plane.any():
        return
    pre_ghost = plane.copy()
    for ghost in ghosts:
        amplitude = float(scene_value(ghost, 'amplitude', 0.0))
        if amplitude == 0.0:
            continue
        dv = float(scene_value(ghost, 'dv_px', 0.0)) * oversample
        du = float(scene_value(ghost, 'du_px', 0.0)) * oversample
        defocus_sigma = float(scene_value(ghost, 'defocus_sigma', 0.0)) * oversample
        copy = ndimage.shift(pre_ghost, (dv, du), order=1, mode='constant', cval=0.0)
        if defocus_sigma > 0.0:
            copy = ndimage.gaussian_filter(
                copy, defocus_sigma, mode='constant', truncate=_DEFOCUS_TRUNCATE_SIGMA
            )
        plane += amplitude * copy


def apply_ghosts(
    frame: SimFrame,
    *,
    ghosts: Sequence[Mapping[str, Any]],
    oversample: int,
) -> None:
    """Add displaced, defocused, scaled copies of the scene in place.

    Parameters:
        frame: The frame whose signal and point-source planes receive the ghosts.
        ghosts: The scene ``optics.ghosts`` list of ghost specifications.
        oversample: The render-grid oversampling factor (offsets and defocus
            are in detector pixels and scale to the render grid).
    """
    if not ghosts:
        return
    _add_ghosts_to_plane(frame.signal, ghosts, oversample)
    _add_ghosts_to_plane(frame.point_e, ghosts, oversample)
