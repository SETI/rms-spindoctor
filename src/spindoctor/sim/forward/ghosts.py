"""Ghost reflections for the optics stage.

A ghost is a faint internal reflection of the focal-plane image: a displaced,
defocused, low-amplitude copy of the scene added back onto it.  A bright star
casts a ghost the same way an extended source does, so each ghost reflects both
the intensive signal (bodies, rings, sky) and the point-source plane (stars).
Each ghost copies the pre-ghost planes (so ghosts do not reflect one another),
shifts them by its offset, blurs them by its defocus, scales them by its
amplitude, and adds them back in.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from scipy import ndimage

from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.scene_schema import scene_value
from spindoctor.support.types import NDArrayFloatType

__all__ = ['apply_ghosts']


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
            copy = ndimage.gaussian_filter(copy, defocus_sigma, mode='constant')
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
