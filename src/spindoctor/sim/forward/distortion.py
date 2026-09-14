"""Residual geometric distortion for the optics stage.

The navigator corrects each camera's known distortion model, so the quantity
actually present in the frames the pipeline consumes -- and the only quantity
this stage plants -- is the residual: a low-order radial polynomial about the
optical center plus an optional small non-radial wander.  A limb fitted at the
frame edge then disagrees with a ring fitted through the center by the
differential residual between their positions, which the navigator gets no
model to remove.

The warp maps each output pixel ``p`` to a source position
``center + (p - center) * (1 + k1*rho^2 + k2*rho^4)`` with
``rho = |p - center| / rho_ref`` and ``rho_ref`` half the image diagonal; the
scene is resampled there by cubic interpolation.  The non-radial term adds a
seeded, smooth 2-D displacement field (its own RNG stream) at the commanded
RMS amplitude.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage

from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.seeds import derive_effect_seed
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayFloatType

__all__ = ['apply_distortion']

# Control-grid resolution of the non-radial wander before it is smoothly
# upsampled to the render grid: a handful of points keeps the field low-order
# (crater-scale coherent, not per-pixel jitter).
_NONRADIAL_CONTROL = 4


def _nonradial_field(
    shape: tuple[int, int], rms_px: float, seed: int
) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Build a seeded, smooth non-radial displacement field at a target RMS.

    Parameters:
        shape: The render-grid ``(V*os, U*os)`` shape.
        rms_px: Target RMS displacement magnitude in render-grid pixels.
        seed: Seed for the field's own RNG stream.

    Returns:
        The ``(field_v, field_u)`` displacement planes.
    """
    rng = np.random.default_rng(seed)
    coarse_v = rng.normal(size=(_NONRADIAL_CONTROL, _NONRADIAL_CONTROL))
    coarse_u = rng.normal(size=(_NONRADIAL_CONTROL, _NONRADIAL_CONTROL))
    zoom = (shape[0] / _NONRADIAL_CONTROL, shape[1] / _NONRADIAL_CONTROL)
    field_v = ndimage.zoom(coarse_v, zoom, order=3)[: shape[0], : shape[1]]
    field_u = ndimage.zoom(coarse_u, zoom, order=3)[: shape[0], : shape[1]]
    magnitude_rms = float(np.sqrt(np.mean(field_v**2 + field_u**2)))
    if magnitude_rms > 0.0:
        scale = rms_px / magnitude_rms
        field_v *= scale
        field_u *= scale
    return field_v, field_u


def apply_distortion(
    frame: SimFrame,
    *,
    params: Mapping[str, Any],
    oversample: int,
    distortion: Mapping[str, Any] | None = None,
) -> None:
    """Warp the signal and point-source planes by the residual distortion.

    A disabled block (all-zero radial coefficients and no non-radial wander) is
    a true no-op, so a scene that names a distortion block without amplitude
    renders bit-identically to one without it.

    Parameters:
        frame: The frame whose planes are warped in place.
        params: The full scene mapping; supplies the scene ``random_seed`` for
            the non-radial field's stream and, when ``distortion`` is not passed,
            the ``optics.distortion`` block.
        oversample: The render-grid oversampling factor (center and amplitude
            are in detector pixels and scale to the render grid).
        distortion: An explicit distortion block; when None the block is read
            from ``params['optics']['distortion']`` (the instrument-defaults
            residual is passed here explicitly).
    """
    if distortion is None:
        optics = params.get('optics') or {}
        distortion = optics.get('distortion')
    if not isinstance(distortion, dict):
        return
    k1 = float(distortion.get('k1', 0.0))
    k2 = float(distortion.get('k2', 0.0))
    nonradial_rms = float(distortion.get('nonradial_rms_px', 0.0))
    if k1 == 0.0 and k2 == 0.0 and nonradial_rms == 0.0:
        return

    size_v, size_u = frame.signal.shape
    # A scene states a center in pixel-corner coordinates on the detector grid;
    # the sampling grid below is pixel-centric on the oversampled one.  Scaling
    # a corner coordinate is a plain multiply, and the half pixel takes it the
    # rest of the way.  The default is the frame's own center, which lands on
    # (size - 1) / 2 by the same route.
    center_v = (
        float(distortion.get('center_v', (size_v / oversample) / 2.0)) * oversample
        - PIXEL_CENTER_TO_CORNER_PX
    )
    center_u = (
        float(distortion.get('center_u', (size_u / oversample) / 2.0)) * oversample
        - PIXEL_CENTER_TO_CORNER_PX
    )
    rho_ref = 0.5 * float(np.hypot(size_v, size_u))

    vv, uu = np.mgrid[0:size_v, 0:size_u].astype(np.float64)
    rv = vv - center_v
    ru = uu - center_u
    rho2 = (rv**2 + ru**2) / (rho_ref**2)
    factor = 1.0 + k1 * rho2 + k2 * rho2**2
    src_v = center_v + rv * factor
    src_u = center_u + ru * factor

    if nonradial_rms > 0.0:
        seed = derive_effect_seed(int(params.get('random_seed', 42)), 'optics/distortion')
        field_v, field_u = _nonradial_field((size_v, size_u), nonradial_rms * oversample, seed)
        src_v = src_v + field_v
        src_u = src_u + field_u

    coords = np.stack([src_v, src_u])
    frame.signal[:] = ndimage.map_coordinates(
        frame.signal, coords, order=3, mode='constant', cval=0.0
    )
    if float(np.count_nonzero(frame.point_e)) > 0.0:
        frame.point_e[:] = ndimage.map_coordinates(
            frame.point_e, coords, order=3, mode='constant', cval=0.0
        )
