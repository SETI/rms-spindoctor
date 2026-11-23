"""Motion smear for the optics stage: exposure-time averaging of the scene.

During an exposure the scene drifts across the detector, so the recorded frame
is the average of the radiance over the drift track.  A single ``object_class:
all`` entry smears the whole scene by one motion vector; several entries give
differential smear, where each object class (stars, bodies, rings) carries its
own vector -- the fast-flyby regime where a tracked target stays sharp while
the star field trails, or vice versa.

Differential smear composites the per-class radiance layers the radiance stage
records.  The body and ring layers are intensive-signal layers (they sum back
into ``frame.signal``); the star layer is a point-source layer in the electron /
DN domain (it sums back into ``frame.point_e``), because stars never pass through
the detector's intensive conversion.  Cross-class occlusion is resolved before
smear (each class is smeared in isolation, then the classes are summed), which
is a stated approximation where two classes overlap; at present fidelity rings
sit behind bodies and neither overlaps the star field, so the approximation is
exact for the scenes it serves.  Smear runs first among the optics sub-stages,
on the un-blurred layers, so the downstream PSF and distortion form the image of
the time-averaged radiance.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from spindoctor.sim.render.roll import turn_vector
from spindoctor.sim.render.stages import SimFrame
from spindoctor.sim.scene.fields import scene_value
from spindoctor.support.stable_math import stable_hypot
from spindoctor.support.types import NDArrayFloatType

__all__ = ['apply_smear', 'smear_kernel', 'smear_reach_px']

# The intensive-signal layers (summed back into frame.signal) and the
# point-source layer (summed back into frame.point_e), kept separate because the
# star layer carries electron / DN weights that never enter the signal plane.
_SIGNAL_LAYER_CLASSES: tuple[str, ...] = ('rings', 'bodies')
_POINT_LAYER_CLASS: str = 'stars'


def smear_kernel(dv_px: float, du_px: float) -> NDArrayFloatType | None:
    """Build a normalized line-segment motion-blur kernel.

    The kernel is the drift track from ``-(dv, du)/2`` to ``+(dv, du)/2`` about
    the center, sampled and bilinearly splatted so the smear is centered (the
    centroid does not move).

    Parameters:
        dv_px: Total drift along v in pixels of the render grid.
        du_px: Total drift along u in pixels of the render grid.

    Returns:
        A normalized 2-D kernel, or None when the drift is negligible.
    """
    length = float(stable_hypot(dv_px, du_px))
    if length < 1e-9:
        return None
    radius = _kernel_radius(dv_px, du_px)
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    n_samples = max(2, int(np.ceil(length)) + 1)
    weight = 1.0 / n_samples
    for t in np.linspace(-0.5, 0.5, n_samples):
        pos_v = radius + t * dv_px
        pos_u = radius + t * du_px
        v0 = int(np.floor(pos_v))
        u0 = int(np.floor(pos_u))
        fv = pos_v - v0
        fu = pos_u - u0
        kernel[v0, u0] += (1.0 - fv) * (1.0 - fu) * weight
        kernel[v0 + 1, u0] += fv * (1.0 - fu) * weight
        kernel[v0, u0 + 1] += (1.0 - fv) * fu * weight
        kernel[v0 + 1, u0 + 1] += fv * fu * weight
    kernel /= kernel.sum()
    return kernel


def _kernel_radius(dv_px: float, du_px: float) -> int:
    """The half-width of the smear kernel for a drift, in render-grid pixels.

    Parameters:
        dv_px: Total drift along v in pixels of the render grid.
        du_px: Total drift along u in pixels of the render grid.

    Returns:
        The half-width; the kernel spreads no light farther from its center.
    """
    return int(np.ceil(max(abs(dv_px), abs(du_px)) / 2.0)) + 1


def smear_reach_px(
    smear: Sequence[Mapping[str, Any]], *, oversample: int, roll_deg: float = 0.0
) -> int:
    """How far the scene's smear spreads light, in render-grid pixels.

    A drift track runs half the drift to each side of the center, and the bilinear
    sample at each end touches the pixels on either side of it, so light reaches
    half the drift rounded up to a whole pixel.  The kernel array is a pixel wider on
    each side, which holds no light.  The reach is the farthest among the
    whole-scene vector and the vector each object class is smeared by, each turned
    by the roll as :func:`apply_smear` turns it.

    Parameters:
        smear: The scene ``optics.smear`` list of per-class motion entries.
        oversample: The render-grid oversampling factor.
        roll_deg: The planted camera roll in degrees, from ``+v`` toward ``+u``.

    Returns:
        The reach; 0 when nothing is smeared.
    """
    (all_dv, all_du), per_class = _turned_smear(smear, roll_deg)
    vectors = [(all_dv, all_du)] + [(all_dv + dv, all_du + du) for dv, du in per_class.values()]
    reach = 0
    for dv, du in vectors:
        # The same negligible-drift test smear_kernel applies.
        if float(stable_hypot(dv * oversample, du * oversample)) >= 1e-9:
            half = max(abs(dv), abs(du)) * oversample / 2.0
            reach = max(reach, math.ceil(half))
    return reach


def _parse_smear(
    smear: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, float], dict[str, tuple[float, float]]]:
    """Split the smear list into a whole-scene vector and per-class vectors."""
    all_dv = 0.0
    all_du = 0.0
    per_class: dict[str, tuple[float, float]] = {}
    for entry in smear:
        dv = float(scene_value(entry, 'dv_px', 0.0))
        du = float(scene_value(entry, 'du_px', 0.0))
        object_class = str(entry.get('object_class', 'all'))
        if object_class == 'all':
            all_dv += dv
            all_du += du
        else:
            pdv, pdu = per_class.get(object_class, (0.0, 0.0))
            per_class[object_class] = (pdv + dv, pdu + du)
    return (all_dv, all_du), per_class


def _turned_smear(
    smear: Sequence[Mapping[str, Any]], roll_deg: float
) -> tuple[tuple[float, float], dict[str, tuple[float, float]]]:
    """The whole-scene and per-class smear vectors, turned by the planted roll."""
    (all_dv, all_du), per_class = _parse_smear(smear)
    if roll_deg != 0.0:
        all_dv, all_du = turn_vector(all_dv, all_du, roll_deg=roll_deg)
        per_class = {
            object_class: turn_vector(dv, du, roll_deg=roll_deg)
            for object_class, (dv, du) in per_class.items()
        }
    return (all_dv, all_du), per_class


def _convolve_in_place(plane: NDArrayFloatType, kernel: NDArrayFloatType) -> None:
    """FFT-convolve a plane with a kernel in place (deterministic)."""
    plane[:] = fftconvolve(plane, kernel, mode='same')


def apply_smear(
    frame: SimFrame,
    *,
    smear: Sequence[Mapping[str, Any]],
    oversample: int,
    roll_deg: float = 0.0,
) -> None:
    """Apply whole-scene or differential motion smear in place.

    The drift is the scene's motion across the detector, so a planted camera roll
    turns every motion vector with the rest of the scene.

    Parameters:
        frame: The frame whose signal (and, for whole-scene smear, point-source)
            plane is smeared in place.
        smear: The scene ``optics.smear`` list of per-class motion entries.
        oversample: The render-grid oversampling factor (motion vectors are in
            detector pixels and scale to the render grid).
        roll_deg: The planted camera roll in degrees, from ``+v`` toward ``+u``.
    """
    (all_dv, all_du), per_class = _turned_smear(smear, roll_deg)

    if not per_class:
        kernel = smear_kernel(all_dv * oversample, all_du * oversample)
        if kernel is None:
            return
        _convolve_in_place(frame.signal, kernel)
        if float(np.count_nonzero(frame.point_e)) > 0.0:
            _convolve_in_place(frame.point_e, kernel)
        return

    layers = frame.truth.pop('radiance_layers', None)
    if layers is None:
        # No per-class layers were captured; fall back to the whole-scene
        # vector so the smear still contributes rather than silently vanishing.
        kernel = smear_kernel(all_dv * oversample, all_du * oversample)
        if kernel is not None:
            _convolve_in_place(frame.signal, kernel)
            if float(np.count_nonzero(frame.point_e)) > 0.0:
                _convolve_in_place(frame.point_e, kernel)
        return

    # Body and ring layers recompose the intensive signal; the star layer
    # recomposes the point-source plane, each smeared by its own vector.
    signal_result = np.zeros_like(frame.signal)
    for object_class in _SIGNAL_LAYER_CLASSES:
        layer = np.asarray(layers[object_class], dtype=np.float64).copy()
        pdv, pdu = per_class.get(object_class, (0.0, 0.0))
        kernel = smear_kernel((all_dv + pdv) * oversample, (all_du + pdu) * oversample)
        if kernel is not None:
            layer = fftconvolve(layer, kernel, mode='same')
        signal_result += layer
    # No clip: the signal plane is unclipped through the optics stage on both
    # smear paths (point-mass star deposits legitimately exceed 1.0 before the
    # PSF spreads them); the detector conversion floors it at 0 after the
    # downsample, and the full well saturates it.
    frame.signal[:] = signal_result

    star_layer = np.asarray(layers[_POINT_LAYER_CLASS], dtype=np.float64).copy()
    sdv, sdu = per_class.get(_POINT_LAYER_CLASS, (0.0, 0.0))
    kernel = smear_kernel((all_dv + sdv) * oversample, (all_du + sdu) * oversample)
    if kernel is not None:
        star_layer = fftconvolve(star_layer, kernel, mode='same')
    frame.point_e[:] = star_layer
