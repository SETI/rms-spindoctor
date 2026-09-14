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

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from spindoctor.sim.forward.stages import SimFrame
from spindoctor.support.types import NDArrayFloatType

__all__ = ['apply_smear', 'smear_kernel']

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
    length = float(np.hypot(dv_px, du_px))
    if length < 1e-9:
        return None
    radius = int(np.ceil(max(abs(dv_px), abs(du_px)) / 2.0)) + 1
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


def _parse_smear(
    smear: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, float], dict[str, tuple[float, float]]]:
    """Split the smear list into a whole-scene vector and per-class vectors."""
    all_dv = 0.0
    all_du = 0.0
    per_class: dict[str, tuple[float, float]] = {}
    for entry in smear:
        dv = float(entry.get('dv_px', 0.0))
        du = float(entry.get('du_px', 0.0))
        object_class = str(entry.get('object_class', 'all'))
        if object_class == 'all':
            all_dv += dv
            all_du += du
        else:
            pdv, pdu = per_class.get(object_class, (0.0, 0.0))
            per_class[object_class] = (pdv + dv, pdu + du)
    return (all_dv, all_du), per_class


def _convolve_in_place(plane: NDArrayFloatType, kernel: NDArrayFloatType) -> None:
    """FFT-convolve a plane with a kernel in place (deterministic)."""
    plane[:] = fftconvolve(plane, kernel, mode='same')


def apply_smear(
    frame: SimFrame,
    *,
    smear: Sequence[Mapping[str, Any]],
    oversample: int,
) -> None:
    """Apply whole-scene or differential motion smear in place.

    Parameters:
        frame: The frame whose signal (and, for whole-scene smear, point-source)
            plane is smeared in place.
        smear: The scene ``optics.smear`` list of per-class motion entries.
        oversample: The render-grid oversampling factor (motion vectors are in
            detector pixels and scale to the render grid).
    """
    (all_dv, all_du), per_class = _parse_smear(smear)

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
    # PSF spreads them); the detector conversion clips after the downsample.
    frame.signal[:] = signal_result

    star_layer = np.asarray(layers[_POINT_LAYER_CLASS], dtype=np.float64).copy()
    sdv, sdu = per_class.get(_POINT_LAYER_CLASS, (0.0, 0.0))
    kernel = smear_kernel((all_dv + sdv) * oversample, (all_du + sdu) * oversample)
    if kernel is not None:
        star_layer = fftconvolve(star_layer, kernel, mode='same')
    frame.point_e[:] = star_layer
