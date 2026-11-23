"""Effects of a vidicon's target and its readout beam.

The target keeps a faint image of the frame before, and the beam that reads it bends
toward stored charge.  Both act on the vidicon chain's DN plane.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from spindoctor.support.types import NDArrayFloatType

__all__ = ['add_residual_image', 'apply_beam_bend']


def add_residual_image(
    signal: NDArrayFloatType,
    *,
    amplitude: float,
    prior: str,
    offset_v: int,
    offset_u: int,
) -> dict[str, Any]:
    """Add a faint ghost of a prior frame in place (the erase-cycle residual).

    When the light-flood erase cycle is shortened, a faint copy of the prior
    frame survives into the next.  With no prior frame available the current
    frame stands in for it: ``self_offset`` adds ``amplitude`` times a copy of
    the frame shifted by ``(offset_v, offset_u)``; ``flat`` adds a uniform
    ``amplitude`` times the frame mean.  Applied before the detector noise, so
    the ghost carries the noise the rest of the frame does.  A zero amplitude is
    a no-op.

    Parameters:
        signal: The image plane, modified in place.
        amplitude: Ghost strength as a fraction of the prior frame.
        prior: 'self_offset' (a displaced copy of this frame) or 'flat'.
        offset_v: Row shift of the ghost (self_offset).
        offset_u: Column shift of the ghost (self_offset).

    Returns:
        The realized ghost summary for the truth record.
    """
    if amplitude <= 0.0:
        return {'active': False}
    if prior == 'flat':
        signal += amplitude * float(signal.mean())
        return {'active': True, 'prior': 'flat'}
    ghost = np.roll(signal, shift=(offset_v, offset_u), axis=(0, 1))
    signal += amplitude * ghost
    return {'active': True, 'prior': 'self_offset', 'offset': [offset_v, offset_u]}


def apply_beam_bend(
    signal: NDArrayFloatType,
    *,
    amplitude_px: float,
    full_brightness_dn: float,
) -> dict[str, Any]:
    """Warp the image near bright boundaries by a brightness-dependent bias.

    A vidicon readout beam deflects toward stored charge, so a bright disc's
    limb position shifts by up to a pixel or two, the shift growing with local
    brightness.  This plants that residual geometric error: a smooth vertical
    displacement field whose amplitude scales with the locally-smoothed
    brightness and whose direction follows the brightness gradient (toward the
    brighter side).  It is a deliberately simple, tunable model of a real
    navigation-error source -- the amplitude and sign are knobs a later
    calibration pass fits, not a first-principles beam-physics solution.  A zero
    amplitude is a no-op.

    Local brightness is the smoothed charge over ``full_brightness_dn``, the
    charge a fully lit (signal 1.0) scene stores, capped at 1.  The reference is
    fixed, so a bright spot -- a cosmic-ray hit, say -- bends the beam only
    within reach of the smoothing kernel around it, and not across the frame.

    Parameters:
        signal: The DN image, warped in place.
        amplitude_px: Peak displacement, in pixels, at full local brightness.
        full_brightness_dn: The DN of full local brightness; a non-positive value
            makes the bend a no-op.

    Returns:
        The realized bend summary for the truth record.
    """
    if amplitude_px <= 0.0 or full_brightness_dn <= 0.0:
        return {'active': False}
    size_v, size_u = signal.shape
    smooth = ndimage.gaussian_filter(signal, sigma=2.0)
    brightness = np.clip(smooth / full_brightness_dn, 0.0, 1.0)
    grad_v = np.gradient(brightness, axis=0)
    # Displace toward the brighter side, scaled by local brightness; the beam
    # bends most where the disc is brightest and the gradient steepest.
    disp_v = amplitude_px * brightness * np.sign(grad_v)
    vv, uu = np.mgrid[0:size_v, 0:size_u].astype(np.float64)
    warped = ndimage.map_coordinates(signal, [vv + disp_v, uu], order=1, mode='nearest')
    signal[:] = warped
    return {'active': True, 'max_displacement_px': float(np.abs(disp_v).max())}
