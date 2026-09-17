"""Human-viewable PNG export for simulated images.

The simulator renders detector counts (DN), whose absolute range depends on the
instrument full-scale and on any cosmic-ray spikes, so a raw cast to 8-bit would
be unreadable: a single hot pixel can scale the whole frame to black.  These
helpers stretch a DN image to a visible grayscale PNG with a percentile clip (so
a few outliers do not crush the body / star / ring signal) and an optional gamma
that lifts dim features -- a thin high-phase crescent, faint background stars --
without blowing out a bright disc.

The two entry points are :func:`stretch_to_uint8` (DN array to an 8-bit array)
and :func:`save_png` (write that array to a PNG file).  :func:`render_scene_png`
is the convenience that renders a ``sim_params`` scene and saves it in one call,
used by the documentation-image and sweep-dump tooling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spindoctor.support.types import NDArrayFloatType, NDArrayUint8Type, PathLike

__all__ = ['render_scene_png', 'save_png', 'stretch_to_uint8']


def stretch_to_uint8(
    image: NDArrayFloatType,
    *,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    gamma: float = 1.0,
) -> NDArrayUint8Type:
    """Stretch a DN image to an 8-bit grayscale array for human viewing.

    The black and white points are taken at the requested percentiles of the
    finite pixels (not the absolute min/max), so a handful of cosmic-ray or
    saturated pixels do not collapse the rest of the frame to black.  A ``gamma``
    above 1 brightens the midtones, which makes a dim crescent or a faint star
    field legible alongside a bright disc.  Non-finite pixels (the missing-data
    marker on calibrated frames) are mapped to the black point.

    Parameters:
        image: The DN image array.
        low_percentile: Percentile mapped to black (0).
        high_percentile: Percentile mapped to white (255).
        gamma: Display gamma; values above 1 lift dim features.

    Returns:
        A ``uint8`` array of the same shape, in [0, 255].
    """
    arr = np.asarray(image, dtype=np.float64)
    out_shape = arr.shape
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(out_shape, dtype=np.uint8)
    lo = float(np.percentile(finite, low_percentile))
    hi = float(np.percentile(finite, high_percentile))
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(out_shape, dtype=np.uint8)
    norm = np.clip((np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo) - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        norm = np.power(norm, 1.0 / gamma)
    return (norm * 255.0 + 0.5).astype(np.uint8)


def save_png(
    image: NDArrayFloatType,
    path: PathLike,
    *,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    gamma: float = 1.0,
    upscale: int = 1,
) -> Path:
    """Stretch a DN image and write it as a grayscale PNG.

    Parameters:
        image: The DN image array.
        path: Destination ``.png`` path; parent directories are created.
        low_percentile: Percentile mapped to black.
        high_percentile: Percentile mapped to white.
        gamma: Display gamma; values above 1 lift dim features.
        upscale: Integer nearest-neighbor magnification, so a small frame is
            still legible in a document (1 disables it).

    Returns:
        The written path.
    """
    u8 = stretch_to_uint8(
        image, low_percentile=low_percentile, high_percentile=high_percentile, gamma=gamma
    )
    im = Image.fromarray(u8, mode='L')
    if upscale > 1:
        im = im.resize((u8.shape[1] * upscale, u8.shape[0] * upscale), Image.Resampling.NEAREST)
    out = Path(str(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(out))
    return out


def render_scene_png(
    sim_params: dict[str, Any],
    path: PathLike,
    *,
    ignore_offset: bool = True,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    gamma: float = 1.0,
    upscale: int = 1,
) -> Path:
    """Render a ``sim_params`` scene and save it as a viewable PNG.

    Parameters:
        sim_params: The scene parameters consumed by
            :func:`spindoctor.sim.render.render_combined_model`.
        path: Destination ``.png`` path.
        ignore_offset: Render the unshifted geometry (the planted navigation
            offset is a small sub-pixel shift that is not meant to be visible);
            set ``False`` to render exactly what the navigator sees.
        low_percentile: Percentile mapped to black.
        high_percentile: Percentile mapped to white.
        gamma: Display gamma; values above 1 lift dim features.
        upscale: Integer nearest-neighbor magnification.

    Returns:
        The written path.
    """
    from spindoctor.sim.render import render_combined_model

    img, _ = render_combined_model(sim_params, ignore_offset=ignore_offset)
    return save_png(
        img,
        path,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        gamma=gamma,
        upscale=upscale,
    )
