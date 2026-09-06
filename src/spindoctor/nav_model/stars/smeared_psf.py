"""Smear-aware PSF rendering helpers.

When the spacecraft attitude rate is non-zero during an exposure, stars
smear into trails along the per-pixel motion vector.  ``psfmodel`` exposes
a smear-aware ``eval_rect(..., movement=, movement_granularity=)`` API
that integrates the PSF along the trail.  These helpers build the small
postage stamp returned by that call and choose a sensible
``movement_granularity`` from the smear amplitude.

A second helper computes the per-image smear vector ``(my, mx)`` (in
pixels) from the SPICE pointing brackets at the start and end of the
exposure, exposed as a free function so the orchestrator can drive it
from ``NavContext`` without subclassing the obs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from psfmodel import PSF

    from spindoctor.obs import ObsSnapshot
    from spindoctor.support.types import MutableStar, NDArrayFloatType

__all__ = [
    'compute_smear_vector_px',
    'movement_granularity_px',
    'render_smeared_psf',
    'smear_length_px',
]


def smear_length_px(move_v: float, move_u: float) -> float:
    """Return the smear length ``sqrt(my**2 + mx**2)`` in pixels.

    Parameters:
        move_v: Per-exposure smear amplitude along the V (row) axis.
        move_u: Per-exposure smear amplitude along the U (column) axis.

    Returns:
        Euclidean smear length in pixels (always >= 0).
    """
    return float(np.hypot(move_v, move_u))


def movement_granularity_px(move_v: float, move_u: float, *, max_steps: int = 50) -> float:
    """Choose the ``movement_granularity`` step for ``psf.eval_rect``.

    Targets at most ``max_steps`` integration samples along the smear
    path, clamped to ``[0.1, 1.0]`` pixels per sample.  The lower bound
    keeps the integration tractable for very short smears; the upper
    bound keeps long smears from sub-sampling the PSF.

    Parameters:
        move_v: Per-exposure smear amplitude along V.
        move_u: Per-exposure smear amplitude along U.
        max_steps: Maximum samples along the smear (default 50).

    Returns:
        Step size in pixels per integration sample.
    """
    if max_steps <= 0:
        raise ValueError(f'max_steps must be > 0; got {max_steps!r}')
    coarse = max(abs(move_v) / max_steps, abs(move_u) / max_steps)
    return float(np.clip(coarse, 0.1, 1.0))


def render_smeared_psf(
    psf: PSF,
    *,
    star: MutableStar,
    max_movement_steps: int,
) -> NDArrayFloatType:
    """Render one star's smeared PSF stamp.

    The output stamp shape is
    ``(2 * psf_half_v + 1) x (2 * psf_half_u + 1)`` where each half-size
    accounts for the PSF support plus the rounded smear amplitude.  The
    amplitude is folded into the half-sizes so the integrated stamp
    captures the entire trail without clipping.

    Parameters:
        psf: PSF instance from ``obs.star_psf()``.
        star: Star record carrying ``psf_size``, ``move_v``, ``move_u``,
            ``u``, ``v``, and ``dn``.
        max_movement_steps: Cap on the number of integration steps along
            the smear path; passed through to
            ``movement_granularity_px``.

    Returns:
        ``np.ndarray`` postage stamp containing the rendered PSF, in DN.
    """
    psf_half_u = int(star.psf_size[1] + np.round(abs(star.move_u))) // 2
    psf_half_v = int(star.psf_size[0] + np.round(abs(star.move_v))) // 2
    move_gran = movement_granularity_px(star.move_v, star.move_u, max_steps=max_movement_steps)
    u_idx = star.u
    v_idx = star.v
    u_int = int(u_idx)
    v_int = int(v_idx)
    u_frac = float(u_idx - u_int)
    v_frac = float(v_idx - v_int)
    stamp = psf.eval_rect(
        (psf_half_v * 2 + 1, psf_half_u * 2 + 1),
        offset=(v_frac, u_frac),
        scale=star.dn,
        movement=(star.move_v, star.move_u),
        movement_granularity=move_gran,
    )
    return np.asarray(stamp, dtype=np.float64)


def compute_smear_vector_px(obs: ObsSnapshot) -> tuple[float, float]:
    """Compute the per-image smear vector ``(my, mx)`` in pixels.

    Implements the design's "smear from SPICE bracket" approach: project
    the camera attitude at ``obs.time[0]`` and ``obs.time[1]`` into pixel
    coordinates by re-evaluating the frame centre's RA/DEC through the obs
    FOV at both bracket times, and take the difference.  The result is
    the total per-exposure displacement of a star at the centre of the
    FOV.

    Parameters:
        obs: Observation snapshot.

    Returns:
        Tuple ``(my, mx)`` in pixels (vertical, horizontal).
    """
    # ``obs.uv_from_ra_and_dec`` accepts ``tfrac`` in [0, 1] mapping the
    # exposure window onto a unit interval; tfrac=0 is the start ET and
    # tfrac=1 is the end ET.  Differencing the two projections of one fixed
    # sky direction gives the per-exposure pointing displacement at the FOV
    # centre.  The direction is the centre's own, read off the one-pixel
    # backplane the snapshot already builds.
    center_ra, center_dec = obs.center_ra_dec(apparent=True)
    boresight_uv0 = obs.uv_from_ra_and_dec(
        center_ra,
        center_dec,
        tfrac=0.0,
        apparent=True,
    )
    boresight_uv1 = obs.uv_from_ra_and_dec(
        center_ra,
        center_dec,
        tfrac=1.0,
        apparent=True,
    )
    u0, v0 = boresight_uv0.to_scalars()
    u1, v1 = boresight_uv1.to_scalars()
    return float(v1.vals - v0.vals), float(u1.vals - u0.vals)
