"""Image-side optics stage: what the camera's optical path does to the scene.

The optics stage runs on the oversampled radiance image, in a fixed internal
order chosen to mirror image formation:

1. **Smear** averages the scene radiance over the exposure along the pointing
   drift (whole-scene, or per object class for differential smear).  It runs
   first, while the per-class layers are still separable, so the optics below
   form the image of the time-averaged radiance.
2. **Distortion** warps the geometric image: the residual field-position error
   the navigator does not correct maps where each point of the scene lands.
3. **PSF** blurs the mapped image by the aperture's core-plus-wing kernel; the
   limb, ring-edge, and star profiles all inherit it.
4. **Ghosts** add displaced, defocused, low-amplitude copies of the formed
   focal-plane image (internal reflections).
5. **Stray light** adds the smooth scattered-light background last.

A stage whose scene block is absent contributes nothing.  Only the distortion
non-radial field draws randomness, and it derives its own seeded stream from
the scene seed, so the optics stage does not consume the pipeline generator.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from spindoctor.sim.forward.artifacts_catalog import (
    DISTORTION_RESIDUAL_PARAMS,
    PSF_KERNELS,
)
from spindoctor.sim.forward.distortion import apply_distortion
from spindoctor.sim.forward.ghosts import apply_ghosts
from spindoctor.sim.forward.psf import apply_psf, psf_truncation_for_instrument
from spindoctor.sim.forward.smear import apply_smear
from spindoctor.sim.forward.stages import SimFrame
from spindoctor.sim.instruments import navigator_matched_psf
from spindoctor.support.constants import PIXEL_CENTER_TO_CORNER_PX
from spindoctor.support.types import NDArrayFloatType

__all__ = ['apply_optics', 'apply_stray_light', 'effective_psf', 'instrument_defaults_on']


def instrument_defaults_on(params: Mapping[str, Any]) -> bool:
    """Whether the scene opts into the instrument's physical signal chain."""
    artifacts = params.get('artifacts')
    return isinstance(artifacts, dict) and bool(artifacts.get('instrument_defaults', False))


def effective_psf(params: Mapping[str, Any]) -> dict[str, Any] | None:
    """The PSF block to apply: an explicit optics.psf, else the catalog kernel.

    An explicit ``optics.psf`` block wins; the authored navigator-matched form
    (``{match_navigator: true}``) resolves here to the navigator's own Gaussian
    at the emulated instrument's configured ``star_psf_sigma``.  Otherwise
    ``instrument_defaults`` supplies the instrument's empirical kernel from the
    catalog.  Absent both, there is no PSF (the stage-activation floor).

    Parameters:
        params: The full scene mapping.

    Returns:
        The resolved PSF parameter mapping, or None when no PSF is active.
    """
    optics = params.get('optics') or {}
    explicit = optics.get('psf')
    if isinstance(explicit, dict):
        if explicit.get('match_navigator'):
            # Import at call time keeps the config read out of module import.
            from spindoctor.config import DEFAULT_CONFIG

            return navigator_matched_psf(
                DEFAULT_CONFIG, params.get('instrument'), params.get('instrument_config')
            )
        return explicit
    if instrument_defaults_on(params):
        kernel = PSF_KERNELS.get(str(params.get('instrument')))
        if kernel is not None:
            return dict(kernel)
    return None


def _effective_distortion(params: Mapping[str, Any]) -> dict[str, Any] | None:
    """The distortion block to apply: explicit optics.distortion, else residual.

    An explicit block wins; otherwise ``instrument_defaults`` supplies the
    catalog's measured per-instrument residual distortion -- the radial ``k1`` /
    ``k2`` coefficients and the non-radial wander RMS from the star-field FOV
    distortion analysis.

    Parameters:
        params: The full scene mapping.

    Returns:
        The resolved distortion parameter mapping, or None when no distortion is
        active.
    """
    optics = params.get('optics') or {}
    explicit = optics.get('distortion')
    if isinstance(explicit, dict):
        return explicit
    if not instrument_defaults_on(params):
        return None
    instrument = params.get('instrument')
    if instrument is None:
        return None
    measured = DISTORTION_RESIDUAL_PARAMS.get(str(instrument))
    if measured is None:
        return None
    return dict(measured)


def apply_stray_light(
    img: NDArrayFloatType,
    *,
    amplitude: float,
    direction_deg: float = 0.0,
    model: str = 'linear',
    center_v: float | None = None,
    center_u: float | None = None,
) -> None:
    """Add a smooth low-frequency stray-light field to the signal in place.

    Scattered light raises a slowly-varying background across the frame; the
    navigator's BANDPASS_DOG source-image filter is meant to remove it.  The
    field is additive, so it brightens dark sky as well as lit features (a
    multiplicative field would leave the dark sky -- where the gradient most
    needs suppressing -- untouched).  It is applied to the noise-free signal in
    [0, 1] before the detector stage.

    Parameters:
        img: Normalized [0, 1] signal image, modified in place.
        amplitude: Peak stray-light level added, in normalized signal units.
        direction_deg: Ramp direction for the 'linear' model, in degrees.
        model: 'linear' (a ramp spanning [0, amplitude]) or 'radial' (a bump of
            height amplitude fading to 0 at the farthest corner).
        center_v: Bump center v for 'radial'; frame center when None.
        center_u: Bump center u for 'radial'; frame center when None.

    Raises:
        ValueError: If ``model`` is not 'linear' or 'radial'.
    """
    if amplitude <= 0.0:
        return
    size_v, size_u = img.shape
    vv, uu = np.mgrid[0:size_v, 0:size_u]
    vv = vv.astype(np.float64)
    uu = uu.astype(np.float64)
    if model == 'linear':
        theta = np.radians(direction_deg)
        proj = np.cos(theta) * vv + np.sin(theta) * uu
        proj -= float(proj.min())
        span = float(proj.max())
        field = amplitude * (proj / span) if span > 0.0 else np.zeros_like(proj)
    elif model == 'radial':
        # ``vv`` / ``uu`` are pixel-centric, so the frame's center is
        # ``(size - 1) / 2`` and its outer corners sit half a pixel outside the
        # first and last sample.
        half = PIXEL_CENTER_TO_CORNER_PX
        cv = (size_v - 1) / 2.0 if center_v is None else float(center_v)
        cu = (size_u - 1) / 2.0 if center_u is None else float(center_u)
        radius = np.sqrt((vv - cv) ** 2 + (uu - cu) ** 2)
        v_lo, v_hi = -half, size_v - half
        u_lo, u_hi = -half, size_u - half
        corners = [(v_lo, u_lo), (v_lo, u_hi), (v_hi, u_lo), (v_hi, u_hi)]
        r_max = max(np.hypot(cv - c[0], cu - c[1]) for c in corners)
        if r_max <= 0.0:
            return
        field = amplitude * np.clip(1.0 - radius / r_max, 0.0, 1.0)
    else:
        raise ValueError(f"stray_light model must be 'linear' or 'radial'; got {model!r}")
    img += field


def apply_optics(
    frame: SimFrame,
    *,
    params: Mapping[str, Any],
    rng: np.random.Generator,
) -> None:
    """Optics stage: apply the scene's optical-path effects in place.

    Runs the smear, distortion, PSF, ghost, and stray-light sub-stages in the
    fixed internal order documented at the module level.  A sub-stage whose
    block is absent from the scene ``optics`` mapping contributes nothing.

    Parameters:
        frame: The frame whose signal and point-source planes are modified.
        params: The full scene mapping; reads the ``optics`` block.
        rng: The stage generator (used by the seeded distortion field).
    """
    del rng
    optics = params.get('optics') or {}
    oversample = int(frame.oversample)

    smear = optics.get('smear')
    if smear:
        apply_smear(frame, smear=smear, oversample=oversample)

    apply_distortion(
        frame, params=params, oversample=oversample, distortion=_effective_distortion(params)
    )

    psf = effective_psf(params)
    if isinstance(psf, dict):
        sigma_v = float(psf['sigma_v'])
        sigma_u = float(psf.get('sigma_u', sigma_v))
        apply_psf(
            frame.signal,
            frame.point_e,
            sigma_v=sigma_v,
            sigma_u=sigma_u,
            w=float(psf.get('w', 0.0)),
            r0=float(psf.get('r0', 2.0)),
            n=float(psf.get('n', 3.0)),
            truncation_px=psf_truncation_for_instrument(params.get('instrument')),
            oversample=oversample,
        )

    ghosts = optics.get('ghosts')
    if ghosts:
        apply_ghosts(frame, ghosts=ghosts, oversample=oversample)

    stray = optics.get('stray_light')
    if stray:
        # A scene states a center in pixel-corner coordinates on the detector
        # grid; the signal plane is pixel-centric on the oversampled one.
        center_v = stray.get('center_v')
        center_u = stray.get('center_u')
        if center_v is not None:
            center_v = float(center_v) * oversample - PIXEL_CENTER_TO_CORNER_PX
        if center_u is not None:
            center_u = float(center_u) * oversample - PIXEL_CENTER_TO_CORNER_PX
        apply_stray_light(
            frame.signal,
            amplitude=float(stray.get('amplitude', 0.0)),
            direction_deg=float(stray.get('direction_deg', 0.0)),
            model=str(stray.get('model', 'linear')),
            center_v=center_v,
            center_u=center_u,
        )
