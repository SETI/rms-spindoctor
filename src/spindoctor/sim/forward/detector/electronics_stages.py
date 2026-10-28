"""Detector-electronics artifact mechanics (the registry's detector modes).

Each function here renders one detector/electronics artifact mode from the
artifact-mode registry onto a detector-grid plane in place, and (where a
placement is stochastic) accepts a seeded generator and an optional adversarial
candidate pool.  The chain orchestrator
(:mod:`spindoctor.sim.forward.detector.chain`) wires each one at the physically
right point in the unit chain and records its realized geometry into the frame
truth; these functions carry only the mechanics, so each is unit-testable on a
synthetic plane and is a no-op when its gating amplitude / count is zero (the
stage-activation rule).

Domains follow the physics: fixed-pattern PRNU, vignetting, and dust donuts are
multiplicative on the electron plane before Poisson (a per-pixel response, not
an added signal); the dark ramp and frame-transfer smear add electrons; the
stitch combs, jail bars, and serial tail act in the DN domain after the gain
divide (readout-chain and amplifier structure rides on the digitized signal);
and the Voyager beam bend and residual image are geometric / pre-noise effects
on the scene plane.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = [
    'add_bright_dark_pairs',
    'add_coherent_banding',
    'add_dark_ramp',
    'add_fixed_pattern_dn',
    'add_fixed_pattern_response',
    'add_residual_image',
    'add_serial_tail',
    'apply_beam_bend',
    'apply_exposure_shading',
    'apply_frame_transfer_smear',
]


def add_coherent_banding(
    electrons: NDArrayFloatType,
    *,
    amplitude_e: float,
    period_px: float,
    orientation: str,
    freq_step_factor: float,
    dark_step_dn: float,
    gain_e_per_dn: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Add coherent banding (electrons) in place, horizontal and/or vertical.

    A horizontal family is line-correlated (constant along a row, a sinusoid in
    the row index) -- the Cassini 2 Hz / LORRI striping shape; a vertical family
    is column-correlated (a sinusoid in the column index) -- the Galileo 42-px
    supply-noise comb.  ``both`` lays down a vertical comb plus a horizontal
    band.  A mid-image readout pause steps the horizontal spatial frequency by
    ``freq_step_factor`` and, when ``dark_step_dn`` is set, steps the dark level
    at that line (converted to electrons through ``gain_e_per_dn``).  A zero
    amplitude or non-positive period is a no-op.

    Parameters:
        electrons: The electron image, modified in place.
        amplitude_e: Coherent-banding amplitude in electrons.
        period_px: Sinusoid spatial period, in pixels.
        orientation: 'horizontal', 'vertical', or 'both'.
        freq_step_factor: Horizontal-band period multiplier below the mid-line.
        dark_step_dn: Dark-level step (DN) applied below the mid-line.
        gain_e_per_dn: Gain, converting the DN dark step to electrons.
        rng: The mode's seeded generator.

    Returns:
        The realized banding geometry for the truth record.
    """
    if amplitude_e <= 0.0 or period_px <= 0.0:
        return {'active': False}
    size_v, size_u = electrons.shape
    mid = size_v // 2
    freq = 2.0 * np.pi / period_px
    if orientation in ('horizontal', 'both'):
        freqs = np.full(size_v, freq, dtype=np.float64)
        if freq_step_factor != 1.0 and freq_step_factor > 0.0:
            freqs[mid:] = freq * freq_step_factor
        argument = np.zeros(size_v, dtype=np.float64)
        argument[1:] = np.cumsum(freqs[:-1])
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        band = amplitude_e * np.sin(argument + phase)
        band += 0.3 * amplitude_e * rng.standard_normal(size_v)
        electrons += band[:, None]
    if orientation in ('vertical', 'both'):
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        cols = np.arange(size_u, dtype=np.float64)
        comb = amplitude_e * np.sin(2.0 * np.pi * cols / period_px + phase)
        electrons += comb[None, :]
    if dark_step_dn > 0.0 and gain_e_per_dn > 0.0:
        electrons[mid:, :] += dark_step_dn * gain_e_per_dn
    return {
        'active': True,
        'orientation': orientation,
        'period_px': period_px,
        'dark_step_line': mid if dark_step_dn > 0.0 else None,
    }


def add_dark_ramp(
    electrons: NDArrayFloatType,
    *,
    amplitude_e: float,
    nonlinear: float,
    rbi_column_factor: float,
    hot_columns: NDArrayIntType | None,
) -> dict[str, Any]:
    """Add a dark signal growing with line number (readout gradient) in place.

    The dark ramp accumulates during the line-by-line readout, so it grows from
    line 0 to the last line: ``amplitude_e`` is the extra dark charge at the
    last line, and ``nonlinear`` bends the growth (an exponent != 1 gives the
    vidicon's nonlinear wait-time dependence).  ``rbi_column_factor`` adds the
    Cassini residual-bulk-image flavor: columns read out above a hot pixel carry
    an enhanced ramp, so the listed ``hot_columns`` get an extra factor.  A zero
    amplitude is a no-op.

    Parameters:
        electrons: The electron image, modified in place.
        amplitude_e: Extra dark charge at the last line, in electrons.
        nonlinear: Ramp exponent (1 = linear growth with line number).
        rbi_column_factor: Extra ramp fraction on the enhanced columns.
        hot_columns: Columns carrying the enhanced RBI ramp, or None.

    Returns:
        The realized ramp geometry for the truth record.
    """
    if amplitude_e <= 0.0:
        return {'active': False}
    size_v, _size_u = electrons.shape
    exponent = nonlinear if nonlinear > 0.0 else 1.0
    frac = np.linspace(0.0, 1.0, size_v) ** exponent
    ramp = amplitude_e * frac
    electrons += ramp[:, None]
    enhanced: list[int] = []
    if rbi_column_factor > 0.0 and hot_columns is not None and hot_columns.size > 0:
        cols = np.unique(hot_columns)
        electrons[:, cols] += (rbi_column_factor * ramp)[:, None]
        enhanced = [int(c) for c in cols]
    return {'active': True, 'peak_e': amplitude_e, 'enhanced_columns': enhanced}


def apply_exposure_shading(
    electrons: NDArrayFloatType,
    *,
    top_factor: float,
    bottom_factor: float,
) -> dict[str, Any]:
    """Scale the signal by a line-dependent shutter exposure gradient in place.

    A focal-plane shutter opens and closes line by line, so the effective
    exposure varies from ``top_factor`` at line 0 to ``bottom_factor`` at the
    last line (the Galileo ~1.5 -> ~1.05 ms shading).  Modeled as a
    multiplicative gradient on the accumulated electrons.  Equal factors are a
    no-op.

    Parameters:
        electrons: The electron image, modified in place.
        top_factor: Exposure multiplier at line 0.
        bottom_factor: Exposure multiplier at the last line.

    Returns:
        The realized gradient for the truth record.
    """
    if top_factor <= 0.0 or bottom_factor <= 0.0 or top_factor == bottom_factor:
        return {'active': False}
    size_v, _size_u = electrons.shape
    gradient = np.linspace(top_factor, bottom_factor, size_v) / top_factor
    electrons *= gradient[:, None]
    return {'active': True, 'top_factor': top_factor, 'bottom_factor': bottom_factor}


def apply_frame_transfer_smear(
    electrons: NDArrayFloatType,
    *,
    t_scrub_sec: float,
    t_transfer_sec: float,
    exposure_sec: float,
) -> dict[str, Any]:
    """Add the frame-transfer vertical column pedestal (electrons) in place.

    A shutterless frame-transfer CCD keeps integrating while the image shifts
    through the bright rows during the pre-exposure scrub and the post-exposure
    transfer, so every column carries a pedestal proportional to its signal
    integral times ``(t_scrub + t_transfer) / t_exp``.  The pedestal differs on
    the two sides of the column's flux centroid: rows below the centroid in
    image coordinates (larger line numbers) receive the scrub share, rows above
    it receive the transfer share.  The scrub and transfer times are
    independent knobs, so swapping their values swaps the side assignment.  A
    zero transfer time or exposure is a no-op.

    The desmear-residual behavior (the ground pipeline's desmear failing through
    saturated columns and leaving residual banding) is not modeled here; the
    planted pedestal is the raw smear.

    Parameters:
        electrons: The electron image, modified in place.
        t_scrub_sec: Pre-exposure scrub time (seconds).
        t_transfer_sec: Post-exposure transfer time (seconds).
        exposure_sec: The scene exposure (seconds).

    Returns:
        The realized smear summary for the truth record.
    """
    if exposure_sec <= 0.0 or (t_scrub_sec <= 0.0 and t_transfer_sec <= 0.0):
        return {'active': False}
    size_v, _size_u = electrons.shape
    column_sum = electrons.sum(axis=0)
    ped_scrub = column_sum * t_scrub_sec / (exposure_sec * size_v)
    ped_transfer = column_sum * t_transfer_sec / (exposure_sec * size_v)
    rows = np.arange(size_v, dtype=np.float64)[:, None]
    weights = electrons - electrons.min(axis=0, keepdims=True)
    total = weights.sum(axis=0)
    safe = np.where(total > 0.0, total, 1.0)
    centroid = (rows * weights).sum(axis=0) / safe
    below_centroid = rows > centroid[None, :]
    electrons += np.where(below_centroid, ped_scrub[None, :], ped_transfer[None, :])
    return {
        'active': True,
        'mean_pedestal_e': float((ped_scrub + ped_transfer).mean() * 0.5),
    }


def add_serial_tail(
    dn: NDArrayFloatType,
    *,
    saturation_dn: float,
    saturation_frac: float,
    amplitude_dn: float,
    length_px: int,
    direction: str,
) -> dict[str, Any]:
    """Add a horizontal bright-then-dark serial tail off saturated cores in place.

    An antiblooming CCD shows no column bloom; a hard-saturated compact source
    instead drives an amplifier undershoot along the readout (serial) direction:
    a short bright overshoot immediately after the source, then a longer dark
    undershoot.  Every pixel at or above ``saturation_frac`` of the ADC ceiling
    seeds a tail of ``length_px`` in ``direction``.  A zero amplitude or length,
    or no saturated pixel, is a no-op.

    Parameters:
        dn: The DN image, modified in place.
        saturation_dn: The ADC ceiling (DN).
        saturation_frac: Fraction of the ceiling that counts as saturated.
        amplitude_dn: Peak tail amplitude (DN).
        length_px: Tail length in pixels.
        direction: 'right' (+u readout) or 'left' (-u readout).

    Returns:
        The realized tail summary for the truth record.
    """
    if amplitude_dn <= 0.0 or length_px <= 0:
        return {'active': False, 'sources': 0}
    threshold = saturation_frac * saturation_dn
    vs, us = np.nonzero(dn >= threshold)
    if vs.size == 0:
        return {'active': False, 'sources': 0}
    size_u = dn.shape[1]
    step = 1 if direction == 'right' else -1
    # The pixel just past the source shows a bright overshoot; the pixels beyond
    # it show a dark undershoot decaying back to zero over the tail length.  The
    # profile is sized at least 2 so a 1-px tail (whose loop writes nothing past
    # the source) still builds without indexing past the array.
    profile = np.zeros(max(length_px, 2), dtype=np.float64)
    profile[1] = amplitude_dn
    if length_px > 2:
        decay = np.linspace(1.0, 0.0, length_px - 2, endpoint=False)
        profile[2:] = -amplitude_dn * decay
    for v, u in zip(vs.tolist(), us.tolist(), strict=True):
        for k in range(1, length_px):
            uu = u + step * k
            if 0 <= uu < size_u:
                dn[v, uu] += profile[k]
    return {'active': True, 'sources': int(vs.size), 'direction': direction}


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


def add_fixed_pattern_response(
    electrons: NDArrayFloatType,
    *,
    prnu_rms: float,
    vignetting_frac: float,
    dust_donut_count: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Multiply the electron plane by the static per-pixel response in place.

    The multiplicative fixed pattern is a per-pixel response applied before
    Poisson: photo-response non-uniformity (a per-pixel gain jitter of
    ``prnu_rms``), corner vignetting (a radial falloff reaching ``vignetting_frac``
    at the corners), and ``dust_donut_count`` faint ring-shaped shadows.  The
    pattern is drawn from the mode's seeded stream, so it is the same every
    render of the scene.  All-zero parameters are a no-op.

    Parameters:
        electrons: The electron image, modified in place.
        prnu_rms: RMS of the per-pixel response jitter.
        vignetting_frac: Corner response deficit (0 = none).
        dust_donut_count: Number of dust-donut shadow rings.
        rng: The mode's seeded generator.

    Returns:
        The realized response summary for the truth record.
    """
    if prnu_rms <= 0.0 and vignetting_frac <= 0.0 and dust_donut_count <= 0:
        return {'active': False}
    size_v, size_u = electrons.shape
    response = np.ones((size_v, size_u), dtype=np.float64)
    if prnu_rms > 0.0:
        response += rng.normal(0.0, prnu_rms, size=(size_v, size_u))
    if vignetting_frac > 0.0:
        vv, uu = np.mgrid[0:size_v, 0:size_u].astype(np.float64)
        cv, cu = (size_v - 1) / 2.0, (size_u - 1) / 2.0
        rho2 = ((vv - cv) / cv) ** 2 + ((uu - cu) / cu) ** 2
        response *= 1.0 - vignetting_frac * (rho2 / 2.0)
    donuts = 0
    for _ in range(max(0, dust_donut_count)):
        dv = int(rng.integers(0, size_v))
        du = int(rng.integers(0, size_u))
        radius = float(rng.uniform(3.0, 8.0))
        vv, uu = np.mgrid[0:size_v, 0:size_u].astype(np.float64)
        ring = np.exp(-(((np.hypot(vv - dv, uu - du) - radius) / 1.5) ** 2))
        response *= 1.0 - 0.05 * ring
        donuts += 1
    np.maximum(response, 0.0, out=response)
    electrons *= response
    return {'active': True, 'prnu_rms': prnu_rms, 'dust_donuts': donuts}


def add_fixed_pattern_dn(
    dn: NDArrayFloatType,
    *,
    stitch_period_px: int,
    stitch_amplitude_dn: float,
    jail_bar_dn: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Add the static additive DN fixed pattern in place (combs and jail bars).

    The photolithography stitch comb is a set of bright columns every
    ``stitch_period_px`` pixels raised by ``stitch_amplitude_dn``; the jail bars
    are an even/odd column offset of ``jail_bar_dn`` whose sign is drawn once per
    seed (a power-cycle-dependent bias).  All-zero parameters are a no-op.

    Parameters:
        dn: The DN image, modified in place.
        stitch_period_px: Column period of the stitch comb (0 disables).
        stitch_amplitude_dn: Stitch-comb column amplitude (DN).
        jail_bar_dn: Even/odd column offset amplitude (DN).
        rng: The mode's seeded generator.

    Returns:
        The realized pattern summary for the truth record.
    """
    if (stitch_period_px <= 0 or stitch_amplitude_dn <= 0.0) and jail_bar_dn <= 0.0:
        return {'active': False}
    _size_v, size_u = dn.shape
    stitched: list[int] = []
    if stitch_period_px > 0 and stitch_amplitude_dn > 0.0:
        cols = np.arange(0, size_u, stitch_period_px)
        dn[:, cols] += stitch_amplitude_dn
        stitched = [int(c) for c in cols]
    if jail_bar_dn > 0.0:
        sign = 1.0 if bool(rng.integers(0, 2)) else -1.0
        parity = (np.arange(size_u) % 2).astype(np.float64) * 2.0 - 1.0
        dn += (sign * jail_bar_dn * parity)[None, :]
    return {'active': True, 'stitch_columns': stitched, 'jail_bar_dn': jail_bar_dn}


def add_bright_dark_pairs(
    electrons: NDArrayFloatType,
    *,
    count: int,
    amplitude_e: float,
    rng: np.random.Generator,
    candidate_pool: tuple[NDArrayIntType, NDArrayIntType] | None = None,
) -> dict[str, Any]:
    """Deposit scattered vertical bright/dark pixel pairs (electrons) in place.

    The Cassini anti-blooming mode produces isolated vertical two-pixel pairs in
    unsummed long exposures: one pixel raised by ``amplitude_e``, the pixel below
    it lowered by the same charge.  Placement is uniform, or drawn from
    ``candidate_pool`` for adversarial placement onto the navigation features.  A
    zero count or amplitude is a no-op.

    Parameters:
        electrons: The electron image, modified in place.
        count: Number of bright/dark pairs to deposit.
        amplitude_e: Pair amplitude in electrons.
        rng: The mode's seeded generator.
        candidate_pool: Optional ``(v, u)`` pool for adversarial placement.

    Returns:
        The realized pair geometry for the truth record.
    """
    if count <= 0 or amplitude_e <= 0.0:
        return {'pairs': []}
    size_v, size_u = electrons.shape
    if candidate_pool is not None and candidate_pool[0].size > 0:
        pool_v, pool_u = candidate_pool
        pick = rng.integers(0, pool_v.size, size=count)
        bv = np.clip(pool_v[pick], 0, size_v - 2)
        bu = np.clip(pool_u[pick], 0, size_u - 1)
    else:
        bv = rng.integers(0, size_v - 1, size=count)
        bu = rng.integers(0, size_u, size=count)
    pairs: list[list[int]] = []
    for v, u in zip(bv.tolist(), bu.tolist(), strict=True):
        electrons[v, u] += amplitude_e
        electrons[v + 1, u] -= amplitude_e
        pairs.append([int(v), int(u)])
    return {'pairs': pairs}
