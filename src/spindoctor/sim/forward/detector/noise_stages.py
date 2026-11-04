"""Stochastic and structured detector-noise stages (generic mechanics).

Each function here is a self-contained detector sub-effect: dark current and hot
pixels, coherent horizontal banding, low-order bias structure, and the
morphological cosmic-ray model.  The chain orchestrator
(:mod:`spindoctor.sim.forward.detector.chain`) draws each one an independent RNG
via ``derive_effect_seed(random_seed, 'detector/<effect>')`` so toggling one
never perturbs another's realization, and each stage is a no-op when its gating
amplitude/fraction/rate is zero (the stage-activation rule).  The one exception
is the uniform hot-pixel population, which belongs to the detector rather than
the scene: it is a fixed map keyed on the emulated detector
(:func:`detector_hot_pixel_map`), so the same pixels recur in every scene that
emulates that camera.

Dark current, hot pixels, banding, and cosmic rays act in the electron domain
(the vidicon chain, which has no electron domain, deposits its cosmic rays in DN);
bias structure acts in the DN domain (the amplifier pedestal and the read-out
row/column gradients ride on the digitized signal).  Per-instrument
parameterizations come with the catalog defaults; these are the generic shapes.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = [
    'add_banding',
    'add_bias_structure',
    'add_cosmic_rays',
    'add_dark_current',
    'add_hot_pixels',
    'deposit_morphological_events',
    'detector_hot_pixel_map',
]

# The hot-pixel map hashes each pixel's (v, u) as v * stride + u, so the stride
# bounds the detector width it keys uniquely; every emulated camera is far
# narrower.
_DETECTOR_STRIDE = 1 << 20
# SplitMix64 constants (Steele, Lea, and Flood 2014): a fast, well-mixed
# bijection on 64-bit words, applied element-wise to the pixel keys.
_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_MUL2 = np.uint64(0x94D049BB133111EB)
_UNIT_BITS = 53


def add_dark_current(
    electrons: NDArrayFloatType,
    *,
    rate_e_per_sec: float,
    exposure_sec: float,
) -> None:
    """Add a uniform dark-current pedestal (electrons) in place, pre-Poisson.

    The dark signal accumulates over the exposure.  Because the chain adds it
    before the Poisson stage, it carries its own shot noise whenever that
    stage is on (as it is under ``instrument_defaults``); with Poisson
    explicitly disabled it is a noise-free pedestal.  A zero rate is a no-op.

    Parameters:
        electrons: The electron image, modified in place.
        rate_e_per_sec: Dark current in electrons per second.
        exposure_sec: Exposure time in seconds.
    """
    if rate_e_per_sec <= 0.0:
        return
    electrons += rate_e_per_sec * exposure_sec


def _splitmix64(words: NDArrayIntType) -> NDArrayIntType:
    """One SplitMix64 step on each 64-bit word (wrapping unsigned arithmetic)."""
    z = words + _SPLITMIX_GAMMA
    z = (z ^ (z >> np.uint64(30))) * _SPLITMIX_MUL1
    z = (z ^ (z >> np.uint64(27))) * _SPLITMIX_MUL2
    return z ^ (z >> np.uint64(31))


def _unit_interval(words: NDArrayIntType) -> NDArrayFloatType:
    """Map 64-bit words to uniform floats in [0, 1) from their top 53 bits."""
    return (words >> np.uint64(64 - _UNIT_BITS)).astype(np.float64) * 2.0**-_UNIT_BITS


def detector_hot_pixel_map(
    shape: tuple[int, int], *, fraction: float, detector_id: str
) -> tuple[NDArrayIntType, NDArrayIntType, NDArrayFloatType]:
    """The emulated detector's fixed hot-pixel population over a frame.

    Each detector pixel ``(v, u)`` is hot when a uniform draw keyed on the
    detector and the pixel, never on a scene seed, falls below ``fraction``, so
    every scene emulating the detector sees the same hot pixels at the same
    coordinates, at any frame size (a smaller frame sees the corner of the
    map from ``(0, 0)``).  The count is binomial with mean ``fraction`` times
    the pixel count, and raising ``fraction`` only adds pixels.  Each hot
    pixel's relative amplitude is a fixed exponential draw keyed the same way.

    Parameters:
        shape: The frame's ``(size_v, size_u)`` in detector pixels.
        fraction: Fraction of detector pixels that are hot.
        detector_id: The emulated detector's name; the calibrated and raw
            products of one camera share it.

    Returns:
        ``(hot_v, hot_u, relative_amplitudes)`` in row-major order, the
        amplitudes exponentially distributed with mean 1.
    """
    size_v, size_u = shape
    key = int.from_bytes(hashlib.sha256(f'hot_pixels:{detector_id}'.encode()).digest()[:8], 'big')
    vs, us = np.indices((size_v, size_u), dtype=np.uint64)
    words = _splitmix64((vs * np.uint64(_DETECTOR_STRIDE) + us) ^ np.uint64(key))
    hot_v, hot_u = np.nonzero(_unit_interval(words) < fraction)
    amp_words = _splitmix64(words[hot_v, hot_u])
    relative = -np.log1p(-_unit_interval(amp_words))
    return hot_v.astype(np.intp), hot_u.astype(np.intp), relative


def add_hot_pixels(
    electrons: NDArrayFloatType,
    *,
    fraction: float,
    amplitude_e: float,
    column_factor: float,
    detector_id: str,
    rng: np.random.Generator,
    candidate_pool: tuple[NDArrayIntType, NDArrayIntType] | None = None,
) -> dict[str, Any]:
    """Add a hot-pixel population (electrons) in place.

    A hot pixel holds a large fixed charge; on CCDs read through it, a fraction
    of that charge contaminates the column above it (a warm streak).  The
    uniform population is the detector's fixed map
    (:func:`detector_hot_pixel_map`), the same pixels and amplitudes in every
    scene that emulates the detector.  An adversarial population drawn onto
    the navigation features comes from ``candidate_pool`` instead:
    ``round(fraction * size)`` distinct pool pixels, or the whole pool when it
    holds fewer, drawn without replacement from the scene's seeded ``rng``.  A
    zero fraction or amplitude is a no-op.

    Parameters:
        electrons: The electron image, modified in place.
        fraction: Fraction of pixels that are hot.
        amplitude_e: Hot-pixel amplitude scale in electrons (exponentially
            distributed about this scale, so a few are very hot).
        column_factor: Fraction of a hot pixel's TOTAL charge bled up its
            column: the warm streak's integral is ``column_factor`` times the
            hot pixel's charge, independent of the frame height.
        detector_id: The emulated detector the uniform population belongs to.
        rng: The stage's seeded generator, read only by an adversarial draw.
        candidate_pool: Optional ``(v, u)`` coordinate pool an adversarial
            population is drawn from.  When None or empty, the population is
            the detector's map.

    Returns:
        The realized population for the truth record: the planted ``pixels``
        as ``[v, u]`` pairs and their ``amplitudes_e``, empty on a no-op.
    """
    if fraction <= 0.0 or amplitude_e <= 0.0:
        return {'pixels': [], 'amplitudes_e': []}
    size_v, size_u = electrons.shape
    if candidate_pool is not None and candidate_pool[0].size > 0:
        n_hot = round(fraction * size_v * size_u)
        if n_hot <= 0:
            return {'pixels': [], 'amplitudes_e': []}
        pool_v, pool_u = candidate_pool
        pick = rng.choice(pool_v.size, size=min(n_hot, pool_v.size), replace=False)
        hot_v = pool_v[pick]
        hot_u = pool_u[pick]
        amps = amplitude_e * rng.exponential(1.0, size=pick.size)
    else:
        hot_v, hot_u, relative = detector_hot_pixel_map(
            (size_v, size_u), fraction=fraction, detector_id=detector_id
        )
        amps = amplitude_e * relative
    np.add.at(electrons, (hot_v, hot_u), amps)
    record = {
        'pixels': [[int(v), int(u)] for v, u in zip(hot_v.tolist(), hot_u.tolist(), strict=True)],
        'amplitudes_e': [float(a) for a in amps.tolist()],
    }
    if column_factor <= 0.0:
        return record
    # Warm column: each hot pixel bleeds charge onto the pixels above it
    # (decreasing toward the read register), a CCD readout scar.  The linear
    # ramp is normalized so its INTEGRAL is column_factor * amp: the column
    # carries a fixed fraction of the hot pixel's charge however tall the
    # frame is, keeping the contamination conservative and size-invariant.
    for v, u, amp in zip(hot_v.tolist(), hot_u.tolist(), amps.tolist(), strict=True):
        if v <= 0:
            continue
        weights = np.linspace(1.0, 0.0, v, endpoint=False)
        weights /= weights.sum()
        electrons[:v, u] += column_factor * amp * weights
    return record


def add_banding(
    electrons: NDArrayFloatType,
    *,
    amplitude_e: float,
    period_px: float,
    rng: np.random.Generator,
    freq_step_factor: float = 1.0,
) -> None:
    """Add horizontal coherent + random-phase banding (electrons) in place.

    The banding is line-correlated (constant along a row, varying with row
    index): a coherent sinusoid at ``period_px`` with a per-seed random phase,
    plus a smaller per-row random-phase component.  When ``freq_step_factor``
    differs from 1, the sinusoid's period changes across the image mid-line, the
    readout-pause frequency step some cameras show.  A zero amplitude or
    non-positive period is a no-op.

    Parameters:
        electrons: The electron image, modified in place.
        amplitude_e: Coherent-banding amplitude in electrons.
        period_px: Sinusoid spatial period along the row axis, in pixels.
        rng: The stage's seeded generator.
        freq_step_factor: Period multiplier below the image mid-line (1 = none).
    """
    if amplitude_e <= 0.0 or period_px <= 0.0:
        return
    size_v = electrons.shape[0]
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    freq = 2.0 * np.pi / period_px
    # A mid-image readout pause steps the spatial frequency; keep the sinusoid
    # phase-continuous across the step so the band does not jump.
    mid = size_v // 2
    freqs = np.full(size_v, freq, dtype=np.float64)
    if freq_step_factor != 1.0 and freq_step_factor > 0.0:
        freqs[mid:] = freq * freq_step_factor
    argument = np.zeros(size_v, dtype=np.float64)
    argument[1:] = np.cumsum(freqs[:-1])
    coherent = amplitude_e * np.sin(argument + phase)
    random_phase = 0.3 * amplitude_e * rng.standard_normal(size_v)
    band = coherent + random_phase
    electrons += band[:, None]


def add_bias_structure(
    dn: NDArrayFloatType,
    *,
    pedestal_sigma_dn: float,
    row_gradient_dn: float,
    col_gradient_dn: float,
    rng: np.random.Generator,
) -> None:
    """Add a bias pedestal offset and low-order bias gradients (DN) in place.

    A per-image pedestal offset (a single seeded draw) rides on the flat bias
    level, and shallow row and column gradients model the read-out bias
    structure.  A no-op when every amplitude is zero.

    Parameters:
        dn: The DN image, modified in place.
        pedestal_sigma_dn: Standard deviation of the per-image pedestal (DN).
        row_gradient_dn: Peak-to-peak row (v-axis) bias gradient (DN).
        col_gradient_dn: Peak-to-peak column (u-axis) bias gradient (DN).
        rng: The stage's seeded generator.
    """
    if pedestal_sigma_dn <= 0.0 and row_gradient_dn <= 0.0 and col_gradient_dn <= 0.0:
        return
    size_v, size_u = dn.shape
    if pedestal_sigma_dn > 0.0:
        dn += float(rng.normal(0.0, pedestal_sigma_dn))
    if row_gradient_dn > 0.0:
        row_ramp = np.linspace(-0.5 * row_gradient_dn, 0.5 * row_gradient_dn, size_v)
        dn += row_ramp[:, None]
    if col_gradient_dn > 0.0:
        col_ramp = np.linspace(-0.5 * col_gradient_dn, 0.5 * col_gradient_dn, size_u)
        dn += col_ramp[None, :]


def add_cosmic_rays(
    electrons: NDArrayFloatType,
    *,
    events_per_pixel_per_sec: float,
    exposure_sec: float,
    amplitude_e: float,
    rng: np.random.Generator,
) -> None:
    """Deposit morphological cosmic-ray events in place.

    Events are point hits (the common case, a degenerate single pixel), short
    streaks at a random angle whose length is drawn from an incidence-angle
    distribution, and rare multi-pixel splatters.  Each deposits at least
    ``amplitude_e`` (the full well) so the digitized frame clips at the ADC
    ceiling and the orchestrator's cosmic-ray/saturation masks catch it.  The
    event count is a Poisson draw whose mean is ``events_per_pixel_per_sec *
    exposure_sec`` times the frame's pixel count.  A zero rate is a no-op.

    Parameters:
        electrons: The image, modified in place: electrons on the CCD chain, DN
            on the vidicon chain.
        events_per_pixel_per_sec: Expected cosmic-ray events per pixel per second
            of exposure.
        exposure_sec: Exposure time in seconds.
        amplitude_e: Deposit scale per pixel of an event, in the image's units
            (the CCD's full well in electrons, the vidicon's DN full scale).
        rng: The stage's seeded generator.
    """
    if events_per_pixel_per_sec <= 0.0 or amplitude_e <= 0.0:
        return
    size_v, size_u = electrons.shape
    expected = events_per_pixel_per_sec * exposure_sec * size_v * size_u
    if expected <= 0.0:
        return
    n_events = int(rng.poisson(expected))
    deposit_morphological_events(
        electrons, n_events=n_events, amplitude_e=amplitude_e, rng=rng, amplitude_dist='lognormal'
    )


def deposit_morphological_events(
    electrons: NDArrayFloatType,
    *,
    n_events: int,
    amplitude_e: float,
    rng: np.random.Generator,
    amplitude_dist: str = 'lognormal',
) -> None:
    """Deposit a fixed count of morphological charge events (electrons) in place.

    The event-type mix is the same for cosmic rays and for the Galileo radiation
    regime -- mostly single-pixel point hits, some grazing streaks, a few
    multi-pixel splatters -- but the amplitude distribution differs: cosmic-ray
    events are lognormal about the deposit scale, while the radiation regime's
    amplitudes fall steeply from a few DN (an exponential draw).  A zero count or
    amplitude is a no-op.

    Parameters:
        electrons: The electron image, modified in place.
        n_events: The number of events to deposit.
        amplitude_e: The charge-deposit scale in electrons.
        rng: The stage's seeded generator.
        amplitude_dist: 'lognormal' (cosmic rays) or 'exponential' (radiation,
            steeply-falling amplitudes).
    """
    if n_events <= 0 or amplitude_e <= 0.0:
        return
    size_v, size_u = electrons.shape
    kinds = rng.choice(3, size=n_events, p=[0.80, 0.15, 0.05])
    for kind in kinds.tolist():
        v0 = int(rng.integers(0, size_v))
        u0 = int(rng.integers(0, size_u))
        if amplitude_dist == 'exponential':
            charge = amplitude_e * float(rng.exponential(0.25))
        else:
            charge = amplitude_e * (1.0 + float(rng.lognormal(0.0, 1.0)))
        if kind == 0:
            electrons[v0, u0] += charge
        elif kind == 1:
            _deposit_streak(electrons, v0, u0, charge=charge, rng=rng)
        else:
            _deposit_splatter(electrons, v0, u0, charge=charge, rng=rng)


def _deposit_streak(
    electrons: NDArrayFloatType,
    v0: int,
    u0: int,
    *,
    charge: float,
    rng: np.random.Generator,
) -> None:
    """Deposit a grazing cosmic-ray streak from a random angle and length."""
    size_v, size_u = electrons.shape
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    # A grazing incidence gives a long track; length follows an exponential so
    # most streaks are short and a few are long.
    length = 1 + int(rng.exponential(4.0))
    dv = np.sin(angle)
    du = np.cos(angle)
    for step in range(length):
        v = round(v0 + dv * step)
        u = round(u0 + du * step)
        if 0 <= v < size_v and 0 <= u < size_u:
            electrons[v, u] += charge


def _deposit_splatter(
    electrons: NDArrayFloatType,
    v0: int,
    u0: int,
    *,
    charge: float,
    rng: np.random.Generator,
) -> None:
    """Deposit a rare compact multi-pixel cosmic-ray splatter."""
    size_v, size_u = electrons.shape
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            v = v0 + dv
            u = u0 + du
            if 0 <= v < size_v and 0 <= u < size_u:
                # Center pixel takes the full charge; neighbors a fraction.
                weight = 1.0 if (dv == 0 and du == 0) else float(rng.uniform(0.2, 0.6))
                electrons[v, u] += charge * weight
