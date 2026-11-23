"""Transients: the hits of cosmic rays and of a radiation environment.

The two are different particle populations that leave the same shapes (points,
streaks, and splatters).  Cosmic rays arrive at a rate per pixel per second and
deposit at least a set charge, with a lognormal tail above it.  A radiation
environment's hits arrive at a rate per second over the whole frame, and their charge
falls off steeply from a few DN.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage

from spindoctor.support.types import NDArrayFloatType

__all__ = ['add_cosmic_rays', 'deposit_morphological_events']


def add_cosmic_rays(
    electrons: NDArrayFloatType,
    *,
    events_per_pixel_per_sec: float,
    exposure_sec: float,
    amplitude_e: float,
    rng: np.random.Generator,
    readout_dwell_sec: float = 0.0,
    spread_px: float = 0.0,
) -> None:
    """Deposit morphological cosmic-ray events in place.

    Events are point hits (the common case, a degenerate single pixel), short
    streaks at a random angle whose length is drawn from an incidence-angle
    distribution, and rare multi-pixel splatters.  Each event deposits
    ``amplitude_e * (1 + L)`` with ``L`` a standard lognormal draw, so
    ``amplitude_e`` is the smallest deposit: the full well makes every hit clip
    at the ADC ceiling, a smaller scale gives the modest hits a CCD collects.
    The detector collects hits during the exposure and while it waits to be
    read, so the event count is a Poisson draw whose mean is
    ``events_per_pixel_per_sec * (exposure_sec + readout_dwell_sec)`` times
    the frame's pixel count.  A positive ``spread_px`` diffuses each event's
    charge with a Gaussian of that sigma, as charge spreads between CCD pixels.
    A zero rate is a no-op.

    Parameters:
        electrons: The image, modified in place: electrons on the CCD chain, DN
            on the vidicon chain.
        events_per_pixel_per_sec: Expected cosmic-ray events per pixel per second
            the detector collects.
        exposure_sec: Exposure time in seconds.
        amplitude_e: Smallest deposit per event, in the image's units.
        rng: The stage's seeded generator.
        readout_dwell_sec: Effective seconds the detector collects hits outside
            the exposure, while the frame waits to be read.
        spread_px: Charge-diffusion sigma in frame pixels; 0 keeps each deposit
            where it lands.
    """
    if events_per_pixel_per_sec <= 0.0 or amplitude_e <= 0.0:
        return
    size_v, size_u = electrons.shape
    collect_sec = max(exposure_sec, 0.0) + max(readout_dwell_sec, 0.0)
    expected = events_per_pixel_per_sec * collect_sec * size_v * size_u
    if expected <= 0.0:
        return
    n_events = int(rng.poisson(expected))
    if spread_px <= 0.0:
        deposit_morphological_events(
            electrons,
            n_events=n_events,
            amplitude_e=amplitude_e,
            rng=rng,
            amplitude_dist='lognormal',
        )
        return
    hits = np.zeros_like(electrons)
    deposit_morphological_events(
        hits, n_events=n_events, amplitude_e=amplitude_e, rng=rng, amplitude_dist='lognormal'
    )
    electrons += scipy.ndimage.gaussian_filter(hits, sigma=spread_px, mode='constant')


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
