"""The dark signal: charge the detector collects without light.

The dark current builds up over the exposure and is the same on every line.  The
readout dark builds up while each line waits its turn to be read, whatever the
exposure, so it grows down the frame.  On a CCD both are added before the shot noise,
which they then carry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spindoctor.support.types import NDArrayFloatType, NDArrayIntType

__all__ = ['add_dark_current', 'add_readout_dark', 'readout_dark_profile']


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


def readout_dark_profile(
    size_v: int, *, amplitude: float, nonlinear: float, binning: int
) -> NDArrayFloatType:
    """The dark charge each line collects while it waits its turn to be read.

    The first line is read at once and the last one waits longest, so the charge grows
    with line number whatever the exposure.

    Parameters:
        size_v: The frame's line count.
        amplitude: The charge a native pixel of the last line collects, in the
            image's units.
        nonlinear: The exponent of the growth with line number; 1 is linear.
        binning: The readout mode's binning factor ``b``; a frame pixel sums the
            charge of ``b**2`` native pixels.

    Returns:
        The per-line charge of a frame pixel, zeros when ``amplitude`` is 0.
    """
    if amplitude <= 0.0:
        return np.zeros(size_v, dtype=np.float64)
    exponent = nonlinear if nonlinear > 0.0 else 1.0
    result: NDArrayFloatType = amplitude * binning**2 * np.linspace(0.0, 1.0, size_v) ** exponent
    return result


def add_readout_dark(
    plane: NDArrayFloatType,
    profile: NDArrayFloatType,
    *,
    rbi_column_factor: float,
    rbi_columns: NDArrayIntType,
) -> dict[str, Any]:
    """Add the readout dark in place.

    The residual bulk image of a Cassini camera makes some columns collect more dark
    charge than the rest, so each of the ``rbi_columns`` gets ``rbi_column_factor`` of
    the profile on top of it.  A profile of zeros is a no-op.

    Parameters:
        plane: The image, modified in place: electrons on the CCD chain, DN on the
            vidicon chain.
        profile: The per-line charge (:func:`readout_dark_profile`).
        rbi_column_factor: The extra fraction of the profile on the enhanced columns.
        rbi_columns: The enhanced columns.

    Returns:
        The realized readout dark for the truth record.
    """
    peak = float(profile.max())
    if peak <= 0.0:
        return {'active': False}
    plane += profile[:, None]
    enhanced: list[int] = []
    if rbi_column_factor > 0.0 and rbi_columns.size > 0:
        cols = np.unique(rbi_columns)
        plane[:, cols] += (rbi_column_factor * profile)[:, None]
        enhanced = [int(c) for c in cols]
    return {'active': True, 'peak': peak, 'enhanced_columns': enhanced}
