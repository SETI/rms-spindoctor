"""Unit tests for the detection-probability model per readout mode and gain state."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import scipy.special
from util.star_photometry.star_completeness import fit_gain_limits

#: The planted DN at which each gain state's detection rises (half the stars detected).
RISE_DN = {'LOW': 100.0, 'HIGH': 30.0}

#: The DN of each gain state per DN of ``LOW``, the reference state.
GAIN_RATIO = {'LOW': 1.0, 'HIGH': 3.0, 'MID': 2.0}


def _rows(gain: str, seed: int) -> list[dict[str, Any]]:
    """Return 400 full-frame star rows of one gain state whose detection rises at its DN.

    Parameters:
        gain: The gain state, a key of ``RISE_DN``.
        seed: The random seed.

    Returns:
        The rows, five stars per frame.
    """
    rng = np.random.default_rng(seed)
    log_dn = rng.uniform(0.0, 4.0, 400)
    rise = math.log10(RISE_DN[gain])
    detected = rng.uniform(size=400) < 0.5 * (1.0 + scipy.special.erf((log_dn - rise) / 0.2))
    return [
        {
            'pred_dn': 10**x,
            'detected': bool(d),
            'mode': 'FULL',
            'gain': gain,
            'image_id': f'{gain}{k // 5}',
        }
        for k, (x, d) in enumerate(zip(log_dn, detected, strict=True))
    ]


def _fit() -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, str]], dict[str, str]]:
    """Fit the planted rows of ``LOW`` and ``HIGH``; ``MID`` has no stars.

    Returns:
        What :func:`fit_gain_limits` returns.
    """
    return fit_gain_limits(_rows('LOW', 1) + _rows('HIGH', 2), GAIN_RATIO)


def test_each_gain_state_with_stars_gets_its_own_limit() -> None:
    """A gain state whose detections rise at a third of the DN gets a third of the limit."""
    limits, _, _ = _fit()
    ratio = limits['FULL']['LOW'][0] / limits['FULL']['HIGH'][0]
    assert ratio == pytest.approx(100.0 / 30.0, rel=0.25)


def test_a_gain_state_without_stars_is_scaled_from_the_nearest_fitted_one() -> None:
    """MID, at twice LOW's DN per electron, is nearer HIGH and takes two thirds of its DN."""
    limits, scaled_from, _ = _fit()
    assert scaled_from['FULL'] == {'MID': 'HIGH'}
    assert limits['FULL']['MID'][0] == pytest.approx(limits['FULL']['HIGH'][0] * 2.0 / 3.0)


def test_a_gain_state_whose_stars_cannot_place_the_rise_is_reported() -> None:
    """Forty-five stars detected at one DN leave the pair unfitted, with the reason."""
    rows = (
        _rows('LOW', 1)
        + [{'pred_dn': 100.0, 'detected': True, 'mode': 'FULL', 'gain': 'HIGH', 'image_id': 'H'}]
        * 45
    )
    _, scaled_from, reasons = fit_gain_limits(rows, GAIN_RATIO)
    assert reasons == {'FULL/HIGH': '45 detected and 0 undetected stars'}
    assert scaled_from['FULL']['HIGH'] == 'LOW'
