"""The closed-form means of the ring photometry where optical depth changes linearly."""

import math

import numpy as np
import pytest

from spindoctor.sim.render.rings.photometry import (
    ring_reflection_factor,
    ring_reflection_mean,
    ring_transmission_mean,
)
from spindoctor.support.types import NDArrayFloatType

_SAMPLES = 200_000


def _array(value: float) -> NDArrayFloatType:
    return np.array([value], dtype=np.float64)


def _dense_reflection_mean(tau_lo: float, tau_hi: float, mu: float, mu0: float, lit: bool) -> float:
    """The mean of the point form over a linear tau, by dense midpoint sampling."""
    tau = tau_lo + (np.arange(_SAMPLES) + 0.5) / _SAMPLES * (tau_hi - tau_lo)
    return float(np.mean(ring_reflection_factor(tau, mu, mu0, lit=lit)))


@pytest.mark.parametrize(
    ('mu', 'mu0', 'lit'),
    [(0.5, 0.3, True), (0.5, 0.3, False), (0.4, 0.4, False)],
    ids=['lit', 'unlit', 'unlit_matched_mu'],
)
def test_reflection_mean_matches_dense_sampling(mu: float, mu0: float, lit: bool) -> None:
    """The closed-form mean over tau 0.2 to 1.7 matches the sampled mean."""
    mean = float(ring_reflection_mean(_array(0.2), _array(1.7), mu, mu0, lit=lit)[0])
    assert mean == pytest.approx(_dense_reflection_mean(0.2, 1.7, mu, mu0, lit), rel=1e-9, abs=0.0)


@pytest.mark.parametrize(
    ('mu', 'mu0', 'lit'),
    [(0.5, 0.3, True), (0.5, 0.3, False), (0.4, 0.4, False)],
    ids=['lit', 'unlit', 'unlit_matched_mu'],
)
def test_reflection_mean_of_a_point_is_the_point_value(mu: float, mu0: float, lit: bool) -> None:
    """With no change in tau, the mean is the point form."""
    mean = float(ring_reflection_mean(_array(0.9), _array(0.9), mu, mu0, lit=lit)[0])
    point = float(ring_reflection_factor(_array(0.9), mu, mu0, lit=lit)[0])
    assert mean == pytest.approx(point, rel=1e-14, abs=0.0)


@pytest.mark.parametrize('spread', [0.003, 0.0041], ids=['series', 'direct'])
def test_matched_mu_mean_near_its_series_limit_matches_sampling(spread: float) -> None:
    """Spreads on either side of the series limit (0.004 at mu 0.4) match sampling."""
    mu = 0.4
    mean = float(ring_reflection_mean(_array(0.0), _array(spread), mu, mu, lit=False)[0])
    assert mean == pytest.approx(
        _dense_reflection_mean(0.0, spread, mu, mu, False), rel=1e-9, abs=0.0
    )


def test_transmission_mean_matches_the_integral() -> None:
    """The mean of exp(-tau/mu) over tau 1 to 3 is mu (e^-1/mu - e^-3/mu) / 2."""
    mu = 0.5
    mean = float(ring_transmission_mean(_array(1.0), _array(3.0), mu)[0])
    assert mean == pytest.approx(mu * (math.exp(-2.0) - math.exp(-6.0)) / 2.0, rel=1e-14, abs=0.0)


def test_steep_reflection_mean_stays_finite_and_exact() -> None:
    """A spread of 50 in tau at mu 0.001 (rates near 1e5) gives the exact mean."""
    mu, mu0 = 0.001, 0.003
    mean = float(ring_reflection_mean(_array(0.0), _array(50.0), mu, mu0, lit=False)[0])
    exact = mu0 / (mu0 - mu) * (mu0 / 50.0 - mu / 50.0)
    assert mean == pytest.approx(exact, rel=1e-12, abs=0.0)
