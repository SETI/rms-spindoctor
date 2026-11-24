"""Unit tests for the finite-column haze reflectance and its helpers."""

import math

import numpy as np
import pytest

from spindoctor.sim.render.bodies.haze_scattering import (
    chandrasekhar_h,
    effective_cosine,
    grazing_constant,
    haze_reflectance,
    mean_cosine,
    twilight_weight,
)


def test_h_function_of_a_conservative_haze_is_one_plus_two_x() -> None:
    """At albedo 1 Hapke's H-function is 1 + 2x."""
    x = np.array([0.0, 0.3, 1.0])
    np.testing.assert_allclose(chandrasekhar_h(x, 1.0), 1.0 + 2.0 * x, rtol=1e-12)


def test_h_function_of_an_absorbing_haze_is_one() -> None:
    """At albedo 0 nothing scatters twice, so the H-function is 1."""
    np.testing.assert_allclose(chandrasekhar_h(np.array([0.0, 0.5, 1.0]), 0.0), 1.0)


def test_a_thin_column_reflects_the_single_scattering_result() -> None:
    """A thin column of optical depth tau reflects albedo * P * tau / 4."""
    tau = 1e-5
    reflectance = haze_reflectance(
        np.array(tau), np.array(0.4), np.array(0.7), albedo=0.9, g=0.6, phase_function=2.5
    )
    assert float(reflectance) == pytest.approx(0.9 * 2.5 * tau / 4.0, rel=1e-4)


def test_a_thick_column_reflects_the_semi_infinite_value() -> None:
    """A thick column reflects Hapke's semi-infinite value at the similarity albedo."""
    albedo = 0.95
    g = 0.6
    mu = 0.4
    mu0 = 0.7
    phase_function = 2.5
    reflectance = haze_reflectance(
        np.array(50.0),
        np.array(mu),
        np.array(mu0),
        albedo=albedo,
        g=g,
        phase_function=phase_function,
    )
    similar = albedo * (1.0 - g) / (1.0 - albedo * g)
    gamma = math.sqrt(1.0 - similar)
    h_mu = (1.0 + 2.0 * mu) / (1.0 + 2.0 * gamma * mu)
    h_mu0 = (1.0 + 2.0 * mu0) / (1.0 + 2.0 * gamma * mu0)
    expected = albedo / 4.0 * mu0 / (mu0 + mu) * (phase_function + h_mu * h_mu0 - 1.0)
    assert float(reflectance) == pytest.approx(expected, rel=1e-9)


def test_the_effective_cosine_overhead_is_one() -> None:
    """Straight overhead the effective cosine is the cosine itself, 1."""
    assert float(effective_cosine(np.array(1.0), 0.05)) == pytest.approx(1.0, rel=1e-12)


def test_the_grazing_column_is_the_chapman_enhancement() -> None:
    """Toward the horizon the column is sqrt(pi R / (2 H)) times the vertical one."""
    scale_height = 4.0
    radius = 90.0
    c = grazing_constant(scale_height, radius)
    enhancement = 1.0 / float(effective_cosine(np.array(0.0), c))
    assert enhancement == pytest.approx(math.sqrt(math.pi * radius / (2.0 * scale_height)))


def test_the_twilight_weight_is_one_in_sunlight() -> None:
    """Wherever the sun is above the horizon the whole haze column is lit."""
    np.testing.assert_array_equal(twilight_weight(np.array([0.0, 0.2, 1.0]), 0.1), 1.0)


def test_the_twilight_weight_falls_by_e_at_one_wrap_width() -> None:
    """At a solar depression of one wrap width the weight is exp(-1)."""
    delta = 0.1
    weight = twilight_weight(np.array(-math.sin(delta)), delta)
    assert float(weight) == pytest.approx(math.exp(-1.0), rel=1e-12)


def test_the_mean_cosine_weights_the_two_terms_by_their_shares() -> None:
    """A two-term phase function's mean cosine is its terms' share-weighted mean."""
    assert mean_cosine(0.9, -0.2, 0.1) == pytest.approx(0.79, rel=1e-12)
