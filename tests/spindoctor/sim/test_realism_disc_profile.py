"""Tests of the disc-profile measurement the realism match's law study makes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spindoctor.sim.realism.disc_profile import (
    MIN_FIT_PIXELS,
    equator_profile,
    fit_lunar_lambert,
    fit_minnaert,
    photometric_coordinates,
    profile_residual,
    select_disc_pixels,
)
from spindoctor.sim.render.bodies.photometry import incidence_cosines, shade_surface

_PHASE_DEG = 40.0


def _sphere(
    size: int = 161, radius: float = 70.0, phase_deg: float = _PHASE_DEG
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A sphere lit from the right: which pixels it covers and the two cosines there."""
    rows, cols = np.indices((size, size), dtype=np.float64)
    center = size / 2.0
    normal_v = (rows + 0.5 - center) / radius
    normal_u = (cols + 0.5 - center) / radius
    squared = 1.0 - normal_v**2 - normal_u**2
    on_body = squared > 0.0
    normal_z = np.sqrt(np.clip(squared, 0.0, None))
    mu0 = incidence_cosines(
        normal_v,
        normal_u,
        normal_z,
        illumination_angle=math.pi / 2,
        phase_angle=math.radians(phase_deg),
    )
    return on_body, np.where(on_body, mu0, -1.0), np.where(on_body, normal_z, -1.0)


def _disc(law: str, **kwargs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The counted pixels of a sphere shaded with one law: brightness and both cosines."""
    on_body, mu0, mu = _sphere()
    keep = select_disc_pixels(on_body, mu0, mu, usable=np.ones(on_body.shape, dtype=bool))
    intensity = 0.4 * shade_surface(
        mu0[keep], mu[keep], law=law, phase_angle=math.radians(_PHASE_DEG), **kwargs
    )
    return intensity, mu0[keep], mu[keep]


def test_a_minnaert_disc_gives_its_exponent() -> None:
    """A disc shaded with Minnaert's law at k = 0.7 fits k = 0.7."""
    intensity, mu0, mu = _disc('minnaert', minnaert_k=0.7)
    fit = fit_minnaert(intensity, mu0, mu)
    assert fit is not None
    assert fit.k == pytest.approx(0.7, abs=1e-6)


def test_the_exponent_does_not_depend_on_the_brightness_scale() -> None:
    """Scaling a disc's brightness leaves its exponent and scales its B0."""
    intensity, mu0, mu = _disc('minnaert', minnaert_k=0.7)
    fit = fit_minnaert(3.0 * intensity, mu0, mu)
    assert fit is not None
    assert fit.b0 == pytest.approx(1.2, rel=1e-6)


def test_bright_markings_do_not_move_the_exponent() -> None:
    """A bright patch where the Sun is high, a quarter of those pixels, is rejected."""
    intensity, mu0, mu = _disc('minnaert', minnaert_k=0.7)
    marked = intensity.copy()
    high = np.flatnonzero(mu0 * mu > 0.6)
    marked[high[::4]] *= 3.0
    fit = fit_minnaert(marked, mu0, mu)
    assert fit is not None
    assert fit.k == pytest.approx(0.7, abs=1e-3)


def test_a_lambert_disc_fits_the_exponent_one() -> None:
    """Lambert's law is Minnaert's at k = 1."""
    intensity, mu0, mu = _disc('lambert')
    fit = fit_minnaert(intensity, mu0, mu)
    assert fit is not None
    assert fit.k == pytest.approx(1.0, abs=1e-6)


def test_a_lunar_lambert_disc_gives_its_blend() -> None:
    """A disc at 40 deg phase shaded with McEwen's law fits McEwen's blend there."""
    intensity, mu0, mu = _disc('lunar_lambert')
    fit = fit_lunar_lambert(intensity, mu0, mu)
    alpha = _PHASE_DEG
    mcewen = 1.0 - 0.019 * alpha + 2.42e-4 * alpha**2 - 1.46e-6 * alpha**3
    assert fit is not None
    assert fit.blend == pytest.approx(mcewen, abs=1e-6)


@pytest.mark.parametrize('n', [1, MIN_FIT_PIXELS - 1])
def test_too_few_pixels_give_no_fit(n: int) -> None:
    """With one pixel, or one fewer than the minimum, the Minnaert fit returns None."""
    intensity, mu0, mu = _disc('lambert')
    assert fit_minnaert(intensity[:n], mu0[:n], mu[:n]) is None


@pytest.mark.parametrize('n', [1, MIN_FIT_PIXELS - 1])
def test_too_few_pixels_give_no_blend(n: int) -> None:
    """With one pixel, or one fewer than the minimum, the lunar-Lambert fit returns None."""
    intensity, mu0, mu = _disc('lambert')
    assert fit_lunar_lambert(intensity[:n], mu0[:n], mu[:n]) is None


def test_pixels_near_the_edge_of_the_lit_region_are_left_out() -> None:
    """A lit pixel nearer the limb than the clearance does not count."""
    on_body, mu0, mu = _sphere()
    keep = select_disc_pixels(on_body, mu0, mu, usable=np.ones(on_body.shape, dtype=bool))
    row = 80
    lit_cols = np.flatnonzero(on_body[row] & (mu0[row] > 0.0))
    assert not keep[row, lit_cols.max()]


def test_pixels_deep_in_the_disc_count() -> None:
    """A lit pixel far from every edge counts."""
    on_body, mu0, mu = _sphere()
    keep = select_disc_pixels(on_body, mu0, mu, usable=np.ones(on_body.shape, dtype=bool))
    assert keep[80, 95]


def test_unusable_pixels_are_left_out() -> None:
    """A pixel the caller marks unusable does not count."""
    on_body, mu0, mu = _sphere()
    usable = np.ones(on_body.shape, dtype=bool)
    usable[80, 95] = False
    keep = select_disc_pixels(on_body, mu0, mu, usable=usable)
    assert not keep[80, 95]


def test_the_sub_solar_point_lies_at_the_phase_longitude() -> None:
    """Where the Sun is overhead the luminance longitude equals the phase angle."""
    alpha = math.radians(50.0)
    longitude, _latitude = photometric_coordinates(
        np.array([1.0]), np.array([math.cos(alpha)]), alpha
    )
    assert longitude[0] == pytest.approx(50.0)


def test_the_sub_solar_point_lies_on_the_equator() -> None:
    """The sub-solar point has latitude 0."""
    alpha = math.radians(50.0)
    _longitude, latitude = photometric_coordinates(
        np.array([1.0]), np.array([math.cos(alpha)]), alpha
    )
    assert latitude[0] == pytest.approx(0.0, abs=1e-6)


def test_the_profile_bins_by_longitude_away_from_zero_phase() -> None:
    """Above the low-phase limit the profile has the 36 longitude bins."""
    intensity, mu0, mu = _disc('lambert')
    profile = equator_profile(intensity, mu0, mu, math.radians(_PHASE_DEG))
    assert profile.size == 36


def test_the_profile_bins_by_emission_near_zero_phase() -> None:
    """Below the low-phase limit the profile has the 20 emission-cosine bins."""
    intensity, mu0, mu = _disc('lambert')
    profile = equator_profile(intensity, mu0, mu, math.radians(2.0))
    assert profile.size == 20


def test_a_profile_bin_with_too_few_pixels_is_empty() -> None:
    """A longitude bin beyond the terminator is empty, so it is NaN."""
    intensity, mu0, mu = _disc('lambert')
    profile = equator_profile(intensity, mu0, mu, math.radians(_PHASE_DEG))
    assert math.isnan(profile[0])


def test_profiles_of_one_shape_have_no_residual() -> None:
    """Two profiles that differ only in scale have a residual of 0."""
    real = np.array([1.0, 2.0, 3.0, np.nan])
    assert profile_residual(real, 5.0 * real) == pytest.approx(0.0)


def test_profiles_of_different_shapes_have_a_residual() -> None:
    """The residual is the rms of the median-normalized ratio less 1."""
    real = np.array([1.0, 1.0, 1.0, 2.0])
    twin = np.ones(4)
    assert profile_residual(real, twin) == pytest.approx(0.5)


def test_profiles_sharing_two_bins_have_no_residual() -> None:
    """Fewer than 3 shared bins give no residual."""
    real = np.array([1.0, 2.0, np.nan, np.nan])
    twin = np.array([1.0, 2.0, 3.0, 4.0])
    assert profile_residual(real, twin) is None
