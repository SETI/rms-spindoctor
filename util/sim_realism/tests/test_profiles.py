"""Unit tests for the FOM 2-4 profile machinery (stars, limbs, ring edges)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage
from util.sim_realism.profiles import (
    edge_normal_profiles,
    ee_radius,
    encircled_energy,
    profile_rise_width,
    radial_profile,
)


def _gaussian_star(sigma: float, *, size: int = 41, amp: float = 100.0) -> np.ndarray:
    vv, uu = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    return np.asarray(amp * np.exp(-((vv - c) ** 2 + (uu - c) ** 2) / (2.0 * sigma**2)))


def test_radial_profile_monotone_for_gaussian() -> None:
    """A Gaussian star's azimuthal profile decreases with radius."""
    image = _gaussian_star(1.2)
    _radius, intensity = radial_profile(image, (20.0, 20.0))
    finite = intensity[np.isfinite(intensity)]
    assert finite.size >= 10
    assert np.all(np.diff(finite[:8]) <= 0.0)


def test_radial_profile_off_image_is_nan() -> None:
    """A cutout hanging off the frame yields all-NaN intensity."""
    image = _gaussian_star(1.0)
    _radius, intensity = radial_profile(image, (1.0, 1.0))
    assert np.all(np.isnan(intensity))


def test_encircled_energy_reaches_one() -> None:
    """The EE curve is normalized to 1 at the outer radius."""
    image = _gaussian_star(1.0)
    radius, intensity = radial_profile(image, (20.0, 20.0))
    _r, ee = encircled_energy(radius, intensity)
    assert ee[-1] == pytest.approx(1.0)
    assert np.all(np.diff(ee) >= 0.0)


def test_ee50_tracks_psf_width() -> None:
    """EE50 grows with the Gaussian sigma (wider PSF, wider half-energy)."""
    widths = []
    for sigma in (0.8, 1.4, 2.0):
        image = _gaussian_star(sigma)
        radius, intensity = radial_profile(image, (20.0, 20.0))
        r, ee = encircled_energy(radius, intensity)
        widths.append(ee_radius(r, ee, 0.5))
    assert widths[0] < widths[1]
    assert widths[1] < widths[2]


def test_ee_radius_unreached_fraction_is_nan() -> None:
    """A fraction the curve never reaches yields NaN."""
    r = np.linspace(0.0, 4.0, 9)
    ee = np.linspace(0.0, 0.4, 9)
    assert np.isnan(ee_radius(r, ee, 0.5))


def _step_edge(sigma: float, *, size: int = 41, edge_u: float = 20.0) -> np.ndarray:
    uu = np.mgrid[0:size, 0:size][1].astype(np.float64)
    step = np.where(uu < edge_u, 1.0, 0.0)
    return np.asarray(scipy.ndimage.gaussian_filter(step, sigma), dtype=np.float64)


def test_edge_normal_profiles_shapes_and_bounds() -> None:
    """Profiles are sampled per vertex; off-image tracks are dropped."""
    image = _step_edge(1.0)
    vertices = np.array([[20.0, 19.5], [20.0, 2.0]])  # second track exits the left edge
    normals = np.array([[0.0, 1.0], [0.0, 1.0]])
    profiles = edge_normal_profiles(image, vertices, normals, half_length_px=8.0, n_samples=33)
    assert profiles.shape == (1, 33)


def test_edge_profile_descends_inside_to_outside() -> None:
    """Sampled along the outward normal, the profile falls from lit to sky."""
    image = _step_edge(1.0)
    vertices = np.array([[20.0, 19.5]])
    normals = np.array([[0.0, 1.0]])
    profiles = edge_normal_profiles(image, vertices, normals, half_length_px=6.0, n_samples=25)
    assert profiles[0, 0] > 0.9
    assert profiles[0, -1] < 0.1


def test_rise_width_scales_with_blur() -> None:
    """The 10-90% width tracks the known analytic width of a blurred step.

    For a Gaussian-blurred step the 10-90% distance is 2.563 * sigma.
    """
    for sigma in (0.8, 1.5):
        image = _step_edge(sigma)
        vertices = np.array([[v, 19.5] for v in np.arange(10.0, 30.0)])
        normals = np.tile(np.array([[0.0, 1.0]]), (20, 1))
        profiles = edge_normal_profiles(image, vertices, normals, half_length_px=8.0, n_samples=65)
        spacing = 16.0 / 64.0
        widths = [profile_rise_width(p, spacing_px=spacing) for p in profiles]
        mean_width = float(np.nanmean(widths))
        assert mean_width == pytest.approx(2.563 * sigma, rel=0.15)


def test_rise_width_flat_profile_is_nan() -> None:
    """A profile with no edge yields NaN."""
    assert np.isnan(profile_rise_width(np.ones(33), spacing_px=0.5))


def test_rise_width_reversed_polarity() -> None:
    """A dark-inside / bright-outside edge is measured the same way."""
    image = 1.0 - _step_edge(1.0)
    vertices = np.array([[v, 19.5] for v in np.arange(10.0, 30.0)])
    normals = np.tile(np.array([[0.0, 1.0]]), (20, 1))
    profiles = edge_normal_profiles(image, vertices, normals, half_length_px=8.0, n_samples=65)
    widths = [profile_rise_width(p, spacing_px=0.25) for p in profiles]
    assert float(np.nanmean(widths)) == pytest.approx(2.563, rel=0.15)
