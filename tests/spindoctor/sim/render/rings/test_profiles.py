"""The ring features' sharp radial profiles and the pixel-footprint segments."""

from typing import Any

import numpy as np
import pytest

from spindoctor.sim.render.rings.profiles import (
    FeatureBoundaries,
    feature_boundaries,
    feature_sharp_radii,
    feature_tau,
    footprint_samples,
)
from spindoctor.sim.shared.ring import ring_orbit_from_mapping


def _feature(kind: str, **keys: Any) -> dict[str, Any]:
    """A feature of ``kind`` with tau 2 at orbit radius 10 px, plus ``keys``."""
    feature: dict[str, Any] = {'kind': kind, 'tau': 2.0, 'orbit': {'a_px': 10.0}}
    feature.update(keys)
    return feature


def _bounds(feature: dict[str, Any], count: int) -> FeatureBoundaries:
    """The feature's radii at ``count`` longitudes of zero."""
    orbit = ring_orbit_from_mapping(feature['orbit'])
    return feature_boundaries(feature, orbit, np.zeros(count), os=1, epoch=0.0, time=0.0)


def test_one_crossing_cuts_the_footprint_into_two_segments() -> None:
    """A boundary at +0.2 px splits the footprint into 0.7 and 0.3 of its length."""
    samples = footprint_samples(np.array([7]), np.array([0.2]))
    np.testing.assert_allclose(samples.weight, [0.7, 0.3], atol=1e-15)
    np.testing.assert_allclose(samples.offset, [-0.15, 0.35], atol=1e-15)


def test_two_crossings_in_one_pixel_cut_three_segments() -> None:
    """Crossings listed out of order still cut the footprint in radial order."""
    samples = footprint_samples(np.array([3, 3]), np.array([0.25, -0.25]))
    np.testing.assert_allclose(samples.weight, [0.25, 0.5, 0.25], atol=1e-15)


def test_segment_weights_sum_to_one_for_each_pixel() -> None:
    """Every listed pixel's segments cover its whole footprint."""
    samples = footprint_samples(np.array([9, 2, 9, 5]), np.array([0.1, -0.3, -0.4, 0.0]))
    totals = np.bincount(samples.slot, weights=samples.weight)
    np.testing.assert_allclose(totals, np.ones(3), atol=1e-15)


def test_listed_pixels_are_sorted_and_unique() -> None:
    """The pixel list holds each pixel once, in increasing order."""
    samples = footprint_samples(np.array([9, 2, 9, 5]), np.array([0.1, -0.3, -0.4, 0.0]))
    np.testing.assert_array_equal(samples.pixels, [2, 5, 9])


def test_ringlet_profile_is_sharp_at_both_ends() -> None:
    """A ringlet carries its full tau up to each edge and nothing past it."""
    feature = _feature('ringlet', width_px=5.0)
    radii = np.array([9.999, 10.0, 15.0, 15.001])
    tau = feature_tau(feature, radii, _bounds(feature, radii.size), os=1)
    np.testing.assert_array_equal(tau, [0.0, 2.0, 2.0, 0.0])


def test_ramp_profile_is_sharp_only_at_its_tau_end() -> None:
    """An outward ramp rises linearly and drops straight to zero past its outer end."""
    feature = _feature('ramp', width_px=4.0, side='out')
    radii = np.array([11.0, 14.0, 14.001])
    tau = feature_tau(feature, radii, _bounds(feature, radii.size), os=1)
    np.testing.assert_allclose(tau, [0.5, 2.0, 0.0], atol=1e-15)


def test_outward_ramp_steps_only_at_its_outer_end() -> None:
    """Only the ramp end that carries tau is a sharp boundary."""
    feature = _feature('ramp', width_px=4.0, side='out')
    sharp = feature_sharp_radii(feature, _bounds(feature, 1))
    np.testing.assert_allclose(np.concatenate(sharp), [14.0])


def test_wave_has_no_sharp_boundary() -> None:
    """A density wave is continuous at its launch radius."""
    feature = _feature('wave', wavelength_px=4.0, density_wave_damping_px=8.0)
    assert feature_sharp_radii(feature, _bounds(feature, 1)) == []


def test_unknown_kind_is_not_renderable() -> None:
    """A kind outside the profile set raises with the kind in the message."""
    feature = _feature('spiral')
    with pytest.raises(ValueError, match="kind 'spiral' is not renderable"):
        feature_tau(feature, np.array([10.0]), _bounds(feature, 1), os=1)
