"""Tests for the pair photometry in ``_star_unique_two_star``."""

from __future__ import annotations

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import DrawGaussianStarFactory, NavFeatureFactory

from spindoctor.nav_technique._star_unique_two_star import peak_per_snr, source_peak


def test_source_peak_is_the_brightest_pixel_above_the_local_background(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A Gaussian of peak 100 DN on a 50 DN floor stands 100 DN above it."""
    image = np.full((60, 60), 50.0)
    draw_gaussian_star(image, (30.0, 30.0), peak_dn=100.0)
    assert source_peak(image, (30.4, 29.6), box_half_px=3) == pytest.approx(100.0)


def test_source_peak_is_zero_when_the_box_leaves_the_image() -> None:
    """A box and ring that do not fit inside the image hold no peak."""
    image = np.full((60, 60), 50.0)
    assert source_peak(image, (3.0, 30.0), box_half_px=3) == 0.0


def test_peak_per_snr_is_the_geometric_mean_of_the_stars_quotients(
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Two stars at the offset, 100 DN and 400 DN at SNR 10, give 20 DN per unit SNR."""
    image = np.zeros((200, 200))
    draw_gaussian_star(image, (52.0, 47.0), peak_dn=100.0)
    draw_gaussian_star(image, (152.0, 147.0), peak_dn=400.0)
    stars = [
        make_star_feature('star:UCAC4:0', predicted_vu=(50.0, 50.0), predicted_snr=10.0),
        make_star_feature('star:UCAC4:1', predicted_vu=(150.0, 150.0), predicted_snr=10.0),
    ]
    scale = peak_per_snr(
        image, stars, (2.0, -3.0), detect_at=lambda star, target: target, box_half_px=3
    )
    assert scale == pytest.approx(20.0)


def test_peak_per_snr_is_none_when_a_star_is_not_found(
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A star absent where the offset puts it leaves no scale."""
    image = np.zeros((200, 200))
    draw_gaussian_star(image, (52.0, 47.0), peak_dn=100.0)
    stars = [
        make_star_feature('star:UCAC4:0', predicted_vu=(50.0, 50.0), predicted_snr=10.0),
        make_star_feature('star:UCAC4:1', predicted_vu=(150.0, 150.0), predicted_snr=10.0),
    ]
    scale = peak_per_snr(
        image,
        stars,
        (2.0, -3.0),
        detect_at=lambda star, target: target if target[0] < 100.0 else None,
        box_half_px=3,
    )
    assert scale is None
