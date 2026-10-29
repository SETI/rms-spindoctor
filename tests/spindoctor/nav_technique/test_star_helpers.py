"""Tests for the window detection of ``spindoctor.nav_technique._star_helpers``."""

import numpy as np
import pytest
from tests.spindoctor.star_streak_images import draw_streak

from spindoctor.nav_technique._star_helpers import local_centroid

_BACKGROUND = 100.0
_NOISE_SIGMA = 1.0


def _bright_floor() -> np.ndarray:
    """Return a 41x41 frame on a uniform floor far above the noise, with mild texture."""
    rng = np.random.default_rng(7)
    return _BACKGROUND + rng.normal(0.0, _NOISE_SIGMA, size=(41, 41))


def test_a_bright_floor_without_a_star_holds_no_detection() -> None:
    """A window whose pixels all clear the threshold only by their floor finds nothing."""
    detection, _peak = local_centroid(
        _bright_floor(),
        (20.0, 20.0),
        search_window_px=6.0,
        centroid_box_half_px=3,
        image_noise_sigma=_NOISE_SIGMA,
        detection_sigma=5.0,
    )
    assert detection is None


def test_a_star_on_a_bright_floor_is_detected_where_it_lies() -> None:
    """A star standing above the floor by more than the threshold is found."""
    image = _bright_floor()
    vv, uu = np.mgrid[0:41, 0:41]
    image += 40.0 * np.exp(-((vv - 22.0) ** 2 + (uu - 18.0) ** 2) / (2.0 * 0.8**2))
    detection, _peak = local_centroid(
        image,
        (20.0, 20.0),
        search_window_px=6.0,
        centroid_box_half_px=3,
        image_noise_sigma=_NOISE_SIGMA,
        detection_sigma=5.0,
    )
    assert detection is not None
    assert detection[0] == pytest.approx(22.0, abs=0.2)
    assert detection[1] == pytest.approx(18.0, abs=0.2)


def test_the_brightest_pixel_of_a_sloping_floor_holds_no_detection() -> None:
    """Glare that rises across the window peaks at its edge but stands above nothing."""
    columns = np.mgrid[0:41, 0:41][1]
    image = _bright_floor() + 2.0 * columns
    detection, _peak = local_centroid(
        image,
        (20.0, 20.0),
        search_window_px=6.0,
        centroid_box_half_px=3,
        image_noise_sigma=_NOISE_SIGMA,
        detection_sigma=5.0,
    )
    assert detection is None


def test_a_star_on_a_sloping_floor_is_centroided_where_it_lies() -> None:
    """Glare rising across the centroid box does not pull the centroid uphill."""
    image = _bright_floor()
    vv, uu = np.mgrid[0:41, 0:41]
    image += 4.0 * uu + 40.0 * np.exp(-((vv - 22.0) ** 2 + (uu - 18.0) ** 2) / (2.0 * 0.8**2))
    detection, _peak = local_centroid(
        image,
        (20.0, 20.0),
        search_window_px=6.0,
        centroid_box_half_px=3,
        image_noise_sigma=_NOISE_SIGMA,
        detection_sigma=5.0,
    )
    assert detection is not None
    assert detection[1] == pytest.approx(18.0, abs=0.2)


def test_a_cosmic_ray_beside_the_box_does_not_tilt_its_background() -> None:
    """A spike in the ring around the centroid box is left out of the background's slope."""
    image = _bright_floor()
    vv, uu = np.mgrid[0:41, 0:41]
    image += 40.0 * np.exp(-((vv - 22.0) ** 2 + (uu - 18.0) ** 2) / (2.0 * 0.8**2))
    image[22, 23] += 2000.0
    detection, _peak = local_centroid(
        image,
        (22.0, 18.0),
        search_window_px=2.0,
        centroid_box_half_px=3,
        image_noise_sigma=_NOISE_SIGMA,
        detection_sigma=5.0,
    )
    assert detection is not None
    assert detection[1] == pytest.approx(18.0, abs=0.2)


def test_the_background_level_is_the_box_floor_not_the_ring() -> None:
    """Glow filling the centroid box above the floor around it is background, not star."""
    image = _bright_floor()
    vv, uu = np.mgrid[0:41, 0:41]
    image += 40.0 * np.exp(-((vv - 22.4) ** 2 + (uu - 18.0) ** 2) / (2.0 * 0.8**2))
    image[19:26, 15:22] += 20.0
    detection, _peak = local_centroid(
        image,
        (22.0, 18.0),
        search_window_px=2.0,
        centroid_box_half_px=3,
        image_noise_sigma=_NOISE_SIGMA,
        detection_sigma=5.0,
    )
    assert detection is not None
    assert detection[0] == pytest.approx(22.4, abs=0.15)


def test_a_noisy_streak_is_centroided_at_its_center() -> None:
    """A streak is measured at its center wherever the noise puts its brightest pixel."""
    rng = np.random.default_rng(765)
    image = rng.normal(0.0, _NOISE_SIGMA, size=(41, 41))
    draw_streak(image, (20.45, 19.6), (8.0, -1.0), flux=600.0)
    detection, _peak = local_centroid(
        image,
        (21.0, 20.0),
        search_window_px=6.0,
        centroid_box_half_px=3,
        image_noise_sigma=_NOISE_SIGMA,
        detection_sigma=4.0,
        streak_vu=(8.0, -1.0),
    )
    assert detection is not None
    assert detection[0] == pytest.approx(20.45, abs=0.15)
    assert detection[1] == pytest.approx(19.6, abs=0.15)


def test_a_streak_on_steep_glare_is_centroided_at_its_center() -> None:
    """A slope steeper than the streak's light does not draw the search uphill."""
    columns = np.mgrid[0:41, 0:41][1]
    image = 100.0 + 4.0 * columns + np.random.default_rng(7).normal(0.0, _NOISE_SIGMA, (41, 41))
    draw_streak(image, (20.3, 18.0), (6.0, 1.0), flux=300.0)
    detection, _peak = local_centroid(
        image,
        (21.0, 19.0),
        search_window_px=6.0,
        centroid_box_half_px=3,
        image_noise_sigma=_NOISE_SIGMA,
        detection_sigma=5.0,
        streak_vu=(6.0, 1.0),
    )
    assert detection is not None
    assert detection[0] == pytest.approx(20.3, abs=0.15)
    assert detection[1] == pytest.approx(18.0, abs=0.15)


def test_sloping_glare_without_a_star_holds_no_streak() -> None:
    """On a star-free slope the streak's own pixels stand no higher than the fitted plane."""
    columns = np.mgrid[0:41, 0:41][1]
    image = _bright_floor() + 1.0 * columns
    detection, _peak = local_centroid(
        image,
        (20.0, 20.0),
        search_window_px=6.0,
        centroid_box_half_px=3,
        image_noise_sigma=_NOISE_SIGMA,
        detection_sigma=5.0,
        streak_vu=(1.0, 6.0),
    )
    assert detection is None


def test_a_streak_too_short_to_matter_is_centroided_as_a_point() -> None:
    """A streak too short to move the centroid leaves the point measurement."""
    image = _bright_floor()
    draw_streak(image, (22.0, 18.0), (3.0, 0.0), flux=300.0)
    short = local_centroid(
        image,
        (20.0, 20.0),
        search_window_px=6.0,
        centroid_box_half_px=3,
        image_noise_sigma=_NOISE_SIGMA,
        detection_sigma=5.0,
        streak_vu=(3.0, 0.0),
    )
    point = local_centroid(
        image,
        (20.0, 20.0),
        search_window_px=6.0,
        centroid_box_half_px=3,
        image_noise_sigma=_NOISE_SIGMA,
        detection_sigma=5.0,
    )
    assert short == point


def test_light_beside_a_streak_does_not_pull_its_centroid() -> None:
    """A source inside the streak's box but clear of the streak itself is left out."""
    rng = np.random.default_rng(765)
    image = rng.normal(0.0, _NOISE_SIGMA, size=(41, 41))
    draw_streak(image, (20.3, 19.6), (6.5, -0.8), flux=400.0)
    draw_streak(image, (14.0, 24.0), (0.0, 0.0), flux=150.0)
    detection, _peak = local_centroid(
        image,
        (21.0, 20.0),
        search_window_px=6.0,
        centroid_box_half_px=3,
        image_noise_sigma=_NOISE_SIGMA,
        detection_sigma=4.0,
        streak_vu=(6.5, -0.8),
    )
    assert detection is not None
    assert detection[0] == pytest.approx(20.3, abs=0.2)
    assert detection[1] == pytest.approx(19.6, abs=0.2)
