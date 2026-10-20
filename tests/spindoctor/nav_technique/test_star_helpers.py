"""Tests for the window detection of ``spindoctor.nav_technique._star_helpers``."""

import numpy as np
import pytest

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
