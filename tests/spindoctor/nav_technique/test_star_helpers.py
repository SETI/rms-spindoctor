"""Tests for the search window and detection rule the star techniques share."""

from __future__ import annotations

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import DrawGaussianStarFactory

from spindoctor.nav_technique._star_helpers import (
    detection_peak_ratio,
    local_centroid,
    search_window_bounds,
)

# The library frame N1530185128_1 (Hyperion against stars): the prediction for its
# brightest star, and the image shape of its extended frame.
_FRAME_PREDICTION_VU = (638.717, 1101.787)
_FRAME_SHAPE = (1124, 1304)


def _star_and_spike(draw_gaussian_star: DrawGaussianStarFactory) -> np.ndarray:
    """Return a frame holding one star and a brighter single-pixel spike near it.

    The star sits 10 px below a prediction at ``(100, 100)``; the spike is one pixel
    1.8 times the star's peak, inside the same 30 px search window, which is the
    ratio a cosmic ray holds over the intended star in N1530185128_1.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    draw_gaussian_star(image, (110.0, 100.0), peak_dn=15.0, sigma=1.2)
    image[80, 118] += 27.0
    return image


def test_the_search_window_is_centered_on_the_pixel_holding_the_prediction() -> None:
    """The window reaches 30 whole pixels either side of the prediction's pixel.

    The prediction falls in pixel (639, 1102), so the window is rows 609 through 669
    and columns 1072 through 1132.  Row 608, which a cosmic ray occupies in
    N1530185128_1, lies 30.717 px from the prediction and is outside it.
    """
    bounds = search_window_bounds(_FRAME_PREDICTION_VU, 30.0, _FRAME_SHAPE)
    assert bounds == (609, 670, 1072, 1133)


@pytest.mark.parametrize('v_pred', [50.2, 50.7])
def test_the_search_window_width_does_not_depend_on_the_fraction(v_pred: float) -> None:
    """A prediction either side of a pixel's center gets the same odd-width window."""
    bounds = search_window_bounds((v_pred, 50.0), 10.0, (200, 200))
    assert bounds is not None
    v_lo, v_hi, _u_lo, _u_hi = bounds
    assert v_hi - v_lo == 21


def test_the_search_window_off_the_image_is_empty() -> None:
    """A prediction farther off the image than the half-width has no window."""
    assert search_window_bounds((-50.0, 100.0), 10.0, (200, 200)) is None


def test_a_brighter_single_pixel_does_not_win_the_stars_window(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """The detection is the star, not the single pixel that outshines its peak."""
    centroid, _peak = local_centroid(
        _star_and_spike(draw_gaussian_star),
        (100.0, 100.0),
        search_window_px=30.0,
        centroid_box_half_px=3,
        psf_sigma_px=1.0,
        image_noise_sigma=1.0,
        detection_sigma=4.0,
        refuse_outshone=False,
    )
    assert centroid is not None
    assert centroid[0] == pytest.approx(110.0, abs=0.1)
    assert centroid[1] == pytest.approx(100.0, abs=0.1)


def test_a_brighter_single_pixel_elsewhere_makes_the_window_ambiguous(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """The peak ratio compares the detection's own peak with the brighter pixel.

    The spike is 1.8 times the star's peak, so the ratio is 15 / 27.
    """
    ratio = detection_peak_ratio(
        _star_and_spike(draw_gaussian_star),
        (100.0, 100.0),
        (110.0, 100.0),
        search_window_px=30.0,
        exclude_half_px=5,
    )
    assert ratio == pytest.approx(15.0 / 27.0, rel=1e-6)


def test_a_star_outside_the_window_is_not_measured_through_its_flank(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A star whose peak lies 3 px past the window's edge yields no detection.

    Its light reaches into the window, and the brightest pixel of the window is on its
    flank, but no pixel of the window is a peak of it.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    draw_gaussian_star(image, (100.0, 67.0), peak_dn=200.0, sigma=1.2)
    centroid, _peak = local_centroid(
        image,
        (100.0, 100.0),
        search_window_px=30.0,
        centroid_box_half_px=3,
        psf_sigma_px=1.0,
        image_noise_sigma=1.0,
        detection_sigma=4.0,
        refuse_outshone=False,
    )
    assert centroid is None


def test_a_bump_on_a_bright_floor_is_no_detection() -> None:
    """A pixel 2 sigma above a uniform 100-sigma floor does not clear 4 sigma.

    Its own value clears the threshold many times over; it stands only 2 noise
    sigmas above its surroundings.
    """
    image = np.full((200, 200), 100.0, dtype=np.float64)
    image[100, 100] += 2.0
    centroid, _peak = local_centroid(
        image,
        (100.0, 100.0),
        search_window_px=30.0,
        centroid_box_half_px=3,
        psf_sigma_px=1.0,
        image_noise_sigma=1.0,
        detection_sigma=4.0,
        refuse_outshone=False,
    )
    assert centroid is None


def test_a_saturated_core_as_tall_as_the_box_is_measured_about_its_middle() -> None:
    """A flat core 7 rows tall and 3 wide centroids to its middle row.

    The response peaks near an end of a flat core, and a box placed there first
    straddles that end.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    image[97:104, 99:102] = 1000.0
    centroid, _peak = local_centroid(
        image,
        (98.0, 100.0),
        search_window_px=30.0,
        centroid_box_half_px=3,
        psf_sigma_px=1.0,
        image_noise_sigma=1.0,
        detection_sigma=4.0,
        refuse_outshone=False,
    )
    assert centroid is not None
    assert centroid[0] == pytest.approx(100.0, abs=0.5)


@pytest.mark.parametrize(('refuse', 'expect_none'), [(True, True), (False, False)])
def test_a_window_outshone_from_outside_is_refused_on_request(
    draw_gaussian_star: DrawGaussianStarFactory, refuse: bool, expect_none: bool
) -> None:
    """A bright star 3 px past the window's edge outshines a 6-sigma bump inside it.

    Asked to refuse such a window, the helper finds nothing; otherwise it measures
    the bump.
    """
    image = np.zeros((40, 40), dtype=np.float64)
    draw_gaussian_star(image, (20.0, 11.0), peak_dn=200.0, sigma=1.2)
    image[21, 21] += 6.0
    centroid, _peak = local_centroid(
        image,
        (20.0, 20.0),
        search_window_px=6.0,
        centroid_box_half_px=3,
        psf_sigma_px=1.0,
        image_noise_sigma=1.0,
        detection_sigma=4.0,
        refuse_outshone=refuse,
    )
    assert (centroid is None) is expect_none
