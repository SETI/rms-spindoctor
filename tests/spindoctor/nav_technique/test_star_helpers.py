"""Tests for the search window and detection rule the star techniques share."""

from __future__ import annotations

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import DrawGaussianStarFactory, NavFeatureFactory

from spindoctor.nav_technique._star_helpers import (
    _flank_outshines,
    _window_peaks,
    detection_peak_ratio,
    gaussian_psf_kernel,
    isolated_response,
    isolation_radii,
    local_centroid,
    search_window_bounds,
    third_star_votes,
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


def test_the_search_window_reaches_whole_pixels_only() -> None:
    """A half-width of 10.5 px reaches 10 whole pixels either side, 21 in all."""
    bounds = search_window_bounds((50.0, 50.0), 10.5, (200, 200))
    assert bounds == (40, 61, 40, 61)


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
        max_peak_sharpness=8.0,
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
        max_peak_sharpness=8.0,
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
        max_peak_sharpness=8.0,
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
        max_peak_sharpness=8.0,
        image_noise_sigma=1.0,
        detection_sigma=4.0,
        refuse_outshone=False,
    )
    assert centroid is not None
    assert centroid[0] == pytest.approx(100.0, abs=0.1)


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
        max_peak_sharpness=8.0,
        image_noise_sigma=1.0,
        detection_sigma=4.0,
        refuse_outshone=refuse,
    )
    assert (centroid is None) is expect_none


def _centroid(image: np.ndarray, prediction: tuple[float, float], **overrides: object) -> object:
    """Return :func:`local_centroid`'s centroid with the star techniques' defaults."""
    kwargs: dict[str, object] = {
        'search_window_px': 30.0,
        'centroid_box_half_px': 3,
        'psf_sigma_px': 1.0,
        'max_peak_sharpness': 8.0,
        'image_noise_sigma': 1.0,
        'detection_sigma': 4.0,
        'refuse_outshone': False,
    }
    kwargs.update(overrides)
    centroid, _peak = local_centroid(image, prediction, **kwargs)  # type: ignore[arg-type]
    return centroid


def _narrow_star_and_spike(draw_gaussian_star: DrawGaussianStarFactory) -> np.ndarray:
    """Return a Cassini NAC-like star (PSF sigma 0.54 px) and a spike twice its peak.

    The star's peak is 10 noise sigmas, so the spike stands 20 noise sigmas above the
    background with nothing in its neighbors.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    draw_gaussian_star(image, (110.0, 100.0), peak_dn=10.0, sigma=0.54)
    image[85, 112] += 20.0
    return image


def test_a_spike_outranks_a_narrow_star_in_the_response_alone(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """With no sharpness limit, a spike twice a sigma-0.54 star's peak takes its window."""
    centroid = _centroid(
        _narrow_star_and_spike(draw_gaussian_star), (100.0, 100.0), max_peak_sharpness=np.inf
    )
    assert centroid == pytest.approx((85.0, 112.0), abs=0.1)


def test_a_spike_too_sharp_for_a_star_is_passed_over(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """The sharpness limit skips the spike, and the narrow star is measured."""
    centroid = _centroid(_narrow_star_and_spike(draw_gaussian_star), (100.0, 100.0))
    assert centroid == pytest.approx((110.0, 100.0), abs=0.1)


def test_a_lone_narrow_star_is_not_too_sharp(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A sigma-0.54 star centered on a pixel is a star, not a spike."""
    image = np.zeros((200, 200), dtype=np.float64)
    draw_gaussian_star(image, (104.0, 97.0), peak_dn=500.0, sigma=0.54)
    centroid = _centroid(image, (100.0, 100.0))
    assert centroid == pytest.approx((104.0, 97.0), abs=0.1)


def test_the_centroid_box_subtracts_its_background(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A star off its box's center on a uniform floor is measured where it is."""
    image = np.full((200, 200), 50.0, dtype=np.float64)
    draw_gaussian_star(image, (104.4, 97.3), peak_dn=200.0, sigma=1.2)
    centroid = _centroid(image, (100.0, 100.0))
    assert centroid == pytest.approx((104.4, 97.3), abs=0.15)


def test_a_box_that_walks_out_of_the_window_holds_no_detection(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Re-centering does not carry a faint star's box onto a bright one beyond the window.

    The window spans rows 14 through 26; the faint star at row 25 is its only peak, and
    the light of the star at row 31 draws the box down the column, step by step.
    """
    image = np.zeros((50, 50), dtype=np.float64)
    draw_gaussian_star(image, (25.0, 20.0), peak_dn=10.0, sigma=1.0)
    draw_gaussian_star(image, (31.0, 20.0), peak_dn=1000.0, sigma=1.0)
    assert _centroid(image, (20.0, 20.0), search_window_px=6.0) is None


def test_a_box_that_steps_just_out_of_the_window_holds_no_detection(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A box drawn 2 px past the window's last row has left the window's source.

    The window spans rows 14 through 26.  The narrow star at row 25 is its only peak;
    the broad source 3 px below it, outside the window, draws the box to row 27.
    """
    image = np.zeros((50, 50), dtype=np.float64)
    draw_gaussian_star(image, (25.0, 20.0), peak_dn=100.0, sigma=0.8)
    draw_gaussian_star(image, (28.0, 20.0), peak_dn=80.0, sigma=2.0)
    assert _centroid(image, (20.0, 20.0), search_window_px=6.0) is None


def test_a_star_on_the_last_row_of_the_image_is_measured(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A star whose brightest pixel is on the image's edge has no sharpness to judge."""
    image = np.zeros((60, 60), dtype=np.float64)
    draw_gaussian_star(image, (59.0, 30.0), peak_dn=200.0, sigma=1.0)
    centroid = _centroid(image, (55.0, 30.0))
    assert centroid is not None


def test_a_box_that_walks_past_its_half_width_holds_no_detection(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Re-centering does not carry a narrow source's box 6 px onto a broad one.

    The narrow source is the strongest peak; the broad one beside it draws the box
    along the row, inside the window, farther than the box's own half-width.
    """
    image = np.zeros((80, 80), dtype=np.float64)
    draw_gaussian_star(image, (40.0, 40.0), peak_dn=100.0, sigma=0.8)
    draw_gaussian_star(image, (40.0, 46.0), peak_dn=90.0, sigma=2.5)
    assert _centroid(image, (40.0, 40.0)) is None


def test_a_star_beyond_the_padding_is_not_measured_through_its_flank(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A broad source 6 px past the window's edge leaves no peak inside the window.

    The response at the window's edge is judged against the response 3 px beyond it,
    which must itself see 3 px farther out to be the response a whole-image filter
    gives there.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    draw_gaussian_star(image, (100.0, 136.0), peak_dn=500.0, sigma=3.0)
    assert _centroid(image, (100.0, 100.0)) is None


def test_the_peak_ratio_of_a_detection_off_the_window_is_zero() -> None:
    """A detection whose exclusion box misses the window has no ratio."""
    image = np.ones((200, 200), dtype=np.float64)
    ratio = detection_peak_ratio(
        image, (100.0, 100.0), (180.0, 180.0), search_window_px=30.0, exclude_half_px=5
    )
    assert ratio == 0.0


@pytest.mark.parametrize(
    ('bright_vu', 'bright_dn', 'peak_vu', 'outshone'),
    [
        ((0, 6), 20.0, (11, 11), True),
        ((2, 6), 20.0, (11, 11), True),
        ((3, 6), 20.0, (11, 11), False),
        ((6, 6), 20.0, (11, 11), False),
        ((0, 6), 10.0, (11, 11), False),
        ((0, 6), 20.0, (3, 6), False),
    ],
)
def test_a_window_is_outshone_by_a_brighter_pixel_at_its_edge_with_no_peak_near(
    bright_vu: tuple[int, int], bright_dn: float, peak_vu: tuple[int, int], outshone: bool
) -> None:
    """Only a pixel brighter than the detection's peak, at the edge, with no peak near.

    The window is 13 px square and the box half-width 3 px, so the edge zone is the
    outer 3 rows and columns.  A pixel equal to the detection's peak (10 DN) does not
    outshine it, and one with a peak within 3 px is a source inside the window.
    """
    window = np.zeros((13, 13), dtype=np.float64)
    window[bright_vu] = bright_dn
    peaks = np.full((13, 13), -np.inf)
    peaks[peak_vu] = 1.0
    assert _flank_outshines(window, peaks, 10.0, half=3) is outshone


def test_a_broad_source_just_past_the_window_leaves_it_no_peak() -> None:
    """A sigma-4 source centered 1 px past the window's last column puts no peak in it.

    Whether the last column is a peak is judged against the response 3 px beyond it,
    which is the whole-image response only when the image is seen 3 px farther still.
    """
    vv, uu = np.mgrid[0:60, 0:60]
    image = 100.0 * np.exp(-((vv - 30.0) ** 2 + (uu - 43.0) ** 2) / (2.0 * 4.0 * 4.0))
    bounds = search_window_bounds((30.0, 30.0), 12.0, image.shape)
    assert bounds is not None
    peaks = _window_peaks(image, bounds, kernel=gaussian_psf_kernel(1.0, size=7))
    assert not np.isfinite(peaks).any()


def test_the_isolation_annulus_moves_out_by_half_the_smear() -> None:
    """A star smeared over 8.5 px gets an annulus 5 px farther out."""
    assert isolation_radii(4, 6, 8.5) == (9, 11)


def test_a_peak_on_a_raised_response_is_isolated() -> None:
    """Isolation is judged above the annulus mean, so a uniform offset changes nothing."""
    vv, uu = np.mgrid[-10:11, -10:11]
    response = 10.0 + 5.0 * np.exp(-(vv * vv + uu * uu) / 2.0)
    isolated = isolated_response(response, inner_px=4, outer_px=6, max_ratio=0.5)
    assert bool(isolated[10, 10])


def test_a_star_with_no_detection_casts_no_vote(
    make_star_feature: NavFeatureFactory,
) -> None:
    """Stars whose windows hold nothing neither agree nor disagree."""
    others = [
        make_star_feature(f'star:UCAC4:{k}', predicted_vu=(50.0 * k, 50.0), predicted_snr=10.0)
        for k in range(3)
    ]
    votes = third_star_votes(others, (2.0, -3.0), detect=lambda star: None, tolerance_px=2.4)
    assert votes == (0, 0)
