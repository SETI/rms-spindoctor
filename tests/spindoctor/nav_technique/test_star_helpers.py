"""Tests for the search window and detection rule the star techniques share."""

from __future__ import annotations

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import DrawGaussianStarFactory, NavFeatureFactory

from spindoctor.nav_technique._star_helpers import (
    _flank_outshines,
    _flat_top_region,
    _window_peaks,
    detection_peak_ratio,
    distinct_stars,
    gaussian_psf_kernel,
    local_centroid,
    search_window_bounds,
    window_sources,
)
from spindoctor.nav_technique._star_isolation import (
    is_isolated_point_source,
    isolated_response,
    isolation_lookup,
    isolation_radii,
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
    bounds = search_window_bounds(_FRAME_PREDICTION_VU, (30.0, 30.0), _FRAME_SHAPE)
    assert bounds == (609, 670, 1072, 1133)


@pytest.mark.parametrize('v_pred', [50.2, 50.7])
def test_the_search_window_width_does_not_depend_on_the_fraction(v_pred: float) -> None:
    """A prediction either side of a pixel's center gets the same odd-width window."""
    bounds = search_window_bounds((v_pred, 50.0), (10.0, 10.0), (200, 200))
    assert bounds is not None
    v_lo, v_hi, _u_lo, _u_hi = bounds
    assert v_hi - v_lo == 21


def test_the_search_window_reaches_whole_pixels_only() -> None:
    """A half-width of 10.5 px reaches 10 whole pixels either side, 21 in all."""
    bounds = search_window_bounds((50.0, 50.0), (10.5, 10.5), (200, 200))
    assert bounds == (40, 61, 40, 61)


def test_the_search_window_reaches_each_axis_by_its_own_half_width() -> None:
    """Half-widths of 5 and 10 px give a window 11 rows tall and 21 columns wide."""
    bounds = search_window_bounds((50.0, 50.0), (5.0, 10.0), (200, 200))
    assert bounds == (45, 56, 40, 61)


def test_the_search_window_off_the_image_is_empty() -> None:
    """A prediction farther off the image than the half-width has no window."""
    assert search_window_bounds((-50.0, 100.0), (10.0, 10.0), (200, 200)) is None


def test_a_brighter_single_pixel_does_not_win_the_stars_window(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """The detection is the star, not the single pixel that outshines its peak."""
    centroid, _peak = local_centroid(
        _star_and_spike(draw_gaussian_star),
        (100.0, 100.0),
        search_half_vu=(30.0, 30.0),
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
        search_half_vu=(30.0, 30.0),
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
        search_half_vu=(30.0, 30.0),
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
        search_half_vu=(30.0, 30.0),
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

    The response peaks near an end of a flat core, where a box first straddles it; the
    core is as long as the box, so it is measured whole.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    image[97:104, 99:102] = 1000.0
    centroid, _peak = local_centroid(
        image,
        (98.0, 100.0),
        search_half_vu=(30.0, 30.0),
        centroid_box_half_px=3,
        psf_sigma_px=1.0,
        max_peak_sharpness=8.0,
        image_noise_sigma=1.0,
        detection_sigma=4.0,
        refuse_outshone=False,
    )
    assert centroid is not None
    assert centroid[0] == pytest.approx(100.0, abs=0.1)


def test_a_saturated_core_longer_than_the_box_is_measured_about_its_middle() -> None:
    """A core 13 rows tall and 4 wide, flat within the tolerance, centroids to its middle.

    A box re-centered on it stops wherever it lies wholly on the flat top, so the
    centroid is measured over the whole flat top instead.  The core's pixels differ by
    3 DN, so a flat top taken without the tolerance breaks into single pixels.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    image[94:107, 99:103] = 1000.0
    image[94:107:2, 99:103:2] -= 3.0
    image[95:107:2, 100:103:2] -= 3.0
    centroid = _centroid(image, (95.0, 100.0))
    assert centroid == pytest.approx((100.0, 100.5), abs=0.01)


def test_a_flat_streak_across_the_window_is_no_source() -> None:
    """A flat streak that runs out of the window is no point source of it."""
    image = np.zeros((200, 200), dtype=np.float64)
    image[99:102, :] = 1000.0
    assert _centroid(image, (100.0, 100.0)) is None


def test_a_flat_core_centered_outside_the_window_is_no_source() -> None:
    """A flat core wider than the box whose middle lies past the window's edge is refused."""
    image = np.zeros((200, 200), dtype=np.float64)
    image[105:109, 93:108] = 1000.0
    assert _centroid(image, (100.0, 100.0), search_half_vu=(6.0, 6.0)) is None


def _clipped_core_region(image: np.ndarray, candidate: tuple[int, int]) -> object:
    """Return :func:`_flat_top_region` for a candidate in a 61 px window around (100, 100)."""
    bounds = search_window_bounds((100.0, 100.0), (30.0, 30.0), image.shape)
    assert bounds is not None
    return _flat_top_region(
        image, candidate[0], candidate[1], half=3, tolerance_dn=4.0, bounds=bounds
    )


def test_a_flat_core_as_wide_as_the_box_is_a_clipped_core() -> None:
    """A flat core 13 rows tall and 7 wide, as wide as the box, is a clipped core."""
    image = np.zeros((200, 200), dtype=np.float64)
    image[94:107, 97:104] = 1000.0
    assert _clipped_core_region(image, (94, 97)) == (91, 110, 94, 107)


def test_a_flat_ridge_with_gentle_sides_is_no_clipped_core() -> None:
    """A flat line 9 px long whose neighbors lie 6 tolerances (24 DN) below it is no core.

    A ring or a limb is flat within the tolerance along its crest, but its sides slope
    away gently, where a clipped core stands on the steep wall of a star's profile.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    image[96:105, 99:102] = 976.0
    image[96:105, 100] = 1000.0
    assert _clipped_core_region(image, (96, 100)) is None


def test_a_clipped_core_beside_the_candidate_is_not_its_own() -> None:
    """A core whose nearest column is 3 px from the candidate belongs to another source."""
    image = np.zeros((200, 200), dtype=np.float64)
    image[94:107, 103:106] = 1000.0
    assert _clipped_core_region(image, (100, 100)) is None


def test_a_flat_block_wider_than_the_box_both_ways_is_no_clipped_core() -> None:
    """A flat 12 by 12 px block, a saturated body rather than a star, is no clipped core."""
    image = np.zeros((200, 200), dtype=np.float64)
    image[94:106, 94:106] = 1000.0
    assert _clipped_core_region(image, (94, 94)) is None


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
        search_half_vu=(6.0, 6.0),
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
        'search_half_vu': (30.0, 30.0),
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
    assert _centroid(image, (20.0, 20.0), search_half_vu=(6.0, 6.0)) is None


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
    assert _centroid(image, (20.0, 20.0), search_half_vu=(6.0, 6.0)) is None


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
        image, (100.0, 100.0), (180.0, 180.0), search_half_vu=(30.0, 30.0), exclude_half_px=5
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
    bounds = search_window_bounds((30.0, 30.0), (12.0, 12.0), image.shape)
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


def _sources(image: np.ndarray, *, max_sources: int = 5) -> list[tuple[float, float]]:
    """Return the isolated point sources :func:`window_sources` finds around (100, 100)."""
    found = window_sources(
        image,
        (100.0, 100.0),
        search_half_vu=(40.0, 40.0),
        centroid_box_half_px=3,
        psf_sigma_px=1.0,
        max_peak_sharpness=8.0,
        image_noise_sigma=1.0,
        detection_sigma=4.0,
        accept=lambda c: is_isolated_point_source(
            image, c, sigma_px=1.0, kernel_half_px=3, inner_px=4, outer_px=6, max_ratio=0.5
        ),
        max_sources=max_sources,
    )
    return [s.centroid_vu for s in found]


def test_a_saturated_core_longer_than_the_box_is_one_source() -> None:
    """A flat core 13 rows tall is one source at its middle, not one near an end."""
    image = np.zeros((200, 200), dtype=np.float64)
    image[94:107, 99:103] = 1000.0
    found = window_sources(
        image,
        (100.0, 100.0),
        search_half_vu=(40.0, 40.0),
        centroid_box_half_px=3,
        psf_sigma_px=1.0,
        max_peak_sharpness=8.0,
        image_noise_sigma=1.0,
        detection_sigma=4.0,
        accept=lambda _c: True,
        max_sources=5,
    )
    assert [s.centroid_vu for s in found] == [pytest.approx((100.0, 100.5), abs=0.01)]


def test_window_sources_are_listed_strongest_first(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Three stars of 50, 200, and 100 DN are listed brightest first, at most two."""
    image = np.zeros((200, 200))
    for vu, peak in (((90.0, 90.0), 50.0), ((110.0, 120.0), 200.0), ((80.0, 120.0), 100.0)):
        draw_gaussian_star(image, vu, peak_dn=peak)
    found = _sources(image, max_sources=2)
    assert len(found) == 2
    assert found[0] == pytest.approx((110.0, 120.0), abs=0.1)
    assert found[1] == pytest.approx((80.0, 120.0), abs=0.1)


def test_window_sources_pass_over_a_ridge_to_the_star_beside_it(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A ridge brighter than the star holds no source; the star is the window's first."""
    image = np.zeros((200, 200))
    for u in np.arange(60.0, 140.0, 0.5):
        draw_gaussian_star(image, (85.0, u), peak_dn=60.0, sigma=1.5)
    draw_gaussian_star(image, (102.0, 97.0), peak_dn=40.0)
    found = _sources(image)
    assert found[0] == pytest.approx((102.0, 97.0), abs=0.1)


def test_window_sources_pass_over_a_spike_too_sharp_for_a_star() -> None:
    """A single hot pixel is no source."""
    image = np.zeros((200, 200))
    image[110, 95] = 500.0
    assert _sources(image) == []


def test_window_sources_off_the_image_are_none() -> None:
    """A window wholly off the image holds no source."""
    found = window_sources(
        np.zeros((50, 50)),
        (300.0, 300.0),
        search_half_vu=(10.0, 10.0),
        centroid_box_half_px=3,
        psf_sigma_px=1.0,
        max_peak_sharpness=8.0,
        image_noise_sigma=1.0,
        detection_sigma=4.0,
        accept=lambda c: True,
        max_sources=5,
    )
    assert found == []


def test_window_sources_count_a_source_reached_from_two_peaks_once(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Two equal stars 4 px apart leave two response peaks whose boxes settle on one centroid."""
    image = np.zeros((200, 200))
    draw_gaussian_star(image, (100.0, 98.0), peak_dn=50.0, sigma=0.8)
    draw_gaussian_star(image, (100.0, 102.0), peak_dn=50.0, sigma=0.8)
    found = window_sources(
        image,
        (100.0, 100.0),
        search_half_vu=(40.0, 40.0),
        centroid_box_half_px=3,
        psf_sigma_px=1.0,
        max_peak_sharpness=8.0,
        image_noise_sigma=1.0,
        detection_sigma=4.0,
        accept=lambda c: True,
        max_sources=5,
    )
    assert len(found) == 1


@pytest.mark.parametrize('smear_px', [0.0, 40.0])
def test_isolation_lookup_answers_as_the_single_detection_test(
    draw_gaussian_star: DrawGaussianStarFactory, smear_px: float
) -> None:
    """At 300 points of a window the window-wide test agrees with the one-detection test."""
    rng = np.random.default_rng(5)
    image = rng.normal(0.0, 1.0, (160, 220))
    for vu in rng.uniform((5.0, 5.0), (155.0, 215.0), (40, 2)):
        draw_gaussian_star(image, (float(vu[0]), float(vu[1])), peak_dn=30.0)
    image[60, 20:120] += 40.0
    inner, outer = isolation_radii(4, 6, smear_px)
    bounds = search_window_bounds((80.0, 110.0), (70.0, 100.0), image.shape)
    assert bounds is not None
    lookup = isolation_lookup(
        image, bounds, sigma_px=1.0, kernel_half_px=3, inner_px=inner, outer_px=outer, max_ratio=0.5
    )
    points = rng.uniform((bounds[0], bounds[2]), (bounds[1] - 1, bounds[3] - 1), (300, 2))
    disagree = [
        p
        for p in points
        if lookup((float(p[0]), float(p[1])))
        != is_isolated_point_source(
            image,
            (float(p[0]), float(p[1])),
            sigma_px=1.0,
            kernel_half_px=3,
            inner_px=inner,
            outer_px=outer,
            max_ratio=0.5,
        )
    ]
    assert disagree == []


def test_window_sources_pass_over_a_bump_below_the_threshold(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A smooth bump 3 noise sigmas high is no source at a 4-sigma threshold."""
    image = np.zeros((200, 200))
    draw_gaussian_star(image, (102.0, 97.0), peak_dn=3.0, sigma=1.2)
    assert _sources(image) == []


def test_distinct_stars_keep_the_first_of_two_entries_for_one_star(
    make_star_feature: NavFeatureFactory,
) -> None:
    """Entries 0.5 px apart are one star; one 5 px away is another."""
    stars = [
        make_star_feature('star:YBSC:1', predicted_vu=(100.0, 100.0), predicted_snr=60.0),
        make_star_feature('star:TYCHO2:1', predicted_vu=(100.3, 100.4), predicted_snr=59.0),
        make_star_feature('star:TYCHO2:2', predicted_vu=(105.0, 100.0), predicted_snr=50.0),
    ]
    kept = distinct_stars(stars)
    assert [s.feature_id for s in kept] == ['star:YBSC:1', 'star:TYCHO2:2']


def test_isolation_lookup_answers_as_the_single_detection_test_at_the_image_corner(
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """At every star of a window that runs off the image's corner the two tests agree."""
    rng = np.random.default_rng(9)
    image = rng.normal(0.0, 0.2, (120, 140))
    stars = [(float(v), float(u)) for v, u in rng.uniform((2.0, 2.0), (60.0, 70.0), (25, 2))]
    for vu in stars:
        draw_gaussian_star(image, vu, peak_dn=30.0)
    bounds = search_window_bounds((20.0, 25.0), (40.0, 45.0), image.shape)
    assert bounds is not None
    lookup = isolation_lookup(
        image, bounds, sigma_px=1.0, kernel_half_px=3, inner_px=4, outer_px=6, max_ratio=0.5
    )
    answers = [
        (
            lookup(vu),
            is_isolated_point_source(
                image, vu, sigma_px=1.0, kernel_half_px=3, inner_px=4, outer_px=6, max_ratio=0.5
            ),
        )
        for vu in stars
    ]
    assert sum(single for _, single in answers) >= 5
    assert [window for window, _ in answers] == [single for _, single in answers]


_FLOOR_DN = 100.0


def _bright_floor() -> np.ndarray:
    """Return a 41x41 frame on a uniform floor far above the noise, with mild texture."""
    rng = np.random.default_rng(7)
    return _FLOOR_DN + rng.normal(0.0, 1.0, size=(41, 41))


def _star_on(image: np.ndarray, center_vu: tuple[float, float]) -> np.ndarray:
    """Return ``image`` with a 40 DN star of sigma 0.8 px added at ``center_vu``."""
    vv, uu = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    v, u = center_vu
    return np.asarray(image + 40.0 * np.exp(-((vv - v) ** 2 + (uu - u) ** 2) / (2.0 * 0.8**2)))


def test_a_star_on_a_sloping_floor_is_centroided_where_it_lies() -> None:
    """Glare rising across the centroid box does not pull the centroid uphill."""
    columns = np.mgrid[0:41, 0:41][1]
    image = _star_on(_bright_floor() + 4.0 * columns, (22.0, 18.0))
    centroid = _centroid(image, (20.0, 20.0), search_half_vu=(6.0, 6.0))
    assert isinstance(centroid, tuple)
    assert centroid[1] == pytest.approx(18.0, abs=0.2)


def test_a_cosmic_ray_beside_the_box_does_not_tilt_its_background() -> None:
    """A spike in the ring around the centroid box is left out of the background's slope."""
    image = _star_on(_bright_floor(), (22.0, 18.0))
    image[22, 23] += 2000.0
    centroid = _centroid(image, (22.0, 18.0), search_half_vu=(2.0, 2.0))
    assert isinstance(centroid, tuple)
    assert centroid[1] == pytest.approx(18.0, abs=0.2)


def test_the_background_level_is_the_box_floor_not_the_ring() -> None:
    """Glow filling the centroid box above the floor around it is background, not star."""
    image = _star_on(_bright_floor(), (22.4, 18.0))
    image[19:26, 15:22] += 20.0
    centroid = _centroid(image, (22.0, 18.0), search_half_vu=(2.0, 2.0))
    assert isinstance(centroid, tuple)
    assert centroid[0] == pytest.approx(22.4, abs=0.15)
