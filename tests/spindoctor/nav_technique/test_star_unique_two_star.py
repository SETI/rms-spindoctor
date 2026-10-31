"""Tests for the pair photometry in ``_star_unique_two_star``."""

from __future__ import annotations

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import DrawGaussianStarFactory, NavFeatureFactory

from spindoctor.feature.feature import NavFeature
from spindoctor.nav_technique._star_unique_two_star import (
    TwoStarAssignment,
    absent_stars,
    pair_hypotheses,
    peak_per_snr,
    source_peak,
    star_like_source,
    third_star_votes,
)

# The solution of a pair that moves every prediction by 2 rows and -3 columns.
_SHIFT = TwoStarAssignment(offset_vu=(2.0, -3.0), residual_px=0.0, rotation_rad=None).place


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
    scale = peak_per_snr(image, stars, _SHIFT, detect_at=lambda star, target: target, box_half_px=3)
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
        _SHIFT,
        detect_at=lambda star, target: target if target[0] < 100.0 else None,
        box_half_px=3,
    )
    assert scale is None


def test_pair_hypotheses_count_every_star_found_at_the_offset() -> None:
    """Three stars with a source each at (2, -3) back the pair; a stray source does not."""
    predictions = [(50.0, 50.0), (150.0, 50.0), (100.0, 150.0)]
    sources = [[(52.0, 47.0), (70.0, 70.0)], [(152.1, 47.0)], [(101.9, 147.0)]]
    best = pair_hypotheses(predictions, sources, max_residual_px=1.2, fit_rotation=False)[0]
    assert best.support == 3
    assert best.solved.offset_vu == pytest.approx((2.0, -3.0), abs=0.1)


def test_pair_hypotheses_put_the_best_supported_pair_first() -> None:
    """A coincidence between two strongest sources ranks below a pair three stars back."""
    predictions = [(50.0, 50.0), (150.0, 50.0), (100.0, 150.0)]
    sources = [[(60.0, 65.0), (52.0, 47.0)], [(160.0, 65.0), (152.0, 47.0)], [(102.0, 147.0)]]
    found = pair_hypotheses(predictions, sources, max_residual_px=1.2, fit_rotation=False)
    assert found[0].solved.offset_vu == pytest.approx((2.0, -3.0))
    assert found[-1].support == 2


def test_pair_hypotheses_do_not_pair_one_source_with_itself() -> None:
    """A source in two stars' windows is not paired with itself."""
    predictions = [(50.0, 50.0), (50.5, 50.0)]
    sources = [[(52.0, 47.0)], [(52.0, 47.0)]]
    assert pair_hypotheses(predictions, sources, max_residual_px=1.2, fit_rotation=False) == []


def test_pair_hypotheses_refuse_a_residual_over_the_tolerance() -> None:
    """Two stars whose offsets differ by 3 px have a residual of 1.5 px and do not pass."""
    predictions = [(50.0, 50.0), (150.0, 50.0)]
    sources = [[(52.0, 47.0)], [(155.0, 47.0)]]
    assert pair_hypotheses(predictions, sources, max_residual_px=1.2, fit_rotation=False) == []


def test_star_like_source_skips_a_source_unlike_the_star(
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """At 10 DN per unit SNR, an SNR-10 star's source is the 100 DN one, not the 500 DN one."""
    image = np.zeros((200, 200))
    draw_gaussian_star(image, (50.0, 50.0), peak_dn=500.0)
    draw_gaussian_star(image, (120.0, 120.0), peak_dn=100.0)
    star = make_star_feature('star:UCAC4:0', predicted_vu=(80.0, 80.0), predicted_snr=10.0)
    found = star_like_source(
        image,
        star,
        [(50.0, 50.0), (120.0, 120.0)],
        peak_per_snr=10.0,
        peak_factor=4.0,
        box_half_px=3,
    )
    assert found == (120.0, 120.0)


@pytest.mark.parametrize(('n_sources', 'located'), [(1, True), (2, False)])
def test_star_like_source_without_a_scale_needs_a_lone_source(
    make_star_feature: NavFeatureFactory, n_sources: int, located: bool
) -> None:
    """With no scale a window's one source stands for the star, and two leave it unlocated."""
    star = make_star_feature('star:UCAC4:0', predicted_vu=(80.0, 80.0), predicted_snr=10.0)
    found = star_like_source(
        np.zeros((200, 200)),
        star,
        [(50.0, 50.0), (120.0, 120.0)][:n_sources],
        peak_per_snr=None,
        peak_factor=4.0,
        box_half_px=3,
    )
    assert (found is not None) is located


def test_star_like_source_of_several_as_bright_as_the_star_is_none(
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """Two sources that could each be an SNR-10 star at 10 DN per unit SNR locate it nowhere."""
    image = np.zeros((200, 200))
    draw_gaussian_star(image, (50.0, 50.0), peak_dn=90.0)
    draw_gaussian_star(image, (120.0, 120.0), peak_dn=110.0)
    star = make_star_feature('star:UCAC4:0', predicted_vu=(80.0, 80.0), predicted_snr=10.0)
    found = star_like_source(
        image,
        star,
        [(50.0, 50.0), (120.0, 120.0)],
        peak_per_snr=10.0,
        peak_factor=4.0,
        box_half_px=3,
    )
    assert found is None


def _absent_count(image: np.ndarray, stars: list[NavFeature], *, faint_snr: float = 10.0) -> int:
    """Count absent stars at offset (2, -3), 10 DN per unit SNR, 20 px padding."""
    return absent_stars(
        image,
        stars,
        _SHIFT,
        faint_snr=faint_snr,
        peak_per_snr=10.0,
        peak_factor=4.0,
        margin_vu=(20, 20),
        noise_sigma=1.0,
        box_half_px=3,
    )


@pytest.mark.parametrize(('drawn_dn', 'absent'), [(0.0, 1), (100.0, 0)])
def test_absent_stars_count_a_star_missing_from_clear_sky(
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    drawn_dn: float,
    absent: int,
) -> None:
    """An SNR-10 star expected at 100 DN is absent when nothing is there."""
    image = np.random.default_rng(1).normal(0.0, 1.0, (200, 200))
    draw_gaussian_star(image, (102.0, 97.0), peak_dn=drawn_dn)
    star = make_star_feature('star:UCAC4:0', predicted_vu=(100.0, 100.0), predicted_snr=10.0)
    assert _absent_count(image, [star]) == absent


@pytest.mark.parametrize(
    ('where', 'snr'), [('in-padding', 10.0), ('fainter-than-the-pair', 5.0), ('on-a-ring', 10.0)]
)
def test_absent_stars_judge_only_a_bright_star_on_clear_sky_in_the_image(
    make_star_feature: NavFeatureFactory,
    where: str,
    snr: float,
) -> None:
    """A star in the padding, fainter than the pair, or on a textured ring is not judged."""
    rng = np.random.default_rng(2)
    image = rng.normal(0.0, 1.0, (200, 200))
    pred = (8.0, 100.0) if where == 'in-padding' else (100.0, 100.0)
    if where == 'on-a-ring':
        image[80:120, :] += rng.normal(50.0, 10.0, (40, 200))
    star = make_star_feature('star:UCAC4:0', predicted_vu=pred, predicted_snr=snr)
    assert _absent_count(image, [star]) == 0


def test_a_star_with_no_detection_casts_no_vote(
    make_star_feature: NavFeatureFactory,
) -> None:
    """Stars whose windows hold nothing neither agree nor disagree."""
    others = [
        make_star_feature(f'star:UCAC4:{k}', predicted_vu=(50.0 * k, 50.0), predicted_snr=10.0)
        for k in range(3)
    ]
    votes = third_star_votes(
        others,
        _SHIFT,
        detect=lambda star: None,
        detect_near=lambda star, target: None,
        tolerance_px=2.4,
    )
    assert votes == (0, 0)


@pytest.mark.parametrize(
    ('near', 'votes'),
    [((52.1, 46.9), (1, 0)), (None, (0, 1))],
    ids=['found-at-the-offset', 'absent-at-the-offset'],
)
def test_a_star_found_at_the_offset_agrees_whatever_else_its_window_holds(
    make_star_feature: NavFeatureFactory,
    near: tuple[float, float] | None,
    votes: tuple[int, int],
) -> None:
    """A brighter source elsewhere in the window votes only when the star is absent."""
    star = make_star_feature('star:UCAC4:0', predicted_vu=(50.0, 50.0), predicted_snr=10.0)
    result = third_star_votes(
        [star],
        _SHIFT,
        detect=lambda feature: (70.0, 30.0),
        detect_near=lambda feature, target: near,
        tolerance_px=2.4,
    )
    assert result == votes


def test_a_star_is_sought_where_the_offset_puts_it(
    make_star_feature: NavFeatureFactory,
) -> None:
    """The near search is centered on the prediction moved by the offset."""
    star = make_star_feature('star:UCAC4:0', predicted_vu=(50.0, 50.0), predicted_snr=10.0)
    targets: list[tuple[float, float]] = []

    def detect_near(feature: object, target: tuple[float, float]) -> tuple[float, float] | None:
        targets.append(target)
        return None

    third_star_votes(
        [star],
        _SHIFT,
        detect=lambda feature: None,
        detect_near=detect_near,
        tolerance_px=2.4,
    )
    assert targets == [(52.0, 47.0)]


@pytest.mark.parametrize('distance_px', [2.6, 3.3])
def test_a_source_beyond_the_tolerance_of_the_offset_does_not_agree(
    make_star_feature: NavFeatureFactory,
    distance_px: float,
) -> None:
    """A near detection beyond the tolerance from the moved prediction casts no vote."""
    star = make_star_feature('star:UCAC4:0', predicted_vu=(50.0, 50.0), predicted_snr=10.0)
    result = third_star_votes(
        [star],
        _SHIFT,
        detect=lambda feature: None,
        detect_near=lambda feature, target: (target[0] + distance_px, target[1]),
        tolerance_px=2.4,
    )
    assert result == (0, 0)


@pytest.mark.parametrize(
    ('second_detection', 'votes'),
    [((70.5, 30.4), (0, 1)), ((71.2, 30.0), (0, 2))],
    ids=['same-source', 'distinct-sources'],
)
def test_one_source_in_two_windows_disagrees_once(
    make_star_feature: NavFeatureFactory,
    second_detection: tuple[float, float],
    votes: tuple[int, int],
) -> None:
    """Two stars whose windows return one source cast one disagreeing vote between them."""
    stars = [
        make_star_feature(
            f'star:UCAC4:{k}', predicted_vu=(50.0 + 20.0 * k, 50.0), predicted_snr=10.0
        )
        for k in range(2)
    ]
    detections = {stars[0].feature_id: (70.0, 30.0), stars[1].feature_id: second_detection}
    result = third_star_votes(
        stars,
        _SHIFT,
        detect=lambda feature: detections[feature.feature_id],
        detect_near=lambda feature, target: None,
        tolerance_px=2.4,
    )
    assert result == votes
