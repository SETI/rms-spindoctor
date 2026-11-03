"""Tests for ``StarUniqueMatchNav``'s lone-star search over the whole offset window."""

from __future__ import annotations

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    DrawGaussianStarFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.nav_technique._star_unique_one_star import (
    LoneStarHypothesis,
    choose_lone_star,
)
from spindoctor.nav_technique.nav_technique_star_unique_match import StarUniqueMatchNav


def _hyp(
    star_index: int,
    offset_vu: tuple[float, float],
    peak: float,
    peak_per_snr: float | None,
    *,
    star_like: bool = True,
    refutation: str | None = None,
) -> LoneStarHypothesis:
    """Build a hypothesis whose source sits at its offset from a prediction at the origin.

    Parameters:
        star_index: Index of the star among the searched stars.
        offset_vu: The hypothesis's offset.
        peak: The source's peak.
        peak_per_snr: The peak per unit of the star's predicted SNR.
        star_like: Whether the source is no sharper than a star.
        refutation: Why the other stars refute the offset, or None.

    Returns:
        The hypothesis.
    """
    return LoneStarHypothesis(
        star_index=star_index,
        source_vu=offset_vu,
        offset_vu=offset_vu,
        peak=peak,
        peak_per_snr=peak_per_snr,
        star_like=star_like,
        refutation=refutation,
    )


def _choose(hypotheses: list[LoneStarHypothesis]) -> tuple[LoneStarHypothesis | None, str]:
    """Choose with the shipped tolerance (2.4 px), runner-up ratio 1.5, and rival factor 4."""
    choice = choose_lone_star(hypotheses, tolerance_px=2.4, runner_up_ratio=1.5, rival_factor=4.0)
    return choice.best, choice.reason or ''


def test_the_survivor_brightest_for_its_predicted_snr_is_chosen() -> None:
    """Of two stars' survivors, the one brightest per unit of predicted SNR wins.

    Fails if the stronger source or the brighter star wins instead: the first star's
    source is the stronger, the second's the brighter for its star.
    """
    best, _ = _choose([_hyp(0, (0.0, 0.0), 100.0, 2.0), _hyp(1, (30.0, 0.0), 90.0, 9.0)])
    assert best is not None
    assert best.star_index == 1


def test_a_refuted_hypothesis_is_not_chosen() -> None:
    """A source the other stars refute is passed over for a weaker one.

    Fails if refutation is ignored: the refuted source is the stronger.
    """
    best, _ = _choose(
        [
            _hyp(0, (0.0, 0.0), 100.0, 10.0, refutation='1 other star(s) show nothing'),
            _hyp(1, (30.0, 0.0), 20.0, 1.0),
        ]
    )
    assert best is not None
    assert best.offset_vu == (30.0, 0.0)


def test_a_source_sharper_than_a_star_is_not_chosen() -> None:
    """A source sharper than a star is no lone star, and naming it explains the refusal.

    Fails if sharpness is ignored.
    """
    best, reason = _choose([_hyp(0, (0.0, 0.0), 100.0, 10.0, star_like=False)])
    assert best is None
    assert 'no_lone_star_candidate: 1 source(s), 1 sharper or brighter than a star' in reason


def test_a_refuted_source_nearly_as_bright_in_the_window_makes_it_ambiguous() -> None:
    """Every source of the chosen star's window competes, refuted or not.

    Fails if only survivors compete: the rival is refuted, yet a window holding two
    sources this close in brightness does not say which is the star.
    """
    best, reason = _choose(
        [
            _hyp(0, (0.0, 0.0), 100.0, 10.0),
            _hyp(0, (20.0, 0.0), 90.0, 9.0, refutation='1 other star(s) show nothing'),
        ]
    )
    assert best is None
    assert 'ambiguous_detection' in reason


@pytest.mark.parametrize(
    ('rival_peak_per_snr', 'ambiguous'), [(3.0, True), (2.5, True), (2.0, False)]
)
def test_another_stars_survivor_is_a_rival_only_within_the_factor(
    rival_peak_per_snr: float, ambiguous: bool
) -> None:
    """Another star's survivor elsewhere makes the choice ambiguous within a factor of 4.

    Fails if every other star's survivor is a rival (the 2.0 case), if none is (3.0), or if
    one exactly at the factor is not (2.5).
    """
    best, _ = _choose(
        [_hyp(0, (0.0, 0.0), 100.0, 10.0), _hyp(1, (30.0, 0.0), 100.0, rival_peak_per_snr)]
    )
    assert (best is None) is ambiguous


def test_the_same_source_twice_is_no_runner_up() -> None:
    """A second source within the tolerance of the answer is the same answer.

    Fails if the runner-up ratio ignores the tolerance: the two are nearly as bright.
    """
    best, _ = _choose([_hyp(0, (0.0, 0.0), 100.0, 10.0), _hyp(0, (1.0, 0.0), 95.0, 9.5)])
    assert best is not None


def test_a_brighter_source_sharper_than_a_star_still_makes_the_window_ambiguous() -> None:
    """Every source of the window competes, including one too sharp to be the star.

    Fails if only star-like sources compete.
    """
    best, reason = _choose(
        [_hyp(0, (0.0, 0.0), 100.0, 10.0), _hyp(0, (20.0, 0.0), 300.0, 30.0, star_like=False)]
    )
    assert best is None
    assert 'ambiguous_detection' in reason


def test_another_stars_survivor_at_the_same_offset_is_no_rival() -> None:
    """Another star's survivor at the answer's offset agrees with it.

    Fails if the rival test ignores where the other survivor puts the offset.
    """
    best, _ = _choose([_hyp(0, (0.0, 0.0), 100.0, 10.0), _hyp(1, (1.0, 0.0), 90.0, 9.0)])
    assert best is not None


def test_a_survivor_of_unknown_brightness_ranks_last() -> None:
    """A star that cannot say how bright it should be yields to one that can.

    The two agree on the offset, so neither is a rival.  Fails if unknown brightness
    ranks first.
    """
    best, _ = _choose([_hyp(1, (0.0, 0.0), 100.0, None), _hyp(0, (1.0, 0.0), 90.0, 9.0)])
    assert best is not None
    assert best.star_index == 0


def test_a_survivor_of_unknown_brightness_elsewhere_is_a_rival() -> None:
    """Another star's survivor of unknown brightness elsewhere cannot be ranked against.

    Fails if a survivor of unknown brightness is never a rival.
    """
    best, reason = _choose([_hyp(0, (0.0, 0.0), 100.0, 10.0), _hyp(1, (30.0, 0.0), 1.0, None)])
    assert best is None
    assert 'ambiguous_lone_stars' in reason


def test_lone_star_comes_from_a_fainter_star_when_the_brightest_is_off_the_image(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A fainter star is matched alone when the brightest one shows nothing in the image.

    The brightest star is predicted in the padding, where its window holds nothing, and
    the offset that puts the second star on its source leaves the brightest outside the
    image.  Fails if only the brightest star is searched for alone.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    draw_gaussian_star(image, (120.0, 120.0), peak_dn=200.0, sigma=1.2)
    bright = make_star_feature('star:UCAC4:A', predicted_vu=(5.0, 100.0), predicted_snr=60.0)
    faint = make_star_feature('star:UCAC4:B', predicted_vu=(119.0, 118.0), predicted_snr=20.0)
    result = StarUniqueMatchNav().navigate([bright, faint], make_nav_context(image))
    assert result.feature_ids == ('star:UCAC4:B',)
    assert result.offset_px[0] == pytest.approx(1.0, abs=0.3)
    assert result.offset_px[1] == pytest.approx(2.0, abs=0.3)


def test_lone_star_offset_is_refuted_by_a_star_absent_on_clear_sky(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A second star the offset puts on clear sky in the image, showing nothing, refutes it.

    The second star is nearly as bright as the first, so the first's source says how
    bright it should be; it is predicted where its own window cannot hold that source.
    Fails if absent stars are not counted.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    draw_gaussian_star(image, (101.0, 100.0), peak_dn=200.0, sigma=1.2)
    lone = make_star_feature('star:UCAC4:A', predicted_vu=(99.0, 103.0), predicted_snr=40.0)
    absent = make_star_feature('star:UCAC4:B', predicted_vu=(50.0, 150.0), predicted_snr=30.0)
    result = StarUniqueMatchNav().navigate([lone, absent], make_nav_context(image))
    assert result.spurious is True
    assert '1 refuted; 1 other star(s) show nothing' in capsys.readouterr().out


def test_lone_star_refuses_a_source_a_comparable_star_could_be(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A star nearly as bright whose window holds the source could be it, so it is refused.

    Only the brightest star is searched, so the second is no hypothesis of its own; at
    the first star's offset it falls in the padding, where nothing can refute it.  Fails
    if no star outside the searched ones is asked whether it could be the source.
    """
    tuning = dict(StarUniqueMatchNav.tuning)
    tuning['pair_search_stars'] = 1
    monkeypatch.setattr(StarUniqueMatchNav, 'tuning', tuning)
    image = np.zeros((200, 200), dtype=np.float64)
    draw_gaussian_star(image, (40.0, 100.0), peak_dn=200.0, sigma=1.2)
    first = make_star_feature('star:UCAC4:A', predicted_vu=(39.0, 99.0), predicted_snr=20.0)
    second = make_star_feature('star:UCAC4:B', predicted_vu=(19.0, 99.0), predicted_snr=18.0)
    result = StarUniqueMatchNav().navigate([first, second], make_nav_context(image))
    assert result.spurious is True
    assert 'ambiguous_identity: star:UCAC4:B' in capsys.readouterr().out


def test_lone_star_refuses_a_source_sharper_than_a_star(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A source 7.5 times brighter than its neighbors is no lone star.

    The noise allowance of max_peak_sharpness lets it through as a source; the lone-star
    limit of 7 has none.  Fails if every source is taken as star-like.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    image[100, 100] = 75.0
    for dv, du in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        image[100 + dv, 100 + du] = 10.0
    feature = make_star_feature('star:UCAC4:A', predicted_vu=(98.0, 101.0), predicted_snr=40.0)
    result = StarUniqueMatchNav().navigate([feature], make_nav_context(image))
    assert result.spurious is True
    assert '1 sharper or brighter than a star' in capsys.readouterr().out


def test_star_unique_match_rejects_a_lone_star_sharpness_below_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction rejects a ``one_star_max_peak_sharpness`` below 1."""
    tuning = dict(StarUniqueMatchNav.tuning)
    tuning['one_star_max_peak_sharpness'] = 0.5
    monkeypatch.setattr(StarUniqueMatchNav, 'tuning', tuning)
    with pytest.raises(ValueError, match='one_star_max_peak_sharpness must be finite and >= 1'):
        StarUniqueMatchNav()


def test_lone_star_is_found_beside_a_brighter_ridge(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
) -> None:
    """A brighter ridge in the window is no star and does not hide the lone star.

    The window's strongest peak lies on the ridge, which is not an isolated point source;
    the star is the window's one isolated source.  Fails if the 1-star path measures only
    the window's strongest peak, as head 7617c81e did.
    """
    image = np.zeros((200, 200), dtype=np.float64)
    for step in np.linspace(0.0, 20.0, 41):
        draw_gaussian_star(image, (80.0 + step, 120.0), peak_dn=400.0, sigma=1.0)
    draw_gaussian_star(image, (102.0, 97.0), peak_dn=150.0, sigma=1.2)
    feature = make_star_feature('star:UCAC4:A', predicted_vu=(100.0, 100.0), predicted_snr=40.0)
    result = StarUniqueMatchNav().navigate([feature], make_nav_context(image))
    assert result.spurious is False
    assert result.offset_px[0] == pytest.approx(2.0, abs=0.3)
    assert result.offset_px[1] == pytest.approx(-3.0, abs=0.3)


@pytest.mark.parametrize(('bound', 'spurious'), [(4.0, True), (6.0, False)])
def test_lone_star_refuses_a_source_brighter_than_the_camera_bound(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    bound: float,
    spurious: bool,
) -> None:
    """A source brighter per unit of predicted SNR than the camera's bound is no lone star.

    The source peaks near 5 noise sigmas per unit of the star's predicted SNR.  Fails if the
    camera's ``star_max_peak_per_predicted_snr`` is ignored (the 4.0 case) or applied the
    wrong way (6.0).
    """
    image = np.zeros((200, 200), dtype=np.float64)
    draw_gaussian_star(image, (101.0, 102.0), peak_dn=200.0, sigma=1.2)
    feature = make_star_feature('star:UCAC4:A', predicted_vu=(100.0, 100.0), predicted_snr=40.0)
    context = make_nav_context(image)
    context.obs.inst_config['star_max_peak_per_predicted_snr'] = bound  # type: ignore[attr-defined]
    result = StarUniqueMatchNav().navigate([feature], context)
    assert result.spurious is spurious
