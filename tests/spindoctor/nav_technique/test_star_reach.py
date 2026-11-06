"""Tests for the star reach mask and its use by ``StarFieldFromCatalogNav``."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import (
    DrawGaussianStarFactory,
    NavContextFactory,
    NavFeatureFactory,
)

from spindoctor.feature.feature import NavFeature
from spindoctor.nav_technique._star_reach import reach_mask
from spindoctor.nav_technique.diagnostics import StarFieldDiagnostics
from spindoctor.nav_technique.nav_technique_star_field import (
    StarFieldFromCatalogNav,
    _source_area_px2,
    _wide_offset_false_lock_expectation,
)

# ---------------------------------------------------------------------------
# reach_mask
# ---------------------------------------------------------------------------


def test_reach_mask_bounds_are_inclusive() -> None:
    """A whole-pixel prediction reaches exactly its half-widths on each side."""
    mask = reach_mask((40, 50), [(10.0, 20.0)], reach_vu=(2.0, 3.0))
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.tolist() == [8, 9, 10, 11, 12]
    assert cols.tolist() == [17, 18, 19, 20, 21, 22, 23]
    assert int(mask.sum()) == 5 * 7


def test_reach_mask_rounds_a_fractional_prediction_inward() -> None:
    """Only whole pixels within the half-widths of a fractional prediction are marked."""
    mask = reach_mask((40, 50), [(10.4, 20.6)], reach_vu=(1.0, 1.0))
    assert np.flatnonzero(mask.any(axis=1)).tolist() == [10, 11]
    assert np.flatnonzero(mask.any(axis=0)).tolist() == [20, 21]


def test_reach_mask_clips_a_prediction_off_the_image() -> None:
    """A prediction off the image marks only the part of its rectangle on it."""
    mask = reach_mask((40, 50), [(-2.0, 25.0), (100.0, 25.0)], reach_vu=(3.0, 1.0))
    assert np.flatnonzero(mask.any(axis=1)).tolist() == [0, 1]
    assert int(mask.sum()) == 2 * 3


def test_reach_mask_unions_the_predictions() -> None:
    """Overlapping rectangles mark their union once."""
    mask = reach_mask((40, 50), [(10.0, 10.0), (10.0, 12.0)], reach_vu=(1.0, 1.0))
    assert int(mask.sum()) == 3 * 5


def test_reach_mask_without_predictions_is_empty() -> None:
    """No prediction reaches any pixel."""
    assert not reach_mask((10, 10), [], reach_vu=(5.0, 5.0)).any()


@pytest.mark.parametrize('reach_vu', [(-1.0, 2.0), (2.0, math.nan), (math.inf, 2.0)])
def test_reach_mask_rejects_a_bad_half_width(reach_vu: tuple[float, float]) -> None:
    """A negative or non-finite half-width raises.

    Parameters:
        reach_vu: The half-widths passed.
    """
    with pytest.raises(ValueError, match='reach_vu must be finite and >= 0'):
        reach_mask((10, 10), [(5.0, 5.0)], reach_vu=reach_vu)


# ---------------------------------------------------------------------------
# StarFieldFromCatalogNav source cohort
# ---------------------------------------------------------------------------

_STAR_CENTERS = [
    (60.0, 80.0),
    (110.0, 130.0),
    (180.0, 90.0),
    (220.0, 200.0),
    (90.0, 240.0),
    (250.0, 70.0),
]

_MARGIN_VU = (6, 12)
"""Search window half-widths; with the 2 px inlier tolerance and 1 px of slack a source
counts within 9 px of a prediction along v and 15 px along u."""


def _features(
    centers: list[tuple[float, float]],
    offset: tuple[float, float],
    *,
    make_feature: NavFeatureFactory,
    snrs: list[float] | None = None,
) -> list[NavFeature]:
    """Return one STAR feature per center, predicted at ``center - offset``.

    Parameters:
        centers: Where the stars are in the image, ``(v, u)``.
        offset: The pointing offset ``(dv, du)`` the technique should recover.
        make_feature: Builds one STAR feature.
        snrs: Predicted SNR per star; a slowly falling default when None.

    Returns:
        The features, in the order of ``centers``.
    """
    if snrs is None:
        snrs = [40.0 - 0.5 * i for i in range(len(centers))]
    return [
        make_feature(
            f'star:UCAC4:{i}', predicted_vu=(cv - offset[0], cu - offset[1]), predicted_snr=snr
        )
        for i, ((cv, cu), snr) in enumerate(zip(centers, snrs, strict=True))
    ]


def _bright_sources_beyond_reach(
    image: np.ndarray, predictions: list[tuple[float, float]], *, draw: DrawGaussianStarFactory
) -> int:
    """Plant bright point sources on a 30 px grid wherever no prediction reaches.

    A source is planted only where it is more than 12 px along v or 18 px along u from
    every prediction, 3 px beyond the reach.

    Parameters:
        image: The frame, changed in place.
        predictions: Predicted star positions ``(v, u)``.
        draw: Stamps a Gaussian source onto a frame in place.

    Returns:
        The number of sources planted.
    """
    n = 0
    for v in range(15, image.shape[0] - 15, 30):
        for u in range(15, image.shape[1] - 15, 30):
            if all(abs(v - pv) > 12.0 or abs(u - pu) > 18.0 for pv, pu in predictions):
                draw(image, (float(v), float(u)), peak_dn=1000.0, sigma=1.2)
                n += 1
    return n


def test_star_field_matches_faint_stars_outshone_by_sources_beyond_reach(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Sources no star can reach do not crowd the faint stars out of the cohort.

    More bright sources than the cohort holds lie where no predicted star can be at any
    offset the window allows; counted, they would fill the cohort and leave the stars
    unmatched.
    """
    planted = (3.0, -2.5)
    image = np.zeros((320, 320), dtype=np.float64)
    for center in _STAR_CENTERS:
        draw_gaussian_star(image, center, peak_dn=200.0, sigma=1.2)
    features = _features(_STAR_CENTERS, planted, make_feature=make_star_feature)
    predictions = [(cv - planted[0], cu - planted[1]) for cv, cu in _STAR_CENTERS]
    technique = StarFieldFromCatalogNav()
    n_bright = _bright_sources_beyond_reach(image, predictions, draw=draw_gaussian_star)
    assert n_bright > technique._max_sources
    result = technique.navigate(features, make_nav_context(image, extfov_margin_vu=_MARGIN_VU))
    assert result.spurious is False
    assert result.offset_px[0] == pytest.approx(planted[0], abs=0.1)
    assert result.offset_px[1] == pytest.approx(planted[1], abs=0.1)
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    assert result.diagnostics.n_detected_sources == len(_STAR_CENTERS)
    assert f'Set aside {n_bright} source(s) beyond the reach' in capsys.readouterr().out


@pytest.mark.parametrize(
    ('distance_vu', 'counted'),
    [((9.0, 0.0), True), ((10.0, 0.0), False), ((0.0, 15.0), True), ((0.0, 16.0), False)],
)
def test_star_field_counts_a_source_only_within_reach(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    distance_vu: tuple[float, float],
    counted: bool,
) -> None:
    """A source counts within each axis's window half-width plus 3 px of a prediction.

    Parameters:
        distance_vu: Where the extra source lies from the nearest prediction, ``(v, u)``.
        counted: Whether the source should be among the detections.
    """
    image = np.zeros((320, 320), dtype=np.float64)
    for center in _STAR_CENTERS:
        draw_gaussian_star(image, center, peak_dn=200.0, sigma=1.2)
    draw_gaussian_star(
        image, (280.0 + distance_vu[0], 280.0 + distance_vu[1]), peak_dn=200.0, sigma=1.2
    )
    centers = [*_STAR_CENTERS, (280.0, 280.0)]
    # The seventh star is predicted at (280, 280) but is not in the image.
    features = _features(centers, (0.0, 0.0), make_feature=make_star_feature)
    result = StarFieldFromCatalogNav().navigate(
        features, make_nav_context(image, extfov_margin_vu=_MARGIN_VU)
    )
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    expected = len(_STAR_CENTERS) + (1 if counted else 0)
    assert result.diagnostics.n_detected_sources == expected


def test_source_area_is_the_reach_in_a_crowded_frame() -> None:
    """A frame denser in sources than the reach leaves the reach as the area."""
    area = _source_area_px2(
        n_kept=3, n_candidates=10000, reach_area_px2=2000.0, image_area_px2=1.0e6
    )
    assert area == pytest.approx(2000.0)


def test_source_area_spreads_a_few_kept_sources_at_the_frame_density() -> None:
    """A frame of few sources spreads the kept ones at its density over more than the reach."""
    area = _source_area_px2(
        n_kept=3, n_candidates=1000, reach_area_px2=2000.0, image_area_px2=1.0e6
    )
    assert area == pytest.approx(3000.0)


def test_source_area_spreads_the_kept_sources_at_the_frame_density() -> None:
    """A frame sparser than the reach spreads the kept sources at the frame's density."""
    area = _source_area_px2(n_kept=3, n_candidates=30, reach_area_px2=2000.0, image_area_px2=1.0e6)
    assert area == pytest.approx(1.0e5)


def test_source_area_without_candidates_is_the_reach() -> None:
    """With no candidate anywhere the reach is the area."""
    area = _source_area_px2(n_kept=0, n_candidates=0, reach_area_px2=2000.0, image_area_px2=1.0e6)
    assert area == pytest.approx(2000.0)


def _three_star_wide_field(
    draw: DrawGaussianStarFactory,
    make_feature: NavFeatureFactory,
    *,
    n_other: int,
    pad_px: int,
) -> tuple[np.ndarray, list[NavFeature], list[tuple[float, float]]]:
    """Return a three-star frame, its features, and the predictions.

    The stars are far apart, so only their own pairings give an offset inside the window.
    ``n_other`` bright sources are planted beyond the reach and off a border ``pad_px``
    wide, the first ``n_other`` points of a 20 px grid that lie there.

    Parameters:
        draw: Stamps a Gaussian source onto a frame in place.
        make_feature: Builds one STAR feature.
        n_other: Number of sources to plant beyond the reach.
        pad_px: Width of the border that holds no source.

    Returns:
        The image, the features, and the predicted positions ``(v, u)``.
    """
    centers = [(60.0, 60.0), (200.0, 300.0), (330.0, 120.0)]
    planted = (2.0, -3.0)
    image = np.zeros((400, 400), dtype=np.float64)
    for center in centers:
        draw(image, center, peak_dn=400.0, sigma=1.2)
    predictions = [(cv - planted[0], cu - planted[1]) for cv, cu in centers]
    n = 0
    for v in range(pad_px + 10, 400 - pad_px - 5, 20):
        for u in range(pad_px + 10, 400 - pad_px - 5, 20):
            if n < n_other and all(
                abs(v - pv) > 12.0 or abs(u - pu) > 18.0 for pv, pu in predictions
            ):
                draw(image, (float(v), float(u)), peak_dn=200.0, sigma=1.2)
                n += 1
    assert n == n_other
    features = _features(centers, planted, make_feature=make_feature, snrs=[400.0, 120.0, 40.0])
    return image, features, predictions


@pytest.mark.parametrize(('n_other', 'pad_px'), [(0, 0), (40, 0), (0, 52), (200, 52)])
def test_wide_offset_density_is_the_lower_of_reach_and_frame(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
    draw_gaussian_star: DrawGaussianStarFactory,
    n_other: int,
    pad_px: int,
) -> None:
    """The false-lock model spreads the three detections at the lower of two densities.

    Three far-apart stars give one in-window seed, so the recorded expectation is the
    model's value for three inliers, one seed, three catalog stars, and three detections
    over the area :func:`_source_area_px2` gives for the frame's candidates.  Both areas
    count sensor pixels only: a border marked off the sensor, as the extended frame's
    padding is, counts in neither, and it cuts into the first star's reach.

    Parameters:
        n_other: Sources planted beyond the reach, which count only in the frame density;
            200 make the frame denser than the reach.
        pad_px: Width of the border off the sensor.
    """
    image, features, predictions = _three_star_wide_field(
        draw_gaussian_star, make_star_feature, n_other=n_other, pad_px=pad_px
    )
    sensor = np.zeros(image.shape, dtype=np.bool_)
    sensor[pad_px : image.shape[0] - pad_px, pad_px : image.shape[1] - pad_px] = True
    context = replace(make_nav_context(image, extfov_margin_vu=_MARGIN_VU), sensor_mask_ext=sensor)
    technique = StarFieldFromCatalogNav()
    result = technique.navigate(features, context)
    assert isinstance(result.diagnostics, StarFieldDiagnostics)
    assert result.diagnostics.wide_offset_lock is True
    slack = technique._inlier_tolerance_px + 1.0
    reach = reach_mask(
        image.shape,
        predictions,
        reach_vu=(_MARGIN_VU[0] + slack, _MARGIN_VU[1] + slack),
    )
    area = _source_area_px2(
        n_kept=3,
        n_candidates=3 + n_other,
        reach_area_px2=float((reach & sensor).sum()),
        image_area_px2=float(sensor.sum()),
    )
    expected = _wide_offset_false_lock_expectation(
        n_inliers=3,
        n_seeds=1,
        n_catalog=3,
        n_detected=3,
        tolerance_px=technique._inlier_tolerance_px,
        image_area_px2=area,
    )
    assert result.diagnostics.wide_offset_false_lock_expectation == pytest.approx(expected)
