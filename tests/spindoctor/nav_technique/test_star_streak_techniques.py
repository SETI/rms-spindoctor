"""Each star technique measures the stars of a streaked frame at their streaks' centers."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from psfmodel import GaussianPSF
from tests.spindoctor.nav_technique.conftest import FakeObs, NavContextFactory, NavFeatureFactory
from tests.spindoctor.star_streak_images import draw_streak

from spindoctor.feature.feature import NavFeature
from spindoctor.nav_technique.diagnostics import StarUniqueMatchDiagnostics
from spindoctor.nav_technique.nav_technique import NavTechnique
from spindoctor.nav_technique.nav_technique_star_field import StarFieldFromCatalogNav
from spindoctor.nav_technique.nav_technique_star_refine import StarRefineNav
from spindoctor.nav_technique.nav_technique_star_unique_match import (
    StarUniqueMatchNav,
    _found_at,
)

_STREAK_VU = (8.0, -1.0)
"""The streak every planted star draws."""

_TRUE_OFFSET = (3.0, -2.0)
"""Planted ``(dv, du)`` between the predictions and the streaks' centers."""

_CENTERS = [
    (52.3, 61.6),
    (148.7, 83.2),
    (97.1, 171.4),
    (171.6, 152.9),
    (63.4, 140.8),
    (128.2, 41.7),
]
"""Centers of the planted streaks, spread so any three form a usable triangle."""


class _PsfObs(FakeObs):
    """Observation stand-in that also supplies the star point-spread function."""

    def star_psf(self) -> GaussianPSF:
        """Return a Gaussian star PSF of sigma 0.6 px."""
        return GaussianPSF(sigma=0.6)


def _streak_frame(
    make_star_feature: NavFeatureFactory, *, flux: float
) -> tuple[np.ndarray, list[NavFeature]]:
    """Return a noisy frame of planted streaks and the STAR features predicting them.

    Parameters:
        make_star_feature: The STAR feature factory.
        flux: Light of the brightest streak; each later one has 10% less of it.

    Returns:
        The image and the features, brightest first.
    """
    image = np.random.default_rng(765).normal(0.0, 1.0, size=(220, 220))
    features = []
    for i, (cv, cu) in enumerate(_CENTERS):
        draw_streak(image, (cv, cu), _STREAK_VU, flux=flux * (1.0 - 0.1 * i))
        features.append(
            make_star_feature(
                f'star:UCAC4:{i}',
                predicted_vu=(cv - _TRUE_OFFSET[0], cu - _TRUE_OFFSET[1]),
                predicted_snr=60.0 - 6.0 * i,
                streak_vu=_STREAK_VU,
            )
        )
    return image, features


@pytest.mark.parametrize(
    ('technique', 'prior'),
    [
        (StarFieldFromCatalogNav(), None),
        (StarUniqueMatchNav(), None),
        (StarRefineNav(), (3.4, -2.4)),
    ],
    ids=['star_field', 'unique_match', 'refine'],
)
def test_streaked_stars_give_the_true_offset(
    technique: NavTechnique,
    prior: tuple[float, float] | None,
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
) -> None:
    """On a frame of noisy 8 px streaks each technique recovers the planted offset."""
    image, features = _streak_frame(make_star_feature, flux=4000.0)
    context = make_nav_context(image)
    if prior is not None:
        context = context.with_prior(offset_px=prior, covariance_px2=np.eye(2))
    result = technique.navigate(features, context)
    assert result.offset_px[0] == pytest.approx(_TRUE_OFFSET[0], abs=0.15)
    assert result.offset_px[1] == pytest.approx(_TRUE_OFFSET[1], abs=0.15)


def test_star_field_keeps_streak_centroids_out_of_the_psf_refit(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
) -> None:
    """Faint streaks keep their moment centroids though a star PSF is on hand to refit."""
    image, features = _streak_frame(make_star_feature, flux=400.0)
    context = dataclasses.replace(make_nav_context(image), obs=_PsfObs(extfov_margin_vu=(32, 32)))
    result = StarFieldFromCatalogNav().navigate(features, context)
    assert result.offset_px[0] == pytest.approx(_TRUE_OFFSET[0], abs=0.15)
    assert result.offset_px[1] == pytest.approx(_TRUE_OFFSET[1], abs=0.15)


def test_unique_match_keeps_a_long_streak_out_of_its_own_runner_up(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
) -> None:
    """A lone 14 px streak is not its own runner-up in the one-star peak ratio.

    The streak reaches 7 px from its center, past the exclusion square an unsmeared
    star needs, so unless the square grows by the streak's half extent the streak's
    far pixels rival its peak and the detection reads as ambiguous.
    """
    streak_vu = (14.0, 0.0)
    image = np.random.default_rng(768).normal(0.0, 1.0, size=(200, 200))
    draw_streak(image, (100.0, 100.0), streak_vu, flux=3000.0)
    feature = make_star_feature(
        'star:UCAC4:lone', predicted_vu=(98.0, 103.0), predicted_snr=40.0, streak_vu=streak_vu
    )
    result = StarUniqueMatchNav().navigate([feature], make_nav_context(image))
    assert result.spurious is False
    assert result.offset_px[0] == pytest.approx(2.0, abs=0.15)
    assert result.offset_px[1] == pytest.approx(-3.0, abs=0.15)


def test_unique_match_still_counts_a_rival_beside_a_long_streak(
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
) -> None:
    """A star 9 px across a 14 px streak still rivals it in the one-star peak ratio.

    The exclusion box grows only along the streak, so a rival beside it, as bright as
    the streak's own pixels, keeps the detection ambiguous.
    """
    streak_vu = (14.0, 0.0)
    image = np.random.default_rng(768).normal(0.0, 1.0, size=(200, 200))
    draw_streak(image, (100.0, 100.0), streak_vu, flux=3000.0)
    draw_streak(image, (100.0, 109.0), (0.0, 0.0), flux=280.0)
    feature = make_star_feature(
        'star:UCAC4:lone', predicted_vu=(98.0, 103.0), predicted_snr=40.0, streak_vu=streak_vu
    )
    result = StarUniqueMatchNav().navigate([feature], make_nav_context(image))
    assert result.spurious is True
    assert isinstance(result.diagnostics, StarUniqueMatchDiagnostics)
    assert result.diagnostics.detection_peak_ratio < 1.5


def test_corroboration_finds_a_long_streak_at_its_center() -> None:
    """A 14 px streak centered on a corroboration target is found there.

    The brightest pixel lands anywhere along the streak, so a box around it would put
    the centroid up to half the box's reach from the center; the streak centroid is
    taken about the streak's center instead.
    """
    streak_vu = (14.0, 0.0)
    image = np.random.default_rng(1).normal(0.0, 1.0, size=(120, 120))
    draw_streak(image, (60.0, 60.0), streak_vu, flux=3000.0)
    found = _found_at(
        image,
        (60.0, 60.0),
        tolerance_px=1.5,
        box_half_px=3,
        noise_sigma=1.0,
        detection_sigma=4.0,
        streak_vu=streak_vu,
    )
    assert found is True
