"""The ensemble on strongly anisotropic, oblique covariances.

``TitanHazeNav`` reports an error ellipse that is about 0.36 px across the symmetry axis
and 1.02 px along it, rotated into ``(v, u)`` by the symmetry-axis angle.  These tests
pin what the ensemble does with such an ellipse, each against a closed-form answer: the
precision-weighted merge keeps its orientation, the agreement test measures a
disagreement in its metric, and the confidence tier reads the fused ellipse's major
semi-axis, so the tier does not depend on the ellipse's tilt.
"""

import math

import numpy as np
import pytest
from tests.packaged_sections import packaged_ensemble

from spindoctor.nav_orchestrator.ensemble import _combine_precision_weighted, ensemble
from spindoctor.nav_orchestrator.ensemble_consensus import consensus_selection
from spindoctor.nav_orchestrator.ensemble_observability import (
    mahalanobis_distance,
    major_semi_axis_px,
)
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.diagnostics import BodyLimbDiagnostics, StarFieldDiagnostics
from spindoctor.nav_technique.nav_technique import embed_rotation_unobservable
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.types import NDArrayFloatType

#: The haze fit's typical sigma across the symmetry axis (px).
_SIGMA_CROSS_PX = 0.36
#: The haze fit's typical sigma along the symmetry axis (px).
_SIGMA_ALONG_PX = 1.02
#: An oblique symmetry-axis angle in ``(v, u)`` (degrees).
_THETA_DEG = 30.0
#: Distinct stars behind the star-field results, enough that no star-count cap applies.
_STARS = ('star:a', 'star:b', 'star:c', 'star:d', 'star:e')


def _axes(theta_deg: float) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Return the along-axis and cross-axis unit vectors in ``(v, u)``."""
    theta = math.radians(theta_deg)
    along = np.array([math.cos(theta), math.sin(theta)])
    cross = np.array([-math.sin(theta), math.cos(theta)])
    return along, cross


def _ellipse(theta_deg: float, sigma_along_px: float, sigma_cross_px: float) -> NDArrayFloatType:
    """Return the ``(v, u)`` covariance of an ellipse whose major axis lies at ``theta_deg``."""
    along, cross = _axes(theta_deg)
    cov = sigma_along_px**2 * np.outer(along, along) + sigma_cross_px**2 * np.outer(cross, cross)
    symmetric: NDArrayFloatType = 0.5 * (cov + cov.T)
    return symmetric


def _haze_cov(theta_deg: float = _THETA_DEG) -> NDArrayFloatType:
    """Return the haze fit's typical covariance with its symmetry axis at ``theta_deg``."""
    return _ellipse(theta_deg, _SIGMA_ALONG_PX, _SIGMA_CROSS_PX)


def _result(
    technique_name: str,
    offset: NDArrayFloatType | tuple[float, float],
    cov: NDArrayFloatType,
    *,
    confidence: float = 0.8,
    rotation_rad: float | None = None,
) -> NavTechniqueResult:
    """Build a result; a star-field result rests on five stars, any other on one feature."""
    is_star = technique_name == 'StarFieldFromCatalogNav'
    return NavTechniqueResult(
        technique_name=technique_name,
        feature_ids=_STARS if is_star else (f'{technique_name}:f1',),
        offset_px=(float(offset[0]), float(offset[1])),
        covariance_px2=cov,
        confidence=confidence,
        spurious=False,
        at_edge=False,
        diagnostics=(
            StarFieldDiagnostics(n_inliers=len(_STARS)) if is_star else BodyLimbDiagnostics()
        ),
        rotation_rad=rotation_rad,
    )


def _ensemble(results: list[NavTechniqueResult]) -> str:
    """Run the packaged ensemble and return the confidence rank it assigns."""
    nav = ensemble(
        results,
        feature_inventory=[],
        image_classifier=NavImageClassifierResult(
            image_class='clean',
            saturation_frac=0.0,
            missing_frac=0.0,
            unread_line_frac=0.0,
            noise_sigma=1.0,
            max_dn=10.0,
        ),
        provenance=Provenance(
            spindoctor_version='0.5.2',
            image_et=0.0,
            pipeline_run_iso8601='2026-04-26T12:00:00Z',
        ),
        config=packaged_ensemble(),
    )
    return nav.confidence_rank


def _combine(results: list[NavTechniqueResult]) -> tuple[NDArrayFloatType, NDArrayFloatType]:
    """Return the fused ``(dv, du)`` offset and translation covariance."""
    config = packaged_ensemble()
    combined = _combine_precision_weighted(
        results,
        rcond=config.pinvh_rcond,
        max_allowed_rotation_deg=config.max_allowed_rotation_deg,
    )
    return np.array(combined.offset_px), combined.covariance_px2[:2, :2]


# ---------------------------------------------------------------------------
# The precision-weighted merge
# ---------------------------------------------------------------------------


def test_merge_with_a_circle_keeps_the_ellipse_orientation() -> None:
    """An oblique ellipse fused with a circle keeps its axes; each axis adds information."""
    sigma_star = 0.5
    haze = _result('TitanHazeNav', (0.0, 0.0), _haze_cov())
    star = _result('StarFieldFromCatalogNav', (0.0, 0.0), sigma_star**2 * np.eye(2))
    _, cov = _combine([haze, star])
    expected = _ellipse(
        _THETA_DEG,
        1.0 / math.sqrt(_SIGMA_ALONG_PX**-2 + sigma_star**-2),
        1.0 / math.sqrt(_SIGMA_CROSS_PX**-2 + sigma_star**-2),
    )
    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=1e-15)


def test_merge_moves_each_axis_by_its_own_precision_share() -> None:
    """The fused offset moves toward the ellipse by its information share along each axis."""
    sigma_star = 0.5
    along, cross = _axes(_THETA_DEG)
    haze = _result('TitanHazeNav', along + cross, _haze_cov())
    star = _result('StarFieldFromCatalogNav', (0.0, 0.0), sigma_star**2 * np.eye(2))
    offset, _ = _combine([haze, star])
    share_along = _SIGMA_ALONG_PX**-2 / (_SIGMA_ALONG_PX**-2 + sigma_star**-2)
    share_cross = _SIGMA_CROSS_PX**-2 / (_SIGMA_CROSS_PX**-2 + sigma_star**-2)
    np.testing.assert_allclose(offset, share_along * along + share_cross * cross, rtol=1e-12)


def test_merge_of_crossed_ellipses_is_a_circle() -> None:
    """Two equal ellipses at right angles fuse to a circle: their cross terms cancel."""
    first = _result('TitanHazeNav', (0.0, 0.0), _haze_cov(_THETA_DEG))
    second = _result('TechniqueB', (0.0, 0.0), _haze_cov(_THETA_DEG + 90.0))
    _, cov = _combine([first, second])
    variance = 1.0 / (_SIGMA_ALONG_PX**-2 + _SIGMA_CROSS_PX**-2)
    np.testing.assert_allclose(cov, variance * np.eye(2), rtol=1e-12, atol=1e-15)


def test_merge_keeps_the_ellipse_orientation_beside_an_unobservable_rotation() -> None:
    """A rotation the haze fit cannot observe leaves its oblique translation block intact."""
    sigma_star = 0.5
    haze = _result(
        'TitanHazeNav', (0.0, 0.0), embed_rotation_unobservable(_haze_cov()), rotation_rad=0.0
    )
    star_cov = np.diag([sigma_star**2, sigma_star**2, 1.0e-6])
    star = _result('StarFieldFromCatalogNav', (0.0, 0.0), star_cov, rotation_rad=0.0)
    _, cov = _combine([haze, star])
    expected = _ellipse(
        _THETA_DEG,
        1.0 / math.sqrt(_SIGMA_ALONG_PX**-2 + sigma_star**-2),
        1.0 / math.sqrt(_SIGMA_CROSS_PX**-2 + sigma_star**-2),
    )
    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# The agreement test
# ---------------------------------------------------------------------------

#: Sigma of the tight star fix the haze result is compared with (px).
_SIGMA_TIGHT_STAR_PX = 0.1
#: The disagreement the agreement tests place between the two results (px).
_DISAGREEMENT_PX = 2.0


def _dissenting_pair(direction: str) -> list[NavTechniqueResult]:
    """Return a tight star fix and a haze result 2 px from it along or across its axis."""
    along, cross = _axes(_THETA_DEG)
    step = along if direction == 'along' else cross
    star = _result(
        'StarFieldFromCatalogNav', (0.0, 0.0), _SIGMA_TIGHT_STAR_PX**2 * np.eye(2), confidence=0.9
    )
    haze = _result('TitanHazeNav', _DISAGREEMENT_PX * step, _haze_cov(), confidence=0.6)
    return [star, haze]


@pytest.mark.parametrize(
    ('direction', 'sigma_px'),
    [('along', _SIGMA_ALONG_PX), ('across', _SIGMA_CROSS_PX)],
)
def test_disagreement_is_measured_in_the_ellipse_metric(direction: str, sigma_px: float) -> None:
    """2 px along the weak axis scores fewer sigmas than 2 px across the strong axis."""
    star, haze = _dissenting_pair(direction)
    dist = mahalanobis_distance(
        np.array(star.offset_px),
        star.covariance_px2,
        np.array(haze.offset_px),
        haze.covariance_px2,
        rcond=packaged_ensemble().pinvh_rcond,
    )
    expected = _DISAGREEMENT_PX / math.sqrt(sigma_px**2 + _SIGMA_TIGHT_STAR_PX**2)
    assert dist == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ('direction', 'n_excluded'),
    [('along', 0), ('across', 1)],
)
def test_without_the_pixel_floor_only_the_cross_axis_dissenter_is_excluded(
    direction: str, n_excluded: int
) -> None:
    """With the pixel floor off, the haze result groups only when its sigma covers the gap."""
    config = packaged_ensemble()
    selection = consensus_selection(
        _dissenting_pair(direction),
        agreement_sigma=config.agreement_sigma,
        agreement_pixel_floor=0.0,
        shared_feature_agreement_px=config.shared_feature_agreement_px,
        rcond=config.pinvh_rcond,
        max_allowed_rotation_deg=config.max_allowed_rotation_deg,
    )
    assert len(selection.excluded) == n_excluded


def test_the_pixel_floor_groups_a_cross_axis_dissenter_inside_it() -> None:
    """The isotropic pixel floor groups a disagreement inside it in any direction."""
    config = packaged_ensemble()
    selection = consensus_selection(
        _dissenting_pair('across'),
        agreement_sigma=config.agreement_sigma,
        agreement_pixel_floor=config.agreement_pixel_floor,
        shared_feature_agreement_px=config.shared_feature_agreement_px,
        rcond=config.pinvh_rcond,
        max_allowed_rotation_deg=config.max_allowed_rotation_deg,
    )
    assert selection.excluded == []


# ---------------------------------------------------------------------------
# The confidence tier
# ---------------------------------------------------------------------------


def test_the_major_semi_axis_ignores_the_rotation_axis() -> None:
    """The tier's sigma is the ellipse's long axis, read from the translation block only."""
    cov = embed_rotation_unobservable(_haze_cov())
    assert major_semi_axis_px(cov) == pytest.approx(_SIGMA_ALONG_PX, rel=1e-12)


def test_a_haze_only_frame_caps_at_medium() -> None:
    """The haze fit's 1.02 px long axis exceeds the high tier's limit at any angle."""
    haze = _result('TitanHazeNav', (0.0, 0.0), _haze_cov(45.0), confidence=0.95)
    assert _ensemble([haze]) == 'medium'


@pytest.mark.parametrize('theta_deg', [0.0, 45.0])
def test_the_tier_of_a_fused_ellipse_does_not_depend_on_its_tilt(theta_deg: float) -> None:
    """One fused ellipse, 0.52 px by 0.31 px, is medium along an image axis and at 45 degrees.

    The tier compares the major semi-axis with ``max_sigma_px``.  The larger image-axis
    sigma would read 0.43 px at 45 degrees and earn high.
    """
    sigma_star = 0.6
    haze = _result('TitanHazeNav', (0.0, 0.0), _haze_cov(theta_deg), confidence=0.9)
    star = _result('StarFieldFromCatalogNav', (0.0, 0.0), sigma_star**2 * np.eye(2), confidence=0.9)
    assert _ensemble([haze, star]) == 'medium'
