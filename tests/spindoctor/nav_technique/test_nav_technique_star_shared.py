"""Shared behavior tests across the three star-accepting NavTechniques.

The per-technique test files cover technique-specific paths
(``StarUniqueMatchNav`` 1- vs 2-star, ``StarRefineNav`` outlier drop,
``StarFieldFromCatalogNav`` triplet matching).  Tests in this file
parameterize behavior that is identical across all three star
techniques — currently the "blank-image input yields a spurious
result with confidence zero" contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.spindoctor.nav_technique.conftest import NavContextFactory, NavFeatureFactory

from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_technique.nav_technique import NavTechnique
from spindoctor.nav_technique.nav_technique_star_field import StarFieldFromCatalogNav
from spindoctor.nav_technique.nav_technique_star_refine import StarRefineNav
from spindoctor.nav_technique.nav_technique_star_unique_match import StarUniqueMatchNav


def _attach_zero_prior(context: NavContext) -> NavContext:
    """Return ``context`` with a (0, 0) pass-1 prior attached."""
    return context.with_prior(
        offset_px=(0.0, 0.0),
        covariance_px2=np.eye(2, dtype=np.float64),
    )


@pytest.mark.parametrize(
    ('technique_cls', 'feature_count', 'requires_prior'),
    [
        pytest.param(StarUniqueMatchNav, 1, False, id='StarUniqueMatchNav'),
        pytest.param(StarRefineNav, 1, True, id='StarRefineNav'),
        pytest.param(StarFieldFromCatalogNav, 3, False, id='StarFieldFromCatalogNav'),
    ],
)
def test_star_technique_blank_image_returns_spurious(
    technique_cls: type[NavTechnique],
    feature_count: int,
    requires_prior: bool,
    make_nav_context: NavContextFactory,
    make_star_feature: NavFeatureFactory,
) -> None:
    """Each star technique reports spurious + zero confidence on a blank image.

    StarUniqueMatchNav fires the 1-star path against one feature whose
    search window contains nothing but noise floor; StarRefineNav has a
    prior attached but no inliers to refine on; StarFieldFromCatalogNav
    needs ≥ 3 STAR features but cannot match any of them against an
    empty image.  All three return ``spurious=True`` and confidence
    ``0.0`` rather than raising.
    """
    image = np.zeros((220, 220), dtype=np.float64)
    features = [
        make_star_feature(
            f'star:UCAC4:{i}',
            predicted_vu=(60.0 + 30.0 * i, 80.0 + 25.0 * i),
            predicted_snr=30.0,
        )
        for i in range(feature_count)
    ]
    technique = technique_cls()
    context = make_nav_context(image)
    if requires_prior:
        context = _attach_zero_prior(context)
    result = technique.navigate(features, context)
    assert result.spurious is True
    assert result.confidence == pytest.approx(0.0)
