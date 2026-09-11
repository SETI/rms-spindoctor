"""Tests for ``STATUS_REASON_INFO_TEMPLATE`` covering every NavStatusReason."""

import numpy as np
import pytest

from spindoctor.feature.feature import NavFeature, NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.feature.flags import BodyBlobFlags, StarFlags
from spindoctor.feature.geometry import BodyBlobGeometry, StarGeometry
from spindoctor.nav_orchestrator import NavImageClassifierResult
from spindoctor.nav_orchestrator.orchestrator import NavOrchestrator
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_orchestrator.status_reason_info import STATUS_REASON_INFO_TEMPLATE
from spindoctor.support.filters import NavFilterKind, NavFilterSpec
from spindoctor.support.status_reason import NavStatusReason


def test_every_status_reason_has_template() -> None:
    """Every NavStatusReason value has an entry in the template."""
    missing = set(NavStatusReason) - set(STATUS_REASON_INFO_TEMPLATE)
    assert missing == set()


def test_template_lines_non_empty() -> None:
    """Each template's line list is non-empty."""
    for reason, lines in STATUS_REASON_INFO_TEMPLATE.items():
        assert lines, f'{reason!r} has empty template'


def test_template_covers_full_taxonomy() -> None:
    """Template covers the full NavStatusReason taxonomy."""
    assert set(STATUS_REASON_INFO_TEMPLATE) == set(NavStatusReason)


# ---------------------------------------------------------------------------
# A covering body with nothing in view decides the reason
# ---------------------------------------------------------------------------


def _star(*, hidden: bool) -> NavFeature:
    """A star feature, flagged as behind a body when ``hidden``.

    Parameters:
        hidden: Whether the star model flagged the star as inside a body
            silhouette, which it emits at zero reliability.

    Returns:
        The feature.
    """
    return NavFeature(
        feature_id='star:hidden' if hidden else 'star:clear',
        feature_type=NavFeatureType.STAR,
        source_model='stars',
        geometry=StarGeometry(
            predicted_vu=(10.0, 20.0),
            catalog_vu=(10.0, 20.0),
            bbox_extfov_vu=(0, 0, 16, 16),
        ),
        subject_range_km=1.0e10,
        position_cov_px=np.eye(2, dtype=np.float64) * 0.25,
        intensity_sigma_rel=0.05,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.0 if hidden else 0.8,
        reliability_reasons=NavReliabilityBreakdown(predicted_snr=10.0, in_body_silhouette=hidden),
        usable_types=frozenset({NavFeatureType.STAR}),
        flags=StarFlags(in_body_silhouette=hidden),
    )


def _blob() -> NavFeature:
    """A moon's blob feature, the kind something in front of a covering body emits.

    Returns:
        The feature, at a reliability the gate drops.
    """
    return NavFeature(
        feature_id='body_blob:MIMAS',
        feature_type=NavFeatureType.BODY_BLOB,
        source_model='body:MIMAS',
        geometry=BodyBlobGeometry(
            bbox_extfov_vu=(0, 0, 16, 16),
            predicted_center_vu=(8.0, 8.0),
            predicted_diameter_px=8.0,
        ),
        usable_types=frozenset({NavFeatureType.BODY_BLOB}),
        flags=BodyBlobFlags(body_name='MIMAS', predicted_diameter_px=8.0),
        subject_range_km=1.0e8,
        position_cov_px=np.eye(2, dtype=np.float64) * 0.25,
        intensity_sigma_rel=0.05,
        preferred_filter=NavFilterSpec(kind=NavFilterKind.NONE),
        reliability=0.1,
        reliability_reasons=NavReliabilityBreakdown(visible_lit_fraction=0.1),
    )


def _failed_with(
    reported: NavStatusReason,
    model_metadata: dict[str, dict[str, object]],
    features: list[NavFeature],
) -> NavStatusReason:
    """File one failure through a bare orchestrator and return the reason it records.

    Parameters:
        reported: The reason the pipeline reached.
        model_metadata: What the models recorded about themselves.
        features: What the models emitted, gated or not.

    Returns:
        The reason actually filed.
    """
    orch = NavOrchestrator([])
    result = orch._fail(
        status_reason=reported,
        image_classifier=NavImageClassifierResult(
            image_class='clean',
            saturation_frac=0.0,
            missing_frac=0.0,
            noise_sigma=1.0,
            max_dn=1.0,
            flags=[],
        ),
        provenance=Provenance(
            spindoctor_version='0.0.0',
            image_et=0.0,
            pipeline_run_iso8601='2026-04-27T00:00:00Z',
            technique_names=(),
            extractor_names=(),
        ),
        model_metadata=model_metadata,
        features=features,
    )
    return result.status_reason


_COVERING: dict[str, dict[str, object]] = {
    'body:SATURN': {'fills_extfov': True, 'edge_in_frame': False}
}


@pytest.mark.parametrize(
    ('reported', 'features'),
    [
        (NavStatusReason.NO_FEATURES_EXTRACTED, []),
        (NavStatusReason.ALL_FEATURES_GATED, [_star(hidden=True)]),
    ],
    ids=['nothing emitted', 'only hidden stars'],
)
def test_a_covering_body_with_nothing_in_view_decides_the_reason(
    reported: NavStatusReason, features: list[NavFeature]
) -> None:
    """Nothing navigable was in view, so the image could not have been navigated.

    The stars behind the body are emitted and gated rather than absent, which
    is what would otherwise file the image under the gate it fell through.

    Parameters:
        reported: The reason the gate would otherwise have given.
        features: What the models emitted.
    """
    assert _failed_with(reported, _COVERING, features) is NavStatusReason.BODY_FILLS_FOV


def test_a_feature_in_front_of_a_covering_body_keeps_the_reason() -> None:
    """A moon in front of the body was in view, so the image could have been navigated."""
    features = [_star(hidden=True), _blob()]
    assert (
        _failed_with(NavStatusReason.ALL_FEATURES_GATED, _COVERING, features)
        is NavStatusReason.ALL_FEATURES_GATED
    )


def test_a_reason_reached_after_a_technique_ran_keeps_it() -> None:
    """A technique ran on something, so the image was not one nothing could navigate."""
    assert (
        _failed_with(NavStatusReason.ALL_TECHNIQUES_SPURIOUS, _COVERING, [_star(hidden=True)])
        is NavStatusReason.ALL_TECHNIQUES_SPURIOUS
    )


@pytest.mark.parametrize(
    'model_metadata',
    [
        {'body:MIMAS': {'fills_extfov': False}},
        {'body:SATURN': {'fills_extfov': True, 'edge_in_frame': True}},
    ],
    ids=['no covering body', 'covering body showing its terminator'],
)
def test_no_declined_body_leaves_the_reason_alone(
    model_metadata: dict[str, dict[str, object]],
) -> None:
    """Only a body model that declined its body as covering the frame substitutes.

    Parameters:
        model_metadata: What the models recorded about themselves.
    """
    assert (
        _failed_with(NavStatusReason.ALL_FEATURES_GATED, model_metadata, [_star(hidden=True)])
        is NavStatusReason.ALL_FEATURES_GATED
    )
