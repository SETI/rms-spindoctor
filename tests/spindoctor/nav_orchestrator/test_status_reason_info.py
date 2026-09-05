"""Tests for ``STATUS_REASON_INFO_TEMPLATE`` covering every NavStatusReason."""

import pytest

from spindoctor.nav_orchestrator import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import NavInternalErrorRecord
from spindoctor.nav_orchestrator.orchestrator import NavOrchestrator
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_orchestrator.status_reason_info import STATUS_REASON_INFO_TEMPLATE
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
# A covering body decides the reason
# ---------------------------------------------------------------------------


def _failed_with(
    reported: NavStatusReason,
    model_metadata: dict[str, dict[str, object]],
    internal_error: NavInternalErrorRecord | None = None,
) -> NavStatusReason:
    """Reason a bare orchestrator reports for one failure and one set of metadata.

    Parameters:
        reported: The reason the pipeline reached.
        model_metadata: What the models recorded about themselves.
        internal_error: The record an internal-error result must carry.

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
        internal_error=internal_error,
    )
    return result.status_reason


_COVERING: dict[str, dict[str, object]] = {'body:SATURN': {'fills_extfov': True}}


@pytest.mark.parametrize(
    'reported',
    [
        NavStatusReason.NO_FEATURES_EXTRACTED,
        NavStatusReason.ALL_FEATURES_GATED,
        NavStatusReason.ALL_TECHNIQUES_SPURIOUS,
        NavStatusReason.NO_FEASIBLE_TECHNIQUES,
        NavStatusReason.FINAL_CONFIDENCE_BELOW_THRESHOLD,
    ],
)
def test_a_covering_body_controls_the_failure_reason(reported: NavStatusReason) -> None:
    """Whatever the pipeline reached, the covering body is what to report.

    It occludes the stars and stands in front of the rings, so features from
    those are not evidence the frame was navigable. Reporting the gate they
    fell through would describe a symptom and hide the cause.

    Parameters:
        reported: The reason the pipeline would otherwise have given.
    """
    assert _failed_with(reported, _COVERING) is NavStatusReason.BODY_FILLS_FOV


@pytest.mark.parametrize(
    'defect', [NavStatusReason.INTERNAL_ERROR, NavStatusReason.CONTRACT_VIOLATION]
)
def test_a_covering_body_never_masks_a_defect(defect: NavStatusReason) -> None:
    """A fact about the geometry must not be filed over a fault in the code.

    Parameters:
        defect: The reason naming a defect rather than the image.
    """
    record = (
        NavInternalErrorRecord(component='body:SATURN.create_model', exception_type='ValueError')
        if defect is NavStatusReason.INTERNAL_ERROR
        else None
    )
    assert _failed_with(defect, _COVERING, record) is defect


def test_no_covering_body_leaves_the_reason_alone() -> None:
    """A frame with no covering body reports what the pipeline reached."""
    assert (
        _failed_with(NavStatusReason.ALL_FEATURES_GATED, {'body:MIMAS': {'fills_extfov': False}})
        is NavStatusReason.ALL_FEATURES_GATED
    )
