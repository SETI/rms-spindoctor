"""Tests for ``spindoctor.nav_orchestrator.curator.build_metadata_dict``."""

import dataclasses
import json
import math
from typing import Any, cast

import numpy as np
import pytest

from spindoctor.feature.feature import NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_orchestrator.curator import (
    assert_diagnostic_fields_present,
    build_metadata_dict,
)
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import (
    ArbitrationCandidate,
    ConflictArbitrationRecord,
    NavInternalErrorRecord,
    NavResult,
)
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.diagnostics import BodyLimbDiagnostics
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.cmatrix import AttitudeBaseline, PointingSolution
from spindoctor.support.status_reason import NavStatusReason


def _json_round_trip(result: NavResult) -> dict[str, Any]:
    """Build the metadata dict for a result and round-trip it through real JSON.

    Asserting on the parsed-back dict proves the metadata is genuinely
    serializable and pins what a reader of the written file sees, not what the
    in-memory objects happened to be.

    Parameters:
        result: The result to build metadata for.

    Returns:
        The metadata dict as it survives JSON serialization.
    """
    return cast(dict[str, Any], json.loads(json.dumps(build_metadata_dict(result))))


def _classifier() -> NavImageClassifierResult:
    return NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=1.234567,
        max_dn=10.0,
    )


def _provenance() -> Provenance:
    return Provenance(
        spindoctor_version='0.5.2',
        image_et=414504000.123456789,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
    )


def _ok_result_with_one_technique() -> NavResult:
    cov = np.diag([0.04, 0.16])
    diag = BodyLimbDiagnostics(visible_arc_px=120.0, dt_fit_rms_px=0.456789)
    tech = NavTechniqueResult(
        technique_name='BodyLimbNav',
        feature_ids=('limb_arc:MIMAS',),
        offset_px=(1.234567, 2.345678),
        covariance_px2=cov,
        confidence=0.876543,
        spurious=False,
        at_edge=False,
        diagnostics=diag,
    )
    inv = [
        NavFeatureSummary(
            feature_id='limb_arc:MIMAS',
            feature_type=NavFeatureType.LIMB_ARC,
            source_model='body:MIMAS',
            reliability=0.9,
            gated=False,
            gate_reason=None,
            bbox_extfov_vu=(100, 200, 300, 400),
        ),
    ]
    return NavResult.success(
        offset_px=(1.234567, 2.345678),
        covariance_px2=cov,
        confidence=0.876543,
        confidence_rank='high',
        status_reason=NavStatusReason.OK,
        per_technique=[tech],
        feature_inventory=inv,
        image_classifier=_classifier(),
        provenance=_provenance(),
    )


def test_metadata_dict_contains_top_level_keys() -> None:
    """The metadata dict has the exact required top-level key set."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    expected = {
        'status',
        'status_reason',
        'offset_px',
        'sigma_px',
        'sigma_along_unobservable_px',
        'confidence',
        'confidence_provisional',
        'confidence_rank',
        'covariance_px2',
        'techniques_used',
        'excluded_from_consensus',
        'feature_count_by_type',
        'per_technique',
        'feature_inventory',
        'image_classifier',
        'provenance',
    }
    assert set(md) == expected


def test_metadata_dict_rounds_offset_to_4_decimals() -> None:
    """offset_px is rounded to 4 decimals."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert md['offset_px'] == [1.2346, 2.3457]


def test_metadata_dict_rounds_confidence_to_3_decimals() -> None:
    """confidence is rounded to 3 decimals."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert md['confidence'] == 0.877


def test_metadata_dict_marks_confidence_provisional() -> None:
    """confidence_provisional is present and literally true.

    The marker flags that confidence values and tiers are calibrated
    against simulated planted-truth recovery only and must not be read
    as probabilities of real-image accuracy; it stays true until a
    calibration against real-image error measurements lands.
    """
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert md['confidence_provisional'] is True


def test_metadata_dict_provenance_carries_config_and_catalog_fields() -> None:
    """The provenance block serializes the config hash, overrides, and catalogs."""
    prov = Provenance(
        spindoctor_version='0.5.2',
        image_et=414504000.123456789,
        pipeline_run_iso8601='2026-04-26T12:00:00Z',
        config_hash='cd' * 32,
        config_overrides=('/etc/spindoctor/site.yaml',),
        star_catalogs={'ucac4': 'gs://bucket/UCAC4'},
    )
    result = NavResult.success(
        offset_px=(0.0, 0.0),
        covariance_px2=np.eye(2, dtype=np.float64),
        confidence=0.5,
        confidence_rank='low',
        status_reason=NavStatusReason.OK,
        per_technique=[],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=prov,
    )
    md = build_metadata_dict(result)
    assert md['provenance']['config_hash'] == 'cd' * 32
    assert md['provenance']['config_overrides'] == ['/etc/spindoctor/site.yaml']
    assert md['provenance']['star_catalogs'] == {'ucac4': 'gs://bucket/UCAC4'}


def test_metadata_dict_provenance_config_fields_default_to_empty() -> None:
    """A minimal provenance still emits the new keys (null hash, empty lists)."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert md['provenance']['config_hash'] is None
    assert md['provenance']['config_overrides'] == []
    assert md['provenance']['star_catalogs'] == {}


def test_metadata_dict_rounds_image_et_to_6_decimals() -> None:
    """image_et is rounded to 6 decimals."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert md['provenance']['image_et'] == 414504000.123457


def test_metadata_dict_is_json_serializable() -> None:
    """The output dict can be JSON-serialized round-trip."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    s = json.dumps(md)
    again = json.loads(s)
    assert again['confidence'] == 0.877


def test_metadata_dict_replaces_inf_with_sentinel() -> None:
    """A rank-1 result's sigma_along_unobservable_px is finite-clamped."""
    cov = np.diag([0.04, 1e10])
    result = NavResult.success(
        offset_px=(0.0, 0.0),
        covariance_px2=cov,
        confidence=0.5,
        confidence_rank='low',
        status_reason=NavStatusReason.RANK_1_ONLY,
        per_technique=[],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
        sigma_along_unobservable_px=math.inf,
    )
    md = build_metadata_dict(result)
    # ``math.inf`` is clamped to the curator's JSON_INF_SENTINEL sentinel.
    from spindoctor.feature.constants import JSON_INF_SENTINEL

    assert md['sigma_along_unobservable_px'] == JSON_INF_SENTINEL


def test_assert_diagnostic_fields_present_passes() -> None:
    """A well-formed NavResult passes the curator allow-list assertion."""
    result = _ok_result_with_one_technique()
    assert_diagnostic_fields_present(result)


def test_assert_diagnostic_fields_present_detects_missing_curator_field() -> None:
    """A diagnostic without CURATOR_FIELDS triggers AssertionError."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class BadDiag:
        x: float = 0.0
        # No CURATOR_FIELDS attribute.

    cov = np.eye(2, dtype=np.float64)
    tech = NavTechniqueResult(
        technique_name='BadTechnique',
        feature_ids=(),
        offset_px=(0.0, 0.0),
        covariance_px2=cov,
        confidence=0.5,
        spurious=False,
        at_edge=False,
        diagnostics=BadDiag(),  # type: ignore[arg-type]
    )
    result = NavResult.success(
        offset_px=(0.0, 0.0),
        covariance_px2=cov,
        confidence=0.5,
        confidence_rank='low',
        status_reason=NavStatusReason.OK,
        per_technique=[tech],
        feature_inventory=[],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    with pytest.raises(AssertionError, match='CURATOR_FIELDS'):
        assert_diagnostic_fields_present(result)


def test_metadata_dict_techniques_used_sorted() -> None:
    """``techniques_used`` matches the per_technique technique names sorted."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert md['techniques_used'] == ['BodyLimbNav']


def _gated_titan_result() -> NavResult:
    """A Titan-only frame whose lone feature the reliability gate dropped."""
    inv = [
        NavFeatureSummary(
            feature_id='titan_limb:TITAN',
            feature_type=NavFeatureType.TITAN_LIMB,
            source_model='titan:TITAN',
            reliability=0.0,
            gated=True,
            gate_reason='reliability_0.000_below_threshold_0.300',
            bbox_extfov_vu=(10, 10, 90, 90),
            reliability_reasons=NavReliabilityBreakdown(
                titan_envelope_diameter_px=31.5,
                titan_occluded_fraction=0.42,
            ),
        ),
    ]
    return NavResult.failed(
        status_reason=NavStatusReason.ALL_FEATURES_GATED,
        feature_inventory=inv,
        image_classifier=_classifier(),
        provenance=_provenance(),
    )


def test_gated_feature_entry_reaches_the_json() -> None:
    """A gated feature is recorded in the emitted metadata, not only in the log."""
    md = _json_round_trip(_gated_titan_result())
    entry = md['feature_inventory'][0]
    assert entry['gated'] is True


def test_gated_feature_entry_names_its_type() -> None:
    """The gate record identifies which feature type was dropped."""
    md = _json_round_trip(_gated_titan_result())
    assert md['feature_inventory'][0]['feature_type'] == 'TITAN_LIMB'


def test_gated_feature_entry_carries_its_breakdown() -> None:
    """The reliability breakdown travels into the JSON so gates are attributable."""
    md = _json_round_trip(_gated_titan_result())
    reasons = md['feature_inventory'][0]['reliability_reasons']
    assert reasons['titan_occluded_fraction'] == pytest.approx(0.42)


def test_breakdown_omits_inapplicable_components() -> None:
    """Components that do not apply to a feature type are left out entirely."""
    md = _json_round_trip(_gated_titan_result())
    reasons = md['feature_inventory'][0]['reliability_reasons']
    assert 'predicted_snr' not in reasons


def test_breakdown_is_empty_when_no_component_was_populated() -> None:
    """A feature whose model populated no component reports an empty mapping."""
    md = _json_round_trip(_ok_result_with_one_technique())
    assert md['feature_inventory'][0]['reliability_reasons'] == {}


def _pointing(*, corrected: bool) -> PointingSolution:
    """Build a PointingSolution with recognizable, non-symmetric matrices.

    Parameters:
        corrected: True to carry a corrected C-matrix alongside the baseline;
            False for a solution recording only the uncorrected attitude.

    Returns:
        The solution.
    """
    original = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    baseline = AttitudeBaseline(
        cmatrix_original=original,
        oops_from_spice=np.eye(3),
        camera_frame='CASSINI_ISS_NAC',
        camera_frame_id=-82360,
        ck_frame_id=-82000,
        start_et=246684087.05644953,
        stop_et=246684087.23644954,
        midtime_et=246684087.14644954,
        exposure_s=0.18,
        sclk_start='1/1572105349.077',
        sclk_midtime='1/1572105349.100',
        sclk_stop='1/1572105349.123',
    )
    cmatrix = None
    if corrected:
        cmatrix = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
    return PointingSolution(baseline=baseline, cmatrix=cmatrix)


def _result_with_pointing(*, corrected: bool) -> NavResult:
    """Return the one-technique success result stamped with a pointing solution.

    Parameters:
        corrected: True to carry a corrected C-matrix alongside the baseline;
            False for a solution recording only the uncorrected attitude.
    """
    return dataclasses.replace(
        _ok_result_with_one_technique(), pointing=_pointing(corrected=corrected)
    )


def test_metadata_dict_omits_pointing_when_none_was_computed() -> None:
    """A result with no pointing solution writes no pointing block."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert 'pointing' not in md


def test_metadata_dict_omits_times_when_none_was_computed() -> None:
    """A result with no pointing solution writes no times block."""
    md = build_metadata_dict(_ok_result_with_one_technique())
    assert 'times' not in md


def test_pointing_block_has_the_declared_key_set() -> None:
    """A corrected result's pointing block carries exactly the declared keys."""
    md = _json_round_trip(_result_with_pointing(corrected=True))
    assert set(md['pointing']) == {
        'cmatrix',
        'cmatrix_original',
        'camera_frame',
        'camera_frame_id',
        'ck_frame_id',
    }


def test_times_block_has_the_declared_key_set() -> None:
    """The times block carries exactly the declared keys."""
    md = _json_round_trip(_result_with_pointing(corrected=True))
    assert set(md['times']) == {
        'start_et',
        'stop_et',
        'midtime_et',
        'exposure_s',
        'sclk_start',
        'sclk_midtime',
        'sclk_stop',
    }


def test_cmatrix_is_serialized_as_nine_row_major_floats() -> None:
    """The corrected C-matrix flattens row by row, not column by column."""
    md = _json_round_trip(_result_with_pointing(corrected=True))
    assert md['pointing']['cmatrix'] == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_cmatrix_original_is_serialized_as_nine_row_major_floats() -> None:
    """The uncorrected C-matrix flattens row by row too."""
    md = _json_round_trip(_result_with_pointing(corrected=True))
    assert md['pointing']['cmatrix_original'] == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_frame_identities_are_serialized_as_recorded() -> None:
    """The camera frame name and both frame ids travel unchanged."""
    md = _json_round_trip(_result_with_pointing(corrected=True))
    assert md['pointing']['camera_frame'] == 'CASSINI_ISS_NAC'
    assert md['pointing']['camera_frame_id'] == -82360
    assert md['pointing']['ck_frame_id'] == -82000


def test_times_are_serialized_unrounded() -> None:
    """Epochs keep full precision, since they define a segment interval exactly."""
    md = _json_round_trip(_result_with_pointing(corrected=True))
    assert md['times']['start_et'] == 246684087.05644953
    assert md['times']['midtime_et'] == 246684087.14644954
    assert md['times']['stop_et'] == 246684087.23644954


def test_sclk_strings_are_serialized_as_recorded() -> None:
    """The three spacecraft-clock strings travel unchanged."""
    md = _json_round_trip(_result_with_pointing(corrected=True))
    assert md['times']['sclk_start'] == '1/1572105349.077'
    assert md['times']['sclk_midtime'] == '1/1572105349.100'
    assert md['times']['sclk_stop'] == '1/1572105349.123'


def test_uncorrectable_result_omits_cmatrix_but_keeps_the_original() -> None:
    """A solution with no corrected attitude still records the uncorrected one."""
    md = _json_round_trip(_result_with_pointing(corrected=False))
    assert 'cmatrix' not in md['pointing']
    assert md['pointing']['cmatrix_original'] == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def test_uncorrectable_result_still_records_its_times() -> None:
    """A solution with no corrected attitude still records the exposure times."""
    md = _json_round_trip(_result_with_pointing(corrected=False))
    assert md['times']['exposure_s'] == 0.18


def _internal_error_result() -> NavResult:
    """A result failed by a ring model that raised while building."""
    return NavResult.failed(
        status_reason=NavStatusReason.INTERNAL_ERROR,
        image_classifier=_classifier(),
        provenance=_provenance(),
        internal_error=NavInternalErrorRecord(
            component='rings:SATURN.create_model',
            exception_type='AttributeError',
        ),
    )


def test_an_internal_error_names_its_component_in_the_json() -> None:
    """The document says which component raised, not only the per-image log.

    Every consumer downstream of navigation -- the results index, the bundle
    stage, the backplane stage -- reads the document.  A failure recorded
    only in the log is a failure they cannot see.
    """
    md = _json_round_trip(_internal_error_result())
    assert md['internal_error']['component'] == 'rings:SATURN.create_model'


def test_an_internal_error_names_its_exception_type_in_the_json() -> None:
    """The document says what class of exception was raised."""
    md = _json_round_trip(_internal_error_result())
    assert md['internal_error']['exception_type'] == 'AttributeError'


def test_an_internal_error_document_carries_the_matching_status_reason() -> None:
    """The block and the reason agree, so neither can be read without the other."""
    md = _json_round_trip(_internal_error_result())
    assert md['status_reason'] == 'internal_error'


def test_a_result_with_no_internal_error_emits_no_block() -> None:
    """An ordinary failure carries no internal_error key at all."""
    md = _json_round_trip(_gated_titan_result())
    assert 'internal_error' not in md


def _arbitrated_result() -> NavResult:
    """A failed result stamped with the record of a pass-2 test of a conflict."""
    record = ConflictArbitrationRecord(
        candidates=(
            ArbitrationCandidate(
                techniques=('StarUniqueMatchNav',),
                offset_px=(-0.78049, 4.32012),
                support=30,
                confirmed=True,
            ),
            ArbitrationCandidate(
                techniques=('RingEdgeNav', 'BodyLimbNav', 'BodyLimbNav'),
                offset_px=(-18.0, -46.0),
                support=1,
                confirmed=False,
            ),
        ),
        refuted_techniques=('RingEdgeNav', 'BodyLimbNav', 'BodyLimbNav'),
    )
    return dataclasses.replace(_gated_titan_result(), conflict_arbitration=record)


def test_the_arbitration_record_reaches_the_json() -> None:
    """Both tested answers, their supports, and the refuted techniques are written.

    The runner-up answer holds a limb measured on two bodies: each measurement is one
    entry, and the refuted names are sorted while each answer keeps its own order.
    """
    md = _json_round_trip(_arbitrated_result())
    assert md['conflict_arbitration'] == {
        'candidates': [
            {
                'techniques': ['StarUniqueMatchNav'],
                'offset_px': [-0.7805, 4.3201],
                'support': 30,
                'confirmed': True,
            },
            {
                'techniques': ['RingEdgeNav', 'BodyLimbNav', 'BodyLimbNav'],
                'offset_px': [-18.0, -46.0],
                'support': 1,
                'confirmed': False,
            },
        ],
        'refuted_techniques': ['BodyLimbNav', 'BodyLimbNav', 'RingEdgeNav'],
    }


def test_a_result_the_test_did_not_run_on_emits_no_arbitration_block() -> None:
    """No conflict_arbitration key is written when the pass-2 test did not run."""
    md = _json_round_trip(_gated_titan_result())
    assert 'conflict_arbitration' not in md
