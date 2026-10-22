"""Curator — turn a NavResult into a JSON-friendly metadata dict.

The curator picks JSON-friendly fields from a NavResult, rounds floats to
documented precision, replaces ``inf`` with the ``JSON_INF_SENTINEL`` finite
sentinel, and emits the ``navigation_result`` block consumed by downstream
readers.

Every per-technique diagnostic field that ships in the JSON appears in the
technique's ``CURATOR_FIELDS`` allow-list; ``assert_diagnostic_fields_present``
fails CI if a new diagnostic field is added without updating the allow-list.
"""

import dataclasses
import math
from typing import Any

import numpy as np

from spindoctor.feature.constants import JSON_INF_SENTINEL
from spindoctor.feature.feature import NavReliabilityBreakdown
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import ConflictArbitrationRecord, NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.support.cmatrix import AttitudeBaseline, PointingSolution
from spindoctor.support.types import NDArrayFloatType

__all__ = [
    'assert_diagnostic_fields_present',
    'build_metadata_dict',
]


# Float-rounding policy: 4 decimals for pixel quantities, 3 for confidence
# scores, 6 for ET timestamps.  Tighter precision than the per-image
# tolerance budget; chosen to give stable byte-identical JSON outputs.
PIXEL_DECIMALS = 4
CONFIDENCE_DECIMALS = 3
ET_DECIMALS = 6


def _round_float(value: float, decimals: int) -> float:
    """Round a float to ``decimals`` digits, mapping non-finite values to a
    sentinel so the JSON stays valid.

    ``inf`` maps to ``+/-JSON_INF_SENTINEL``.  ``nan`` maps to the *positive*
    ``JSON_INF_SENTINEL`` (not ``0.0``): the only path that produces a NaN here
    is a degenerate covariance/variance entry, and rendering that as ``0.0``
    would read as a zero-variance, infinitely-confident value -- the opposite of
    the truth.  Mapping it to the huge sentinel instead conveys "no
    information" / unbounded uncertainty, which is the safe direction.
    """
    if math.isinf(value):
        return JSON_INF_SENTINEL if value > 0 else -JSON_INF_SENTINEL
    if math.isnan(value):
        return JSON_INF_SENTINEL
    return round(value, decimals)


def _round_pair(value: tuple[float, float] | None) -> list[float] | None:
    if value is None:
        return None
    return [_round_float(value[0], PIXEL_DECIMALS), _round_float(value[1], PIXEL_DECIMALS)]


def _round_matrix(matrix: Any) -> list[list[float]] | None:
    """Round an NxN covariance matrix into a JSON-friendly nested list.

    Serializes whatever square shape it is given (2x2 translation-only or 3x3
    rotation-aware), so the JSON ``covariance_px2`` value is 3x3 for a
    rotation-fitted result.
    """
    if matrix is None:
        return None
    rows = []
    for r in range(matrix.shape[0]):
        rows.append(
            [_round_float(float(matrix[r, c]), PIXEL_DECIMALS) for c in range(matrix.shape[1])]
        )
    return rows


def assert_diagnostic_fields_present(result: NavResult) -> None:
    """Verify every per-technique diagnostic field has a CURATOR_FIELDS entry.

    An unmapped field is a programmer error and raises ``AssertionError``
    so CI fails the build before a new diagnostic silently disappears from
    the JSON output.

    Parameters:
        result: NavResult whose per_technique entries are inspected.

    Raises:
        AssertionError: if any field on a diagnostic dataclass is missing
            from its ``CURATOR_FIELDS`` allow-list.
    """
    for tech_result in result.per_technique:
        diag = tech_result.diagnostics
        if not hasattr(diag, 'CURATOR_FIELDS'):
            raise AssertionError(f'{type(diag).__name__} is missing CURATOR_FIELDS class attribute')
        curator_fields = diag.CURATOR_FIELDS
        declared = set(curator_fields)
        actual = {f.name for f in dataclasses.fields(diag)}
        missing = actual - declared
        if missing:
            raise AssertionError(
                f'{type(diag).__name__} has unmapped fields {sorted(missing)!r}; '
                f'add them to CURATOR_FIELDS or set value to None to skip'
            )


def _curate_diagnostics(diag: Any) -> dict[str, Any]:
    """Return a JSON-friendly subset of a diagnostic dataclass."""
    out: dict[str, Any] = {}
    curator_fields = diag.CURATOR_FIELDS
    for attr_name, json_key in curator_fields.items():
        if json_key is None:
            continue
        value = getattr(diag, attr_name)
        if isinstance(value, float):
            out[json_key] = _round_float(value, CONFIDENCE_DECIMALS)
        else:
            out[json_key] = value
    return out


def _curate_technique_result(res: NavTechniqueResult) -> dict[str, Any]:
    """Return a JSON-friendly subset of a NavTechniqueResult."""
    out: dict[str, Any] = {
        'technique_name': res.technique_name,
        'feature_ids': list(res.feature_ids),
        'offset_px': _round_pair(res.offset_px),
        'covariance_px2': _round_matrix(res.covariance_px2),
        'confidence': _round_float(res.confidence, CONFIDENCE_DECIMALS),
        'spurious': res.spurious,
        'at_edge': res.at_edge,
        'diagnostics': _curate_diagnostics(res.diagnostics),
    }
    if res.rotation_rad is not None:
        out['rotation_deg'] = _round_float(math.degrees(res.rotation_rad), CONFIDENCE_DECIMALS)
    if res.sigma_rotation_rad is not None:
        out['sigma_rotation_deg'] = _round_float(
            math.degrees(res.sigma_rotation_rad), CONFIDENCE_DECIMALS
        )
    return out


def _curate_reliability_reasons(breakdown: NavReliabilityBreakdown) -> dict[str, Any]:
    """Return the populated components of a reliability breakdown.

    Components that do not apply to a feature type are ``None`` and are
    omitted entirely, so each entry lists only the quantities that actually
    produced the score.  Generic across feature types: a new component field
    appears in the metadata as soon as some model populates it.
    """
    out: dict[str, Any] = {}
    for f in dataclasses.fields(breakdown):
        value = getattr(breakdown, f.name)
        if value is None:
            continue
        # ``bool`` is a subclass of ``int``, not of ``float``, so the
        # boolean components fall through to the pass-through branch and
        # need no case of their own.
        if isinstance(value, float):
            out[f.name] = _round_float(value, CONFIDENCE_DECIMALS)
        else:
            out[f.name] = value
    return out


def _curate_feature_summary(summary: NavFeatureSummary) -> dict[str, Any]:
    """Return a JSON-friendly entry for one NavFeatureSummary."""
    return {
        'feature_id': summary.feature_id,
        'feature_type': summary.feature_type.value,
        'source_model': summary.source_model,
        'reliability': _round_float(summary.reliability, CONFIDENCE_DECIMALS),
        'gated': summary.gated,
        'gate_reason': summary.gate_reason,
        'bbox_extfov_vu': list(summary.bbox_extfov_vu),
        'reliability_reasons': _curate_reliability_reasons(summary.reliability_reasons),
    }


def _curate_image_classifier(classifier: NavImageClassifierResult) -> dict[str, Any]:
    """Return a JSON-friendly entry for the image-quality classifier result."""
    return {
        'class': classifier.image_class,
        'saturation_frac': _round_float(classifier.saturation_frac, CONFIDENCE_DECIMALS),
        'missing_frac': _round_float(classifier.missing_frac, CONFIDENCE_DECIMALS),
        'noise_sigma': _round_float(classifier.noise_sigma, CONFIDENCE_DECIMALS),
        'max_dn': _round_float(classifier.max_dn, CONFIDENCE_DECIMALS),
        'background_gradient_score': (
            None
            if classifier.background_gradient_score is None
            else _round_float(classifier.background_gradient_score, CONFIDENCE_DECIMALS)
        ),
        'flags': list(classifier.flags),
    }


def _curate_provenance(provenance: Provenance) -> dict[str, Any]:
    """Return a JSON-friendly entry for the provenance envelope."""
    return {
        'spindoctor_version': provenance.spindoctor_version,
        'spindoctor_git_sha': provenance.spindoctor_git_sha,
        'spice_kernels': list(provenance.spice_kernels),
        'spice_kernel_count': provenance.spice_kernel_count,
        'static_data_hashes': dict(provenance.static_data_hashes),
        'technique_names': list(provenance.technique_names),
        'extractor_names': list(provenance.extractor_names),
        'config_hash': provenance.config_hash,
        'config_overrides': list(provenance.config_overrides),
        'star_catalogs': dict(provenance.star_catalogs),
        'image_et': _round_float(provenance.image_et, ET_DECIMALS),
        'pipeline_run_iso8601': provenance.pipeline_run_iso8601,
    }


def _flatten_rotation(matrix: NDArrayFloatType) -> list[float]:
    """Flatten a 3x3 rotation into nine row-major floats.

    Deliberately unrounded: a C-kernel writer identifies the baseline kernel
    an image navigated against by reproducing this matrix to within a
    nanoradian, and rounding would put the recorded value outside that bound.

    Parameters:
        matrix: The 3x3 rotation to flatten.

    Returns:
        Its nine elements in row-major order, as plain floats.
    """
    return [float(v) for v in np.asarray(matrix, np.float64).reshape(9)]


def _curate_conflict_arbitration(record: ConflictArbitrationRecord) -> dict[str, Any]:
    """Return a JSON-friendly entry for the pass-2 test of a conflicted pass 1.

    Parameters:
        record: What the test found.

    Returns:
        ``candidates`` (the consensus answer, then the runner-up answer, each with its
        ``techniques``, ``offset_px``, ``support``, and ``confirmed``) and the sorted
        ``refuted_techniques``.  Both name lists hold one entry per result, so a
        technique that measured two bodies appears twice.
    """
    return {
        'candidates': [
            {
                'techniques': list(c.techniques),
                'offset_px': _round_pair(c.offset_px),
                'support': c.support,
                'confirmed': c.confirmed,
            }
            for c in record.candidates
        ],
        'refuted_techniques': sorted(record.refuted_techniques),
    }


def _curate_pointing(solution: PointingSolution) -> dict[str, Any]:
    """Return a JSON-friendly entry for a corrected-attitude solution.

    ``cmatrix`` is present only when one was computed; ``cmatrix_original``
    and the frame identities are always present.

    Parameters:
        solution: The image's pointing solution.

    Returns:
        The ``pointing`` metadata block.
    """
    baseline = solution.baseline
    out: dict[str, Any] = {}
    if solution.cmatrix is not None:
        out['cmatrix'] = _flatten_rotation(solution.cmatrix)
    out['cmatrix_original'] = _flatten_rotation(baseline.cmatrix_original)
    out['camera_frame'] = baseline.camera_frame
    out['camera_frame_id'] = baseline.camera_frame_id
    out['ck_frame_id'] = baseline.ck_frame_id
    return out


def _curate_times(baseline: AttitudeBaseline) -> dict[str, Any]:
    """Return a JSON-friendly entry for an observation's exposure times.

    Deliberately unrounded, for the same reason as the C-matrices: the epochs
    define a C-kernel segment's interpolation interval exactly.

    Parameters:
        baseline: The observation's attitude baseline, which carries its
            exposure epochs and clock strings.

    Returns:
        The ``times`` metadata block.
    """
    return {
        'start_et': baseline.start_et,
        'stop_et': baseline.stop_et,
        'midtime_et': baseline.midtime_et,
        'exposure_s': baseline.exposure_s,
        'sclk_start': baseline.sclk_start,
        'sclk_midtime': baseline.sclk_midtime,
        'sclk_stop': baseline.sclk_stop,
    }


def build_metadata_dict(result: NavResult) -> dict[str, Any]:
    """Build the additive ``navigation_result`` metadata block.

    Output is a JSON-serializable dict; the orchestrator merges it into the
    existing per-image ``_metadata.json`` schema as the additive
    ``navigation_result`` key.

    Parameters:
        result: The full in-memory NavResult.

    Returns:
        Dict ready for ``json.dump``.

    Raises:
        AssertionError: if any diagnostic field is missing a
            ``CURATOR_FIELDS`` entry.
    """
    assert_diagnostic_fields_present(result)
    sigma_along_unobservable_px: float | None
    if result.sigma_along_unobservable_px is None:
        sigma_along_unobservable_px = None
    else:
        sigma_along_unobservable_px = _round_float(
            result.sigma_along_unobservable_px, PIXEL_DECIMALS
        )
    feature_count_by_type: dict[str, int] = {}
    for entry in result.feature_inventory:
        if entry.gated:
            continue
        key = entry.feature_type.value
        feature_count_by_type[key] = feature_count_by_type.get(key, 0) + 1
    out: dict[str, Any] = {
        'status': result.status,
        'status_reason': result.status_reason.value,
        'offset_px': _round_pair(result.offset_px),
        'sigma_px': _round_pair(result.sigma_px),
        'sigma_along_unobservable_px': sigma_along_unobservable_px,
        'confidence': _round_float(result.confidence, CONFIDENCE_DECIMALS),
        # Literal marker (always true) flagging that confidence values and
        # confidence_rank tiers are calibrated against simulated
        # planted-truth recovery only (sim-anchored; see the provenance
        # note in config_510_techniques.yaml) and must not be read as
        # probabilities of real-image accuracy; it stays true until a
        # calibration against real-image error measurements lands.
        'confidence_provisional': True,
        'confidence_rank': result.confidence_rank,
        'covariance_px2': _round_matrix(result.covariance_px2),
        'techniques_used': sorted({r.technique_name for r in result.per_technique}),
        'excluded_from_consensus': sorted(result.excluded_from_consensus),
        'feature_count_by_type': feature_count_by_type,
        'per_technique': [_curate_technique_result(r) for r in result.per_technique],
        'feature_inventory': [_curate_feature_summary(s) for s in result.feature_inventory],
        'image_classifier': _curate_image_classifier(result.image_classifier),
        'provenance': _curate_provenance(result.provenance),
    }
    if result.rotation_rad is not None:
        out['rotation_deg'] = _round_float(math.degrees(result.rotation_rad), CONFIDENCE_DECIMALS)
    if result.sigma_rotation_rad is not None:
        out['sigma_rotation_deg'] = _round_float(
            math.degrees(result.sigma_rotation_rad), CONFIDENCE_DECIMALS
        )
    if result.internal_error is not None:
        # Present when and only when status_reason is internal_error, which
        # NavResult holds in step, so a consumer that sees the reason can
        # read the component without checking for the key's absence first.
        out['internal_error'] = {
            'component': result.internal_error.component,
            'exception_type': result.internal_error.exception_type,
        }
    if result.conflict_arbitration is not None:
        out['conflict_arbitration'] = _curate_conflict_arbitration(result.conflict_arbitration)
    if result.pointing is not None:
        out['pointing'] = _curate_pointing(result.pointing)
        out['times'] = _curate_times(result.pointing.baseline)
    return out
