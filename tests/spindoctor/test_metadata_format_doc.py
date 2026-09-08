"""Staleness guard for the per-image metadata format chapter.

Contract under test (docs/user_guide/user_guide_metadata.rst): the chapter
specifies every key the metadata writers emit, and its JSON examples parse
and structurally match real writer output. Two directions are enforced:

1. Writer-to-chapter: every key name any writer emits -- across a fully
   populated success result (with and without a fitted rotation), a failed
   result, a load-error document, and an early-return document -- appears in
   the chapter as an inline ``key`` literal. A writer gaining a key the
   chapter lacks fails here.
2. Chapter-to-writer: each example's key structure equals the corresponding
   writer output's key structure, block by block, and the open-vocabulary
   sub-objects (diagnostics, reliability_reasons, feature_count_by_type) use
   only names their in-code vocabularies define. An example claiming a key
   no writer emits fails here.
"""

from __future__ import annotations

import dataclasses
import json
import re
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from filecache import FCPath

import spindoctor.nav_technique.diagnostics as diagnostics_module
from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.feature.feature import NavReliabilityBreakdown
from spindoctor.feature.feature_type import NavFeatureType
from spindoctor.nav_orchestrator.feature_summary import NavFeatureSummary
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import NavInternalErrorRecord, NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.nav_technique.diagnostics import BodyLimbDiagnostics
from spindoctor.nav_technique.technique_result import NavTechniqueResult
from spindoctor.navigate_image_files import (
    build_metadata_from_result,
    build_timing_section,
    navigate_image_files,
)
from spindoctor.support.cmatrix import AttitudeBaseline, PointingSolution
from spindoctor.support.file import json_as_string
from spindoctor.support.status_reason import NavStatusReason

_CHAPTER_PATH = (
    Path(__file__).resolve().parents[2] / 'docs' / 'user_guide' / 'user_guide_metadata.rst'
)

# Keys whose object values carry open-vocabulary content (feature-type
# counts, per-file hashes, per-catalog paths, per-technique diagnostics,
# per-feature reliability components) rather than schema keys.  Key
# collection stops at them; the vocabularies with in-code definitions are
# checked by their own tests below.
_OPEN_MAPPING_KEYS = frozenset(
    {
        'feature_count_by_type',
        'static_data_hashes',
        'star_catalogs',
        'diagnostics',
        'reliability_reasons',
    }
)


def _chapter_text() -> str:
    """Return the chapter source, failing clearly if the file moved."""
    assert _CHAPTER_PATH.is_file(), f'metadata format chapter missing at {_CHAPTER_PATH}'
    return _CHAPTER_PATH.read_text(encoding='utf-8')


def _example_json_blocks() -> list[dict[str, Any]]:
    """Parse every ``.. code-block:: json`` example out of the chapter.

    Returns:
        The parsed JSON documents, in chapter order.
    """
    text = _chapter_text()
    blocks: list[dict[str, Any]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == '.. code-block:: json':
            i += 1
            body: list[str] = []
            # Skip the blank line(s) after the directive, then take every
            # line that is blank or indented deeper than the directive.
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith('    ')):
                body.append(lines[i])
                i += 1
            blocks.append(json.loads(textwrap.dedent('\n'.join(body))))
        else:
            i += 1
    return blocks


def _documented_key_literals() -> set[str]:
    """Every inline ``literal`` in the chapter that is shaped like a JSON key."""
    return set(re.findall(r'``([a-z_][a-z0-9_]*)``', _chapter_text()))


def _leaf_key_names(node: Any) -> set[str]:
    """Collect every dict key name in a document, recursively.

    Descent stops at the open-vocabulary mappings named in
    ``_OPEN_MAPPING_KEYS`` (their keys are content, not schema).
    """
    names: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            names.add(key)
            if key not in _OPEN_MAPPING_KEYS:
                names |= _leaf_key_names(value)
    elif isinstance(node, list):
        for item in node:
            names |= _leaf_key_names(item)
    return names


def _key_structure(node: Any) -> Any:
    """Reduce a document to its key structure for shape comparison.

    Dicts map each key to its reduced value; lists reduce to the sorted set
    of their elements' reduced structures (serialized for hashability);
    scalars reduce to None.  Open-vocabulary mappings reduce to the marker
    string ``'<open>'`` so differing content keys compare equal.
    """
    if isinstance(node, dict):
        return {
            key: ('<open>' if key in _OPEN_MAPPING_KEYS else _key_structure(value))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return sorted({json.dumps(_key_structure(item), sort_keys=True) for item in node})
    return None


def _round_trip(metadata: dict[str, Any]) -> dict[str, Any]:
    """Serialize through the real writer path and parse back.

    Proves the document is what a reader of the written file sees, numpy
    types and all.
    """
    out = json.loads(json_as_string(metadata))
    assert isinstance(out, dict)
    return out


def _classifier() -> NavImageClassifierResult:
    """A populated classifier verdict, including the optional score."""
    return NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=1.2,
        max_dn=100.0,
        background_gradient_score=1.7,
    )


def _provenance() -> Provenance:
    """A populated provenance envelope."""
    return Provenance(
        spindoctor_version='0.0.0',
        image_et=309861208.316457,
        pipeline_run_iso8601='2026-08-08T16:46:29Z',
        spindoctor_git_sha='719cde5',
        spice_kernels=('naif0012.tls',),
        static_data_hashes={'config_220_body_shape.yaml': 'ab' * 32},
        technique_names=('BodyLimbNav',),
        extractor_names=('body:RHEA',),
        config_hash='cd' * 32,
        config_overrides=(),
        star_catalogs={'ucac4': '/catalogs/UCAC4'},
    )


def _technique_result(*, with_rotation: bool) -> NavTechniqueResult:
    """One technique result, optionally carrying a fitted rotation."""
    size = 3 if with_rotation else 2
    return NavTechniqueResult(
        technique_name='BodyLimbNav',
        feature_ids=('limb_arc:RHEA',),
        offset_px=(-1.1201, -5.9495),
        covariance_px2=np.diag([6.8] * size),
        confidence=0.788,
        spurious=False,
        at_edge=False,
        diagnostics=BodyLimbDiagnostics(visible_arc_px=277.0, dt_fit_rms_px=0.278),
        rotation_rad=0.001 if with_rotation else None,
        sigma_rotation_rad=0.0005 if with_rotation else None,
    )


def _feature_summary() -> NavFeatureSummary:
    """One ungated feature-inventory entry with a populated breakdown."""
    return NavFeatureSummary(
        feature_id='limb_arc:RHEA',
        feature_type=NavFeatureType.LIMB_ARC,
        source_model='body:RHEA',
        reliability=0.814,
        gated=False,
        gate_reason=None,
        bbox_extfov_vu=(495, 577, 644, 727),
        reliability_reasons=NavReliabilityBreakdown(visible_arc_fraction=1.0),
    )


def _baseline() -> AttitudeBaseline:
    """A valid attitude baseline (identity rotations, real epoch shapes)."""
    return AttitudeBaseline(
        cmatrix_original=np.eye(3),
        oops_from_spice=np.eye(3),
        camera_frame='CASSINI_ISS_NAC',
        camera_frame_id=-82360,
        ck_frame_id=-82000,
        start_et=309861208.2064568,
        stop_et=309861208.4264568,
        midtime_et=309861208.3164568,
        exposure_s=0.22,
        sclk_start='1/1635282917.129',
        sclk_midtime='1/1635282917.157',
        sclk_stop='1/1635282917.186',
    )


def _with_pointing(result: NavResult, *, corrected: bool) -> NavResult:
    """Stamp a pointing solution onto a result, as the orchestrator does."""
    cmatrix = np.eye(3) if corrected else None
    return dataclasses.replace(
        result, pointing=PointingSolution(baseline=_baseline(), cmatrix=cmatrix)
    )


def _timing() -> dict[str, Any]:
    """A real timing section."""
    start = datetime.now(UTC)
    return build_timing_section(start, datetime.now(UTC))


def _document(result: NavResult, *, shutter_mode: str | None = 'NACONLY') -> dict[str, Any]:
    """Build the full written document for a navigated result.

    Parameters:
        result: The navigation result to curate.
        shutter_mode: The host's shutter mode, or ``None`` for a host whose
            labels carry none (the field is then omitted, as the failed
            example's Galileo host shows).
    """
    return _round_trip(
        build_metadata_from_result(
            result,
            Path('/holdings/N0.IMG'),
            'N0.IMG',
            instrument='coiss',
            camera='NAC',
            shutter_mode=shutter_mode,
            image_shape=(1024, 1024),
            timing=_timing(),
        )
    )


def _success_document() -> dict[str, Any]:
    """The written document for a plain (translation-only) success."""
    result = NavResult.success(
        offset_px=(-1.1201, -5.9495),
        covariance_px2=np.diag([6.8, 6.8]),
        confidence=0.788,
        confidence_rank='low',
        status_reason=NavStatusReason.OK,
        per_technique=[_technique_result(with_rotation=False)],
        feature_inventory=[_feature_summary()],
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    return _document(_with_pointing(result, corrected=True))


def _rotation_document() -> dict[str, Any]:
    """The written document for a success that fitted a camera rotation.

    A fitted rotation records no corrected attitude, so the pointing block
    carries ``cmatrix_original`` only; the rotation keys appear at both the
    combined and the per-technique level.
    """
    result = NavResult.success(
        offset_px=(-1.1201, -5.9495),
        covariance_px2=np.diag([6.8, 6.8, 1e-6]),
        confidence=0.788,
        confidence_rank='low',
        status_reason=NavStatusReason.OK,
        per_technique=[_technique_result(with_rotation=True)],
        feature_inventory=[_feature_summary()],
        image_classifier=_classifier(),
        provenance=_provenance(),
        sigma_along_unobservable_px=float('inf'),
        rotation_rad=0.001,
        sigma_rotation_rad=0.0005,
    )
    return _document(_with_pointing(result, corrected=False))


def _failed_document() -> dict[str, Any]:
    """The written document for a failed navigation, pointing baseline only."""
    result = NavResult.failed(
        status_reason=NavStatusReason.NO_FEATURES_EXTRACTED,
        image_classifier=_classifier(),
        provenance=_provenance(),
    )
    return _document(_with_pointing(result, corrected=False), shutter_mode=None)


def _internal_error_document() -> dict[str, Any]:
    """The written document for a navigation a plugin's exception failed.

    Carried by the writer-to-chapter guard because its ``internal_error``
    block appears on no other document shape: without it the chapter could
    stop documenting the block and every check here would still pass.
    """
    result = NavResult.failed(
        status_reason=NavStatusReason.INTERNAL_ERROR,
        image_classifier=_classifier(),
        provenance=_provenance(),
        internal_error=NavInternalErrorRecord(
            component='rings:SATURN.create_model',
            exception_type='AttributeError',
        ),
    )
    return _document(_with_pointing(result, corrected=False), shutter_mode=None)


class _LoadErrorObsClass:
    """Observation class whose load always fails with a SPICE coverage error."""

    @classmethod
    def from_file(cls, path: Any, **kwargs: Any) -> Any:
        """Raise the coverage error the load-error document classifies.

        Parameters:
            path: The image path the driver resolved; unread.
            kwargs: Further loader options; unread.

        Raises:
            RuntimeError: always, carrying a SPICE coverage hint.
        """
        raise RuntimeError('SPICE(NOFRAMECONNECT) -- insufficient information')


def _image_files(tmp_path: Path, count: int) -> ImageFiles:
    """A batch of ``count`` placeholder images carrying the index camera."""
    img_path = tmp_path / 'N0.IMG'
    img_path.write_bytes(b'\x00')
    entry = ImageFile(
        image_file_url=FCPath(img_path),
        label_file_url=FCPath(img_path),
        results_path_stub='N0',
        camera='NAC',
    )
    return ImageFiles(image_files=[entry] * count)


def _load_error_document(tmp_path: Path) -> dict[str, Any]:
    """The document the driver returns for an image whose load fails."""
    _success, metadata = navigate_image_files(
        _LoadErrorObsClass,  # type: ignore[arg-type]
        _image_files(tmp_path, 1),
        FCPath(tmp_path / 'results'),
        write_output_files=False,
    )
    return _round_trip(metadata)


def _early_return_document(tmp_path: Path) -> dict[str, Any]:
    """The document the driver returns for a malformed (two-image) batch."""
    _success, metadata = navigate_image_files(
        _LoadErrorObsClass,  # type: ignore[arg-type]
        _image_files(tmp_path, 2),
        FCPath(tmp_path / 'results'),
        write_output_files=False,
    )
    return _round_trip(metadata)


class _UnnavigableObsClass:
    """Observation class whose snapshot the orchestrator cannot even start on."""

    @classmethod
    def from_file(cls, path: Any, **kwargs: Any) -> Any:
        """Return an object with none of the attributes a snapshot has.

        Parameters:
            path: The image path the driver resolved; unread.
            kwargs: Further loader options; unread.

        Returns:
            A bare object, so building the models raises inside the driver's
            per-image boundary and the internal-error document is written.
        """
        return object()


def _driver_internal_error_document(tmp_path: Path) -> dict[str, Any]:
    """The document the driver returns for an image whose navigation raised."""
    _success, metadata = navigate_image_files(
        _UnnavigableObsClass,  # type: ignore[arg-type]
        _image_files(tmp_path, 1),
        FCPath(tmp_path / 'results'),
        write_output_files=False,
    )
    return _round_trip(metadata)


# --- writer-to-chapter: every emitted key is documented ---


def test_every_writer_key_is_documented(tmp_path: Path) -> None:
    """Every key any writer emits appears in the chapter as a literal.

    This is the staleness guard's forward direction: a writer gaining a key
    the chapter does not document fails here, naming the missing keys.
    """
    emitted: set[str] = set()
    emitted |= _leaf_key_names(_success_document())
    emitted |= _leaf_key_names(_rotation_document())
    emitted |= _leaf_key_names(_failed_document())
    emitted |= _leaf_key_names(_internal_error_document())
    emitted |= _leaf_key_names(_load_error_document(tmp_path))
    emitted |= _leaf_key_names(_early_return_document(tmp_path))
    missing = emitted - _documented_key_literals()
    assert not missing, (
        f'writer emits keys the metadata chapter never documents: {sorted(missing)}; '
        f'update docs/user_guide/user_guide_metadata.rst'
    )


def test_documented_status_reasons_cover_the_enum() -> None:
    """Every NavStatusReason value is documented in the chapter."""
    documented = _documented_key_literals()
    missing = {reason.value for reason in NavStatusReason} - documented
    assert not missing, f'status_reason values missing from the chapter: {sorted(missing)}'


# --- chapter-to-writer: the examples match real output ---


def test_chapter_carries_one_example_per_document_shape() -> None:
    """The chapter has exactly five JSON examples, in the documented order."""
    blocks = _example_json_blocks()
    assert len(blocks) == 5


def test_success_example_matches_writer_structure() -> None:
    """The success example's key structure equals real success output."""
    example = _example_json_blocks()[0]
    assert _key_structure(example) == _key_structure(_success_document())


def test_failed_example_matches_writer_structure() -> None:
    """The failed example's key structure equals real failed output.

    The failed writer document carries empty ``per_technique`` and
    ``feature_inventory`` lists, whose element structure the success example
    already pins, so list contents compare equal by construction here.
    """
    example = _example_json_blocks()[1]
    assert _key_structure(example) == _key_structure(_failed_document())


def test_load_error_example_matches_writer_structure(tmp_path: Path) -> None:
    """The load-error example's key structure equals real driver output."""
    example = _example_json_blocks()[2]
    assert _key_structure(example) == _key_structure(_load_error_document(tmp_path))


def test_internal_error_example_matches_writer_structure(tmp_path: Path) -> None:
    """The internal-error example's key structure equals real driver output."""
    example = _example_json_blocks()[3]
    document = _driver_internal_error_document(tmp_path)
    assert document['status_error'] == 'internal_error'
    assert _key_structure(example) == _key_structure(document)


def test_early_return_example_matches_writer_structure(tmp_path: Path) -> None:
    """The early-return example's key structure equals real driver output."""
    example = _example_json_blocks()[4]
    assert _key_structure(example) == _key_structure(_early_return_document(tmp_path))


def test_failed_example_has_no_top_level_offset() -> None:
    """The failed example omits the top-level offset key, as the writer does."""
    example = _example_json_blocks()[1]
    assert 'offset' not in example


# --- open-vocabulary content in the examples stays within its vocabularies ---


def _all_curator_json_keys() -> set[str]:
    """Union of every diagnostics CURATOR_FIELDS JSON key name."""
    keys: set[str] = set()
    for name in diagnostics_module.__all__:
        cls = getattr(diagnostics_module, name)
        curator_fields = getattr(cls, 'CURATOR_FIELDS', None)
        if curator_fields is None:
            continue
        keys |= {json_key for json_key in curator_fields.values() if json_key is not None}
    return keys


@pytest.mark.parametrize('example_index', [0, 1])
def test_example_diagnostics_keys_exist_in_some_technique(example_index: int) -> None:
    """Every diagnostics key in the navigated examples is a real curator key."""
    example = _example_json_blocks()[example_index]
    allowed = _all_curator_json_keys()
    seen: set[str] = set()
    for entry in example['navigation_result']['per_technique']:
        seen |= set(entry['diagnostics'])
    unknown = seen - allowed
    assert not unknown, f'example diagnostics keys not in any CURATOR_FIELDS: {sorted(unknown)}'


def test_example_reliability_reason_keys_are_breakdown_fields() -> None:
    """Every reliability_reasons key in the examples is a breakdown field."""
    example = _example_json_blocks()[0]
    allowed = {f.name for f in dataclasses.fields(NavReliabilityBreakdown)}
    seen: set[str] = set()
    for entry in example['navigation_result']['feature_inventory']:
        seen |= set(entry['reliability_reasons'])
    unknown = seen - allowed
    assert not unknown, f'example reliability components not on the breakdown: {sorted(unknown)}'


def test_example_feature_count_types_are_feature_type_names() -> None:
    """Every feature_count_by_type key in the examples is a NavFeatureType."""
    example = _example_json_blocks()[0]
    allowed = {feature_type.name for feature_type in NavFeatureType}
    seen = set(example['navigation_result']['feature_count_by_type'])
    unknown = seen - allowed
    assert not unknown, f'example feature-count types not in NavFeatureType: {sorted(unknown)}'
