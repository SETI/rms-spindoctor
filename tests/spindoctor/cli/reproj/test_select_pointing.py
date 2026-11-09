"""Hermetic tests for the pointing selection ladder and its application.

``select_pointing`` classifies one image's parsed navigation record into a
mechanism (C-matrix, offset, or none) with a per-reason short form, and
``apply_pointing_to_obs`` applies the classification to a real oops
observation.  Every test builds its own record and its own observation on the
built-in SSB path and J2000 frame, so nothing here touches SPICE kernels or
holdings; the instrument table lookup inside the C-matrix reader is injected,
exactly as the reader's own unit tests inject it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import oops
import pdslogger
import pytest
from filecache import FCPath
from oops.observation import Snapshot
from tests.cmatrix_helpers import (
    observation_attitude,
    some_attitude,
    synthetic_baseline,
    synthetic_frame_identity,
)

import spindoctor.support.cmatrix as cmatrix_module
from spindoctor.cli.reproj.offsets import (
    AppliedPointing,
    PointingMechanism,
    PointingSelection,
    apply_pointing_to_obs,
    select_pointing,
)
from spindoctor.config import (
    IMAGE_LOGGER,
    MAIN_LOGGER,
    LogLevels,
    LogSinks,
    build_image_log_handlers,
    build_main_logger,
    set_log_levels,
)
from spindoctor.config.program_names import SD_MOSAIC
from spindoctor.obs import ObsSnapshotInst
from spindoctor.obs.obs_snapshot import ObsSnapshot
from spindoctor.support.cmatrix import _build_pointing_solution

# The synthetic exposure every observation here uses: midtime 100.25, the
# epoch the synthetic baseline records, so a valid record passes the reader's
# midtime gate.
_TSTART = 100.0
_TEXP = 0.5
_MIDTIME = 100.25

# The planted offset: distinct, non-zero, opposite-signed components so an
# axis swap or sign flip cannot go unnoticed.
_OFFSET_DV_DU = (3.0, -4.0)

_SHAPE_VU = (10, 12)
_IDENTITY = np.eye(3)

_STAMP = '2026-08-07T12-00-00'


def _hermetic_obs(frame: Any = 'J2000') -> ObsSnapshot:
    """Build a real snapshot observation on the built-in SSB path.

    Parameters:
        frame: The observation frame: the registered J2000 frame by default,
            or any oops frame object (an unregistered ``Cmatrix`` stands in
            for a pool that answers some other attitude).

    Returns:
        The observation, with the closest-planet scan bypassed.
    """
    size_v, size_u = _SHAPE_VU
    fov = oops.fov.FlatFOV((0.001, 0.001), (size_u, size_v))
    snapshot = Snapshot(
        axes=('v', 'u'),
        cadence=oops.cadence.SnapCadence(_TSTART, _TEXP),
        fov=fov,
        path='SSB',
        frame=frame,
    )
    snapshot.insert_subfield('data', np.zeros(_SHAPE_VU, dtype=np.float32))
    # The subfields an oops host declares, which the reader reads to learn the
    # convention and the frame the observation was built on.
    snapshot.insert_subfield('spice_to_frame', oops.Matrix3(_IDENTITY))
    snapshot.insert_subfield('spice_frame_name', 'J2000')
    snapshot.insert_subfield('spice_frame_id', 1)
    return ObsSnapshot(snapshot, extfov_margin_vu=(0, 0), closest_planet=None)


def _recorded_cmatrix(obs: ObsSnapshot) -> np.ndarray:
    """Produce a corrected attitude the writing half would record for ``obs``.

    Built with the identity flip against an identity baseline, which is what
    the J2000-framed hermetic observation's attitude is, so the record means
    for this observation exactly what a real record means for a real one.

    Parameters:
        obs: The observation the record is for.

    Returns:
        The corrected 3x3 rotation the planted offset implies.
    """
    solution = _build_pointing_solution(
        synthetic_baseline(_IDENTITY), obs.fov, offset_px=_OFFSET_DV_DU, rotation_fitted=False
    )
    assert solution.cmatrix is not None
    return np.asarray(solution.cmatrix, np.float64)


def _flat(matrix: np.ndarray) -> list[float]:
    """Flatten a rotation to the nine row-major floats the schema records.

    Parameters:
        matrix: The rotation.

    Returns:
        Its elements as plain floats.
    """
    return [float(value) for value in np.asarray(matrix, np.float64).reshape(9)]


def _record(
    *,
    cmatrix: Any = None,
    cmatrix_original: Any = None,
    midtime_et: Any = _MIDTIME,
    offset: Any = 'default',
    with_pointing: bool = True,
    with_times: bool = True,
    with_offset_key: bool = True,
) -> dict[str, Any]:
    """Build one success-status navigation record, malleable per test.

    Parameters:
        cmatrix: The recorded corrected attitude, or None to omit the key.
        cmatrix_original: The recorded baseline, or None to omit the key.
        midtime_et: The recorded midtime.
        offset: The recorded offset value; the sentinel ``'default'`` records
            the planted offset.
        with_pointing: Whether the ``pointing`` block exists at all.
        with_times: Whether the ``times`` block exists at all.
        with_offset_key: Whether the top-level ``offset`` key exists at all.

    Returns:
        The record.
    """
    nav_result: dict[str, Any] = {}
    if with_pointing:
        pointing: dict[str, Any] = {}
        if cmatrix is not None:
            pointing['cmatrix'] = cmatrix
        if cmatrix_original is not None:
            pointing['cmatrix_original'] = cmatrix_original
        nav_result['pointing'] = pointing
    if with_times:
        nav_result['times'] = {'midtime_et': midtime_et}
    metadata: dict[str, Any] = {'status': 'success', 'navigation_result': nav_result}
    if with_offset_key:
        metadata['offset'] = list(_OFFSET_DV_DU) if offset == 'default' else offset
    return metadata


def _valid_record() -> dict[str, Any]:
    """Build a record carrying a usable C-matrix pair and offset."""
    obs = _hermetic_obs()
    return _record(cmatrix=_flat(_recorded_cmatrix(obs)), cmatrix_original=_flat(_IDENTITY))


def _inject_identity_flip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the reader see a synthetic instrument whose flip is the identity.

    Parameters:
        monkeypatch: The patcher.
    """
    identity = synthetic_frame_identity(_IDENTITY)
    monkeypatch.setattr(cmatrix_module, '_frame_identity', lambda obs: identity)


# ---------------------------------------------------------------------------
# The selection ladder
# ---------------------------------------------------------------------------


def test_a_usable_cmatrix_record_selects_the_cmatrix_mechanism() -> None:
    """A record with a usable pointing block selects the C-matrix, cleanly."""
    selection = select_pointing(_valid_record())
    assert selection.mechanism is PointingMechanism.CMATRIX


def test_a_clean_cmatrix_selection_carries_no_reason() -> None:
    """The exact mechanism is not a degradation, so there is nothing to count."""
    selection = select_pointing(_valid_record())
    assert selection.reason is None


def test_the_cmatrix_selection_carries_the_parsed_values() -> None:
    """The recorded matrices and midtime ride along for the application."""
    record = _valid_record()
    selection = select_pointing(record)
    assert selection.cmatrix is not None
    assert selection.cmatrix_original is not None
    assert np.array_equal(
        np.asarray(selection.cmatrix).reshape(9),
        np.asarray(record['navigation_result']['pointing']['cmatrix']),
    )
    assert selection.midtime_et == _MIDTIME


def test_the_cmatrix_selection_carries_the_offset_for_fallback() -> None:
    """A gate failure at apply time degrades to the offset, so it rides along."""
    selection = select_pointing(_valid_record())
    assert selection.offset == _OFFSET_DV_DU


def test_a_pointing_block_without_a_cmatrix_selects_the_offset() -> None:
    """A fitted-rotation record -- pointing block, no cmatrix -- takes the offset."""
    record = _record(cmatrix_original=_flat(_IDENTITY))
    selection = select_pointing(record)
    assert selection.mechanism is PointingMechanism.OFFSET
    assert selection.reason == 'no_cmatrix_rotation_fitted'


def test_a_record_without_a_pointing_block_selects_the_offset() -> None:
    """A simulated or pre-pointing-schema record takes the offset."""
    record = _record(with_pointing=False, with_times=False)
    selection = select_pointing(record)
    assert selection.mechanism is PointingMechanism.OFFSET
    assert selection.reason == 'no_pointing_block'


def _malformed_records() -> dict[str, dict[str, Any]]:
    """Build one record per malformed-pointing class.

    Returns:
        Records keyed by the malformation they carry.
    """
    obs = _hermetic_obs()
    good = _flat(_recorded_cmatrix(obs))
    identity = _flat(_IDENTITY)
    nan_matrix = list(good)
    nan_matrix[4] = math.nan
    return {
        'eight-elements': _record(cmatrix=good[:8], cmatrix_original=identity),
        'nan-element': _record(cmatrix=nan_matrix, cmatrix_original=identity),
        'bool-elements': _record(cmatrix=[True] * 9, cmatrix_original=identity),
        # The same promotion one nesting deeper, which is where it survives a
        # reader that judges the nine entries by their own type: the entries
        # are containers, the assembled array is float64, and one ``True``
        # among eight numbers reads as the ``1.0`` that completes an identity.
        'bool-among-numbers-in-rows-of-one': _record(
            cmatrix=[[True], *([value] for value in identity[1:])], cmatrix_original=identity
        ),
        'bool-among-numbers-in-the-baseline': _record(
            cmatrix=good, cmatrix_original=[[True], *([value] for value in identity[1:])]
        ),
        'str-elements': _record(cmatrix=[str(v) for v in good], cmatrix_original=identity),
        'not-a-rotation': _record(
            cmatrix=_flat(np.diag([1.0, 1.0, 2.0])), cmatrix_original=identity
        ),
        'absent-original': _record(cmatrix=good),
        'absent-times': _record(cmatrix=good, cmatrix_original=identity, with_times=False),
        'nan-midtime': _record(cmatrix=good, cmatrix_original=identity, midtime_et=math.nan),
        'bool-midtime': _record(cmatrix=good, cmatrix_original=identity, midtime_et=True),
        'text-midtime': _record(cmatrix=good, cmatrix_original=identity, midtime_et='100.25'),
    }


@pytest.mark.parametrize('malformation', sorted(_malformed_records()))
def test_a_malformed_pointing_block_selects_the_offset(malformation: str) -> None:
    """Each malformed-pointing class degrades to the offset path, classified.

    NaN defeats every comparison -- including a NaN ``midtime_et`` against
    the reader's midtime gate -- booleans and numeric text convert to float64
    without complaint, and eight elements reshape nowhere, so each class is
    probed as its own input domain.

    Parameters:
        malformation: Which malformed record to classify.
    """
    selection = select_pointing(_malformed_records()[malformation])
    assert selection.mechanism is PointingMechanism.OFFSET
    assert selection.reason == 'malformed_pointing'


def test_a_malformed_pointing_block_without_an_offset_selects_none() -> None:
    """With no usable offset either, a malformed record leaves pointing alone."""
    record = _record(cmatrix=[True] * 9, cmatrix_original=_flat(_IDENTITY), offset=None)
    selection = select_pointing(record)
    assert selection.mechanism is PointingMechanism.NONE
    assert selection.reason == 'malformed_pointing'


def test_a_missing_offset_key_is_classified_distinctly() -> None:
    """A success record with no offset key at all is defect-shaped and says so.

    Reading the document, an absent field is distinguishable from one holding
    null, and the reason keeps them apart for the run-level tally.  Neither
    supplies a pointing, so the two build the same product.
    """
    record = _record(with_pointing=False, with_times=False, with_offset_key=False)
    selection = select_pointing(record)
    assert selection.reason == 'missing_offset_key'


def test_a_null_offset_is_classified_apart_from_a_missing_key() -> None:
    """A null offset is a recorded no-answer, not a missing field."""
    record = _record(with_pointing=False, with_times=False, offset=None)
    selection = select_pointing(record)
    assert selection.reason == 'null_offset'


@pytest.mark.parametrize(
    'offset',
    [[1.0, 2.0, 3.0], [1.0], [], [1.0, 2.0, None], {'dv': 1.0, 'du': 2.0}],
    ids=['three', 'one', 'empty', 'three-with-null', 'object'],
)
def test_an_offset_that_is_not_a_pair_is_refused_whole(offset: Any) -> None:
    """Only two recorded values are a pair; the rest is refused, never truncated.

    A reader that took the first two of three would apply a pointing nobody
    recorded, and the store that holds these records would have to make the
    same choice or build a different product from the same document.

    Parameters:
        offset: The recorded value under test.
    """
    record = _record(with_pointing=False, with_times=False, offset=offset)
    selection = select_pointing(record)
    assert selection.reason == 'malformed_offset'


def test_an_offset_of_numeric_strings_is_read_as_the_numbers_it_names() -> None:
    """A pair the reader can convert is a pair the reader applies."""
    record = _record(with_pointing=False, with_times=False, offset=['5.5', '-1.25'])
    selection = select_pointing(record)
    assert selection.offset == (5.5, -1.25)


# ---------------------------------------------------------------------------
# Applying a selection
# ---------------------------------------------------------------------------


def test_a_cmatrix_selection_replaces_the_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """The C-matrix mechanism replaces the frame with the recorded attitude.

    oops rounds the matrix it is given to the nearest exact rotation before it builds the
    frame, which moves the last bit of some elements, so the comparison allows that.
    """
    obs = _hermetic_obs()
    cmatrix = _recorded_cmatrix(obs)
    _inject_identity_flip(monkeypatch)
    selection = select_pointing(_record(cmatrix=_flat(cmatrix), cmatrix_original=_flat(_IDENTITY)))
    applied = apply_pointing_to_obs(cast(ObsSnapshotInst, obs), selection)
    assert applied == AppliedPointing(source='cmatrix', reason=None)
    corrected = observation_attitude(cast(ObsSnapshotInst, obs), _MIDTIME)
    assert corrected == pytest.approx(cmatrix, rel=0.0, abs=1e-15)


def test_the_cmatrix_mechanism_leaves_the_fov_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Precedence: with both fields recorded, the FOV is never wrapped."""
    obs = _hermetic_obs()
    fov_before = obs.fov
    _inject_identity_flip(monkeypatch)
    selection = select_pointing(
        _record(cmatrix=_flat(_recorded_cmatrix(obs)), cmatrix_original=_flat(_IDENTITY))
    )
    apply_pointing_to_obs(cast(ObsSnapshotInst, obs), selection)
    assert obs.fov is fov_before


def test_an_already_corrected_pool_applies_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pool already answering the corrected attitude is left alone.

    The observation frame is built as the corrected attitude itself -- the
    state corrected kernels furnished at load time produce -- so the flip
    gate fires, the probe recognizes the state, and neither the frame nor
    the FOV is touched: either fallback would corrupt an observation that is
    already right.
    """
    plain = _hermetic_obs()
    cmatrix = _recorded_cmatrix(plain)
    obs = _hermetic_obs(frame=oops.frame.Cmatrix(cmatrix))
    frame_before = obs.frame
    fov_before = obs.fov
    _inject_identity_flip(monkeypatch)
    selection = select_pointing(_record(cmatrix=_flat(cmatrix), cmatrix_original=_flat(_IDENTITY)))
    applied = apply_pointing_to_obs(cast(ObsSnapshotInst, obs), selection)
    assert applied == AppliedPointing(source='pool', reason='pool_already_corrected')
    assert obs.frame is frame_before
    assert obs.fov is fov_before


def test_a_foreign_midtime_falls_back_to_the_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A record from another observation degrades to the offset path."""
    obs = _hermetic_obs()
    _inject_identity_flip(monkeypatch)
    selection = select_pointing(
        _record(
            cmatrix=_flat(_recorded_cmatrix(obs)),
            cmatrix_original=_flat(_IDENTITY),
            midtime_et=_MIDTIME + 1.0,
        )
    )
    applied = apply_pointing_to_obs(cast(ObsSnapshotInst, obs), selection)
    assert applied == AppliedPointing(source='offset', reason='cmatrix_foreign_midtime')
    assert isinstance(obs.fov, oops.fov.OffsetFOV)


def test_a_baseline_mismatch_falls_back_to_the_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexplained gate failure ends with the offset applied, never the cmatrix."""
    obs = _hermetic_obs()
    _inject_identity_flip(monkeypatch)
    # The recorded baseline is not this observation's attitude, and the pool
    # does not answer the corrected attitude either: the unexplained class.
    selection = select_pointing(
        _record(
            cmatrix=_flat(_recorded_cmatrix(obs) @ some_attitude()),
            cmatrix_original=_flat(some_attitude()),
        )
    )
    applied = apply_pointing_to_obs(cast(ObsSnapshotInst, obs), selection)
    assert applied == AppliedPointing(source='offset', reason='cmatrix_baseline_mismatch')
    assert isinstance(obs.fov, oops.fov.OffsetFOV)


def test_the_offset_fallback_carries_the_recorded_uv_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback applies the metadata (dv, du) as an OffsetFOV (du, dv)."""
    obs = _hermetic_obs()
    _inject_identity_flip(monkeypatch)
    selection = select_pointing(
        _record(
            cmatrix=_flat(_recorded_cmatrix(obs)),
            cmatrix_original=_flat(_IDENTITY),
            midtime_et=_MIDTIME + 1.0,
        )
    )
    apply_pointing_to_obs(cast(ObsSnapshotInst, obs), selection)
    fov = cast(oops.fov.OffsetFOV, obs.fov)
    assert np.asarray(fov.uv_offset.vals)[0] == _OFFSET_DV_DU[1]
    assert np.asarray(fov.uv_offset.vals)[1] == _OFFSET_DV_DU[0]


def test_a_gate_failure_with_no_offset_leaves_pointing_uncorrected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no usable offset to fall back on, nothing is applied at all."""
    obs = _hermetic_obs()
    fov_before = obs.fov
    _inject_identity_flip(monkeypatch)
    selection = select_pointing(
        _record(
            cmatrix=_flat(_recorded_cmatrix(obs)),
            cmatrix_original=_flat(_IDENTITY),
            midtime_et=_MIDTIME + 1.0,
            offset=None,
        )
    )
    applied = apply_pointing_to_obs(cast(ObsSnapshotInst, obs), selection)
    assert applied == AppliedPointing(source='none', reason='cmatrix_foreign_midtime')
    assert obs.fov is fov_before


def test_an_offset_selection_wraps_the_fov() -> None:
    """The offset mechanism is exactly the OffsetFOV application it always was."""
    obs = _hermetic_obs()
    selection = select_pointing(_record(with_pointing=False, with_times=False))
    applied = apply_pointing_to_obs(cast(ObsSnapshotInst, obs), selection)
    assert applied == AppliedPointing(source='offset', reason='no_pointing_block')
    fov = cast(oops.fov.OffsetFOV, obs.fov)
    assert np.asarray(fov.uv_offset.vals)[0] == _OFFSET_DV_DU[1]
    assert np.asarray(fov.uv_offset.vals)[1] == _OFFSET_DV_DU[0]


def test_a_none_selection_touches_nothing() -> None:
    """A record with no usable pointing leaves the observation alone."""
    obs = _hermetic_obs()
    fov_before = obs.fov
    frame_before = obs.frame
    selection = select_pointing(_record(with_pointing=False, with_times=False, offset=None))
    applied = apply_pointing_to_obs(cast(ObsSnapshotInst, obs), selection)
    assert applied == AppliedPointing(source='none', reason='null_offset')
    assert obs.fov is fov_before
    assert obs.frame is frame_before


def test_a_contradictory_selection_is_refused() -> None:
    """A CMATRIX selection without its values is a caller defect, not a record."""
    obs = _hermetic_obs()
    selection = PointingSelection(
        mechanism=PointingMechanism.CMATRIX,
        cmatrix=None,
        cmatrix_original=None,
        midtime_et=None,
        offset=None,
        reason=None,
    )
    with pytest.raises(ValueError, match='must carry cmatrix'):
        apply_pointing_to_obs(cast(ObsSnapshotInst, obs), selection)


@pytest.mark.parametrize('mechanism', ['cmatrix', 'offset'])
def test_geometry_after_application_is_built_on_the_corrected_observation(
    monkeypatch: pytest.MonkeyPatch, mechanism: str
) -> None:
    """Cached geometry from before the application never leaks into products.

    The observation's ``Backplane`` is primed before the pointing is applied,
    exactly as the closest-planet scan primes one during ``from_file``; the
    probed quantity (right ascension) is not invariant under either the frame
    swap or the FOV wrap, so if ``apply_pointing_to_obs`` stops resetting the
    caches, the stale value comes back unchanged and this test fails.

    Parameters:
        mechanism: Which mechanism to apply.
    """
    obs = _hermetic_obs()
    _inject_identity_flip(monkeypatch)
    before = float(np.mean(np.asarray(obs.bp.right_ascension().vals, np.float64)))
    if mechanism == 'cmatrix':
        record = _record(cmatrix=_flat(_recorded_cmatrix(obs)), cmatrix_original=_flat(_IDENTITY))
    else:
        record = _record(with_pointing=False, with_times=False)
    apply_pointing_to_obs(cast(ObsSnapshotInst, obs), select_pointing(record))
    after = float(np.mean(np.asarray(obs.bp.right_ascension().vals, np.float64)))
    # The planted offset moves the line of sight by a few milliradians, ten
    # orders of magnitude above float noise on an unmoved cache.
    assert abs(after - before) > 1e-5


# ---------------------------------------------------------------------------
# Which log gets which account
# ---------------------------------------------------------------------------


def _apply_and_capture_logs(
    tmp_path_root: str, monkeypatch: pytest.MonkeyPatch, record: dict[str, Any]
) -> tuple[AppliedPointing, str, str]:
    """Apply one record's selection with both logs bound to files.

    Parameters:
        tmp_path_root: Directory the two log files are written under.
        monkeypatch: The patcher for the instrument table.
        record: The navigation record to select and apply.

    Returns:
        Tuple of the application outcome, the image log text, and the run
        log text.
    """
    root = FCPath(tmp_path_root)
    obs = _hermetic_obs()
    _inject_identity_flip(monkeypatch)
    levels = LogLevels()
    set_log_levels(levels)
    main_log_path = build_main_logger(
        MAIN_LOGGER,
        SD_MOSAIC,
        LogSinks(log_root=root / 'runlog', main_console=False),
        levels,
        timestamp=_STAMP,
    )
    handlers, image_log_path = build_image_log_handlers(
        'reproj', 'VOL/IMG1', LogSinks(log_root=root / 'logs'), levels, timestamp=_STAMP
    )
    try:
        with IMAGE_LOGGER.open('REPROJECT', handler=handlers):
            selection = select_pointing(record, subject='IMG1')
            applied = apply_pointing_to_obs(cast(ObsSnapshotInst, obs), selection, subject='IMG1')
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
        # Detached before closing: a handler closed while still attached
        # stays registered under its log path, the leaked-handler state the
        # suite's conftest guards against.
        for handler in list(MAIN_LOGGER.handlers):
            if handler is not pdslogger.NULL_HANDLER:
                MAIN_LOGGER.remove_handler(handler)
                handler.close()
    assert image_log_path is not None
    assert main_log_path is not None
    with image_log_path.open('r') as stream:
        image_text = str(stream.read())
    with main_log_path.open('r') as stream:
        main_text = str(stream.read())
    return applied, image_text, main_text


def test_a_gate_fallback_warns_both_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused C-matrix is warned to the image log and to the run log.

    The detailed refusal belongs to the image; that the product fell back to
    the offset path belongs to the run, which would otherwise have to open
    every image's log to find out.
    """
    obs = _hermetic_obs()
    record = _record(
        cmatrix=_flat(_recorded_cmatrix(obs)),
        cmatrix_original=_flat(_IDENTITY),
        midtime_et=_MIDTIME + 1.0,
    )
    applied, image_text, main_text = _apply_and_capture_logs(str(tmp_path), monkeypatch, record)
    assert applied.reason == 'cmatrix_foreign_midtime'
    assert 'not applied' in image_text
    assert 'cmatrix_foreign_midtime' in image_text
    assert 'not applied' in main_text
    assert 'offset path' in main_text


def test_a_malformed_pointing_block_warns_both_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A defective record is a warning in the image log and a line in the run log."""
    record = _record(cmatrix=[True] * 9, cmatrix_original=_flat(_IDENTITY))
    applied, image_text, main_text = _apply_and_capture_logs(str(tmp_path), monkeypatch, record)
    assert applied.reason == 'malformed_pointing'
    assert 'malformed pointing block' in image_text
    assert 'malformed pointing block' in main_text


def test_the_pool_outcome_is_an_image_log_line_not_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The already-corrected pool is a counted fact, not an alarm."""
    plain = _hermetic_obs()
    cmatrix = _recorded_cmatrix(plain)
    root = FCPath(str(tmp_path))
    obs = _hermetic_obs(frame=oops.frame.Cmatrix(cmatrix))
    _inject_identity_flip(monkeypatch)
    levels = LogLevels()
    set_log_levels(levels)
    handlers, image_log_path = build_image_log_handlers(
        'reproj', 'VOL/IMG1', LogSinks(log_root=root / 'logs'), levels, timestamp=_STAMP
    )
    try:
        with IMAGE_LOGGER.open('REPROJECT', handler=handlers):
            selection = select_pointing(
                _record(cmatrix=_flat(cmatrix), cmatrix_original=_flat(_IDENTITY))
            )
            applied = apply_pointing_to_obs(cast(ObsSnapshotInst, obs), selection)
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    assert applied.source == 'pool'
    assert image_log_path is not None
    with image_log_path.open('r') as stream:
        image_text = str(stream.read())
    assert 'already answers the corrected attitude' in image_text
    assert 'WARNING' not in image_text
