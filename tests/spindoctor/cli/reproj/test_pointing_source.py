"""Reading one image's navigation record from a file and from an index.

The guarantee under test is equivalence: for every record ingest stored, the
two sources classify the same pointing, carry the same recorded values, and
write the same warning to the image's log.  Where they cannot -- because ingest
refused a document, so the index has no record of it -- the difference is
asserted rather than assumed, so a later change that silently widened it fails
here.

Every lookup is filtered on the root as well as the stub, and a two-root fixture
whose second root differs in exactly the value under test is what proves it: a
query that dropped the root would answer from whichever row it found first, and
the two directions of the same assertion cannot both be satisfied by one row.
"""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pdslogger
import pytest
import sqlalchemy
from filecache import FCPath
from sqlalchemy.engine import Engine
from tests.spindoctor.cli.reproj.conftest import (
    CMATRIX,
    CMATRIX_ORIGINAL,
    CMATRIX_STUB,
    FAILED_STUB,
    FITTED_STUB,
    MIDTIME_ET,
    NO_POINTING_STUB,
    NO_STATUS_ERROR_STUB,
    OFFSET,
    POINTING,
    REFUSED_DOCUMENT_REASON,
    REFUSED_DOCUMENT_STUB,
    TIMES,
    UNNAVIGATED_STUB,
    ZERO_EPOCH_STUB,
    ZERO_OFFSET_STUB,
    build_tree,
    document,
    image_file,
    index_for,
)

from spindoctor.cli.reproj.offsets import PointingMechanism, PointingSelection
from spindoctor.cli.reproj.pointing_source import (
    FilePointingSource,
    IndexPointingSource,
    PointingSource,
    build_pointing_source,
)
from spindoctor.config import IMAGE_LOGGER, LogLevels, LogSinks, build_image_log_handlers
from spindoctor.results_index import INGEST_RUNS, normalize_root_url, open_index
from spindoctor.support.nav_record import record_status, record_status_error

_STAMP = '2026-08-08T12-00-00'


def _selection(sources: dict[str, PointingSource], mode: str, stub: str) -> PointingSelection:
    """Look one stub up through one of the two sources.

    Parameters:
        sources: The pair of sources over the fixture tree.
        mode: ``'file'`` or ``'index'``.
        stub: The results path stub to look up.

    Returns:
        The classified selection.
    """
    return sources[mode].load_pointing(image_file(stub))


# ---------------------------------------------------------------------------
# The reasons both paths reach, asserted from the same fixture
# ---------------------------------------------------------------------------

_SAME_REASON = [
    (CMATRIX_STUB, None),
    (FITTED_STUB, 'no_cmatrix_rotation_fitted'),
    (NO_POINTING_STUB, 'no_pointing_block'),
    (FAILED_STUB, 'navigation_did_not_succeed'),
    (ZERO_OFFSET_STUB, 'no_pointing_block'),
    (ZERO_EPOCH_STUB, None),
    (UNNAVIGATED_STUB, 'no_metadata'),
]


@pytest.mark.parametrize(('stub', 'reason'), _SAME_REASON)
@pytest.mark.parametrize('mode', ['file', 'index'])
def test_both_paths_report_the_same_reason(
    sources: dict[str, PointingSource], mode: str, stub: str, reason: str | None
) -> None:
    """Each reachable reason is reported the same whichever storage answers."""
    assert _selection(sources, mode, stub).reason == reason


@pytest.mark.parametrize(
    ('stub', 'mechanism'),
    [
        (CMATRIX_STUB, PointingMechanism.CMATRIX),
        (FITTED_STUB, PointingMechanism.OFFSET),
        (NO_POINTING_STUB, PointingMechanism.OFFSET),
        (FAILED_STUB, PointingMechanism.NONE),
        (ZERO_OFFSET_STUB, PointingMechanism.OFFSET),
        (ZERO_EPOCH_STUB, PointingMechanism.CMATRIX),
        (UNNAVIGATED_STUB, PointingMechanism.NONE),
    ],
)
@pytest.mark.parametrize('mode', ['file', 'index'])
def test_both_paths_select_the_same_mechanism(
    sources: dict[str, PointingSource], mode: str, stub: str, mechanism: PointingMechanism
) -> None:
    """And the mechanism, which is what decides how the product is built."""
    assert _selection(sources, mode, stub).mechanism is mechanism


_SAME_OFFSET: list[tuple[str, tuple[float, float] | None]] = [
    (NO_POINTING_STUB, (OFFSET[0], OFFSET[1])),
    (CMATRIX_STUB, (OFFSET[0], OFFSET[1])),
    (ZERO_OFFSET_STUB, (0.0, 0.0)),
]


@pytest.mark.parametrize(('stub', 'offset'), _SAME_OFFSET)
@pytest.mark.parametrize('mode', ['file', 'index'])
def test_both_paths_carry_the_same_fallback_offset(
    sources: dict[str, PointingSource], mode: str, stub: str, offset: tuple[float, float] | None
) -> None:
    """The offset a gate refusal would fall back to is the same in both paths.

    Carried even under the C-matrix mechanism, so a gate refused at apply time
    -- a foreign midtime, a changed baseline -- degrades both paths to the same
    pointing rather than one to the offset and the other to nothing.
    """
    assert _selection(sources, mode, stub).offset == offset


# ---------------------------------------------------------------------------
# The recorded attitude, bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_the_recorded_cmatrix_survives_bit_for_bit(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """A rotation of full-mantissa float64 comes back exactly as recorded.

    The reader's flip gate holds the recovered rotation to 1e-9, so a storage
    that rounded even the last place would refuse records the file path
    accepts.
    """
    selection = _selection(sources, mode, CMATRIX_STUB)
    assert selection.cmatrix is not None
    assert np.array_equal(selection.cmatrix, np.asarray(CMATRIX).reshape(3, 3))


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_the_recorded_baseline_survives_bit_for_bit(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """So does the as-flown attitude the gate is computed against."""
    selection = _selection(sources, mode, CMATRIX_STUB)
    assert selection.cmatrix_original is not None
    assert np.array_equal(selection.cmatrix_original, np.asarray(CMATRIX_ORIGINAL).reshape(3, 3))


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_the_recorded_midtime_survives_exactly(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """And the midtime, which the reader gates against the observation at 1e-6 s.

    Asserted exactly rather than approximately: a gate that tight leaves the
    stored value no room to be nearly right.
    """
    assert _selection(sources, mode, CMATRIX_STUB).midtime_et == MIDTIME_ET


# ---------------------------------------------------------------------------
# A recorded value that is present and false
# ---------------------------------------------------------------------------
#
# The rebuild asks of every column whether the row carries a value, and every
# such question has a spelling that asks instead whether the value is true.
# The two agree for every record whose numbers happen to be non-zero, which is
# all of them above.  A recorded zero is what separates them, and it separates
# them in the product rather than in the reason: an offset of two zeros becomes
# no offset at all, and a midtime at the J2000 epoch becomes a pointing block
# with no epoch to gate against.


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_a_recorded_offset_of_two_zeros_is_a_pair_in_both_paths(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """An offset of no pixels is an offset, not the absence of one.

    A navigation whose image was already pointed correctly records exactly
    this, and a rebuild that read the pair for its truth would render it as
    null: the row would then be counted under ``null_offset`` and reprojected
    uncorrected while the document was reprojected on the pair it recorded.
    """
    assert _selection(sources, mode, ZERO_OFFSET_STUB).offset == (0.0, 0.0)


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_a_recorded_offset_of_two_zeros_still_selects_the_offset_mechanism(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """The other half of it: which mechanism builds the product."""
    assert _selection(sources, mode, ZERO_OFFSET_STUB).mechanism is PointingMechanism.OFFSET


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_a_recorded_midtime_of_zero_survives_into_both_records(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """The J2000 epoch is an epoch, and the ladder cannot run without one.

    Dropped from the rebuild, the pointing block loses the value its gates are
    computed against and the record is classified ``malformed_pointing``,
    where the document is a clean corrected attitude.
    """
    assert _selection(sources, mode, ZERO_EPOCH_STUB).midtime_et == 0.0


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_a_record_of_zero_epochs_still_applies_its_corrected_attitude(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """The product that follows from it, which is what the difference costs."""
    assert _selection(sources, mode, ZERO_EPOCH_STUB).mechanism is PointingMechanism.CMATRIX


def test_the_index_carries_the_recorded_midtime_not_a_recomputed_one(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """The stored midtime is what the document said, not the shutter midpoint.

    The two agree for the frame the rest of these tests use, which is exactly
    why the distinction needs a record where they do not: an implementation
    that recomputed the value from ``start_et`` and ``stop_et`` would pass
    every other assertion here and then gate real records against an epoch
    nobody recorded.
    """
    displaced = dict(TIMES)
    displaced['midtime_et'] = TIMES['start_et'] + 0.001
    root = tmp_path / 'nav'
    build_tree(
        root,
        {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET, times=displaced, pointing=POINTING)},
    )
    engine = index_for([root], tmp_path / 'index.sqlite3', logger=quiet_ingest_logger)
    try:
        source = IndexPointingSource(engine, normalize_root_url(root))
        selection = source.load_pointing(image_file(CMATRIX_STUB))
        assert selection.midtime_et == displaced['midtime_et']
    finally:
        engine.dispose()


def test_a_document_the_ingest_refused_is_not_reported_as_an_unnavigated_image(
    sources: dict[str, PointingSource],
) -> None:
    """A refusal is an answer the index cannot give, not the absence of one.

    The document under test records a usable corrected attitude and is refused
    whole for want of an ``observation.instrument``, so reading its absence as
    "nothing navigated this image" would reproject it corrected through the
    documents and uncorrected through the index, with nothing said either way.
    """
    with pytest.raises(ValueError, match='could not read'):
        sources['index'].load_pointing(image_file(REFUSED_DOCUMENT_STUB))


def test_the_refusal_names_the_reason_the_ingest_recorded(
    sources: dict[str, PointingSource],
) -> None:
    """Without it an operator is told only that the index cannot answer."""
    with pytest.raises(ValueError) as excinfo:
        sources['index'].load_pointing(image_file(REFUSED_DOCUMENT_STUB))
    assert REFUSED_DOCUMENT_REASON in str(excinfo.value)


def test_the_refusal_names_the_stub_it_is_about(sources: dict[str, PointingSource]) -> None:
    """One image of a batch is refused, so the message has to say which."""
    with pytest.raises(ValueError) as excinfo:
        sources['index'].load_pointing(image_file(REFUSED_DOCUMENT_STUB))
    assert REFUSED_DOCUMENT_STUB in str(excinfo.value)


def test_the_refusal_names_the_index_that_holds_it(sources: dict[str, PointingSource]) -> None:
    """And which index recorded the refusal, since a run may be pointed at any."""
    with pytest.raises(ValueError) as excinfo:
        sources['index'].load_pointing(image_file(REFUSED_DOCUMENT_STUB))
    assert 'index.sqlite3' in str(excinfo.value)


def test_reading_the_whole_record_refuses_that_document_too(
    sources: dict[str, PointingSource],
) -> None:
    """The backplane stage reads the record, not the pointing, and must not skip.

    A skip is what it reports for an image nothing navigated; reporting this
    one that way would silently build no product where the documents build one.
    """
    with pytest.raises(ValueError, match='could not read'):
        sources['index'].read_record(image_file(REFUSED_DOCUMENT_STUB))


def test_the_refusal_is_not_the_exception_a_missing_record_raises(
    sources: dict[str, PointingSource],
) -> None:
    """The backplane driver skips a ``FileNotFoundError`` and fails anything else.

    The two facts are different -- nothing navigated this image, against the
    index cannot say what was recorded for it -- so they must not arrive as one
    exception type.
    """
    with pytest.raises(ValueError) as excinfo:
        sources['index'].read_record(image_file(REFUSED_DOCUMENT_STUB))
    assert not isinstance(excinfo.value, FileNotFoundError)


def test_the_file_path_still_supplies_that_documents_pointing(
    sources: dict[str, PointingSource],
) -> None:
    """The control that makes the refusal necessary rather than cautious.

    Read as a file the same document supplies the corrected attitude it
    records, which is the product the index-backed run would otherwise have
    silently built without.
    """
    selection = _selection(sources, 'file', REFUSED_DOCUMENT_STUB)
    assert selection.mechanism is PointingMechanism.CMATRIX


def test_an_image_nothing_navigated_is_still_reported_as_unnavigated(
    sources: dict[str, PointingSource],
) -> None:
    """The other control: absence with no refusal beside it keeps its meaning.

    A source that refused every miss would fail every image a results root
    holds no document for, which is most of a dataset.
    """
    assert _selection(sources, 'index', UNNAVIGATED_STUB).reason == 'no_metadata'


def _one_root_refused_it(tmp_path: Path, logger: pdslogger.PdsLogger) -> tuple[Path, Path, Engine]:
    """Build two roots, one of which refused the very stub the other never held.

    The stub is absent from the record table under both roots, so every lookup
    of it reaches the refusal table; only the root half of the key separates
    "this root never navigated the image" from "this root holds a document
    nothing could be read from".

    Parameters:
        tmp_path: Directory both roots and the index are written under.
        logger: Logger the ingest reports through.

    Returns:
        The root that navigated something else, the root that refused the stub,
        and the open index over both.
    """
    silent = tmp_path / 'nav_a'
    refused = tmp_path / 'nav_b'
    build_tree(silent, {NO_POINTING_STUB: document(NO_POINTING_STUB, offset=OFFSET)})
    build_tree(refused, {CMATRIX_STUB: document(CMATRIX_STUB, instrument=None, offset=OFFSET)})
    return silent, refused, index_for([silent, refused], tmp_path / 'index.sqlite3', logger=logger)


def test_a_refusal_recorded_under_another_root_does_not_refuse_this_one(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """The refusal lookup is keyed by root as well as by stub.

    One index holds several roots, and a stub one of them refused is routinely
    a stub another simply never navigated; a lookup that dropped the root would
    fail an image on the strength of its neighbor's refusal.
    """
    silent, _refused, engine = _one_root_refused_it(tmp_path, quiet_ingest_logger)
    try:
        source = IndexPointingSource(engine, normalize_root_url(silent))
        assert source.load_pointing(image_file(CMATRIX_STUB)).reason == 'no_metadata'
    finally:
        engine.dispose()


def test_the_root_that_did_record_the_refusal_still_refuses(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """The other direction of the same key, which one row cannot satisfy both of."""
    _silent, refused, engine = _one_root_refused_it(tmp_path, quiet_ingest_logger)
    try:
        source = IndexPointingSource(engine, normalize_root_url(refused))
        with pytest.raises(ValueError, match='could not read'):
            source.load_pointing(image_file(CMATRIX_STUB))
    finally:
        engine.dispose()


def test_a_document_that_is_not_an_object_is_refused_by_the_index(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """Ingest refuses it, so the index says it cannot answer for the image.

    The file path calls the same document ``metadata_not_an_object`` and
    reprojects on uncorrected pointing; the index has no record of it and
    reports that rather than reporting an image nothing navigated.
    """
    root = tmp_path / 'nav'
    path = root / f'{NO_POINTING_STUB}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([1, 2, 3]), encoding='utf-8')
    engine = index_for([root], tmp_path / 'index.sqlite3', logger=quiet_ingest_logger)
    try:
        source = IndexPointingSource(engine, normalize_root_url(root))
        with pytest.raises(ValueError, match='not a JSON object'):
            source.load_pointing(image_file(NO_POINTING_STUB))
    finally:
        engine.dispose()


def test_the_file_path_calls_that_same_document_something_else(tmp_path: Path) -> None:
    """Which is the difference the module docstring records."""
    root = tmp_path / 'nav'
    path = root / f'{NO_POINTING_STUB}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([1, 2, 3]), encoding='utf-8')
    source = FilePointingSource(FCPath(root))
    assert source.load_pointing(image_file(NO_POINTING_STUB)).reason == 'metadata_not_an_object'


# ---------------------------------------------------------------------------
# The two NULL-cmatrix cases
# ---------------------------------------------------------------------------


def test_a_fitted_rotation_row_is_not_read_as_a_missing_pointing_block(
    sources: dict[str, PointingSource],
) -> None:
    """A result that fitted a camera rotation stores no cmatrix and a baseline.

    Both that record and one with no pointing block at all leave ``cmatrix``
    NULL, so the column cannot separate them; the recorded baseline can, and
    the reason is what a run-level tally counts each class under.
    """
    assert _selection(sources, 'index', FITTED_STUB).reason == 'no_cmatrix_rotation_fitted'


def test_a_record_with_no_pointing_block_is_not_read_as_a_fitted_rotation(
    sources: dict[str, PointingSource],
) -> None:
    """The other half of that distinction, which one column alone would lose."""
    assert _selection(sources, 'index', NO_POINTING_STUB).reason == 'no_pointing_block'


# ---------------------------------------------------------------------------
# The outcome a document names, and the one it does not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_an_unsuccessful_record_naming_no_error_reads_back_as_naming_none(
    sources: dict[str, PointingSource], mode: str
) -> None:
    """An absent ``status_error`` is absent, not null.

    Every failed and conflicted navigation writes no such field, and the
    backplane stage reports an absent one as ``unknown`` by defaulting.  A
    rebuilt record carrying the key with a null value would default to nothing
    and report ``None`` instead, on the commonest unsuccessful record there is.
    """
    record = sources[mode].read_record(image_file(NO_STATUS_ERROR_STUB))
    assert 'status_error' not in record


@pytest.mark.parametrize('mode', ['file', 'index'])
def test_a_recorded_error_still_reads_back(sources: dict[str, PointingSource], mode: str) -> None:
    """The control for the absence above, which an empty record would pass."""
    record = sources[mode].read_record(image_file(FAILED_STUB))
    assert record['status_error'] == 'missing_spice_data'


# ---------------------------------------------------------------------------
# What a record says about its own outcome, however it names it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('stub', 'expected'),
    [
        (FAILED_STUB, 'error'),
        (CMATRIX_STUB, 'success'),
    ],
)
@pytest.mark.parametrize('mode', ['file', 'index'])
def test_both_paths_report_the_same_outcome(
    sources: dict[str, PointingSource], mode: str, stub: str, expected: str
) -> None:
    """The backplane skip line and the task result name the same outcome either way."""
    assert record_status(sources[mode].read_record(image_file(stub))) == expected


@pytest.mark.parametrize(
    ('stub', 'expected'),
    [
        (FAILED_STUB, 'missing_spice_data'),
        (NO_STATUS_ERROR_STUB, 'unknown'),
    ],
)
@pytest.mark.parametrize('mode', ['file', 'index'])
def test_both_paths_report_the_same_error(
    sources: dict[str, PointingSource], mode: str, stub: str, expected: str
) -> None:
    """And the same error beside it, including where the record names none.

    An absent ``status_error`` names no error, and every failed and conflicted
    navigation writes none; a reader that reported the absence verbatim would
    say ``None`` through a document and ``unknown`` through a row.
    """
    assert record_status_error(sources[mode].read_record(image_file(stub))) == expected


# ---------------------------------------------------------------------------
# The warnings an image's own log carries
# ---------------------------------------------------------------------------


def _log_of(source: PointingSource, stub: str, log_root: Path) -> str:
    """Look one stub up inside an image scope and return the image's log.

    Parameters:
        source: The source to look up through.
        stub: The results path stub.
        log_root: Root the per-image log is written under.

    Returns:
        The text of the image's log.
    """
    handlers, path = build_image_log_handlers(
        'reproj',
        stub,
        LogSinks(log_root=FCPath(log_root)),
        LogLevels(),
        timestamp=_STAMP,
    )
    try:
        with IMAGE_LOGGER.open('REPROJECT', handler=handlers):
            source.load_pointing(image_file(stub))
    finally:
        for handler in handlers:
            if handler is not pdslogger.NULL_HANDLER:
                handler.close()
    assert path is not None
    with path.open('r') as stream:
        return str(stream.read())


@pytest.mark.parametrize(
    ('stub', 'expected'),
    [
        (UNNAVIGATED_STUB, 'No navigation record for'),
        (FAILED_STUB, "status='error'"),
    ],
)
@pytest.mark.parametrize('mode', ['file', 'index'])
def test_the_image_log_says_the_same_thing_either_way(
    sources: dict[str, PointingSource],
    tmp_path: Path,
    mode: str,
    stub: str,
    expected: str,
) -> None:
    """The per-image warnings keep their message shapes in the index path.

    A reader of one image's log, and anything scraping those logs, sees the
    same line whichever storage the run was pointed at.
    """
    assert expected in _log_of(sources[mode], stub, tmp_path / f'logs_{mode}')


def test_the_missing_record_warning_names_the_storage_that_was_searched(
    sources: dict[str, PointingSource], tmp_path: Path
) -> None:
    """An index that holds no row names itself, not a results root it never read.

    An index is a snapshot of its last ingest, so a row can be absent because
    nothing navigated the image or because the image was navigated since; a
    message naming a results root would rule the second out for a reader when
    it is the likelier of the two.
    """
    log_text = _log_of(sources['index'], UNNAVIGATED_STUB, tmp_path / 'logs_index')
    assert 'a snapshot of its last ingest' in log_text


def test_the_same_warning_names_the_documents_when_documents_were_searched(
    sources: dict[str, PointingSource], tmp_path: Path
) -> None:
    """The other half: reading documents, the message names the results root."""
    log_text = _log_of(sources['file'], UNNAVIGATED_STUB, tmp_path / 'logs_file')
    assert 'the navigation results under' in log_text


# ---------------------------------------------------------------------------
# The whole record, for the backplane stage
# ---------------------------------------------------------------------------


def test_the_record_carries_the_status_the_backplane_stage_skips_on(
    sources: dict[str, PointingSource],
) -> None:
    """The stage reads the status before it decides there is work to do."""
    assert sources['index'].read_record(image_file(FAILED_STUB))['status'] == 'error'


def test_the_record_carries_the_error_the_skip_reports(
    sources: dict[str, PointingSource],
) -> None:
    """And the error it names in the skip it returns."""
    record = sources['index'].read_record(image_file(FAILED_STUB))
    assert record['status_error'] == 'missing_spice_data'


def test_a_stub_absent_from_the_index_raises(sources: dict[str, PointingSource]) -> None:
    """Exactly as a missing document does, so the caller reports it the same way."""
    with pytest.raises(FileNotFoundError) as excinfo:
        sources['index'].read_record(image_file(UNNAVIGATED_STUB))
    assert UNNAVIGATED_STUB in str(excinfo.value)


def test_the_raise_names_the_index_it_asked(sources: dict[str, PointingSource]) -> None:
    """So a reader can tell a wrong index from an unnavigated image."""
    with pytest.raises(FileNotFoundError) as excinfo:
        sources['index'].read_record(image_file(UNNAVIGATED_STUB))
    assert 'index.sqlite3' in str(excinfo.value)


def test_a_missing_document_raises_in_the_file_path(
    sources: dict[str, PointingSource],
) -> None:
    """The behavior the index path is matched against, naming the document."""
    with pytest.raises(FileNotFoundError) as excinfo:
        sources['file'].read_record(image_file(UNNAVIGATED_STUB))
    assert UNNAVIGATED_STUB in str(excinfo.value)


def test_a_stub_that_escapes_the_root_reads_no_record(tmp_path: Path) -> None:
    """A stub resolving outside the root names no record this source may read.

    The same rule both of this class's methods apply, rather than one guarded
    lookup and one that joins whatever it was handed onto the root.
    """
    root = tmp_path / 'nav'
    root.mkdir(parents=True)
    (tmp_path / 'elsewhere_metadata.json').write_text(json.dumps({'status': 'success'}))
    escaping = image_file('../elsewhere')
    with pytest.raises(FileNotFoundError) as excinfo:
        FilePointingSource(FCPath(root)).read_record(escaping)
    assert 'does not name a navigation record' in str(excinfo.value)


def test_a_stub_that_stays_under_the_root_still_reads_its_record(tmp_path: Path) -> None:
    """The control for the refusal above, which a source refusing all would pass."""
    root = tmp_path / 'nav'
    build_tree(root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    record = FilePointingSource(FCPath(root)).read_record(image_file(CMATRIX_STUB))
    assert record['status'] == 'success'


def test_an_index_that_stops_answering_is_reported_as_a_refusal(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """A lost index is a refusal naming it, not the database layer's own exception.

    The callers of this module report a lookup failure against one image, and
    they cannot name SQLAlchemy's exception types: they deliberately do not
    import it.  The index's own tables are dropped here, which is the shape a
    partially restored database has.
    """
    root = tmp_path / 'nav'
    build_tree(root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    database = tmp_path / 'index.sqlite3'
    engine = index_for([root], database, logger=quiet_ingest_logger)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql('DROP TABLE images')
        source = IndexPointingSource(engine, normalize_root_url(root))
        with pytest.raises(ValueError, match='could not be read') as excinfo:
            source.read_record(image_file(CMATRIX_STUB))
        assert 'index.sqlite3' in str(excinfo.value)
    finally:
        engine.dispose()


def test_a_file_source_with_no_root_has_nowhere_to_read(tmp_path: Path) -> None:
    """A reprojection run given no navigation results can supply no record."""
    with pytest.raises(FileNotFoundError) as excinfo:
        FilePointingSource(None).read_record(image_file(CMATRIX_STUB))
    assert 'no navigation results root' in str(excinfo.value)


def test_a_document_that_is_not_an_object_is_refused_by_name(tmp_path: Path) -> None:
    """Reading a record has to return one, so a JSON array is refused here.

    Left to the caller it becomes an attribute error from the middle of a
    batch run, naming neither the file nor what is wrong with it.
    """
    root = tmp_path / 'nav'
    path = root / f'{CMATRIX_STUB}_metadata.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([1, 2, 3]), encoding='utf-8')
    with pytest.raises(ValueError, match='not a JSON object'):
        FilePointingSource(FCPath(root)).read_record(image_file(CMATRIX_STUB))


# ---------------------------------------------------------------------------
# One index, several roots
# ---------------------------------------------------------------------------


def _two_roots(tmp_path: Path, logger: pdslogger.PdsLogger) -> tuple[Path, Path, Engine]:
    """Build two results roots holding the same stub with different offsets.

    Parameters:
        tmp_path: Directory both roots and the index are written under.
        logger: Logger the ingest reports through.

    Returns:
        The two roots and the open index over both of them.
    """
    first = tmp_path / 'nav_a'
    second = tmp_path / 'nav_b'
    build_tree(first, {CMATRIX_STUB: document(CMATRIX_STUB, offset=[1.5, -2.5])})
    build_tree(second, {CMATRIX_STUB: document(CMATRIX_STUB, offset=[9.25, 8.75])})
    return first, second, index_for([first, second], tmp_path / 'index.sqlite3', logger=logger)


def test_a_source_reads_its_own_roots_offset(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """One index can hold several roots, and a lookup answers from one of them."""
    first, _second, engine = _two_roots(tmp_path, quiet_ingest_logger)
    try:
        source = IndexPointingSource(engine, normalize_root_url(first))
        assert source.load_pointing(image_file(CMATRIX_STUB)).offset == (1.5, -2.5)
    finally:
        engine.dispose()


def test_the_other_source_reads_the_other_roots_offset(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """The same assertion from the other side.

    Together with the previous test this is what a query filtering on the stub
    alone cannot satisfy: one row cannot be both answers, so dropping the root
    predicate fails one of the two whichever row the database returns first.
    """
    _first, second, engine = _two_roots(tmp_path, quiet_ingest_logger)
    try:
        source = IndexPointingSource(engine, normalize_root_url(second))
        assert source.load_pointing(image_file(CMATRIX_STUB)).offset == (9.25, 8.75)
    finally:
        engine.dispose()


def test_a_stub_recorded_only_under_another_root_is_absent(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """An image navigated under one root was not navigated under the other.

    The backplane raise is what carries this, and a lookup that forgot the root
    would find the neighbor's row and build a product from it.
    """
    first = tmp_path / 'nav_a'
    second = tmp_path / 'nav_b'
    build_tree(first, {NO_POINTING_STUB: document(NO_POINTING_STUB, offset=OFFSET)})
    build_tree(second, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    engine = index_for([first, second], tmp_path / 'index.sqlite3', logger=quiet_ingest_logger)
    try:
        source = IndexPointingSource(engine, normalize_root_url(first))
        with pytest.raises(FileNotFoundError, match=CMATRIX_STUB):
            source.read_record(image_file(CMATRIX_STUB))
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Choosing a source
# ---------------------------------------------------------------------------


def test_no_url_reads_documents(tmp_path: Path) -> None:
    """No index is every program's default, and it reads the tree."""
    source = build_pointing_source(FCPath(tmp_path), results_index_db_url=None)
    assert isinstance(source, FilePointingSource)


def test_a_url_reads_rows(tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger) -> None:
    """A resolved URL makes the same program read one row per image."""
    root = tmp_path / 'nav'
    build_tree(root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    database = tmp_path / 'index.sqlite3'
    index_for([root], database, logger=quiet_ingest_logger).dispose()
    source = build_pointing_source(
        FCPath(root), results_index_db_url=f'sqlite:///{database.as_posix()}'
    )
    try:
        assert isinstance(source, IndexPointingSource)
    finally:
        source.close()


def test_closing_an_index_backed_source_disposes_the_engine(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """Closing it returns every pooled connection, rather than only dropping a name.

    Both cloud-task workers hold one index open for the worker's lifetime and
    close it at the end, so a close that disposed of nothing would leak a
    connection pool per worker with nothing to say so.  Disposal is observable
    as the pool being replaced: the engine keeps no connection from before it.
    """
    root = tmp_path / 'nav'
    build_tree(root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    database = tmp_path / 'index.sqlite3'
    index_for([root], database, logger=quiet_ingest_logger).dispose()
    # The engine is handed in rather than read back off the source, so what the
    # assertion watches is an object the test owns.
    engine = open_index(f'sqlite:///{database.as_posix()}')
    source = IndexPointingSource(engine, normalize_root_url(FCPath(root)))
    source.load_pointing(image_file(CMATRIX_STUB))
    pooled_before = engine.pool
    source.close()
    assert engine.pool is not pooled_before


def test_an_unopenable_url_fails_rather_than_falling_back(tmp_path: Path) -> None:
    """A misconfigured index is a failed run, not a slow and different one."""
    missing = tmp_path / 'nowhere' / 'index.sqlite3'
    with pytest.raises(ValueError) as excinfo:
        build_pointing_source(
            FCPath(tmp_path), results_index_db_url=f'sqlite:///{missing.as_posix()}'
        )
    assert 'sd_results_index' in str(excinfo.value)


def test_an_unopenable_url_leaves_no_database_behind(tmp_path: Path) -> None:
    """And it does not create the index it was told to read."""
    missing = tmp_path / 'index.sqlite3'
    with pytest.raises(ValueError, match='sd_results_index') as excinfo:
        build_pointing_source(
            FCPath(tmp_path), results_index_db_url=f'sqlite:///{missing.as_posix()}'
        )
    assert 'index.sqlite3' in str(excinfo.value)
    assert not missing.exists()


def test_an_index_with_no_root_to_read_under_is_refused(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """Rows are keyed by root, so an index with no root has nothing to answer."""
    root = tmp_path / 'nav'
    build_tree(root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    database = tmp_path / 'index.sqlite3'
    index_for([root], database, logger=quiet_ingest_logger).dispose()
    with pytest.raises(ValueError, match='no navigation results root'):
        build_pointing_source(None, results_index_db_url=f'sqlite:///{database.as_posix()}')


def test_a_root_nobody_ingested_is_refused(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """Absence of a row means "never navigated" only under a fully ingested root.

    Under one nobody ingested it means nothing at all, so the consumer refuses
    rather than reporting every image as unnavigated.
    """
    ingested = tmp_path / 'nav_a'
    other = tmp_path / 'nav_b'
    build_tree(ingested, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    other.mkdir(parents=True, exist_ok=True)
    database = tmp_path / 'index.sqlite3'
    index_for([ingested], database, logger=quiet_ingest_logger).dispose()
    with pytest.raises(ValueError) as excinfo:
        build_pointing_source(
            FCPath(other), results_index_db_url=f'sqlite:///{database.as_posix()}'
        )
    assert normalize_root_url(other) in str(excinfo.value)


def test_the_refusal_names_the_roots_the_index_does_hold(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """So the reader can see which root they meant to point at."""
    ingested = tmp_path / 'nav_a'
    other = tmp_path / 'nav_b'
    build_tree(ingested, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    other.mkdir(parents=True, exist_ok=True)
    database = tmp_path / 'index.sqlite3'
    index_for([ingested], database, logger=quiet_ingest_logger).dispose()
    with pytest.raises(ValueError) as excinfo:
        build_pointing_source(
            FCPath(other), results_index_db_url=f'sqlite:///{database.as_posix()}'
        )
    assert normalize_root_url(ingested) in str(excinfo.value)


def test_a_run_that_died_halfway_leaves_a_root_unreadable(
    tmp_path: Path, quiet_ingest_logger: pdslogger.PdsLogger
) -> None:
    """A root whose newest ingest never finished has not been ingested.

    Its rows are whatever the dead pass wrote, so absence under it says
    nothing, and a consumer must not read it as an answer.
    """
    root = tmp_path / 'nav'
    build_tree(root, {CMATRIX_STUB: document(CMATRIX_STUB, offset=OFFSET)})
    database = tmp_path / 'index.sqlite3'
    index_for([root], database, logger=quiet_ingest_logger).dispose()
    url = f'sqlite:///{database.as_posix()}'
    engine = open_index(url)
    try:
        with engine.begin() as connection:
            # Restricted to the root under test rather than applied to every
            # row: a fixture that mutates all of them cannot tell a query
            # reading the right row from one reading whichever it found first.
            connection.execute(
                INGEST_RUNS.update()
                .where(INGEST_RUNS.c.root_url == normalize_root_url(root))
                .values(finished_utc=None)
            )
    finally:
        engine.dispose()
    with pytest.raises(ValueError, match='no completed ingest'):
        build_pointing_source(FCPath(root), results_index_db_url=url)


# ---------------------------------------------------------------------------
# What the rebuilt record does not carry
# ---------------------------------------------------------------------------


def test_the_rebuilt_record_carries_the_recorded_frame_identities(
    sources: dict[str, PointingSource],
) -> None:
    """The pointing block the index holds is rebuilt whole, not just its matrices."""
    record = sources['index'].read_record(image_file(CMATRIX_STUB))
    assert record['navigation_result']['pointing']['ck_frame_id'] == POINTING['ck_frame_id']


def test_the_rebuilt_record_omits_the_camera_frame_name(
    sources: dict[str, PointingSource],
) -> None:
    """No reader consults it: the frame identity comes from the observation.

    Asserted so that a reader which started consulting it would be caught here
    rather than by a product built on a name the index never stored.  The
    document it was ingested from is checked to carry the name, so what is
    asserted is that the rebuild drops it rather than that nothing ever had it.
    """
    ingested = sources['file'].read_record(image_file(CMATRIX_STUB))
    assert 'camera_frame' in ingested['navigation_result']['pointing']
    record = sources['index'].read_record(image_file(CMATRIX_STUB))
    assert 'camera_frame' not in record['navigation_result']['pointing']


# ---------------------------------------------------------------------------
# The password an index URL can carry
# ---------------------------------------------------------------------------
#
# The index-backed source names its index in three messages that reach files:
# the refusal a missing record raises, the warning that refusal writes into the
# image's own log, and the translation of a read the database would not answer.
# A connection URL can carry a database password, so each of them names the
# index through the masking rule.
#
# A ``sqlite:`` URL is returned by that rule exactly as it came -- it names a
# filesystem path, which has no credentials -- so no test built on one can hold
# any of these to masking anything.  These build the source over a server URL
# instead, which needs no server: the source reads the URL from the engine, and
# a lookup that finds no row is the shortest path to a message carrying it.

_LEFT = 'sup3r'
"""First half of the password, distinctive enough that finding it is a leak."""

_RIGHT = 's3cr3t'
"""Second half, so a rule that hides only part of a password is still caught."""

_SERVER_HOST = 'db.example:5432/spindoctor'
"""Everything after the credentials, which a reader of the message needs."""

_SERVER_URL = f'postgresql+psycopg://us%40er:{_LEFT}%40%3A%2F%3F%23{_RIGHT}@{_SERVER_HOST}'
"""A server index URL whose password carries every character delimiting a URL.

An ``@`` ends the credentials, a ``:`` starts a port, a ``/`` starts a path, a
``?`` starts a query and a ``#`` starts a fragment; the user name carries an
at-sign of its own, so a rule ending the credentials at the first one would
leave the whole password behind.
"""


class _KnownToNeitherTable:
    """The row a lookup gets for a stub the index holds nothing about."""

    record_stub = None
    refusal_reason = None


class _HoldingNothing:
    """A connection whose every lookup finds neither a record nor a refusal."""

    def __enter__(self) -> '_HoldingNothing':
        """Return itself, so it can be used where a connection is.

        Returns:
            This connection.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close nothing and swallow nothing.

        Parameters:
            *exc_info: The exception leaving the block, if any.
        """

    def execute(self, statement: Any) -> '_HoldingNothing':
        """Run nothing and answer as the result of it.

        Parameters:
            statement: The statement, which is not run.

        Returns:
            This object, which is its own result.
        """
        return self

    def first(self) -> _KnownToNeitherTable:
        """Return the one row the real statement always answers with.

        Returns:
            A row naming neither a record nor a refusal.
        """
        return _KnownToNeitherTable()


class _IndexNamedByAUrl:
    """Stands in for an open index carrying a URL, and holding nothing.

    Parameters:
        url: The connection URL the source names its index by.
        unreadable: Whether connecting fails the way a lost server does.
    """

    def __init__(self, url: str, *, unreadable: bool = False) -> None:
        """Take the URL and whether this index answers at all."""
        self.url = sqlalchemy.engine.url.make_url(url)
        self._unreadable = unreadable

    def connect(self) -> _HoldingNothing:
        """Open a connection that holds no rows, or fail as a lost server does.

        Returns:
            The connection.

        Raises:
            sqlalchemy.exc.OperationalError: When this index is unreadable.
        """
        if self._unreadable:
            raise sqlalchemy.exc.OperationalError(
                'SELECT images.status FROM images', {}, OSError('connection refused')
            )
        return _HoldingNothing()

    def dispose(self) -> None:
        """Release nothing: this index never opened anything."""


def _source_over_a_server_url(*, unreadable: bool = False) -> IndexPointingSource:
    """Build an index-backed source naming a server URL with a password in it.

    Parameters:
        unreadable: Whether the index fails every read.

    Returns:
        The source.
    """
    engine = _IndexNamedByAUrl(_SERVER_URL, unreadable=unreadable)
    return IndexPointingSource(cast(Engine, engine), 'file:///nav')


def _missing_record_message() -> str:
    """Return the refusal an index holding no row for an image raises.

    Returns:
        The text of the exception.
    """
    with pytest.raises(FileNotFoundError) as excinfo:
        _source_over_a_server_url().read_record(image_file(UNNAVIGATED_STUB))
    return str(excinfo.value)


def _unreadable_index_message() -> str:
    """Return the refusal an index that will not answer a read raises.

    Returns:
        The text of the exception.
    """
    with pytest.raises(ValueError) as excinfo:
        _source_over_a_server_url(unreadable=True).read_record(image_file(UNNAVIGATED_STUB))
    return str(excinfo.value)


def test_the_missing_record_refusal_carries_no_password() -> None:
    """It names the index, and the caller writes it into the run's log."""
    assert _LEFT not in _missing_record_message()


def test_no_tail_of_that_password_reaches_the_refusal_either() -> None:
    """A rule stopping at the first URL delimiter would leave a working password."""
    assert _RIGHT not in _missing_record_message()


def test_the_missing_record_refusal_still_names_the_index() -> None:
    """The control: a message naming no index would pass both assertions above."""
    assert _SERVER_HOST in _missing_record_message()


def test_the_missing_record_warning_in_an_image_log_carries_no_password(tmp_path: Path) -> None:
    """The same name reaches one file per image with no record, which is most of them."""
    log_text = _log_of(_source_over_a_server_url(), UNNAVIGATED_STUB, tmp_path / 'logs')
    assert _LEFT not in log_text


def test_no_tail_of_that_password_reaches_the_image_log_either(tmp_path: Path) -> None:
    """And a half-hidden password in a per-image log is a password in a log."""
    log_text = _log_of(_source_over_a_server_url(), UNNAVIGATED_STUB, tmp_path / 'logs')
    assert _RIGHT not in log_text


def test_that_warning_still_names_the_index_it_searched(tmp_path: Path) -> None:
    """The control for those two, on the line a reader of the log is there for."""
    log_text = _log_of(_source_over_a_server_url(), UNNAVIGATED_STUB, tmp_path / 'logs')
    assert _SERVER_HOST in log_text


def test_an_unreadable_index_is_reported_without_its_password() -> None:
    """A failed read names the URL that failed, which is the whole of the diagnosis."""
    assert _LEFT not in _unreadable_index_message()


def test_no_tail_of_that_password_reaches_the_failed_read_either() -> None:
    """The driver's own message is quoted into it, so the URL is quoted twice over."""
    assert _RIGHT not in _unreadable_index_message()


def test_the_failed_read_still_names_the_index_that_would_not_answer() -> None:
    """The control: an operator is reading this to learn which index refused."""
    assert _SERVER_HOST in _unreadable_index_message()
