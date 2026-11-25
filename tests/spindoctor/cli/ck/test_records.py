"""The two ways a C-kernel run gets its navigation records, held to each other.

A run reads every document of one mission under a results root, from the tree
or from a results index.  What the readers downstream do with them must not
depend on which: the same images, in the same order, eligible or omitted for
the same reasons, with the same matrices, epochs, kernels and reported facts.
So the tests here drive both sources over one tree and compare what the
generator's own readers make of each.

The one difference the seam permits is tested too, as a difference: a value the
ingest could not store is rebuilt as one the document never recorded, so the
index path reports an image whose record the tree path refuses outright.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pdslogger
import pytest
import sqlalchemy
from filecache import FCPath
from tests.packaged_sections import packaged_tree_tuning
from tests.spindoctor.cli.ck.ck_helpers import KernelPool, image_metadata
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    write_metadata,
)

from spindoctor.cli.ck.images import ImageEntry
from spindoctor.cli.ck.inputs import RECORD_COLUMNS, read_whole_mission
from spindoctor.cli.ck.report import read_image_report_facts
from spindoctor.nav_records import (
    NavRecord,
    RecordSource,
    Selection,
    TreeRecordSource,
    UnreadableFile,
)
from spindoctor.results_index import IMAGES, IndexRecordSource, open_index, open_record_source

MISSION = 'coiss'
"""The instrument identity the runs below write kernels for."""

OTHER_MISSION = 'vgiss'
"""An instrument identity of another mission's documents in the same tree."""

CORRECTED = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
"""A corrected attitude, whose values must survive the round trip exactly."""

ORIGINAL = np.array(
    [
        [0.9999999, -0.0004472, 0.0],
        [0.0004472, 0.9999999, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
"""The uncorrected attitude beside it, deliberately not the identity."""

BothReadings = tuple[list[NavRecord], list[NavRecord]]
"""One mission's records read from the tree and from the index, in that order."""

KERNELS = ('cas00172.tsc', 'naif0012.tls', '18001_18031ra.bc')
"""The kernel basenames a document records, in the order it records them."""


def null_logger() -> pdslogger.PdsLogger:
    """Return a logger that keeps the ingest quiet.

    Returns:
        A logger discarding everything written to it.
    """
    return pdslogger.NullLogger()


def _navigated(image_name: str, **overrides: Any) -> dict[str, Any]:
    """Build one navigated image's document, in the shape the pipeline writes.

    Parameters:
        image_name: Basename recorded for the image.
        overrides: Fields of :func:`image_metadata` to replace.

    Returns:
        The document.
    """
    fields: dict[str, Any] = {
        'image_name': image_name,
        'cmatrix': CORRECTED,
        'cmatrix_original': ORIGINAL,
        'camera_frame': 'CASSINI_ISS_NAC',
        'ck_frame_id': -82000,
        'start_et': 100.0,
        'stop_et': 102.0,
        'sclk_midtime': '1/1454725799.100',
        'instrument': MISSION,
        'camera': 'NAC',
        'shutter_mode': 'BOTSIM',
        'kernels': KERNELS,
        'offset': (1.5, -2.5),
        'sigma_px': (0.1, 0.2),
        'confidence': 0.87,
        'confidence_rank': 'high',
        'status_reason': 'ok',
    }
    fields.update(overrides)
    return image_metadata(**fields)


def _tree(tmp_path: Path) -> Path:
    """Write a results root holding what one mission's run considers.

    It holds a navigated image with a corrected attitude, one that navigated
    without one, and one belonging to another mission entirely.

    Parameters:
        tmp_path: Directory the root is written under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'COISS_2001/data/N1454725799_1_CALIB', _navigated('N1454725799_1.IMG'))
    write_metadata(
        root,
        'COISS_2001/data/N1454725800_1_CALIB',
        _navigated('N1454725800_1.IMG', cmatrix=None, camera='WAC', offset=None),
    )
    write_metadata(
        root,
        'VGISS_5101/data/C1454725_CALIB',
        _navigated('C1454725.IMG', instrument=OTHER_MISSION, camera=None, shutter_mode=None),
    )
    return root


def _both_sources(tmp_path: Path) -> tuple[TreeRecordSource, IndexRecordSource]:
    """Ingest a tree and return a source reading each of its two storages.

    Parameters:
        tmp_path: Directory the root and the index are written under.

    Returns:
        The tree source and the index source, over the same root.
    """
    root = _tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    index_source = open_record_source(
        [root], results_index_db_url=url, columns=RECORD_COLUMNS, tuning=packaged_tree_tuning()
    )
    assert isinstance(index_source, IndexRecordSource)
    return TreeRecordSource([root], tuning=packaged_tree_tuning()), index_source


def _one_mission(source: RecordSource) -> tuple[list[NavRecord], list[UnreadableFile]]:
    """Read one mission through a source, exactly as the driver reads it.

    Parameters:
        source: The source to read.

    Returns:
        The records and the unreadable files, in the order the driver puts them.
    """
    return read_whole_mission(source.records(Selection(instrument=MISSION)))


@pytest.fixture
def both_readings(tmp_path: Path) -> BothReadings:
    """Read one mission's documents from both sources over the same tree.

    Every parity test below builds the same two sources, reads the same
    mission out of each and closes the index; only what it then compares
    differs.  They stay separate tests rather than one parametrized case
    because the comparisons are not one shape -- some read a field off the
    first entry behind a ``None`` guard, some map over every entry, one needs
    SPICE furnished -- and because each test's docstring says why that
    particular field is load-bearing, which a parameter id cannot.

    The index is closed before the documents are handed over, so there is
    nothing to tear down afterwards: what a test receives is two lists.

    Parameters:
        tmp_path: Directory the root and the index are written under.

    Returns:
        The tree documents and the index documents, in that order.
    """
    tree, index = _both_sources(tmp_path)
    try:
        from_tree, _ = _one_mission(tree)
        from_index, _ = _one_mission(index)
    finally:
        index.close()
    return from_tree, from_index


def _entries(documents: list[NavRecord]) -> list[ImageEntry]:
    """Read the generator's entry for each document.

    Parameters:
        documents: The documents to read.

    Returns:
        One entry per document, in the order given.
    """
    return [ImageEntry.from_metadata(document.metadata) for document in documents]


def test_the_two_sources_return_the_same_images(both_readings: BothReadings) -> None:
    """The mission's images, and only that mission's, from either storage."""
    from_tree, from_index = both_readings
    assert [document.stub for document in from_index] == [document.stub for document in from_tree]


def test_the_run_orders_the_records_rather_than_trusting_their_order(tmp_path: Path) -> None:
    """No query asks for an order, so the run is the only thing imposing one.

    A server sorts text under its own collation, and a locale collation orders
    a separator against an underscore differently from the codepoint order the
    walk produces -- so an ORDER BY would agree with the walk on SQLite and
    disagree on PostgreSQL, which the SQLite-only tier here could never show.
    The rows are therefore put back in an order no walk would produce, which is
    what the unordered query then returns them in: the run must still get them
    in path order, because that is what makes its kernels, its report and its
    log identical whichever storage answered.
    """
    root = _tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    engine = open_index(url)
    try:
        with engine.begin() as connection:
            stored = [dict(row._mapping) for row in connection.execute(sqlalchemy.select(IMAGES))]
            connection.execute(sqlalchemy.delete(IMAGES))
            connection.execute(IMAGES.insert(), list(reversed(stored)))
    finally:
        engine.dispose()
    with open_record_source(
        [root], results_index_db_url=url, columns=RECORD_COLUMNS, tuning=packaged_tree_tuning()
    ) as source:
        # The premise, asserted rather than assumed: an unordered query is the
        # server's to answer how it likes, and one that handed them back already
        # sorted would leave the run's own sort untested and this test unable to
        # fail.
        handed = [
            entry.path.as_posix()
            for entry in source.records(Selection(instrument=MISSION))
            if isinstance(entry, NavRecord)
        ]
        documents, _ = _one_mission(source)
    assert handed != sorted(handed)
    paths = [document.path.as_posix() for document in documents]
    assert paths == sorted(paths)


def test_the_two_sources_agree_on_which_images_are_eligible(both_readings: BothReadings) -> None:
    """Eligibility decides which images get a segment at all."""
    from_tree, from_index = both_readings
    assert [entry.is_eligible for entry in _entries(from_index)] == [
        entry.is_eligible for entry in _entries(from_tree)
    ]


def test_the_two_sources_agree_on_the_omission_reasons(both_readings: BothReadings) -> None:
    """An image with no correction is left out for the same recorded reason."""
    from_tree, from_index = both_readings
    assert [entry.ineligibility_reason for entry in _entries(from_index)] == [
        entry.ineligibility_reason for entry in _entries(from_tree)
    ]


def test_the_two_sources_agree_on_the_corrected_attitude(both_readings: BothReadings) -> None:
    """The matrix a segment is built from must survive the row exactly."""
    from_tree, from_index = both_readings
    tree_pointing = _entries(from_tree)[0].pointing
    index_pointing = _entries(from_index)[0].pointing
    assert tree_pointing is not None
    assert index_pointing is not None
    assert index_pointing.cmatrix.tolist() == tree_pointing.cmatrix.tolist()


def test_the_two_sources_agree_on_the_baseline_attitude(both_readings: BothReadings) -> None:
    """And so must the attitude the correction is measured against."""
    from_tree, from_index = both_readings
    tree_pointing = _entries(from_tree)[0].pointing
    index_pointing = _entries(from_index)[0].pointing
    assert tree_pointing is not None
    assert index_pointing is not None
    assert index_pointing.cmatrix_original.tolist() == tree_pointing.cmatrix_original.tolist()


def test_the_two_sources_agree_on_the_exposure_epochs(both_readings: BothReadings) -> None:
    """A segment covers exactly its exposure, so the epochs decide its extent."""
    from_tree, from_index = both_readings
    tree_pointing = _entries(from_tree)[0].pointing
    index_pointing = _entries(from_index)[0].pointing
    assert tree_pointing is not None
    assert index_pointing is not None
    assert (
        index_pointing.start_et,
        index_pointing.stop_et,
        index_pointing.midtime_et,
        index_pointing.exposure_s,
    ) == (
        tree_pointing.start_et,
        tree_pointing.stop_et,
        tree_pointing.midtime_et,
        tree_pointing.exposure_s,
    )


def test_the_two_sources_agree_on_the_camera_frame(both_readings: BothReadings) -> None:
    """The frame name is what the writer looks up among the furnished kernels."""
    from_tree, from_index = both_readings
    tree_pointing = _entries(from_tree)[0].pointing
    index_pointing = _entries(from_index)[0].pointing
    assert tree_pointing is not None
    assert index_pointing is not None
    assert index_pointing.camera_frame == tree_pointing.camera_frame


def test_the_two_sources_agree_on_the_recorded_kernels(both_readings: BothReadings) -> None:
    """The recorded basenames are what assign a correction to its baseline."""
    from_tree, from_index = both_readings
    assert _entries(from_index)[0].kernel_basenames == _entries(from_tree)[0].kernel_basenames


def test_the_two_sources_agree_on_the_shutter_mode(both_readings: BothReadings) -> None:
    """Two cameras exposed together share a bus attitude, and this is what says so."""
    from_tree, from_index = both_readings
    assert [entry.shutter_mode for entry in _entries(from_index)] == [
        entry.shutter_mode for entry in _entries(from_tree)
    ]


def test_the_two_sources_agree_on_the_camera(both_readings: BothReadings) -> None:
    """The camera decides which member of a simultaneous pair yields."""
    from_tree, from_index = both_readings
    assert [entry.camera for entry in _entries(from_index)] == [
        entry.camera for entry in _entries(from_tree)
    ]


def test_the_two_sources_agree_on_the_reported_facts(
    both_readings: BothReadings, pool: KernelPool
) -> None:
    """Every column of the report an operator reads the run's outcome from.

    Parameters:
        both_readings: The same mission read from the tree and from the index.
        pool: Furnishes the leapseconds kernel the UTC column is converted with.
    """
    from_tree, from_index = both_readings
    assert [read_image_report_facts(document.metadata) for document in from_index] == [
        read_image_report_facts(document.metadata) for document in from_tree
    ]


def test_the_index_source_names_the_file_each_row_was_read_from(
    both_readings: BothReadings,
) -> None:
    """A message about a document names the file an operator would open."""
    from_tree, from_index = both_readings
    assert [document.path.as_posix() for document in from_index] == [
        document.path.as_posix() for document in from_tree
    ]


def test_another_roots_images_are_not_this_runs(tmp_path: Path) -> None:
    """One index serves several roots, and a query blind to the root writes both.

    The second root holds an image of the same mission, so a query that filtered
    only on the instrument would hand this run an image nobody asked for -- and
    write it into a kernel.
    """
    root = _tree(tmp_path)
    other = tmp_path / 'other-results'
    write_metadata(other, 'COISS_2002/data/N1454999999_1_CALIB', _navigated('N1454999999_1.IMG'))
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root, other], logger=null_logger())
    with open_record_source(
        [root], results_index_db_url=url, columns=RECORD_COLUMNS, tuning=packaged_tree_tuning()
    ) as source:
        documents, _unreadable = _one_mission(source)
    assert [document.stub for document in documents] == [
        'COISS_2001/data/N1454725799_1_CALIB',
        'COISS_2001/data/N1454725800_1_CALIB',
    ]


def test_a_document_the_ingest_refused_is_reported_as_unreadable(tmp_path: Path) -> None:
    """It is a file that exists and holds no record, which is what stops the run.

    The file path reports a document it cannot read and exits nonzero; a
    refused document is the index's account of exactly that file, and reporting
    nothing would let a run write a kernel set that silently left it out.
    """
    root = _tree(tmp_path)
    (root / 'COISS_2001' / 'data' / 'junk_metadata.json').write_text('{}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    with open_record_source(
        [root], results_index_db_url=url, columns=RECORD_COLUMNS, tuning=packaged_tree_tuning()
    ) as source:
        _documents, unreadable = _one_mission(source)
    assert [entry.path.as_posix() for entry in unreadable] == [
        (root / 'COISS_2001' / 'data' / 'junk_metadata.json').as_posix()
    ]


def test_the_refusal_reason_travels_with_the_file(tmp_path: Path) -> None:
    """What the ingest could not read is what an operator has to go and fix."""
    root = _tree(tmp_path)
    (root / 'COISS_2001' / 'data' / 'junk_metadata.json').write_text('{}', encoding='utf-8')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    with open_record_source(
        [root], results_index_db_url=url, columns=RECORD_COLUMNS, tuning=packaged_tree_tuning()
    ) as source:
        _documents, unreadable = _one_mission(source)
    assert 'navigation document' in unreadable[0].reason


def test_a_root_with_no_completed_ingest_is_refused(tmp_path: Path) -> None:
    """Absence of a row would otherwise read as a mission with no images."""
    root = _tree(tmp_path)
    other = tmp_path / 'other-results'
    write_metadata(other, 'COISS_2002/data/N1454999999_1_CALIB', _navigated('N1454999999_1.IMG'))
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [other], logger=null_logger())
    with pytest.raises(ValueError, match='no completed ingest'):
        open_record_source(
            [root], results_index_db_url=url, columns=RECORD_COLUMNS, tuning=packaged_tree_tuning()
        )


def test_no_index_url_reads_the_tree(tmp_path: Path) -> None:
    """Reading files is the default, and nothing opens a database to do it."""
    root = _tree(tmp_path)
    with open_record_source(
        [root], results_index_db_url=None, columns=RECORD_COLUMNS, tuning=packaged_tree_tuning()
    ) as source:
        assert isinstance(source, TreeRecordSource)


def test_the_index_source_says_which_index_it_read(tmp_path: Path) -> None:
    """The run log has to say where the records came from, not just how many."""
    _tree, index = _both_sources(tmp_path)
    try:
        described = index.describe()
    finally:
        index.close()
    assert 'results index' in described


def test_the_description_hides_a_password_in_the_index_url(tmp_path: Path) -> None:
    """A run log reaches a console, a file, and whoever is sent one."""
    root = _tree(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    source = IndexRecordSource(
        open_index(url),
        [FCPath(root).as_posix()],
        'postgresql+psycopg://svc:sup3rs3cr3t@db.example/spindoctor',
        RECORD_COLUMNS,
    )
    try:
        described = source.describe()
    finally:
        source.close()
    assert 'sup3rs3cr3t' not in described


def test_a_value_the_ingest_could_not_store_reads_as_one_never_recorded(
    tmp_path: Path, pool: KernelPool
) -> None:
    """The one difference the seam permits, in the direction it permits it.

    An offset of three numbers is a defect in the record, and the tree path
    refuses the document naming it.  The index stores an offset it cannot read
    whole as NULL, exactly as it stores an absent one, so the row cannot say
    which it was and the rebuilt document reads as one that recorded none.

    Parameters:
        tmp_path: Directory the root and the index are written under.
        pool: Furnishes the leapseconds kernel the UTC column is converted with.
    """
    root = tmp_path / 'results'
    write_metadata(
        root,
        'COISS_2001/data/N1454725799_1_CALIB',
        _navigated('N1454725799_1.IMG', offset=(1.5, -2.5, 0.5)),
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    with open_record_source(
        [root], results_index_db_url=url, columns=RECORD_COLUMNS, tuning=packaged_tree_tuning()
    ) as source:
        documents, _unreadable = _one_mission(source)
    assert read_image_report_facts(documents[0].metadata).offset_dv is None


def test_the_tree_path_refuses_the_same_document(tmp_path: Path, pool: KernelPool) -> None:
    """The control for the difference above: the file path stops the run on it.

    Parameters:
        tmp_path: Directory the root is written under.
        pool: Furnishes the leapseconds kernel the UTC column is converted with.
    """
    root = tmp_path / 'results'
    write_metadata(
        root,
        'COISS_2001/data/N1454725799_1_CALIB',
        _navigated('N1454725799_1.IMG', offset=(1.5, -2.5, 0.5)),
    )
    documents, _unreadable = _one_mission(TreeRecordSource([root], tuning=packaged_tree_tuning()))
    with pytest.raises(ValueError, match='offset'):
        read_image_report_facts(documents[0].metadata)
