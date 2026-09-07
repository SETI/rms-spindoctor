"""The selection filters answered from a results index, against the same tree.

The parity matrix is here: every filter flag asked of the tree and of the index
over the one fixture tree, each held to a stated answer, plus the assertions that
the index path reads no file at all and leaves nothing of the index open.  So is
every document shape the two storages could plausibly read differently and do
not, each asked of both over a tree of its own.

Four files carry the rest: which records the filter asks each storage about,
what it refuses outright, the answers the index gives differently from the tree,
each with a test of its own, and what the filter reports about the pass that
filled the index.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from filecache import FCPath
from sqlalchemy.pool import QueuePool
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
)
from tests.spindoctor.dataset.conftest import (
    ERROR_WITHOUT_STATUS_ERROR,
    FATAL_ERRORS,
    NO_RESULT,
    NONSPICE_ERROR,
    SECOND_SUCCESS,
    SPICE_ERROR,
    VOLUMES,
    WITH_A_DOCUMENT,
    WITHOUT_A_FATAL_ERROR,
    null_logger,
    one_image_tree,
    select_from,
    selection_of,
)

from spindoctor.cli.results_index import UnwritableRowError, store
from spindoctor.dataset.dataset import ImageFile
from spindoctor.dataset.results_filter import SPICE_STATUS_ERROR, ResultsFilter, SelectionError

_MATRIX = [
    pytest.param({'has_offset_file': True}, list(WITH_A_DOCUMENT), id='offset-file'),
    pytest.param({'has_no_offset_file': True}, [NO_RESULT], id='no-offset-file'),
    pytest.param({'has_offset_error': True}, list(FATAL_ERRORS), id='offset-error'),
    pytest.param({'has_no_offset_error': True}, list(WITHOUT_A_FATAL_ERROR), id='no-offset-error'),
    pytest.param({'has_offset_spice_error': True}, [SPICE_ERROR], id='spice-error'),
    pytest.param(
        {'has_offset_nonspice_error': True},
        [NONSPICE_ERROR, ERROR_WITHOUT_STATUS_ERROR],
        id='nonspice-error',
    ),
    pytest.param(
        {'has_offset_error': True, 'has_offset_file': True},
        list(FATAL_ERRORS),
        id='offset-error-and-offset-file',
    ),
    pytest.param(
        {'has_offset_spice_error': True, 'has_offset_file': True},
        [SPICE_ERROR],
        id='spice-error-and-offset-file',
    ),
    pytest.param(
        {'has_no_offset_error': True, 'has_offset_file': True},
        list(WITHOUT_A_FATAL_ERROR),
        id='no-offset-error-and-offset-file',
    ),
]
"""Every filter flag, alone and paired, with the selection it makes.

The flags ask the seam one of two questions, and both have to land on the same
answer whichever storage answers them: the presence and absence filters are
settled by a listing, which opens no document, and the error filters are settled
by what each document records.  An error filter folds presence in, since a
document that is not there records nothing.

The pairings are the combinations a user has a reason to write.  ``images this
run navigated to a result`` is the last of them, and it is the pair rather than
a flag because presence and outcome are separate questions in this vocabulary;
that each error filter folds presence in, so that naming it changes nothing, is
what the first two pin.
"""


@pytest.mark.parametrize(('flags', 'expected'), _MATRIX)
def test_the_tree_answers_each_filter(
    tree: Path, flags: dict[str, Any], expected: list[str]
) -> None:
    """What reading the results tree selects, stated rather than compared."""
    assert selection_of(tree, flags, results_index_db_url=None) == expected


@pytest.mark.parametrize(('flags', 'expected'), _MATRIX)
def test_the_index_answers_each_filter_the_same_way(
    tree: Path, indexed: str, flags: dict[str, Any], expected: list[str]
) -> None:
    """One query reaches the answer the walk and the per-image reads reach."""
    assert selection_of(tree, flags, results_index_db_url=indexed) == expected


@pytest.mark.parametrize(('flags', 'expected'), _MATRIX)
def test_the_index_path_reads_no_file_at_all(
    tree: Path,
    indexed: str,
    flags: dict[str, Any],
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every route to the results tree is broken, and the answer is unchanged.

    The saving is the whole point of the index: no directory listed, no
    existence check, no metadata download.  Counting round trips would prove the
    same thing more weakly, since a path taken once still costs a round trip per
    enumeration on a cloud root.

    Parameters:
        tree: The results root under test.
        indexed: The index answering the filters.
        flags: The selection flags to apply.
        expected: The stubs the filter selects.
        monkeypatch: Fixture the storage layer is broken through.
    """

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError('the index path must not read the results tree')

    monkeypatch.setattr(FCPath, 'iterdir_metadata', refuse)
    monkeypatch.setattr(FCPath, 'walk', refuse)
    monkeypatch.setattr(FCPath, 'exists', refuse)
    monkeypatch.setattr(FCPath, 'retrieve', refuse)
    assert selection_of(tree, flags, results_index_db_url=indexed) == expected


def test_an_image_with_no_document_is_not_one_recording_no_error_in_the_tree(tree: Path) -> None:
    """The negative error filter asks what a document records, so it needs one.

    An image nothing has been written for is what ``has_no_offset_file``
    selects.  Reading its absence as an outcome would put it in both selections
    at once and leave no way to ask for either without the other.
    """
    kept = selection_of(tree, {'has_no_offset_error': True}, results_index_db_url=None)
    assert NO_RESULT not in kept


ERROR_FLAGS = [
    'has_offset_error',
    'has_no_offset_error',
    'has_offset_spice_error',
    'has_offset_nonspice_error',
]
"""Every flag that asks what a document records rather than whether one is there."""


@pytest.mark.parametrize('flag', ERROR_FLAGS)
def test_an_error_filter_passes_over_an_image_with_no_document(tree: Path, flag: str) -> None:
    """Every error filter folds presence in, and this is where that shows.

    An image nothing has been written for records no error and records no
    outcome either, so it belongs to none of the four selections.  The filter
    phrased in the negative is the one that would otherwise take it: read as
    "records no fatal error", absence would put an image in this selection and
    in the one ``has_no_offset_file`` makes, leaving no way to ask for either
    without the other.

    Parameters:
        tree: The results root under test.
        flag: The error filter, one per flag that reads a document.
    """
    flags: dict[str, Any] = {flag: True}
    results_filter = ResultsFilter(
        VOLUMES, str(tree), logger=null_logger(), results_index_db_url=None, **flags
    )
    assert results_filter.passes(NO_RESULT) is False


def test_an_image_with_no_row_is_not_one_recording_no_error_in_the_index(
    tree: Path, indexed: str
) -> None:
    """Absence of a row is absence of a document, and reads the same way here."""
    kept = selection_of(tree, {'has_no_offset_error': True}, results_index_db_url=indexed)
    assert NO_RESULT not in kept


_NO_OUTCOME_STUB = 'COISS_2001/data/b/N1000000011_1_CALIB'
"""The one image of the tree written for the document naming no outcome.

Under a selected volume, since a stub outside one is answered by neither path
and would make every expectation below the empty selection for the wrong
reason.  It is not a stub of the shared fixture tree: that tree states an answer
per filter for ten documents at once, and what is asked here is what one
document shape answers.
"""

_NO_STATUS_FIELD = object()
"""Stands for the document that carries no top-level ``status`` field at all."""

_NO_OUTCOME_STATUSES = [
    pytest.param(_NO_STATUS_FIELD, id='absent'),
    pytest.param(None, id='null'),
    pytest.param('', id='empty'),
    pytest.param(42, id='not-a-string'),
]
"""Every shape of a top-level ``status`` that names no outcome.

The four are kept apart because one reader tells them apart by type and another
by SQL: a document carrying no field, one carrying null, one carrying the empty
string and one carrying a number reach the store as four different values and
have to leave it as one.
"""

_NO_OUTCOME_MATRIX = [
    pytest.param({'has_offset_file': True}, [_NO_OUTCOME_STUB], id='offset-file'),
    pytest.param({'has_offset_error': True}, [], id='offset-error'),
    pytest.param({'has_no_offset_error': True}, [_NO_OUTCOME_STUB], id='no-offset-error'),
    pytest.param({'has_offset_spice_error': True}, [], id='spice-error'),
    pytest.param({'has_offset_nonspice_error': True}, [], id='nonspice-error'),
]
"""What every filter that reads a document answers about one naming no outcome.

The presence filter leads, because it is what says the document is there to be
read: without it every empty answer below would also be the answer for a root
holding nothing.  ``has_no_offset_error`` carries the same weight from the other
side -- a document the ingest refused matches it no more than it matches the
rest, so an answer of the image itself is an answer read off a stored outcome.
"""


def _naming_no_outcome(root: Path, status: Any) -> list[ImageFile]:
    """Write a document that names no outcome of its own, and return its image.

    The outcome sits in the nested ``navigation_result`` copy instead, which is
    what makes the answers evidence of anything: a path that read the nested
    copy wherever the top-level field names nothing would answer the error
    filters with this image and the negative one without it.

    Parameters:
        root: The results root to write into.
        status: The top-level ``status`` the document carries, or
            :data:`_NO_STATUS_FIELD` for one carrying no such field.

    Returns:
        The one candidate image, ready to filter.
    """
    document = metadata_document(image_name='N1000000011_1.IMG', offset=None)
    if status is _NO_STATUS_FIELD:
        del document['status']
    else:
        document['status'] = status
    document['navigation_result']['status'] = 'error'
    write_metadata(root, _NO_OUTCOME_STUB, document)
    return [
        ImageFile(
            image_file_url=FCPath(root / f'{_NO_OUTCOME_STUB}.IMG'),
            label_file_url=FCPath(root / f'{_NO_OUTCOME_STUB}.LBL'),
            results_path_stub=_NO_OUTCOME_STUB,
        )
    ]


@pytest.mark.parametrize('status', _NO_OUTCOME_STATUSES)
@pytest.mark.parametrize(('flags', 'expected'), _NO_OUTCOME_MATRIX)
def test_the_tree_reads_a_document_naming_no_outcome(
    tmp_path: Path, status: Any, flags: dict[str, Any], expected: list[str]
) -> None:
    """The walk reads the top-level field and no other, so it finds no outcome.

    Parameters:
        tmp_path: Directory the root is written under.
        status: The shape of top-level ``status`` the document carries.
        flags: The selection flags to apply.
        expected: The stubs the filter selects.
    """
    root = tmp_path / 'results'
    images = _naming_no_outcome(root, status)
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_index_db_url=None, **flags
    )
    assert select_from(results_filter, images) == expected


@pytest.mark.parametrize('status', _NO_OUTCOME_STATUSES)
@pytest.mark.parametrize(('flags', 'expected'), _NO_OUTCOME_MATRIX)
def test_the_index_reads_a_document_naming_no_outcome_the_same_way(
    tmp_path: Path, status: Any, flags: dict[str, Any], expected: list[str]
) -> None:
    """The stored status is the document's own field, so the query answers alike.

    Every shape above is stored as the one value a record naming no outcome is
    read as, which is an outcome the error filters name nowhere.  A store that
    borrowed the nested copy would be making a classification no reader of the
    document could arrive at, and -- since the pointing readers rebuild a record
    from these same columns -- would hand a corrected attitude to an image whose
    document supplies no pointing at all.

    Parameters:
        tmp_path: Directory the root and the index are written under.
        status: The shape of top-level ``status`` the document carries.
        flags: The selection flags to apply.
        expected: The stubs the filter selects.
    """
    root = tmp_path / 'results'
    images = _naming_no_outcome(root, status)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, **flags
    )
    assert select_from(results_filter, images) == expected


# ---------------------------------------------------------------------------
# A JSON object that is not a navigation document
# ---------------------------------------------------------------------------

NOT_A_NAVIGATION_DOCUMENT = [
    pytest.param({'status': 'error', 'status_error': SPICE_STATUS_ERROR}, id='fatal-error'),
    pytest.param({'status': 'success'}, id='plain-outcome'),
]
"""Two JSON objects that read perfectly and are no navigation result of any schema.

Each carries the fields the error filters name and nothing else a document has
-- no image, no mission, no navigation result -- and between them they carry
both sides of the vocabulary, so a filter phrased in the positive and one
phrased in the negative each have a candidate that would satisfy them if the two
fields were read out of whatever could be parsed.
"""

STORAGES = [pytest.param(False, id='tree'), pytest.param(True, id='index')]
"""Whether the filter is answered from an ingested index rather than the tree.

Both halves are asked, and each is held to the stated answer rather than to the
other's: two storages that are wrong in the same way agree.
"""


def _selecting_one_object(
    tmp_path: Path, document: dict[str, Any], *, from_an_index: bool, **flags: Any
) -> list[str]:
    """Write one JSON object as an image's metadata file and answer one filter.

    Parameters:
        tmp_path: Directory the root and any index are written under.
        document: Exactly what the metadata file holds.
        from_an_index: Whether the filter reads an ingested index rather than
            the tree.
        flags: The selection flags to apply.

    Returns:
        The stubs that passed.
    """
    root = tmp_path / 'results'
    write_metadata(root, SPICE_ERROR, document)
    images = [
        ImageFile(
            image_file_url=FCPath(root / f'{SPICE_ERROR}.IMG'),
            label_file_url=FCPath(root / f'{SPICE_ERROR}.LBL'),
            results_path_stub=SPICE_ERROR,
        )
    ]
    url = None
    if from_an_index:
        url = index_url(tmp_path / 'index.sqlite3')
        ingest_tree(url, [root], logger=null_logger())
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, **flags
    )
    return select_from(results_filter, images)


@pytest.mark.parametrize('document', NOT_A_NAVIGATION_DOCUMENT)
@pytest.mark.parametrize('from_an_index', STORAGES)
def test_an_object_that_is_no_navigation_document_is_present_to_both_storages(
    tmp_path: Path, from_an_index: bool, document: dict[str, Any]
) -> None:
    """Presence is a question about the file, so whatever is in it, it is there.

    A results root holds whatever an operator has put there, and the presence
    filters are what a resume idiom is spelled from: a file that exists and
    reads as nothing has still been written, and offering its image up to be
    navigated again would overwrite it.

    Parameters:
        tmp_path: Directory the root and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
        document: The object the metadata file holds.
    """
    kept = _selecting_one_object(
        tmp_path, document, from_an_index=from_an_index, has_offset_file=True
    )
    assert kept == [SPICE_ERROR]


@pytest.mark.parametrize('document', NOT_A_NAVIGATION_DOCUMENT)
@pytest.mark.parametrize('from_an_index', STORAGES)
@pytest.mark.parametrize('flag', ERROR_FLAGS)
def test_an_object_that_is_no_navigation_document_matches_no_error_filter(
    tmp_path: Path, flag: str, from_an_index: bool, document: dict[str, Any]
) -> None:
    """What such a file records is unknown, not known to be an outcome.

    The error filters ask what a document records about its image, and nothing
    here records anything about an image: the fields are two words that happen
    to be spelled like a navigator's.  The filter phrased in the negative is one
    of the four for that reason -- selecting this image would claim its
    navigation ran to a result, and nothing in the file says an image was
    navigated at all.

    Parameters:
        tmp_path: Directory the root and any index are written under.
        flag: The error filter, one per flag that reads a document.
        from_an_index: Whether the filter reads an index rather than the tree.
        document: The object the metadata file holds.
    """
    kept = _selecting_one_object(tmp_path, document, from_an_index=from_an_index, **{flag: True})
    assert kept == []


# ---------------------------------------------------------------------------
# A document that has left the tree
# ---------------------------------------------------------------------------


def _index_after_a_document_left_the_tree(
    tmp_path: Path,
) -> tuple[Path, list[ImageFile], str]:
    """Ingest a root, delete one of its documents, and ingest it again.

    This was a divergence while a pass that could not list one directory
    completed anyway and removed no row: the deleted document's row then
    outlived any number of finished passes.  A pass that meets such a directory
    now stops instead, so every completed pass is one that listed the whole
    root and every completed pass prunes, and the two paths answer alike.

    Parameters:
        tmp_path: Directory the root and the index are written under.

    Returns:
        The root, the two candidate images, and the connection URL of the index.
    """
    root = tmp_path / 'results'
    write_metadata(root, SECOND_SUCCESS, metadata_document(image_name='N1000000002_1.IMG'))
    write_metadata(root, SPICE_ERROR, metadata_document(image_name='N1000000004_1.IMG'))
    images = [
        ImageFile(
            image_file_url=FCPath(root / f'{stub}.IMG'),
            label_file_url=FCPath(root / f'{stub}.LBL'),
            results_path_stub=stub,
        )
        for stub in (SECOND_SUCCESS, SPICE_ERROR)
    ]
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=null_logger())
    (root / f'{SPICE_ERROR}_metadata.json').unlink()
    ingest_tree(url, [root], logger=null_logger())
    return root, images, url


def test_a_document_that_left_the_tree_reads_as_absent_in_the_tree(tmp_path: Path) -> None:
    """The walk finds what is there now, which is the answer the index is held to."""
    root, images, _url = _index_after_a_document_left_the_tree(tmp_path)
    results_filter = ResultsFilter(VOLUMES, str(root), logger=null_logger(), has_offset_file=True)
    assert select_from(results_filter, images) == [SECOND_SUCCESS]


def test_a_document_that_left_the_tree_reads_as_absent_in_the_index(tmp_path: Path) -> None:
    """The pass that listed the whole root had the evidence to remove the row, and did.

    Presence of a row means the tree still holds the result, which is what makes
    absence of one mean that nothing navigated the image.
    """
    root, images, url = _index_after_a_document_left_the_tree(tmp_path)
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, has_offset_file=True
    )
    assert select_from(results_filter, images) == [SECOND_SUCCESS]


def test_the_tree_offers_a_document_that_left_it_to_the_absence_filter(tmp_path: Path) -> None:
    """Nothing has been written for that image now, so the resume idiom picks it up."""
    root, images, _url = _index_after_a_document_left_the_tree(tmp_path)
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), has_no_offset_file=True
    )
    assert select_from(results_filter, images) == [SPICE_ERROR]


def test_the_index_offers_a_document_that_left_the_tree_to_the_absence_filter(
    tmp_path: Path,
) -> None:
    """The other direction of the same row, and the costlier one to get wrong.

    ``--has-no-offset-file`` is the resume idiom, so a row that outlived its
    document is an image the run silently declines to navigate again.
    """
    root, images, url = _index_after_a_document_left_the_tree(tmp_path)
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, has_no_offset_file=True
    )
    assert select_from(results_filter, images) == [SPICE_ERROR]


def test_a_document_the_database_would_not_store_leaves_no_index_to_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is not one of the answers the two give differently, because there is no answer.

    A row the database refuses ends the pass where it happened, so the root has
    no completed ingest run and every filter answered from the index refuses the
    root rather than reading absence under it as "this image was never
    navigated".

    Parameters:
        tmp_path: Directory the tree and the index live under.
        monkeypatch: Fixture the image write is replaced through.
    """
    root, _images = one_image_tree(tmp_path)

    def refuse(connection: Any, rows: Any) -> None:
        raise sqlalchemy.exc.IntegrityError('INSERT', {}, Exception('refused'))

    url = index_url(tmp_path / 'index.sqlite3')
    monkeypatch.setattr(store, '_write_image', refuse)
    with pytest.raises(UnwritableRowError, match='would not accept its rows'):
        ingest_tree(url, [root], logger=null_logger())


def test_a_document_the_database_would_not_store_leaves_the_root_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state the failed pass leaves, which is what makes the absence unreadable.

    Ending the pass is only half of it: what stops a filter reading the gap as
    "this image was never navigated" is that the root keeps no completed ingest
    run, so a filter answered from the index refuses it outright.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        monkeypatch: Fixture the image write is replaced through.
    """
    root, _images = one_image_tree(tmp_path)

    def refuse(connection: Any, rows: Any) -> None:
        raise sqlalchemy.exc.IntegrityError('INSERT', {}, Exception('refused'))

    url = index_url(tmp_path / 'index.sqlite3')
    monkeypatch.setattr(store, '_write_image', refuse)
    with pytest.raises(UnwritableRowError):
        ingest_tree(url, [root], logger=null_logger())
    monkeypatch.undo()
    with pytest.raises(SelectionError, match='no completed ingest'):
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, has_offset_file=True
        )


# ---------------------------------------------------------------------------
# What the answer leaves open
# ---------------------------------------------------------------------------


@dataclass
class _EngineBuilt:
    """One engine built to answer a filter, and what became of the pool it started with.

    Parameters:
        engine: The engine, as answering the filter left it.
        pool: The pool it held when it was built.
        checked_out_at_disposal: How many connections were still out of that pool
            at the moment disposal was asked for, and None if it was never
            disposed of.  Disposal installs a fresh pool and leaves whatever is
            checked out of the old one checked out, so this is the last moment at
            which a connection the answer kept is visible at all.
    """

    engine: sqlalchemy.Engine
    pool: QueuePool
    checked_out_at_disposal: int | None = None


def _engines_built_answering(
    tree: Path, indexed: str, monkeypatch: pytest.MonkeyPatch
) -> list[_EngineBuilt]:
    """Answer a filter, recording every engine built for it and how the answer left it.

    Parameters:
        tree: The results root under test.
        indexed: The index answering the filter.
        monkeypatch: Fixture the recording hooks are installed through.

    Returns:
        One record per engine, in the order they were built.  There is always at
        least one, since a filter answered from an index opens one.
    """
    built: list[_EngineBuilt] = []
    create_engine = sqlalchemy.create_engine
    disposing = sqlalchemy.Engine.dispose

    def recording(*args: Any, **kwargs: Any) -> sqlalchemy.Engine:
        made = create_engine(*args, **kwargs)
        # A file-backed SQLite engine pools its connections, and only a pooling
        # implementation can report what it is holding.
        assert isinstance(made.pool, QueuePool)
        built.append(_EngineBuilt(made, made.pool))
        return made

    def counting(engine: sqlalchemy.Engine, *args: Any, **kwargs: Any) -> None:
        # What is checked out has to be counted before the pool goes, and
        # disposal is where it goes: it installs a fresh one and leaves whatever
        # was checked out of the old one checked out, so afterwards neither pool
        # reports a connection the answer kept.
        for record in built:
            if record.engine is engine:
                record.checked_out_at_disposal = record.pool.checkedout()
        disposing(engine, *args, **kwargs)

    monkeypatch.setattr(sqlalchemy, 'create_engine', recording)
    monkeypatch.setattr(sqlalchemy.Engine, 'dispose', counting)
    ResultsFilter(
        VOLUMES, str(tree), logger=null_logger(), results_index_db_url=indexed, has_offset_file=True
    )
    assert built != []
    return built


def test_answering_the_filter_leaves_no_index_open(
    tree: Path, indexed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answer outlives the connection, and nothing else does.

    A filter is built once per enumeration and holds its answer for the whole of
    it, so an undisposed pool keeps a SQLite connection, or a server session, for
    the length of a navigation run.  Disposal replaces the pool, which is the
    observable proof that it happened.

    Parameters:
        tree: The results root under test.
        indexed: The index answering the filter.
        monkeypatch: Fixture the recording hooks are installed through.
    """
    left_open = [
        record.engine
        for record in _engines_built_answering(tree, indexed, monkeypatch)
        if record.engine.pool is record.pool
    ]
    assert left_open == []


def test_answering_the_filter_returns_every_connection(
    tree: Path, indexed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stream holding a cursor has to release it, or a run leaks one per query.

    The count is read at the moment disposal is asked for, which is the last
    moment it says anything: disposal replaces the pool without reclaiming what
    is checked out of it, so afterwards neither the old pool nor the new one
    reports the connection the answer kept.

    Parameters:
        tree: The results root under test.
        indexed: The index answering the filter.
        monkeypatch: Fixture the recording hooks are installed through.
    """
    checked_out = [
        record.checked_out_at_disposal
        for record in _engines_built_answering(tree, indexed, monkeypatch)
    ]
    assert checked_out == [0] * len(checked_out)
