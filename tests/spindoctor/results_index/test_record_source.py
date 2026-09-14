"""The index-backed half of the record seam, and the factory that opens either half.

Three calls, two of them streams, all three keyed by ``(root_url,
results_path_stub)``.  Two things are tested throughout and are worth naming
because both have been real defects here.

**The root half of the key.**  One index serves several roots, so every query
carries a root term and a query missing it answers with another root's files.
Every test of a query therefore runs against an index holding two roots whose
rows differ in exactly the value that test reads, so a term that dropped the
root would hand back the wrong value rather than none.

**The two storages agree.**  A consumer cannot see which half answered, so the
parity tests drive both halves over one tree and compare what each yields.  The
differences the seam declares -- the columns a consumer selected, a value no
column can hold, a stub the index never heard of -- are tested as differences,
in :mod:`tests.spindoctor.cli.ck.test_records` and in the reprojection tests
where their consumers are.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from sqlalchemy.engine import Engine
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
    write_refusal,
)

from spindoctor.nav_records import (
    METADATA_SUFFIX,
    ImageFacts,
    ListedRecord,
    NavRecord,
    Selection,
    TreeRecordSource,
    UnreadableFile,
    normalize_root_url,
)
from spindoctor.results_index import (
    FAILED_FILES,
    IMAGES,
    IndexRecordSource,
    open_index,
    open_index_for_roots,
    open_record_source,
    roots,
)

COLUMNS = (
    IMAGES.c.status,
    IMAGES.c.instrument,
    IMAGES.c.offset_dv,
    IMAGES.c.offset_du,
    IMAGES.c.midtime_et,
)
"""A consumer's columns, standing in for any consumer's."""

MISSION = 'coiss'
"""The instrument identity the mission-filtered reads below keep."""

OTHER_MISSION = 'vgiss'
"""An instrument identity of another mission's documents in the same tree."""

FIRST_STUB = 'VOL1/N1454725799_1_CALIB'
"""The image every per-image test below reads, held by both roots."""

SECOND_STUB = 'VOL2/N1454725800_1_CALIB'
"""An image of the same mission in the other subtree, held by both roots."""

OTHER_MISSION_STUB = 'VOL1/C1454725_CALIB'
"""An image of another mission, held by both roots."""

FIRST_MIDTIME = 100.0
"""The exposure midtime the first image records."""

SECOND_MIDTIME = 300.0
"""The exposure midtime the second image records, well after the first."""


def _document(name: str, *, instrument: str, offset: list[float], midtime: float) -> dict[str, Any]:
    """Build one navigated image's document.

    Parameters:
        name: The recorded image name.
        instrument: The mission the document names.
        offset: The recorded offset, which is what tells the two roots apart.
        midtime: The recorded exposure midtime.

    Returns:
        The document.
    """
    return metadata_document(
        image_name=name,
        instrument=instrument,
        offset=offset,
        times={'midtime_et': midtime, 'start_et': midtime - 1.0, 'stop_et': midtime + 1.0},
    )


def _root(tmp_path: Path, name: str, *, offset: list[float], refused: str) -> Path:
    """Write one results root holding what every test below reads.

    Parameters:
        tmp_path: Directory the root is written under.
        name: The root's directory name.
        offset: The offset both of this root's mission images record, which is
            the value a root-blind query would read out of the wrong root.
        refused: The stub this root's refused document sits at, which differs
            between the two roots for the same reason.

    Returns:
        The results root.
    """
    root = tmp_path / name
    write_metadata(
        root,
        FIRST_STUB,
        _document('N1454725799_1.IMG', instrument=MISSION, offset=offset, midtime=FIRST_MIDTIME),
    )
    write_metadata(
        root,
        SECOND_STUB,
        _document('N1454725800_1.IMG', instrument=MISSION, offset=offset, midtime=SECOND_MIDTIME),
    )
    write_metadata(
        root,
        OTHER_MISSION_STUB,
        _document('C1454725.IMG', instrument=OTHER_MISSION, offset=offset, midtime=FIRST_MIDTIME),
    )
    write_refusal(root, refused)
    return root


FIRST_OFFSET = [1.5, -2.5]
"""What the first root's images record."""

SECOND_OFFSET = [9.5, -8.5]
"""What the second root's images record, so the two are told apart by value."""

FIRST_REFUSAL_STUB = 'VOL1/junk_first'
"""Where the first root's refused document sits."""

SECOND_REFUSAL_STUB = 'VOL2/junk_second'
"""Where the second root's sits, in the other subtree and under another name."""


@pytest.fixture
def quiet_logger() -> pdslogger.PdsLogger:
    """Return a logger that writes nowhere.

    Returns:
        The logger, so an ingest driven by a test says nothing.
    """
    return pdslogger.NullLogger()


class TwoRoots:
    """An index holding two results roots that differ in every value read here.

    Parameters:
        first: The root the tests read.
        second: The root nothing asks for, holding the same stubs with other
            values, a stub the first does not hold, and its refusal somewhere
            else.  A query that dropped the root half of the key answers out of
            whichever of the two the server reached first.
        url: The index both were ingested into.
    """

    def __init__(self, first: Path, second: Path, url: str) -> None:
        self.first = first
        self.second = second
        self.url = url

    @property
    def first_url(self) -> str:
        """The normalized URL of the root the tests read.

        Returns:
            The root URL, as the index records it.
        """
        return normalize_root_url(self.first)

    @property
    def second_url(self) -> str:
        """The normalized URL of the root nothing asks for.

        Returns:
            The root URL, as the index records it.
        """
        return normalize_root_url(self.second)


@pytest.fixture
def two_roots(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> TwoRoots:
    """Ingest two results roots into one index.

    Parameters:
        tmp_path: Directory the roots and the index are written under.
        quiet_logger: Logger the ingest reports through.

    Returns:
        The two roots and the index holding both.
    """
    first = _root(tmp_path, 'results-first', offset=FIRST_OFFSET, refused=FIRST_REFUSAL_STUB)
    second = _root(tmp_path, 'results-second', offset=SECOND_OFFSET, refused=SECOND_REFUSAL_STUB)
    write_metadata(
        second,
        'VOL2/N9999999999_1_CALIB',
        _document('N9999999999_1.IMG', instrument=MISSION, offset=SECOND_OFFSET, midtime=500.0),
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first, second], logger=quiet_logger)
    return TwoRoots(first, second, url)


def _index_over(held: TwoRoots, *which: Path) -> IndexRecordSource:
    """Open an index-backed source over some of the two ingested roots.

    Parameters:
        held: The two roots and their index.
        which: The roots the source is to hold, in order.

    Returns:
        The source, which the caller closes.
    """
    source = open_record_source(list(which), results_index_db_url=held.url, columns=COLUMNS)
    assert isinstance(source, IndexRecordSource)
    return source


def _stubs(found: Iterator[Any]) -> list[str]:
    """Return the stubs of everything a stream yielded, in the order it came.

    Parameters:
        found: What the stream yielded.

    Returns:
        The stubs.
    """
    return [entry.stub for entry in found]


def _facts_stub(one: ImageFacts | UnreadableFile) -> str:
    """Return the stub of one thing a stream of facts yielded.

    Parameters:
        one: The facts, or the file no facts came out of.

    Returns:
        The stub.  An image carries its own as a column and a file no facts came
        out of carries it beside its path.
    """
    if isinstance(one, UnreadableFile):
        return one.stub
    return str(one.image['results_path_stub'])


def _offsets(found: Iterator[NavRecord | UnreadableFile]) -> list[Any]:
    """Return the offset of every record a stream yielded.

    Parameters:
        found: What the stream yielded.

    Returns:
        The offsets, in the order the records arrived.
    """
    return [entry.metadata['offset'] for entry in found if isinstance(entry, NavRecord)]


# ---------------------------------------------------------------------------
# One image, by its stub
# ---------------------------------------------------------------------------


def test_one_image_is_rebuilt_from_its_row(two_roots: TwoRoots) -> None:
    """The per-image shape, which is what a stub is a key for.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        assert source.record(FIRST_STUB).metadata['offset'] == FIRST_OFFSET


def test_the_other_roots_row_is_not_this_ones(two_roots: TwoRoots) -> None:
    """The same stub under the other root, whose record says something else.

    Read against the second root, so that a lookup dropping the root half of the
    key is caught whichever of the two rows the server happens to reach first.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.second) as source:
        assert source.record(FIRST_STUB).metadata['offset'] == SECOND_OFFSET


def test_a_source_holding_two_roots_refuses_a_bare_stub(two_roots: TwoRoots) -> None:
    """A stub is a key under a root and does not say which root.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with (
        _index_over(two_roots, two_roots.first, two_roots.second) as source,
        pytest.raises(ValueError, match='a key under one root'),
    ):
        source.record(FIRST_STUB)


def test_the_refusal_of_a_bare_stub_names_every_root_held(two_roots: TwoRoots) -> None:
    """So the caller can see which of them to ask.

    Naming one of the two would leave the reader unable to see the ambiguity
    that is the whole of what the refusal is about.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with (
        _index_over(two_roots, two_roots.first, two_roots.second) as source,
        pytest.raises(ValueError) as excinfo,
    ):
        source.record(FIRST_STUB)
    assert two_roots.first_url in str(excinfo.value)
    assert two_roots.second_url in str(excinfo.value)


def test_an_image_with_no_row_is_reported_as_one_nothing_recorded(two_roots: TwoRoots) -> None:
    """The exception a caller reads as "this image was never navigated".

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with (
        _index_over(two_roots, two_roots.first) as source,
        pytest.raises(FileNotFoundError, match='VOL1/N0000000000'),
    ):
        source.record('VOL1/N0000000000')


def test_an_image_only_the_other_root_holds_has_no_row_here(two_roots: TwoRoots) -> None:
    """Absence is read per root, or one root's images answer for another's.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with (
        _index_over(two_roots, two_roots.first) as source,
        pytest.raises(FileNotFoundError, match='VOL2/N9999999999_1_CALIB'),
    ):
        source.record('VOL2/N9999999999_1_CALIB')


def test_a_document_the_ingest_refused_fails_the_image(two_roots: TwoRoots) -> None:
    """Not the same exception as an absent row: this image was navigated.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with (
        _index_over(two_roots, two_roots.first) as source,
        pytest.raises(ValueError, match='could not read'),
    ):
        source.record(FIRST_REFUSAL_STUB)


def test_another_roots_refusal_is_not_this_roots(two_roots: TwoRoots) -> None:
    """The refusal table is keyed by root as well, and is read that way.

    The second root's refused document sits at a stub the first root holds no
    file of at all, so a refusal lookup blind to the root would fail an image
    this root has nothing to say about.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with (
        _index_over(two_roots, two_roots.first) as source,
        pytest.raises(FileNotFoundError, match=SECOND_REFUSAL_STUB),
    ):
        source.record(SECOND_REFUSAL_STUB)


def test_the_record_names_the_document_the_ingest_read(two_roots: TwoRoots) -> None:
    """A message about a record names a file an operator can open.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = source.record(FIRST_STUB)
    assert found.path.as_posix() == f'{two_roots.first_url}/{FIRST_STUB}{METADATA_SUFFIX}'


def test_the_record_carries_only_the_columns_its_consumer_named(two_roots: TwoRoots) -> None:
    """Which is what makes a row cheaper than a document, and is a real difference.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        rebuilt = source.record(FIRST_STUB).metadata
    assert 'timing' not in rebuilt


# ---------------------------------------------------------------------------
# A stream of records
# ---------------------------------------------------------------------------


def test_a_stream_yields_every_image_of_the_selected_root(two_roots: TwoRoots) -> None:
    """And nothing of the root beside it, which holds a stub this one does not.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = sorted(_stubs(source.records(Selection())))
    assert found == sorted([FIRST_STUB, SECOND_STUB, OTHER_MISSION_STUB, FIRST_REFUSAL_STUB])


def test_a_stream_reads_the_selected_roots_values(two_roots: TwoRoots) -> None:
    """The rows are the root's own, not the other root's rows of the same stubs.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.second) as source:
        found = _offsets(source.records(Selection(instrument=MISSION)))
    assert found == [SECOND_OFFSET, SECOND_OFFSET, SECOND_OFFSET]


def test_a_selection_narrows_a_source_holding_both_roots_to_one(two_roots: TwoRoots) -> None:
    """A source may hold several roots, and a selection may name some of them.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first, two_roots.second) as source:
        found = _offsets(
            source.records(Selection(roots=(two_roots.second_url,), instrument=MISSION))
        )
    assert found == [SECOND_OFFSET, SECOND_OFFSET, SECOND_OFFSET]


def test_a_selection_naming_a_root_the_source_does_not_hold_is_refused(
    two_roots: TwoRoots,
) -> None:
    """Rather than answered with the roots it does hold.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with (
        _index_over(two_roots, two_roots.first) as source,
        pytest.raises(ValueError, match='does not hold'),
    ):
        list(source.records(Selection(roots=(two_roots.second_url,))))


def test_a_stream_keeps_one_mission(two_roots: TwoRoots) -> None:
    """The mission is matched against the column the document's field filled.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = [
            entry.stub
            for entry in source.records(Selection(instrument=MISSION))
            if isinstance(entry, NavRecord)
        ]
    assert sorted(found) == sorted([FIRST_STUB, SECOND_STUB])


def test_a_stream_keeps_one_subtree(two_roots: TwoRoots) -> None:
    """Which is the restriction a walk applies by descending one directory.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = [
            entry.stub
            for entry in source.records(Selection(subtrees=('VOL2',)))
            if isinstance(entry, NavRecord)
        ]
    assert found == [SECOND_STUB]


def test_a_subtree_selection_still_reads_only_the_selected_root(two_roots: TwoRoots) -> None:
    """The other root holds a second image in that subtree, and it is not this one's.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = [
            entry.stub
            for entry in source.records(Selection(subtrees=('VOL2',)))
            if isinstance(entry, NavRecord)
        ]
    assert 'VOL2/N9999999999_1_CALIB' not in found


def test_a_stream_keeps_a_span_of_time(two_roots: TwoRoots) -> None:
    """Bounded on the recorded exposure midtime, inclusive at both ends.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = [
            entry.stub
            for entry in source.records(
                Selection(instrument=MISSION, start_et=FIRST_MIDTIME, stop_et=FIRST_MIDTIME)
            )
            if isinstance(entry, NavRecord)
        ]
    assert found == [FIRST_STUB]


def test_a_time_bound_drops_a_row_recording_no_midtime(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """It cannot be shown to satisfy a bound, which is the walk's answer too.

    Parameters:
        tmp_path: Directory the root and the index are written under.
        quiet_logger: Logger the ingest reports through.
    """
    root = tmp_path / 'results'
    write_metadata(root, FIRST_STUB, metadata_document(instrument=MISSION, times=None))
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    with open_record_source([root], results_index_db_url=url, columns=COLUMNS) as source:
        found = list(source.records(Selection(start_et=0.0)))
    assert found == []


def test_a_stream_reports_a_document_the_ingest_refused(two_roots: TwoRoots) -> None:
    """A file that exists and holds no record is not an image nobody navigated.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        refused = [
            entry for entry in source.records(Selection()) if isinstance(entry, UnreadableFile)
        ]
    assert [entry.stub for entry in refused] == [FIRST_REFUSAL_STUB]


def test_a_reported_refusal_carries_the_reason_the_ingest_recorded(two_roots: TwoRoots) -> None:
    """What could not be read is what an operator has to go and fix.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        refused = [
            entry for entry in source.records(Selection()) if isinstance(entry, UnreadableFile)
        ]
    assert 'navigation document' in refused[0].reason


def test_a_reported_refusal_names_the_file_it_is_about(two_roots: TwoRoots) -> None:
    """Under this root, not under the other one, which refused a file of its own.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        refused = [
            entry for entry in source.records(Selection()) if isinstance(entry, UnreadableFile)
        ]
    assert refused[0].path.as_posix().startswith(two_roots.first_url)


def test_a_mission_filtered_stream_still_reports_a_refusal(two_roots: TwoRoots) -> None:
    """A refused file names no mission, so no mission's run may pass over it.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        refused = [
            entry.stub
            for entry in source.records(Selection(instrument=MISSION))
            if isinstance(entry, UnreadableFile)
        ]
    assert refused == [FIRST_REFUSAL_STUB]


def test_a_subtree_restriction_narrows_the_refusals_too(two_roots: TwoRoots) -> None:
    """The refusal table carries the subtree for exactly this.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        refused = [
            entry.stub
            for entry in source.records(Selection(subtrees=('VOL2',)))
            if isinstance(entry, UnreadableFile)
        ]
    assert refused == []


# ---------------------------------------------------------------------------
# A stream naming its own stubs
# ---------------------------------------------------------------------------


def test_a_selection_naming_stubs_reads_exactly_those(two_roots: TwoRoots) -> None:
    """Which is what a queue task carries.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.records(Selection(stubs=(SECOND_STUB, FIRST_STUB))))
    assert found == [SECOND_STUB, FIRST_STUB]


def test_named_stubs_are_read_in_the_order_they_were_named(two_roots: TwoRoots) -> None:
    """Naming an image is not a narrowing, so the order is the caller's.

    Asked for in an order the server has no reason to produce, and in enough
    numbers to cross the batch the keys are asked for in, so a source that
    handed back whatever order the rows arrived in would be caught.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = tuple([SECOND_STUB, FIRST_STUB] * 40)
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.records(Selection(stubs=named)))
    assert found == list(named)


def test_a_named_stub_the_ingest_refused_is_reported_as_unreadable(two_roots: TwoRoots) -> None:
    """A worker handed a refused file must not report it as a clean read.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = list(source.records(Selection(stubs=(FIRST_REFUSAL_STUB,))))
    assert [type(entry) for entry in found] == [UnreadableFile]


def test_a_named_stub_reads_the_selected_roots_row(two_roots: TwoRoots) -> None:
    """The other root holds the same stub, recording something else.

    Read against the *first* root, whose rows were written first.  The batch
    read builds what it found with a dictionary update, so a query that dropped
    the root half of the key would be answered by whichever row came back last
    -- and asking for the root that was ingested last would be answered
    correctly by the defect.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = _offsets(source.records(Selection(stubs=(FIRST_STUB,))))
    assert found == [FIRST_OFFSET]


def test_a_named_stub_the_index_never_heard_of_yields_nothing(two_roots: TwoRoots) -> None:
    """Under a completed ingest that is an image nothing navigated.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = list(source.records(Selection(stubs=('VOL1/N0000000000',))))
    assert found == []


def test_a_named_stub_only_the_other_root_holds_yields_nothing(two_roots: TwoRoots) -> None:
    """Naming a key does not stop it being a key under one root.

    The stub is one only the other root holds, so a query that bound the keys
    and dropped the root would hand this run an image nobody asked for -- and
    do it whichever order the server returned its rows in, which naming a stub
    both roots hold cannot show.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = list(source.records(Selection(stubs=('VOL2/N9999999999_1_CALIB',))))
    assert found == []


def test_a_named_stub_only_the_other_root_refused_yields_nothing(two_roots: TwoRoots) -> None:
    """The refusal half of a named-stub read carries its own root term.

    The other root refused a file at a stub this root holds nothing at, so a
    refusal query blind to the root would report this run a file it does not
    have.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = list(source.records(Selection(stubs=(SECOND_REFUSAL_STUB,))))
    assert found == []


def test_named_stubs_still_honour_the_mission(two_roots: TwoRoots) -> None:
    """A selection is a narrowing whatever else it names.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(
            source.records(Selection(stubs=(FIRST_STUB, OTHER_MISSION_STUB), instrument=MISSION))
        )
    assert found == [FIRST_STUB]


def test_named_stubs_need_one_root_to_be_keys_under(two_roots: TwoRoots) -> None:
    """Two roots would hand a caller two records per image it named.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with (
        _index_over(two_roots, two_roots.first, two_roots.second) as source,
        pytest.raises(ValueError, match='selection of keys'),
    ):
        list(source.records(Selection(stubs=(FIRST_STUB,))))


# ---------------------------------------------------------------------------
# A listing
# ---------------------------------------------------------------------------


def test_a_listing_reports_every_file_the_index_records(two_roots: TwoRoots) -> None:
    """Both tables record a file, and a listing answers what is there.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = sorted(_stubs(source.listing(Selection())))
    assert found == sorted([FIRST_STUB, SECOND_STUB, OTHER_MISSION_STUB, FIRST_REFUSAL_STUB])


def test_a_listing_reads_only_the_selected_root(two_roots: TwoRoots) -> None:
    """The other root holds a stub this one does not, and its own refusal.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.listing(Selection()))
    assert 'VOL2/N9999999999_1_CALIB' not in found


def test_a_listing_reads_only_the_selected_roots_refusals(two_roots: TwoRoots) -> None:
    """The refusal arm of the listing carries its own root term.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.listing(Selection()))
    assert SECOND_REFUSAL_STUB not in found


def test_a_listing_narrows_to_a_subtree(two_roots: TwoRoots) -> None:
    """The restriction a walk applies by descending one directory.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.listing(Selection(subtrees=('VOL1',))))
    assert sorted(found) == sorted([FIRST_STUB, OTHER_MISSION_STUB, FIRST_REFUSAL_STUB])


def test_a_listing_of_named_stubs_covers_only_those_named(two_roots: TwoRoots) -> None:
    """The question a caller enumerating candidates asks: which of these is recorded.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.listing(Selection(stubs=(FIRST_STUB,))))
    assert found == [FIRST_STUB]


def test_a_listing_of_named_stubs_answers_in_the_order_it_was_asked(two_roots: TwoRoots) -> None:
    """Naming files is asking about them in that order, whichever storage answers.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.listing(Selection(stubs=(SECOND_STUB, FIRST_STUB))))
    assert found == [SECOND_STUB, FIRST_STUB]


def test_a_named_stub_the_index_records_nothing_for_is_absent(two_roots: TwoRoots) -> None:
    """Under a root with a completed ingest that is an image nothing navigated.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.listing(Selection(stubs=(FIRST_STUB, 'VOL1/never_navigated'))))
    assert found == [FIRST_STUB]


def test_a_listing_of_named_stubs_reports_a_file_the_ingest_refused(two_roots: TwoRoots) -> None:
    """A document the ingest refused is a file that is there, and a listing says what is.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.listing(Selection(stubs=(FIRST_REFUSAL_STUB,))))
    assert found == [FIRST_REFUSAL_STUB]


def test_a_listing_of_named_stubs_reads_only_the_selected_root(two_roots: TwoRoots) -> None:
    """One stub is a key under every root, so a query without its root answers wrongly.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.listing(Selection(stubs=('VOL2/N9999999999_1_CALIB',))))
    assert found == []


def test_a_listing_of_named_stubs_reads_only_the_selected_roots_refusals(
    two_roots: TwoRoots,
) -> None:
    """The refusal arm carries the root term as surely as the image arm does.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.listing(Selection(stubs=(SECOND_REFUSAL_STUB,))))
    assert found == []


def test_the_second_root_answers_a_listing_of_named_stubs_for_itself(two_roots: TwoRoots) -> None:
    """The other side of the same key, so neither answer would survive dropping it.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.second) as source:
        found = _stubs(source.listing(Selection(stubs=('VOL2/N9999999999_1_CALIB',))))
    assert found == ['VOL2/N9999999999_1_CALIB']


def test_the_two_storages_answer_a_listing_of_named_stubs_alike(two_roots: TwoRoots) -> None:
    """A caller cannot see which half answered, so the two must cover the same files.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    named = Selection(stubs=(FIRST_STUB, FIRST_REFUSAL_STUB, 'VOL1/never_navigated'))
    with _index_over(two_roots, two_roots.first) as source:
        from_index = _stubs(source.listing(named))
    assert from_index == _stubs(_tree_over(two_roots).listing(named))


def test_a_listing_carries_the_metrics_the_skip_rule_needs(two_roots: TwoRoots) -> None:
    """Without both of them a later pass cannot tell a changed file from one that is not.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = {entry.stub: entry for entry in source.listing(Selection())}
    assert found[FIRST_STUB].has_metrics


def test_a_listing_carries_the_metrics_of_a_refused_file_too(two_roots: TwoRoots) -> None:
    """A refused file that has not changed is not read again either.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        found = {entry.stub: entry for entry in source.listing(Selection())}
    assert found[FIRST_REFUSAL_STUB].has_metrics


def test_a_listing_names_where_each_file_is(two_roots: TwoRoots) -> None:
    """Under the root it belongs to, which a two-root listing has to say.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first, two_roots.second) as source:
        found = {
            (entry.stub, entry.path.as_posix())
            for entry in source.listing(Selection(subtrees=('VOL1',)))
        }
    assert (FIRST_STUB, f'{two_roots.second_url}/{FIRST_STUB}{METADATA_SUFFIX}') in found


def _forget_where_the_documents_were_read(url: str) -> None:
    """Clear the recorded source file of every image row.

    A row that records no path is what a reader falls back from, and the
    fall-back rebuilds the path out of the row's own key -- so it is the one
    shape in which a stream can name a document under a root that is not the
    row's.

    Parameters:
        url: The index to rewrite.
    """
    engine = open_index(url)
    try:
        with engine.begin() as connection:
            connection.execute(sqlalchemy.update(IMAGES).values(source_file=None))
    finally:
        engine.dispose()


def test_a_row_recording_no_path_is_named_under_its_own_root(two_roots: TwoRoots) -> None:
    """The fall-back rebuilds a path from the key, and a key carries its root.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    _forget_where_the_documents_were_read(two_roots.url)
    with _index_over(two_roots, two_roots.first, two_roots.second) as source:
        found = {
            entry.path.as_posix()
            for entry in source.records(Selection(subtrees=('VOL1',)))
            if isinstance(entry, NavRecord)
        }
    assert f'{two_roots.second_url}/{FIRST_STUB}{METADATA_SUFFIX}' in found


def test_a_listing_names_each_roots_refusal_under_its_own_root(two_roots: TwoRoots) -> None:
    """A refused file records no path, so its path is rebuilt from its own key.

    Rebuilt from the root the row carries rather than from the source's first
    root, which a listing over two roots is what shows: the other root refused a
    file at a stub this one holds nothing at, and naming it under the wrong root
    would send an operator to a file that is not there.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first, two_roots.second) as source:
        found = {entry.stub: entry.path.as_posix() for entry in source.listing(Selection())}
    assert (
        found[SECOND_REFUSAL_STUB]
        == f'{two_roots.second_url}/{SECOND_REFUSAL_STUB}{METADATA_SUFFIX}'
    )


def test_a_stream_names_each_roots_refusal_under_its_own_root(two_roots: TwoRoots) -> None:
    """The same rebuild, on the call that reports a refusal to a run.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first, two_roots.second) as source:
        found = {
            entry.stub: entry.path.as_posix()
            for entry in source.records(Selection())
            if isinstance(entry, UnreadableFile)
        }
    assert (
        found[SECOND_REFUSAL_STUB]
        == f'{two_roots.second_url}/{SECOND_REFUSAL_STUB}{METADATA_SUFFIX}'
    )


@pytest.mark.parametrize(
    ('selection', 'named'),
    [
        pytest.param(Selection(instrument=MISSION), 'instrument', id='instrument'),
        pytest.param(Selection(start_et=0.0), 'start_et', id='start_et'),
        pytest.param(Selection(stop_et=0.0), 'stop_et', id='stop_et'),
    ],
)
def test_a_listing_refuses_what_it_cannot_answer(
    two_roots: TwoRoots, selection: Selection, named: str
) -> None:
    """The index could answer these from its columns and deliberately does not.

    A call meaning one thing over one storage and another over the next is not a
    seam, so the restriction is refused here exactly as the walk refuses it.

    Parameters:
        two_roots: The two ingested roots and their index.
        selection: The selection carrying one restriction a listing cannot honor.
        named: The restriction the refusal has to name.
    """
    with _index_over(two_roots, two_roots.first) as source, pytest.raises(ValueError, match=named):
        list(source.listing(selection))


# ---------------------------------------------------------------------------
# The two storages agree
# ---------------------------------------------------------------------------


def _tree_over(held: TwoRoots) -> TreeRecordSource:
    """Build a document-backed source over the first of the two roots.

    Parameters:
        held: The two roots and their index.

    Returns:
        The source.
    """
    return TreeRecordSource([held.first])


def test_the_two_storages_list_the_same_files(two_roots: TwoRoots) -> None:
    """Down to the refused ones, which are files the tree still holds.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        from_index = sorted(_stubs(source.listing(Selection())))
    from_tree = sorted(_stubs(_tree_over(two_roots).listing(Selection())))
    assert from_index == from_tree


def _listed(found: Iterator[ListedRecord]) -> dict[str, tuple[str, int | None, int | None]]:
    """Return what a listing said about each file, keyed by stub.

    Parameters:
        found: What the listing yielded.

    Returns:
        The path and the two metrics of each entry.
    """
    return {
        entry.stub: (entry.path.as_posix(), entry.mtime_ns, entry.size_bytes) for entry in found
    }


def test_the_two_storages_report_the_same_metrics(two_roots: TwoRoots) -> None:
    """The skip rule compares these, so a difference re-reads a whole root.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        from_index = _listed(source.listing(Selection()))
    from_tree = _listed(_tree_over(two_roots).listing(Selection()))
    assert from_index == from_tree


def test_the_two_storages_stream_the_same_images(two_roots: TwoRoots) -> None:
    """One mission out of one root, from either storage.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        from_index = sorted(_stubs(source.records(Selection(instrument=MISSION))))
    from_tree = sorted(_stubs(_tree_over(two_roots).records(Selection(instrument=MISSION))))
    assert from_index == from_tree


def test_the_two_storages_stream_the_same_offsets(two_roots: TwoRoots) -> None:
    """The value a product is built from, rather than a name counted beside it.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        from_index = sorted(_offsets(source.records(Selection(instrument=MISSION))))
    from_tree = sorted(_offsets(_tree_over(two_roots).records(Selection(instrument=MISSION))))
    assert from_index == from_tree


def test_the_two_storages_name_the_same_document_for_a_record(two_roots: TwoRoots) -> None:
    """A message about a record has to name one file, however it was read.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        from_index = source.record(FIRST_STUB).path.as_posix()
    from_tree = _tree_over(two_roots).record(FIRST_STUB).path.as_posix()
    assert from_index == from_tree


def test_the_two_storages_agree_on_a_span_of_time(two_roots: TwoRoots) -> None:
    """The epoch a bound is applied to is the one the document recorded.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    selection = Selection(instrument=MISSION, start_et=SECOND_MIDTIME)
    with _index_over(two_roots, two_roots.first) as source:
        from_index = sorted(_stubs(source.records(selection)))
    from_tree = sorted(_stubs(_tree_over(two_roots).records(selection)))
    assert from_index == from_tree


def test_a_row_that_cannot_be_placed_in_time_is_not_a_shortfall_here(
    two_roots: TwoRoots,
) -> None:
    """A declared difference, and the reason for it is what each storage holds.

    The walk has only the document, so one recording no usable midtime is a file
    it can say nothing about and it reports it.  The index has a row: a row
    whose midtime is absent satisfies no bound, exactly as a row outside the
    span does not, and it is not selected.  A run that has to account for those
    images reads the documents.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    engine = open_index(two_roots.url)
    try:
        with engine.begin() as connection:
            connection.execute(
                IMAGES.update()
                .where(
                    IMAGES.c.root_url == two_roots.first_url,
                    IMAGES.c.results_path_stub == FIRST_STUB,
                )
                .values(midtime_et=None)
            )
    finally:
        engine.dispose()
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.records(Selection(start_et=0.0, stop_et=1000.0)))
    assert FIRST_STUB not in found


def test_the_two_storages_refuse_a_bare_stub_alike(two_roots: TwoRoots) -> None:
    """One rule about what a stub is a key under, applied by both.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    tree = TreeRecordSource([two_roots.first, two_roots.second])
    with (
        _index_over(two_roots, two_roots.first, two_roots.second) as source,
        pytest.raises(ValueError) as from_index,
    ):
        source.record(FIRST_STUB)
    with pytest.raises(ValueError) as from_tree:
        tree.record(FIRST_STUB)
    assert str(from_index.value) == str(from_tree.value)


def test_the_two_storages_refuse_a_listing_restriction_alike(two_roots: TwoRoots) -> None:
    """Refusing differently would make the call mean two things.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    selection = Selection(instrument=MISSION, stop_et=1.0)
    with _index_over(two_roots, two_roots.first) as source, pytest.raises(ValueError) as from_index:
        list(source.listing(selection))
    with pytest.raises(ValueError) as from_tree:
        list(_tree_over(two_roots).listing(selection))
    assert str(from_index.value) == str(from_tree.value)


# ---------------------------------------------------------------------------
# What the source holds open, and what it says about itself
# ---------------------------------------------------------------------------


def test_no_statement_orders_on_a_column(two_roots: TwoRoots) -> None:
    """A server sorts text under its own collation, so no query here may sort.

    Read off the statements the source actually issued rather than off the
    source code, so a sort reintroduced through any of the three calls is
    caught.  A caller wanting a total order sorts the stream it received, which
    is the one key both storages share.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    issued: list[str] = []
    engine = open_index(two_roots.url)
    sqlalchemy.event.listen(
        engine,
        'before_cursor_execute',
        lambda conn, cursor, statement, *rest: issued.append(statement),
    )
    with IndexRecordSource(engine, [two_roots.first], two_roots.url, COLUMNS) as source:
        list(source.listing(Selection()))
        list(source.records(Selection(instrument=MISSION)))
        list(source.records(Selection(stubs=(FIRST_STUB,))))
        source.record(FIRST_STUB)
    assert [one for one in issued if 'ORDER BY' in one.upper()] == []


def _pool_events_of_a_whole_stream(url: str, root: Path) -> list[str]:
    """Read a whole stream, recording what it did to the connection pool.

    Recorded through the engine's own events rather than off the pool's
    counters, which are a particular pool implementation's and not every
    backend's.

    Parameters:
        url: The index to read.
        root: The results root to read under.

    Returns:
        One entry per connection taken out of the pool and one per connection
        given back, in the order they happened.
    """
    events: list[str] = []
    engine = open_index(url)
    sqlalchemy.event.listen(engine, 'checkout', lambda *_args: events.append('out'))
    sqlalchemy.event.listen(engine, 'checkin', lambda *_args: events.append('in'))
    with IndexRecordSource(engine, [root], url, COLUMNS) as source:
        list(source.records(Selection()))
    return events


def test_a_stream_takes_a_connection_out_of_the_pool(two_roots: TwoRoots) -> None:
    """Without which the balance the next test asserts would hold over nothing.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    events = _pool_events_of_a_whole_stream(two_roots.url, two_roots.first)
    assert events.count('out') > 0


def test_an_exhausted_stream_gives_its_connection_back(two_roots: TwoRoots) -> None:
    """A generator holding a cursor has to release it, or a run leaks one per query.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    events = _pool_events_of_a_whole_stream(two_roots.url, two_roots.first)
    assert events.count('in') == events.count('out')


def test_the_source_says_which_index_it_read(two_roots: TwoRoots) -> None:
    """The run log has to say where the records came from, not just how many.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        described = source.describe()
    assert 'results index' in described


def test_the_source_says_which_roots_it_answers_for(two_roots: TwoRoots) -> None:
    """One index serves several, so naming the index alone says too little.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        described = source.describe()
    assert two_roots.first_url in described


def test_the_description_hides_a_password_in_the_index_url(two_roots: TwoRoots) -> None:
    """A run log reaches a console, a file, and whoever is sent one.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    source = IndexRecordSource(
        open_index(two_roots.url),
        [two_roots.first],
        'postgresql+psycopg://svc:sup3rs3cr3t@db.example/spindoctor',
        COLUMNS,
    )
    try:
        described = source.describe()
    finally:
        source.close()
    assert 'sup3rs3cr3t' not in described


def test_a_source_over_no_root_is_refused(two_roots: TwoRoots) -> None:
    """It could answer nothing, and absence under it would mean nothing.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with pytest.raises(ValueError, match='at least one root'):
        IndexRecordSource(open_index(two_roots.url), [], two_roots.url, COLUMNS)


# ---------------------------------------------------------------------------
# Opening a source
# ---------------------------------------------------------------------------


def test_no_index_url_reads_the_documents(tmp_path: Path) -> None:
    """Reading the documents is every program's default, and opens no database.

    Parameters:
        tmp_path: Directory the root is written under.
    """
    root = tmp_path / 'results'
    write_metadata(root, FIRST_STUB, metadata_document())
    with open_record_source([root]) as source:
        assert isinstance(source, TreeRecordSource)


def test_an_index_url_reads_the_index(two_roots: TwoRoots) -> None:
    """The other half of the choice, so the default above is a choice.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with open_record_source(
        [two_roots.first], results_index_db_url=two_roots.url, columns=COLUMNS
    ) as source:
        assert isinstance(source, IndexRecordSource)


def test_a_source_opened_over_no_root_at_all_is_refused(tmp_path: Path) -> None:
    """An empty spelling and an empty list are both a root nobody named.

    Parameters:
        tmp_path: Directory the test would otherwise write under.
    """
    with pytest.raises(ValueError, match='at least one results root'):
        open_record_source([])


def test_every_root_is_checked_before_anything_is_read(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A root nobody ingested cannot answer, and it is the second one here.

    A check that stopped at the first root would open a source whose absences
    mean nothing under half of what it holds.

    Parameters:
        tmp_path: Directory the roots and the index are written under.
        quiet_logger: Logger the ingest reports through.
    """
    first = _root(tmp_path, 'results-first', offset=FIRST_OFFSET, refused=FIRST_REFUSAL_STUB)
    second = _root(tmp_path, 'results-second', offset=SECOND_OFFSET, refused=SECOND_REFUSAL_STUB)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first], logger=quiet_logger)
    with pytest.raises(ValueError, match='no completed ingest'):
        open_record_source([first, second], results_index_db_url=url, columns=COLUMNS)


def test_opening_for_a_root_nobody_ingested_is_refused(tmp_path: Path) -> None:
    """Absence of a row means nothing under a root no pass has walked.

    Parameters:
        tmp_path: Directory the root and the index are written under.
    """
    root = tmp_path / 'results'
    write_metadata(root, FIRST_STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    open_index(url, create=True).dispose()
    with pytest.raises(ValueError, match='no completed ingest'):
        open_record_source([root], results_index_db_url=url, columns=COLUMNS)


class _WatchedEngine:
    """An engine that counts how many times it was disposed of.

    Everything else is the engine's own: a caller of this stands in for the
    ceremony's caller, which connects and queries as usual.

    Parameters:
        engine: The engine to stand in front of.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self.disposals = 0

    def __getattr__(self, name: str) -> Any:
        """Return whatever the engine has under that name.

        Parameters:
            name: The attribute wanted.

        Returns:
            The engine's own attribute, so this stands in for one.
        """
        return getattr(self._engine, name)

    def dispose(self) -> None:
        """Dispose of the engine, and count it."""
        self.disposals += 1
        self._engine.dispose()


def test_a_refused_open_disposes_the_engine_it_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal happens after the open, so the ceremony has to clean up.

    A caller that never received an engine cannot dispose of one, and a leaked
    pool per refused run is a connection nothing closes.  The disposal is watched
    directly rather than inferred from the database file: on this platform a file
    can be unlinked while it is still open, so a test that unlinked it would pass
    whether or not anything was disposed of.

    Parameters:
        tmp_path: Directory the root and the index are written under.
        monkeypatch: Fixture the opener is replaced through.
    """
    root = tmp_path / 'results'
    write_metadata(root, FIRST_STUB, metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    watched = _WatchedEngine(open_index(url, create=True))
    monkeypatch.setattr(roots, 'open_index', lambda *_args, **_kwargs: watched)
    with pytest.raises(ValueError, match='no completed ingest'):
        open_index_for_roots(url, [normalize_root_url(root)])
    assert watched.disposals == 1


def test_the_documents_are_read_through_the_callers_own_logger(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one line a source has to say reaches the log of the run it belongs to.

    A source constructs no logger, so a run that lent it none is told nothing --
    which is what a directory the walk declined to descend a second time would
    otherwise be.  ``pdslogger`` writes through its own stream handler, so what
    it wrote is read off the captured stream.

    Parameters:
        tmp_path: Directory the root is written under.
        capsys: Fixture the logger's own stream is captured through.
    """
    root = tmp_path / 'results'
    write_metadata(root, FIRST_STUB, metadata_document())
    (root / 'VOL2').mkdir(parents=True, exist_ok=True)
    (root / 'VOL2' / 'again').symlink_to(root / 'VOL1', target_is_directory=True)
    logger = pdslogger.PdsLogger('record_source_test', lognames=False)
    with open_record_source([root], logger=logger) as source:
        list(source.listing(Selection()))
    assert 'reached a second way' in capsys.readouterr().out


def test_an_index_backed_run_reads_the_columns_it_declared(two_roots: TwoRoots) -> None:
    """A column nobody selected is a field the rebuilt record does not have.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with open_record_source(
        [two_roots.first], results_index_db_url=two_roots.url, columns=(IMAGES.c.status,)
    ) as source:
        rebuilt = source.record(FIRST_STUB).metadata
    assert 'offset' not in rebuilt


def test_a_stream_over_a_table_that_cannot_be_read_is_reported(two_roots: TwoRoots) -> None:
    """A stream's query runs when the caller reads, so the translation is inside it.

    Every other read is translated by the call that issues it; a stream issues
    its query lazily, so a refusal that arrived as the database layer's own
    exception class would reach the caller out of ``next()`` -- past the
    translation, in a vocabulary ``nav_records`` deliberately never names.  What
    kind of failure it is does not matter, only that it happens while the caller
    is reading, so a dropped table stands in for a lost connection.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    with _index_over(two_roots, two_roots.first) as source:
        stream = source.records(Selection())
        engine = open_index(two_roots.url)
        try:
            with engine.begin() as connection:
                connection.execute(sqlalchemy.text(f'DROP TABLE {FAILED_FILES.name}'))
        finally:
            engine.dispose()
        with pytest.raises(ValueError, match='could not be read'):
            list(stream)


def test_a_stub_that_escapes_the_root_is_refused_by_the_document_source(tmp_path: Path) -> None:
    """A stub is a key, and a key holding ``..`` names a file the root does not hold.

    The planted document would load, so a source that joined the key and read
    what came out would hand this run a record from outside the root it was
    opened over.

    Parameters:
        tmp_path: Directory the root is written under.
    """
    root = tmp_path / 'results'
    write_metadata(root, FIRST_STUB, metadata_document())
    (tmp_path / f'elsewhere{METADATA_SUFFIX}').write_text(json.dumps({'status': 'success'}))
    with (
        open_record_source([root]) as source,
        pytest.raises(FileNotFoundError, match='names a parent directory'),
    ):
        source.record('../elsewhere')


# ---------------------------------------------------------------------------
# A stub both tables record
# ---------------------------------------------------------------------------


def _refuse_an_already_recorded_file(held: TwoRoots, root: Path, stub: str) -> None:
    """Leave a refusal beside a record for one stub, as two passes can.

    The ingest writes the two tables independently, and a pass divided into
    shares can have one worker record a file another worker had refused.  The
    seam states that such a stub is read as the record it is, so the state has
    to be built to test that, since one ingest pass does not produce it.

    Parameters:
        held: The two roots and their index.
        root: The root to leave the refusal under.
        stub: The stub to leave it at, which already carries an image row.
    """
    engine = open_index(held.url)
    try:
        with engine.begin() as connection:
            connection.execute(
                FAILED_FILES.insert(),
                [
                    {
                        'root_url': normalize_root_url(root),
                        'results_path_stub': stub,
                        'reason': 'not a current-schema navigation document',
                        'subtree': stub.split('/')[0],
                        'mtime_ns': 1_700_000_000_000_000_000,
                        'size_bytes': 64,
                    }
                ],
            )
    finally:
        engine.dispose()


def test_a_stub_both_tables_record_is_yielded_once_by_a_whole_root_stream(
    two_roots: TwoRoots,
) -> None:
    """One image counted as navigated and as a shortfall in one pass is two answers.

    The stream runs its two statements independently, so the refusal arm is the
    one that has to know: without that, ``sd_create_ck`` writes a segment for
    this image and reports it as a file it could not read, in the same run.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    _refuse_an_already_recorded_file(two_roots, two_roots.first, FIRST_STUB)
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.records(Selection()))
    assert found.count(FIRST_STUB) == 1


def test_a_stub_both_tables_record_is_read_as_the_record_by_a_whole_root_stream(
    two_roots: TwoRoots,
) -> None:
    """A count of one would pass if the refusal were the one that survived.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    _refuse_an_already_recorded_file(two_roots, two_roots.first, FIRST_STUB)
    with _index_over(two_roots, two_roots.first) as source:
        found = [entry for entry in source.records(Selection()) if entry.stub == FIRST_STUB]
    assert [type(entry) for entry in found] == [NavRecord]


def test_a_stub_both_tables_record_is_yielded_once_by_a_stream_of_facts(
    two_roots: TwoRoots,
) -> None:
    """One image counted as navigated and as a shortfall in one pass is two answers.

    The facts run their image statement and their refusal statement
    independently, exactly as a stream of records does, so the refusal arm is
    again the one that has to know about the other.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    _refuse_an_already_recorded_file(two_roots, two_roots.first, FIRST_STUB)
    with _index_over(two_roots, two_roots.first) as source:
        found = [_facts_stub(one) for one in source.facts(Selection())]
    assert found.count(FIRST_STUB) == 1


def test_a_stub_both_tables_record_is_read_as_the_image_by_a_stream_of_facts(
    two_roots: TwoRoots,
) -> None:
    """A count of one would pass if the refusal were the one that survived.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    _refuse_an_already_recorded_file(two_roots, two_roots.first, FIRST_STUB)
    with _index_over(two_roots, two_roots.first) as source:
        found = [one for one in source.facts(Selection()) if _facts_stub(one) == FIRST_STUB]
    assert [type(one) for one in found] == [ImageFacts]


def test_a_stale_refusal_of_another_root_is_still_reported(two_roots: TwoRoots) -> None:
    """The record that cancels a refusal is the one under the refusal's own root.

    The first root holds a record at this stub; the second root's refusal at the
    same stub is nothing to do with it, and a cancellation blind to the root
    would drop a real shortfall out of the second root's run.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    _refuse_an_already_recorded_file(two_roots, two_roots.second, SECOND_STUB)
    engine = open_index(two_roots.url)
    try:
        with engine.begin() as connection:
            connection.execute(
                IMAGES.delete().where(
                    IMAGES.c.root_url == two_roots.second_url,
                    IMAGES.c.results_path_stub == SECOND_STUB,
                )
            )
    finally:
        engine.dispose()
    with _index_over(two_roots, two_roots.second) as source:
        found = [entry.stub for entry in source.records(Selection())]
    assert found.count(SECOND_STUB) == 1


def test_a_stub_both_tables_record_is_listed_once(two_roots: TwoRoots) -> None:
    """A listing answers what files are there, and both rows are about one file.

    A walk of the same tree finds that file once, so a listing that reported it
    twice would make the two storages disagree about how much is there.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    _refuse_an_already_recorded_file(two_roots, two_roots.first, FIRST_STUB)
    with _index_over(two_roots, two_roots.first) as source:
        found = _stubs(source.listing(Selection()))
    assert found.count(FIRST_STUB) == 1


def test_a_listed_stub_both_tables_record_carries_the_records_metrics(
    two_roots: TwoRoots,
) -> None:
    """The entry that survives is the record's, whose metrics are the document's.

    A later pass decides whether to read a file again by comparing these against
    what the listing of the tree reports, so an entry carrying the refusal row's
    metrics would either re-read a file that has not changed or skip one that
    has.

    Parameters:
        two_roots: The two ingested roots and their index.
    """
    document = two_roots.first / f'{FIRST_STUB}{METADATA_SUFFIX}'
    _refuse_an_already_recorded_file(two_roots, two_roots.first, FIRST_STUB)
    with _index_over(two_roots, two_roots.first) as source:
        found = [entry for entry in source.listing(Selection()) if entry.stub == FIRST_STUB]
    assert [entry.size_bytes for entry in found] == [document.stat().st_size]
