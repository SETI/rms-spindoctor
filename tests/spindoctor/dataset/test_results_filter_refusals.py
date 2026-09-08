"""What the filter refuses, and how, when it cannot give a selection at all.

Four states are refusals rather than answers: a combination of selection flags
no image could satisfy, a results index that will not open, one that opens and
will not answer, and one holding no completed pass over the results root.  Each
is a run that is misconfigured rather than a run that went wrong, and each is
refused in one type, so a program reports the sentence that says what to change
instead of tracing back out of an enumeration.

The flags are checked before anything is opened, which is why the contradictions
below are asserted with an index URL that names nothing: a refusal that reported
the database instead would send a user to fix the wrong thing.

Whatever a refusal reaching an operator says, what it leaves behind is a
storage: a refusal raised after the storage was opened comes out of the
constructor, so the caller receives no filter and has nothing to close.  The last
tests here watch the storage rather than the message, because nothing about a
refusal's wording depends on that.
"""

import itertools
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
import sqlalchemy
from tests.spindoctor.conftest import index_url
from tests.spindoctor.dataset.conftest import (
    VOLUMES,
    index_without_a_table,
    null_logger,
    one_image_tree,
)

from spindoctor.dataset import results_filter
from spindoctor.dataset.results_filter import ResultsFilter, SelectionError
from spindoctor.nav_records import ImageFacts, ListedRecord, Selection, UnreadableFile

CONTRADICTORY_PAIRS = [
    pytest.param({'has_offset_file': True, 'has_no_offset_file': True}, id='offset-file-pair'),
    pytest.param(
        {'has_offset_spice_error': True, 'has_offset_nonspice_error': True}, id='error-pair'
    ),
    pytest.param(
        {'has_offset_error': True, 'has_no_offset_file': True}, id='error-and-no-offset-file'
    ),
    pytest.param({'has_offset_error': True, 'has_no_offset_error': True}, id='error-and-no-error'),
    pytest.param(
        {'has_offset_spice_error': True, 'has_no_offset_error': True},
        id='spice-error-and-no-error',
    ),
    pytest.param(
        {'has_offset_nonspice_error': True, 'has_no_offset_error': True},
        id='nonspice-error-and-no-error',
    ),
    pytest.param(
        {'has_no_offset_error': True, 'has_no_offset_file': True},
        id='no-error-and-no-offset-file',
    ),
    pytest.param(
        {'has_offset_spice_error': True, 'has_no_offset_file': True},
        id='spice-error-and-no-offset-file',
    ),
    pytest.param(
        {'has_offset_nonspice_error': True, 'has_no_offset_file': True},
        id='nonspice-error-and-no-offset-file',
    ),
]
"""Every pair of selection flags no image could satisfy.

That it is every one of them is asserted rather than claimed: the six flags make
fifteen pairs, and the test below puts each of the fifteen to the constructor and
holds the ones it refuses to exactly this list.  A list short by a pair leaves
the message that pair produces unasserted, which is how two of the four
contradictions came to have a path through the message builder nothing read.
"""

CONTRADICTION_REFUSAL = r'mutually exclusive|cannot be combined with'
"""The two shapes a refusal of contradictory selection flags takes.

Two flags that exclude each other and nothing else are mutually exclusive.  One
flag that excludes several which are satisfiable together is named against the
ones it excludes instead: ``has_offset_error`` and ``has_offset_spice_error``
are a pair the constructor accepts, so a message calling the three of them
mutually exclusive would assert an exclusion between two flags that have none.
"""

SELECTION_FLAGS = (
    'has_offset_file',
    'has_no_offset_file',
    'has_offset_error',
    'has_no_offset_error',
    'has_offset_spice_error',
    'has_offset_nonspice_error',
)
"""The six results-file selection flags."""

COMBINATIONS = [
    names
    for size in range(2, len(SELECTION_FLAGS) + 1)
    for names in itertools.combinations(SELECTION_FLAGS, size)
]
"""Every combination of two or more of them, which is what a user may type.

A user types flags rather than pairs, and a refusal is about the combination
they typed: three of these carry two contradictions at once, and the
contradiction a run happens to notice first is not the whole of what is wrong
with the selection.
"""


def _contradicted_within(names: Sequence[str]) -> set[str]:
    """Return the flags of one combination that another flag of it contradicts.

    Parameters:
        names: The flags the user typed.

    Returns:
        Every flag belonging to a contradictory pair both of whose flags are in
        the combination.  A flag that contradicts nothing else present is not
        one of them: it is a narrowing the selection could have satisfied, and
        naming it in a refusal would send its user to change the wrong thing.
    """
    given = set(names)
    return {
        name
        for case in CONTRADICTORY_PAIRS
        for pair in [set(cast(dict[str, bool], case.values[0]))]
        if pair <= given
        for name in pair
    }


def _refusal_of(tree: Path, names: Sequence[str]) -> str | None:
    """Build a filter over the given flags and return what it refused, if it did.

    Parameters:
        tree: The results root, read only by the combinations that are not
            refused.
        names: The flags to turn on.

    Returns:
        The refusal's message, or None when the combination was accepted.
    """
    flags: dict[str, Any] = dict.fromkeys(names, True)
    try:
        ResultsFilter(VOLUMES, str(tree), logger=null_logger(), results_index_db_url=None, **flags)
    except SelectionError as exc:
        return str(exc)
    return None


def test_exactly_the_combinations_holding_a_contradictory_pair_are_refused(tree: Path) -> None:
    """The list of contradictions is the whole of what the constructor refuses.

    Asserted over every combination rather than over the pairs alone, so that a
    contradiction reachable only by three flags together, and a combination
    refused for no pair anybody wrote down, are both failures here.
    """
    refused = {names for names in COMBINATIONS if _refusal_of(tree, names) is not None}
    assert refused == {names for names in COMBINATIONS if _contradicted_within(names)}


def test_a_refusal_names_every_flag_of_the_combination_it_cannot_satisfy(tree: Path) -> None:
    """A flag left out of the message reads as one the run accepted.

    A combination carrying two contradictions is refused for both, because a
    user who removes the flag the first one named and runs again would otherwise
    meet the second, and a run refused a pair at a time costs a run per pair.
    """
    unnamed = {
        names: sorted(name for name in _contradicted_within(names) if name not in (message or ''))
        for names in COMBINATIONS
        if (message := _refusal_of(tree, names)) is not None
    }
    assert {names: missing for names, missing in unnamed.items() if missing} == {}


def _pairs_called_mutually_exclusive(tree: Path) -> set[frozenset[str]]:
    """Return every pair of flags a refusal claims cannot hold together.

    "Mutually exclusive" is a claim about each pair of the flags it leads, so a
    clause leading three of them claims three pairs.

    Parameters:
        tree: The results root the combinations are put to.

    Returns:
        One frozen pair per claim any refusal makes.
    """
    claimed: set[frozenset[str]] = set()
    for names in COMBINATIONS:
        message = _refusal_of(tree, names)
        if message is None:
            continue
        for clause in message.split('; '):
            if 'mutually exclusive' not in clause:
                continue
            lead = clause.split(':')[0]
            named = sorted(name for name in SELECTION_FLAGS if name in lead)
            claimed.update(frozenset(pair) for pair in itertools.combinations(named, 2))
    return claimed


def test_no_refusal_calls_a_satisfiable_pair_mutually_exclusive(tree: Path) -> None:
    """A refusal that says more than it means sends its user to change the wrong flag.

    ``has_offset_error`` and ``has_offset_spice_error`` are a pair the
    constructor accepts, so a message that named them and
    ``has_no_offset_file`` together as mutually exclusive would tell a user that
    a selection this program answers is impossible.  The refusal that names one
    flag against the ones it excludes says only what is true, and this is what
    holds every "mutually exclusive" clause to a pair that really is one.
    """
    exclusive = {frozenset(cast(dict[str, bool], case.values[0])) for case in CONTRADICTORY_PAIRS}
    unfounded = _pairs_called_mutually_exclusive(tree) - exclusive
    assert sorted(sorted(pair) for pair in unfounded) == []


@pytest.mark.parametrize('flags', CONTRADICTORY_PAIRS)
def test_a_contradictory_pair_is_refused_before_the_index_is_opened(
    tree: Path, tmp_path: Path, flags: dict[str, Any]
) -> None:
    """The flags are validated first, so the refusal is the same with or without one.

    The URL names a database that does not exist, so a constructor that opened
    the index before checking its flags would report that instead.
    """
    absent = index_url(tmp_path / 'not-an-index.sqlite3')
    with pytest.raises(ValueError, match=CONTRADICTION_REFUSAL) as excinfo:
        ResultsFilter(
            VOLUMES, str(tree), logger=null_logger(), results_index_db_url=absent, **flags
        )
    assert 'not-an-index.sqlite3' not in str(excinfo.value)


@pytest.mark.parametrize('flags', CONTRADICTORY_PAIRS)
def test_a_refusal_names_every_flag_that_made_the_selection_impossible(
    tree: Path, flags: dict[str, Any]
) -> None:
    """The message is the whole diagnosis, so it names what the user typed.

    A message naming a category rather than a flag -- "the offset-error
    filters" -- leaves the user to work out which of six flags belongs to it,
    and one of the six is named for the absence of an error, so the category
    reads as something they did not ask for.

    Parameters:
        tree: The results root, which is never read: the flags are refused
            first.
        flags: The contradictory pair.
    """
    with pytest.raises(SelectionError) as excinfo:
        ResultsFilter(VOLUMES, str(tree), logger=null_logger(), results_index_db_url=None, **flags)
    assert [name for name in flags if name not in str(excinfo.value)] == []


def test_a_root_with_no_completed_ingest_is_refused(tree: Path, indexed: str) -> None:
    """Absence of a row is only an answer under a root somebody ingested."""
    other_root = tree.parent / 'never-ingested'
    with pytest.raises(ValueError, match='no completed ingest') as excinfo:
        ResultsFilter(
            VOLUMES,
            str(other_root),
            logger=null_logger(),
            results_index_db_url=indexed,
            has_offset_file=True,
        )
    assert other_root.as_posix() in str(excinfo.value)


def test_an_index_that_cannot_be_opened_is_not_a_reason_to_read_files(
    tree: Path, tmp_path: Path
) -> None:
    """A misconfigured run fails; it does not become a slow, silently different one."""
    absent = index_url(tmp_path / 'not-an-index.sqlite3')
    with pytest.raises(ValueError, match='sd_results_index') as excinfo:
        ResultsFilter(
            VOLUMES,
            str(tree),
            logger=null_logger(),
            results_index_db_url=absent,
            has_offset_file=True,
        )
    assert 'not-an-index.sqlite3' in str(excinfo.value)


def test_an_index_that_will_not_answer_refuses_the_selection(tmp_path: Path) -> None:
    """An index that opened and then failed is a misconfigured run like any other.

    The type is the one a program reporting the message catches, so a database
    failure reaches an operator as the sentence that says what to change rather
    than as a traceback out of an enumeration.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'failed_files')
    with pytest.raises(SelectionError, match='could not be read'):
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, has_offset_file=True
        )


def test_an_index_that_will_not_answer_raises_no_database_exception(tmp_path: Path) -> None:
    """This module never imports the database layer, so it may not raise its types."""
    root, _images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'failed_files')
    with pytest.raises(SelectionError) as excinfo:
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, has_offset_file=True
        )
    assert not isinstance(excinfo.value, sqlalchemy.exc.SQLAlchemyError)


def test_an_index_that_cannot_be_opened_refuses_the_selection(tmp_path: Path) -> None:
    """The three refusals are one type, so a program catches one and reports all three."""
    root, _images = one_image_tree(tmp_path)
    absent = index_url(tmp_path / 'not-an-index.sqlite3')
    with pytest.raises(SelectionError, match='sd_results_index'):
        ResultsFilter(
            VOLUMES,
            str(root),
            logger=null_logger(),
            results_index_db_url=absent,
            has_offset_file=True,
        )


def test_a_root_the_index_does_not_cover_refuses_the_selection(tree: Path, indexed: str) -> None:
    """Absence under a root nobody ingested is the third of the three."""
    with pytest.raises(SelectionError, match='no completed ingest'):
        ResultsFilter(
            VOLUMES,
            str(tree.parent / 'never-ingested'),
            logger=null_logger(),
            results_index_db_url=indexed,
            has_offset_file=True,
        )


def test_a_contradictory_pair_refuses_the_selection(tree: Path) -> None:
    """The flags are the fourth, and the one that needs no index at all."""
    with pytest.raises(SelectionError, match='mutually exclusive'):
        ResultsFilter(
            VOLUMES,
            str(tree),
            logger=null_logger(),
            has_offset_file=True,
            has_no_offset_file=True,
        )


def test_the_refusal_does_not_repeat_the_query_that_failed(tmp_path: Path) -> None:
    """The wrapper exists so that the sentence to act on is what a reader meets.

    The database layer renders a failed statement with its SQL, its bound
    parameters and a link to its own documentation.  Reported through a program
    that catches this type, that puts the advice under a page of machinery.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'failed_files')
    with pytest.raises(SelectionError) as excinfo:
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, has_offset_file=True
        )
    assert 'SELECT' not in str(excinfo.value)


def test_the_refusal_carries_what_the_driver_said(tmp_path: Path) -> None:
    """Which table is missing is the whole of what makes the failure actionable."""
    root, _images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'failed_files')
    with pytest.raises(SelectionError, match='no such table: failed_files'):
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, has_offset_file=True
        )


def test_a_failure_of_the_bookkeeping_query_is_translated_too(tmp_path: Path) -> None:
    """The refusal covers every query the answer costs, not only the last one.

    Reading a root's records is two questions of the database: whether anybody
    finished a pass over the root, and what it holds.  The first is asked before
    the second and against a table of its own, so a translation wrapped around
    only the query that reads the records lets this one out as the database
    layer's own exception.

    Parameters:
        tmp_path: Directory the tree and the index live under.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'ingest_runs')
    with pytest.raises(SelectionError, match='could not be read'):
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, has_offset_file=True
        )


def test_a_failure_of_the_bookkeeping_query_raises_no_database_exception(
    tmp_path: Path,
) -> None:
    """This module never imports the database layer, so it may not raise its types.

    Parameters:
        tmp_path: Directory the tree and the index live under.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'ingest_runs')
    with pytest.raises(SelectionError) as excinfo:
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, has_offset_file=True
        )
    assert not isinstance(excinfo.value, sqlalchemy.exc.SQLAlchemyError)


def test_a_failure_of_the_bookkeeping_query_names_the_table(tmp_path: Path) -> None:
    """Which table is missing is the whole of what makes the failure actionable.

    Parameters:
        tmp_path: Directory the tree and the index live under.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'ingest_runs')
    with pytest.raises(SelectionError, match='no such table: ingest_runs'):
        ResultsFilter(
            VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, has_offset_file=True
        )


def test_an_index_that_stops_answering_a_scan_of_candidates_refuses_the_selection(
    tmp_path: Path,
) -> None:
    """A scan that names its candidates asks as the enumeration runs, and so fails there.

    Its question is put once per batch rather than once at construction, so the
    boundary that turns a database failure into the sentence an operator can act
    on has to be around that question too.

    Parameters:
        tmp_path: Directory the tree and the index live under.
    """
    root, images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'failed_files')
    scan = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, has_no_offset_file=True
    )
    with scan, pytest.raises(SelectionError, match='could not be read'):
        scan.filter_batch(images)


def test_an_index_that_stops_answering_a_scan_of_candidates_raises_no_database_exception(
    tmp_path: Path,
) -> None:
    """This module never imports the database layer, so it may not raise its types.

    Parameters:
        tmp_path: Directory the tree and the index live under.
    """
    root, images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'failed_files')
    scan = ResultsFilter(
        VOLUMES, str(root), logger=null_logger(), results_index_db_url=url, has_no_offset_file=True
    )
    with scan, pytest.raises(SelectionError) as excinfo:
        scan.filter_batch(images)
    assert not isinstance(excinfo.value, sqlalchemy.exc.SQLAlchemyError)


def test_a_scan_of_candidates_refused_at_the_bookkeeping_query_leaves_no_storage_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A constructor that raises hands the caller no filter, so it releases what it opened.

    A scan that names its candidates holds its storage across the enumeration,
    so it is opened before the age of the index's answer is read -- and that
    read can refuse the selection.

    Parameters:
        tmp_path: Directory the tree and the index live under.
        monkeypatch: Fixture the bookkeeping query is wrapped through.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_without_a_table(tmp_path, root, 'ingest_runs')
    source = _CountingSource()

    def opening(roots: Sequence[Any], **kwargs: Any) -> Any:
        return source

    monkeypatch.setattr(results_filter, 'open_record_source', opening)
    with pytest.raises(SelectionError, match='could not be read'):
        ResultsFilter(
            VOLUMES,
            str(root),
            logger=null_logger(),
            results_index_db_url=url,
            has_no_offset_file=True,
        )
    assert source.closes == 1


class _CountingSource:
    """A record source that holds nothing, notes its closes, and can refuse to list.

    Parameters:
        refusal: What a listing of this source raises, or None to list nothing.
    """

    def __init__(self, refusal: BaseException | None = None) -> None:
        self.closes = 0
        self._refusal = refusal

    def __enter__(self) -> '_CountingSource':
        """Return this source, since there is nothing to open.

        Returns:
            This source.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release nothing, since nothing was opened."""

    def close(self) -> None:
        """Count this, since being released is what these tests watch."""
        self.closes += 1

    def listing(self, selection: Selection) -> Iterator[ListedRecord]:
        """Refuse the way this source was built to, or report no documents.

        Parameters:
            selection: What the filter asked for.

        Returns:
            An empty stream, for a source built to refuse nothing.

        Raises:
            BaseException: Whatever this source was built to refuse with.
        """
        if self._refusal is not None:
            raise self._refusal
        return iter(())

    def facts(self, selection: Selection) -> Iterator[ImageFacts | UnreadableFile]:
        """Report no facts, since no test here gets as far as a batch.

        Parameters:
            selection: What the filter asked for.

        Returns:
            An empty stream.
        """
        return iter(())


def _opening(monkeypatch: pytest.MonkeyPatch, source: _CountingSource) -> None:
    """Give the filter a stand-in storage in place of the one it would open.

    Parameters:
        monkeypatch: Fixture the stand-in is installed through.
        source: The storage every open hands back.
    """

    def opening(roots: Sequence[Any], **kwargs: Any) -> _CountingSource:
        return source

    monkeypatch.setattr(results_filter, 'open_record_source', opening)


def test_a_listing_that_refuses_releases_the_storage_it_was_read_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal comes out of the constructor, so the caller is handed nothing to close.

    Over an index the storage is a connection pool, and a run whose selection
    is refused is one an operator retries: each attempt would leave another
    pool behind, released by nothing until the interpreter collected it.

    Parameters:
        tmp_path: Directory standing in for the results root, which the
            stand-in storage never reads.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    source = _CountingSource(ValueError('the index stopped answering'))
    _opening(monkeypatch, source)
    with pytest.raises(SelectionError, match='stopped answering'):
        ResultsFilter(VOLUMES, str(tmp_path), logger=null_logger(), has_offset_error=True)
    assert source.closes == 1


def test_a_listing_that_fails_outright_releases_the_storage_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The release belongs to the opening, not to the one failure that is translated.

    A storage released only where a refusal is turned into the selection type
    holds on to everything a fault leaves behind.

    Parameters:
        tmp_path: Directory standing in for the results root.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    source = _CountingSource(RuntimeError('the storage layer fell over'))
    _opening(monkeypatch, source)
    with pytest.raises(RuntimeError, match='fell over'):
        ResultsFilter(VOLUMES, str(tmp_path), logger=null_logger(), has_offset_error=True)
    assert source.closes == 1


def test_a_refusal_from_the_report_releases_the_storage_as_well(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reporting the answer is the last thing that can refuse, and it refuses the same way.

    An index is asked when its pass finished after it is asked what it holds, so
    a connection lost between the two questions raises here.  An error filter
    keeps its storage across the enumeration, so this is the one refusal raised
    with somewhere to put the storage other than the caller's hands -- and the
    caller never gets the filter, so putting it there releases nothing.

    Parameters:
        tmp_path: Directory standing in for the results root.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    source = _CountingSource()
    _opening(monkeypatch, source)

    def refusing(results_index_db_url: str, root: Any) -> str:
        raise ValueError('this results index could not be read')

    monkeypatch.setattr(results_filter, 'snapshot_finish_time', refusing)
    with pytest.raises(SelectionError, match='could not be read'):
        ResultsFilter(
            VOLUMES,
            str(tmp_path),
            logger=null_logger(),
            results_index_db_url='sqlite+pysqlite:///nothing-is-opened',
            has_offset_error=True,
        )
    assert source.closes == 1
