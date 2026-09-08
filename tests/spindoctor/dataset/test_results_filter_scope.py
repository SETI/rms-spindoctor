"""Which of a results root's documents each storage is asked about.

An enumeration selects volumes, and the filter asks the record seam about those
volumes and no others.  Two things follow, and both are asserted of the tree and
of the index over one fixture: a document outside the selected volumes is not in
the answer, and a volume that is selected and cannot be answered for is a
refusal rather than an empty answer.

Every fixture here builds two results roots, because the key a document is
recorded under is its root and its stub together.  The second root holds a
document for stubs the root under test has none of, and records the opposite
outcome for the stubs it does hold, so any answer read without its root differs
from the stated one.  No single-root fixture can see that happen.

The volumes selected include one the root under test has no directory for, which
is the ordinary state of a results root part way through a campaign: a volume
nobody has navigated yet.  It contributes nothing and ends nothing, which is why
every stated answer below is the answer for the volumes that are there.
"""

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
)
from tests.spindoctor.dataset.conftest import (
    null_logger,
    select_from,
    write_bytes,
)

from spindoctor.dataset import results_filter
from spindoctor.dataset.dataset import ImageFile
from spindoctor.dataset.results_filter import SPICE_STATUS_ERROR, ResultsFilter
from spindoctor.nav_records import (
    ImageFacts,
    ListedRecord,
    Selection,
    TreeTuning,
    UnreadableFile,
)

VOLUMES = ['COISS_2001', 'COISS_2002']
"""The volumes the enumeration selected.

The second of them is a volume the root under test has no directory for, so
every answer below is also an answer about what an unnavigated volume costs.
"""

OTHER_VOLUME = 'COISS_2003'
"""A volume of the same root that the enumeration did not select."""

SELECTED = 'COISS_2001/data/a/N1000000001_1_CALIB'
"""A document under a selected volume, recording a run that succeeded."""

ERRORED_HERE = 'COISS_2001/data/b/N1000000007_1_CALIB'
"""A document under a selected volume, recording a fatal SPICE error.

The second root records a run that finished for this stub, which is what makes
the filter phrased in the negative root-sensitive: the shared decoy's fatal
errors change only the answers phrased in the positive.
"""

FINISHED_HERE = 'COISS_2001/data/b/N1000000008_1_CALIB'
"""A document under a selected volume, recording a run that reached no offset.

The second root records a fatal error for this stub, so the disagreement runs
both ways and an answer read without its root is short as well as long.
"""

UNSELECTED_VOLUME = f'{OTHER_VOLUME}/data/a/N1000000002_1_CALIB'
"""A document of the root under test, under a volume nobody selected."""

UNSELECTED_REFUSED = f'{OTHER_VOLUME}/data/a/N1000000003_1_CALIB'
"""A file no record can be read out of, under a volume nobody selected.

A root whose unreadable files outnumber its results is exactly the root the
refusal bookkeeping exists for, so the restriction to the selected volumes has
to reach those as surely as it reaches the documents.
"""

NO_VOLUME = 'N1000000004_1_CALIB'
"""A document directly under the results root, with no volume above it.

It sits outside every selected directory, so neither storage may offer it: the
walk descends the volumes it was named and this is under none of them, and a
query narrowing on the first path component finds this one has none.
"""

ONLY_IN_THE_OTHER_ROOT = 'COISS_2001/data/c/N1000000005_1_CALIB'
"""A stub only the second root holds a document for, under a selected volume."""

ONLY_REFUSED_IN_THE_OTHER_ROOT = 'COISS_2001/data/c/N1000000006_1_CALIB'
"""A stub only the second root holds an unreadable file for.

No image row anywhere carries it, so a refusal read without its root shows up as
a document under a root that has none.
"""

ONLY_IN_THE_OTHER_ROOTS_OTHER_VOLUME = f'{OTHER_VOLUME}/data/c/N1000000009_1_CALIB'
"""A stub only the second root holds, under the volume nobody selected.

It is what makes the answer to "select the other volume" a root-aware one: the
two roots otherwise agree about that volume, so an answer that dropped its root
would be identical without it.
"""

CANDIDATES = (
    SELECTED,
    ERRORED_HERE,
    FINISHED_HERE,
    UNSELECTED_VOLUME,
    UNSELECTED_REFUSED,
    NO_VOLUME,
    ONLY_IN_THE_OTHER_ROOT,
    ONLY_REFUSED_IN_THE_OTHER_ROOT,
    ONLY_IN_THE_OTHER_ROOTS_OTHER_VOLUME,
)
"""Every stub offered to the filter, in the order an enumeration yields them."""

STORAGES = [pytest.param(False, id='tree'), pytest.param(True, id='index')]
"""Whether the filter reads an ingested index rather than the documents.

Each is held to the stated answer rather than to the other's, because two
storages that are wrong in the same way agree with each other.
"""

MALFORMED = b'{"status": "error"'
"""A metadata file that is not JSON, so no record can be read out of it."""


def _write_roots(tmp_path: Path) -> Path:
    """Write the root under test and the second root beside it.

    Parameters:
        tmp_path: Directory the two roots are written under.

    Returns:
        The results root under test.
    """
    root = tmp_path / 'results'
    write_metadata(root, SELECTED, metadata_document(image_name='N1000000001_1.IMG'))
    write_metadata(
        root,
        ERRORED_HERE,
        metadata_document(
            image_name='N1000000007_1.IMG',
            status='error',
            status_error=SPICE_STATUS_ERROR,
            offset=None,
        ),
    )
    write_metadata(
        root,
        FINISHED_HERE,
        metadata_document(image_name='N1000000008_1.IMG', status='failure', offset=None),
    )
    write_metadata(root, UNSELECTED_VOLUME, metadata_document(image_name='N1000000002_1.IMG'))
    write_bytes(root, UNSELECTED_REFUSED, MALFORMED)
    write_metadata(root, NO_VOLUME, metadata_document(image_name='N1000000004_1.IMG'))

    other = tmp_path / 'other-results'
    write_metadata(other, ONLY_IN_THE_OTHER_ROOT, metadata_document(image_name='N1000000005_1.IMG'))
    write_bytes(other, ONLY_REFUSED_IN_THE_OTHER_ROOT, MALFORMED)
    write_metadata(
        other,
        SELECTED,
        metadata_document(
            image_name='N1000000001_1.IMG',
            status='error',
            status_error=SPICE_STATUS_ERROR,
            offset=None,
        ),
    )
    write_metadata(
        other,
        ERRORED_HERE,
        metadata_document(image_name='N1000000007_1.IMG', status='failure', offset=None),
    )
    write_metadata(
        other,
        FINISHED_HERE,
        metadata_document(
            image_name='N1000000008_1.IMG',
            status='error',
            status_error=SPICE_STATUS_ERROR,
            offset=None,
        ),
    )
    write_metadata(
        other,
        ONLY_IN_THE_OTHER_ROOTS_OTHER_VOLUME,
        metadata_document(image_name='N1000000009_1.IMG'),
    )
    return root


def _ingested(tmp_path: Path, root: Path) -> str:
    """Ingest both roots into one index and return its URL.

    The second root is passed over after the first, so its pass is the newest in
    the index.

    Parameters:
        tmp_path: Directory the index file is written into.
        root: The results root under test.

    Returns:
        The connection URL of the index.
    """
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root, tmp_path / 'other-results'], logger=null_logger())
    return url


def _candidate_files(root: Path) -> list[ImageFile]:
    """Build the images an enumeration would offer the filter.

    Parameters:
        root: The results root, only so the stand-in URLs point somewhere.

    Returns:
        One :class:`ImageFile` per candidate, in enumeration order.
    """
    return [
        ImageFile(
            image_file_url=FCPath(root / f'{stub}.IMG'),
            label_file_url=FCPath(root / f'{stub}.LBL'),
            results_path_stub=stub,
        )
        for stub in CANDIDATES
    ]


def _answering(
    tmp_path: Path,
    *,
    from_an_index: bool,
    volumes: Iterable[str] = tuple(VOLUMES),
    **flags: Any,
) -> list[str]:
    """Build both roots and answer one filter combination over the first of them.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an ingested index rather than
            the documents.
        volumes: The volumes the enumeration selected.
        flags: The selection flags to apply.

    Returns:
        The stubs that passed, in enumeration order.
    """
    root = _write_roots(tmp_path)
    url = _ingested(tmp_path, root) if from_an_index else None
    results_filter = ResultsFilter(
        volumes, str(root), logger=null_logger(), results_index_db_url=url, **flags
    )
    return select_from(results_filter, _candidate_files(root))


# ---------------------------------------------------------------------------
# Which volumes are asked about
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_the_selected_volumes_are_answered_for(tmp_path: Path, from_an_index: bool) -> None:
    """The whole answer, stated, so that every exclusion below is a real one.

    Without it each of the assertions that a stub is absent would also pass for
    a filter that answered nothing at all, and the volume the root has no
    directory for would be indistinguishable from a volume that ended the run.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_offset_file=True)
    assert kept == [SELECTED, ERRORED_HERE, FINISHED_HERE]


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_a_document_under_an_unselected_volume_is_not_answered_for(
    tmp_path: Path, from_an_index: bool
) -> None:
    """A run restricted to two volumes may not be told about a third.

    The restriction is what an enumeration of one volume pays for: without it a
    run over one volume of an archive is answered by reading every volume of it.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_offset_file=True)
    assert UNSELECTED_VOLUME not in kept


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_an_unreadable_file_under_an_unselected_volume_is_not_answered_for(
    tmp_path: Path, from_an_index: bool
) -> None:
    """A file no record comes out of is under a volume like any other.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_offset_file=True)
    assert UNSELECTED_REFUSED not in kept


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_a_document_with_no_volume_above_it_is_not_answered_for(
    tmp_path: Path, from_an_index: bool
) -> None:
    """It lies outside every selected directory, so no selection of volumes reaches it.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_offset_file=True)
    assert NO_VOLUME not in kept


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_selecting_the_other_volume_answers_for_that_volume(
    tmp_path: Path, from_an_index: bool
) -> None:
    """The volume nobody selected is readable when it is the one selected.

    Stated as the whole answer, and the second root holds a file of that volume
    the first does not, so the restriction to a volume is pinned together with
    the restriction to a root: an answer restricted by volume alone carries the
    second root's file as well.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(
        tmp_path, from_an_index=from_an_index, volumes=[OTHER_VOLUME], has_offset_file=True
    )
    assert kept == [UNSELECTED_VOLUME, UNSELECTED_REFUSED]


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_selecting_no_volume_at_all_answers_for_nothing(
    tmp_path: Path, from_an_index: bool
) -> None:
    """An enumeration that selected no volume asks about no directory.

    No volume means none, and the storage layers below read an empty restriction
    as no restriction -- the whole root.  So this is pinned where the two
    vocabularies meet: a filter that handed its empty list of volumes straight
    down would answer with every document of the root and keep every image of an
    enumeration that selected nothing.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, volumes=[], has_offset_file=True)
    assert kept == []


# ---------------------------------------------------------------------------
# Which root is answered for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_a_document_only_the_other_root_holds_is_absent_here(
    tmp_path: Path, from_an_index: bool
) -> None:
    """A record is keyed by its root and its stub together, and this is the stub half.

    ``--has-no-offset-file`` is the resume idiom, so a document borrowed from
    another root is an image this run silently declines to navigate.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_offset_file=True)
    assert ONLY_IN_THE_OTHER_ROOT not in kept


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_a_file_only_the_other_root_could_not_read_is_absent_here(
    tmp_path: Path, from_an_index: bool
) -> None:
    """The unreadable files are keyed by root and stub together, as the records are.

    No record anywhere carries this stub, so a refusal read without its root
    hands the enumeration a file that belongs to a different tree.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_offset_file=True)
    assert ONLY_REFUSED_IN_THE_OTHER_ROOT not in kept


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_an_error_filter_answers_for_this_root_only(tmp_path: Path, from_an_index: bool) -> None:
    """The other root records a fatal SPICE error for a stub that succeeded here.

    Stated as the whole answer, so a filter reading the other root's outcome
    both gains that stub and, since the other root records a run that finished
    for the stub that failed here, loses the one it should have.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_offset_spice_error=True)
    assert kept == [ERRORED_HERE]


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_the_negative_error_filter_answers_for_this_root_only(
    tmp_path: Path, from_an_index: bool
) -> None:
    """The disagreement runs the other way for this one, and it has to be tested that way.

    A second root that recorded a fatal error for every stub would change no
    answer phrased in the negative, so the two stubs it disagrees with this root
    about disagree in opposite directions: one errored here and finished there,
    the other finished here and errored there.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_no_offset_error=True)
    assert kept == [SELECTED, FINISHED_HERE]


def test_a_volume_the_index_holds_no_rows_for_does_not_end_the_enumeration(
    tmp_path: Path,
) -> None:
    """A volume nobody has navigated has no rows, which is an answer and not a failure.

    The tree reads it as a directory that is not there; the index reads it as a
    root that holds nothing under that name.  Both are the ordinary state of a
    campaign in progress, and the volumes after it are the ones the run is
    about.

    Parameters:
        tmp_path: Directory the roots and the index are written under.
    """
    kept = _answering(
        tmp_path,
        from_an_index=True,
        volumes=['COISS_2000', 'COISS_2001'],
        has_offset_file=True,
    )
    assert kept == [SELECTED, ERRORED_HERE, FINISHED_HERE]


def test_a_relative_results_root_names_the_root_the_ingest_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative root is a documented spelling of the option, not a second root.

    An index records one spelling of a root, and an enumeration handed another
    spelling of the same root would otherwise find no rows under it -- which the
    absence filters read as "nothing here has ever been navigated".

    Parameters:
        tmp_path: Directory the roots and the index are written under.
        monkeypatch: Fixture the working directory is changed through.
    """
    root = _write_roots(tmp_path)
    url = _ingested(tmp_path, root)
    monkeypatch.chdir(tmp_path)
    results_filter = ResultsFilter(
        VOLUMES, root.name, logger=null_logger(), results_index_db_url=url, has_offset_file=True
    )
    assert select_from(results_filter, _candidate_files(root)) == [
        SELECTED,
        ERRORED_HERE,
        FINISHED_HERE,
    ]


def test_a_root_named_through_a_parent_segment_is_the_root_the_ingest_recorded(
    tmp_path: Path,
) -> None:
    """A root reached by going up and back down again is the root it lands on.

    It is the shape an operator writes when a results root sits beside the tree
    they started from, and one more spelling of the same rule the relative name
    is: what the index recorded and what the enumeration was pointed at have to
    render alike or the filter answers about nothing.

    Parameters:
        tmp_path: Directory the roots and the index are written under.
    """
    root = _write_roots(tmp_path)
    url = _ingested(tmp_path, root)
    results_filter = ResultsFilter(
        VOLUMES,
        f'{tmp_path.as_posix()}/other-results/../results',
        logger=null_logger(),
        results_index_db_url=url,
        has_offset_file=True,
    )
    assert select_from(results_filter, _candidate_files(root)) == [
        SELECTED,
        ERRORED_HERE,
        FINISHED_HERE,
    ]


# ---------------------------------------------------------------------------
# What a scan of the candidates themselves answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_an_absence_filter_keeps_the_candidates_this_root_holds_nothing_for(
    tmp_path: Path, from_an_index: bool
) -> None:
    """The whole answer, stated, so that every exclusion below is a real one.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_no_offset_file=True)
    assert kept == [
        ONLY_IN_THE_OTHER_ROOT,
        ONLY_REFUSED_IN_THE_OTHER_ROOT,
        ONLY_IN_THE_OTHER_ROOTS_OTHER_VOLUME,
    ]


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_an_absence_filter_drops_a_candidate_this_root_holds_a_document_for(
    tmp_path: Path, from_an_index: bool
) -> None:
    """The image the run is not to navigate again.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_no_offset_file=True)
    assert SELECTED not in kept


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_an_absence_filter_drops_a_candidate_whose_file_says_nothing_readable(
    tmp_path: Path, from_an_index: bool
) -> None:
    """Presence is a question about the file and not about what is in it.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_no_offset_file=True)
    assert UNSELECTED_REFUSED not in kept


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_an_absence_filter_keeps_a_candidate_only_the_other_root_holds(
    tmp_path: Path, from_an_index: bool
) -> None:
    """A stub is a key under every root there is, and only this one's answers.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_no_offset_file=True)
    assert ONLY_IN_THE_OTHER_ROOT in kept


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_an_absence_filter_keeps_a_file_only_the_other_root_refused(
    tmp_path: Path, from_an_index: bool
) -> None:
    """The other half of the key, over the table that records a refused file.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_no_offset_file=True)
    assert ONLY_REFUSED_IN_THE_OTHER_ROOT in kept


@pytest.mark.parametrize('from_an_index', STORAGES)
def test_an_absence_filter_answers_about_a_candidate_under_no_selected_volume(
    tmp_path: Path, from_an_index: bool
) -> None:
    """It asks about the image, so where the image sits narrows nothing.

    An enumeration only offers images from the volumes it walked, so naming
    those volumes as well would narrow nothing that is ever offered -- and a
    scan that named them would answer about this candidate out of a listing that
    never covered it.

    Parameters:
        tmp_path: Directory the roots and any index are written under.
        from_an_index: Whether the filter reads an index rather than the tree.
    """
    kept = _answering(tmp_path, from_an_index=from_an_index, has_no_offset_file=True)
    assert UNSELECTED_VOLUME not in kept


# ---------------------------------------------------------------------------
# What each storage is asked, rather than what it answers
# ---------------------------------------------------------------------------


class _RecordingSource:
    """A record source that holds nothing and notes what it is asked about.

    Parameters:
        asked: List a listing of subtrees appends the subtrees it named to.
        named: List a question about named files appends the stubs it named to,
            whether it asked what those files record or only whether they are
            there.
    """

    def __init__(self, asked: list[tuple[str, ...]], named: list[tuple[str, ...]]) -> None:
        self._asked = asked
        self._named = named
        self.closes = 0

    def __enter__(self) -> '_RecordingSource':
        """Return this source, since there is nothing to open.

        Returns:
            This source.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release nothing, since nothing was opened."""

    def close(self) -> None:
        """Count this, since being closed is what is under test."""
        self.closes += 1

    def listing(self, selection: Selection) -> Iterator[ListedRecord]:
        """Note what this question named and report no documents.

        Parameters:
            selection: What the filter asked for.

        Returns:
            An empty stream.
        """
        if selection.stubs:
            self._named.append(selection.stubs)
        else:
            self._asked.append(selection.subtrees)
        return iter(())

    def facts(self, selection: Selection) -> Iterator[ImageFacts | UnreadableFile]:
        """Note the stubs this question named and report no facts.

        Parameters:
            selection: What the filter asked for.

        Returns:
            An empty stream.
        """
        self._named.append(selection.stubs)
        return iter(())


def _subtrees_asked_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flags: Any
) -> list[tuple[str, ...]]:
    """Build a filter over a stand-in storage and return what it asked that storage.

    The volumes are handed over as an iterator, which is what a caller is free
    to supply, so what the constructor makes of them before asking anything is
    under test here along with the questions themselves.

    Parameters:
        tmp_path: Directory standing in for the results root, which the
            stand-in storage never reads.
        monkeypatch: Fixture the stand-in is installed through.
        flags: The selection flags to apply.

    Returns:
        The subtrees of each question, in the order they were asked.
    """
    asked: list[tuple[str, ...]] = []

    def recording(roots: Sequence[Any], **kwargs: Any) -> _RecordingSource:
        return _RecordingSource(asked, [])

    monkeypatch.setattr(results_filter, 'open_record_source', recording)
    ResultsFilter(
        iter(VOLUMES), str(tmp_path), logger=null_logger(), results_index_db_url=None, **flags
    )
    return asked


def _stubs_named_by_a_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubs: Sequence[str]
) -> list[tuple[str, ...]]:
    """Build an error filter over a stand-in storage and batch some candidates at it.

    Parameters:
        tmp_path: Directory standing in for the results root, which the
            stand-in storage never reads.
        monkeypatch: Fixture the stand-in is installed through.
        stubs: The candidates to offer the batch.

    Returns:
        The stubs of each question about a document, in the order asked.
    """
    named: list[tuple[str, ...]] = []

    def recording(roots: Sequence[Any], **kwargs: Any) -> _RecordingSource:
        return _RecordingSource([], named)

    monkeypatch.setattr(results_filter, 'open_record_source', recording)
    with ResultsFilter(
        VOLUMES, str(tmp_path), logger=null_logger(), has_offset_error=True
    ) as results_filter_under_test:
        results_filter_under_test.filter_batch(
            [
                ImageFile(
                    image_file_url=FCPath(tmp_path / 'x.IMG'),
                    label_file_url=FCPath(tmp_path / 'x.LBL'),
                    results_path_stub=stub,
                )
                for stub in stubs
            ]
        )
    return named


def _stubs_listed_by_a_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stubs: Sequence[str]
) -> list[tuple[str, ...]]:
    """Build an absence filter over a stand-in storage and batch some candidates at it.

    Parameters:
        tmp_path: Directory standing in for the results root, which the
            stand-in storage never reads.
        monkeypatch: Fixture the stand-in is installed through.
        stubs: The candidates to offer the batch.

    Returns:
        The stubs of each question about which files are there, in the order
        asked.
    """
    named: list[tuple[str, ...]] = []

    def recording(roots: Sequence[Any], **kwargs: Any) -> _RecordingSource:
        return _RecordingSource([], named)

    monkeypatch.setattr(results_filter, 'open_record_source', recording)
    with ResultsFilter(
        VOLUMES, str(tmp_path), logger=null_logger(), has_no_offset_file=True
    ) as results_filter_under_test:
        results_filter_under_test.filter_batch(
            [
                ImageFile(
                    image_file_url=FCPath(tmp_path / 'x.IMG'),
                    label_file_url=FCPath(tmp_path / 'x.LBL'),
                    results_path_stub=stub,
                )
                for stub in stubs
            ]
        )
    return named


def test_a_presence_filter_asks_about_one_selected_volume_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One question per volume, so a volume the root lacks costs only itself.

    A single question naming every volume ends at the first directory it cannot
    read, and the volumes after it go unasked.  For an enumeration over volumes
    nobody has navigated yet that is every volume after the first, which is the
    ordinary state of a results root part way through a campaign.

    Parameters:
        tmp_path: Directory standing in for the results root.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    asked = _subtrees_asked_about(tmp_path, monkeypatch, has_offset_file=True)
    assert asked == [(VOLUMES[0],), (VOLUMES[1],)]


def test_an_error_filter_lists_one_selected_volume_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error filter needs the same listing, since its images must have a document.

    Parameters:
        tmp_path: Directory standing in for the results root.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    asked = _subtrees_asked_about(tmp_path, monkeypatch, has_offset_error=True)
    assert asked == [(VOLUMES[0],), (VOLUMES[1],)]


def test_an_error_filter_asks_about_its_candidates_and_not_about_a_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading a document is what an error filter costs, so it reads the candidates.

    Asked of the selected volumes instead, a run whose other constraints keep
    one image in a hundred would read every document under them and discard
    almost all of it -- on a cloud results root, one paid download apiece.

    Parameters:
        tmp_path: Directory standing in for the results root.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    named = _stubs_named_by_a_batch(tmp_path, monkeypatch, [SELECTED, ERRORED_HERE])
    assert named == [(SELECTED, ERRORED_HERE)]


def test_an_absence_filter_lists_no_volume_when_it_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It keeps the images the root has nothing for, so a listing of a volume is waste.

    Every entry such a listing produces is one to reject from, so a run whose
    other constraints name ten images would pay for a whole volume to answer
    about ten.

    Parameters:
        tmp_path: Directory standing in for the results root.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    assert _subtrees_asked_about(tmp_path, monkeypatch, has_no_offset_file=True) == []


def test_an_absence_filter_asks_about_its_candidates_and_not_about_a_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One question per batch, naming the images the run might still keep.

    Parameters:
        tmp_path: Directory standing in for the results root.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    named = _stubs_listed_by_a_batch(tmp_path, monkeypatch, [SELECTED, ERRORED_HERE])
    assert named == [(SELECTED, ERRORED_HERE)]


def _source_of(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **flags: Any) -> _RecordingSource:
    """Build a filter over a stand-in storage and hand that storage back.

    Parameters:
        tmp_path: Directory standing in for the results root, which the
            stand-in storage never reads.
        monkeypatch: Fixture the stand-in is installed through.
        flags: The selection flags to apply.

    Returns:
        The stand-in storage the filter was given.
    """
    source = _RecordingSource([], [])

    def recording(roots: Sequence[Any], **kwargs: Any) -> _RecordingSource:
        return source

    monkeypatch.setattr(results_filter, 'open_record_source', recording)
    with ResultsFilter(
        VOLUMES, str(tmp_path), logger=null_logger(), results_index_db_url=None, **flags
    ):
        pass
    return source


def test_leaving_an_error_filter_closes_the_storage_it_held_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error filter keeps a storage open between batches and must give it back.

    Over an index that is a connection pool, and an enumeration is one of many
    a long run makes.  Nothing about an answer depends on this, so only a test
    that watches the storage can tell a filter that releases it from one that
    leaks it until the interpreter collects it.

    Parameters:
        tmp_path: Directory standing in for the results root.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    assert _source_of(tmp_path, monkeypatch, has_offset_error=True).closes == 1


def test_leaving_an_absence_filter_closes_the_storage_it_held_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It asks a question per batch too, so it holds its storage for the enumeration.

    Parameters:
        tmp_path: Directory standing in for the results root.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    assert _source_of(tmp_path, monkeypatch, has_no_offset_file=True).closes == 1


def test_a_filter_with_nothing_left_to_ask_closes_the_storage_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A presence filter has asked everything it will ask by the time it is built.

    Holding a storage open across an enumeration that will never use it again
    is a connection held for the length of a run for nothing.

    Parameters:
        tmp_path: Directory standing in for the results root.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    assert _source_of(tmp_path, monkeypatch, has_offset_file=True).closes == 1


@pytest.mark.parametrize(
    ('flags', 'batches'),
    [
        pytest.param({'has_offset_file': True}, False, id='offset-file'),
        pytest.param({'has_no_offset_file': True}, True, id='no-offset-file'),
        pytest.param({'has_offset_error': True}, True, id='offset-error'),
        pytest.param({'has_no_offset_error': True}, True, id='no-offset-error'),
        pytest.param({'has_offset_spice_error': True}, True, id='spice-error'),
        pytest.param({'has_offset_nonspice_error': True}, True, id='nonspice-error'),
    ],
)
def test_only_a_filter_that_reads_documents_asks_the_enumeration_to_batch(
    tmp_path: Path, flags: dict[str, Any], batches: bool
) -> None:
    """The enumeration buffers candidates only where buffering buys something.

    Nothing about an answer depends on this, which is why it needs a test of its
    own: a filter that reported the wrong thing here would answer every
    selection correctly and pay for it one candidate at a time, asking a
    question per image where one question covers sixty-four.  An error filter
    has a document to read per batch and an absence filter has a question to ask
    about one; ``has_offset_file`` is settled by the listing taken when the
    filter was built.

    Parameters:
        tmp_path: Directory standing in for the results root.
        flags: The selection flags to apply.
        batches: Whether this filter has anything left to ask per batch.
    """
    with ResultsFilter(
        VOLUMES, str(tmp_path), logger=null_logger(), results_index_db_url=None, **flags
    ) as results_filter_under_test:
        assert results_filter_under_test.needs_batch_filtering is batches


def test_a_batch_of_no_candidates_asks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An enumeration that accepted nothing must not pay for a question about it.

    Parameters:
        tmp_path: Directory standing in for the results root.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    assert _stubs_named_by_a_batch(tmp_path, monkeypatch, []) == []


def test_the_tuning_the_filter_is_given_reaches_the_storage_it_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The enumeration resolves the tuning once, and the filter hands it on unchanged.

    Parameters:
        tmp_path: Directory standing in for the results root.
        monkeypatch: Fixture the stand-in storage is installed through.
    """
    handed: list[Any] = []

    def recording(roots: Sequence[Any], **kwargs: Any) -> _RecordingSource:
        handed.append(kwargs.get('tuning'))
        return _RecordingSource([], [])

    monkeypatch.setattr(results_filter, 'open_record_source', recording)
    tuning = TreeTuning(walk_threads=3, walk_directories_at_once=3)
    with ResultsFilter(
        VOLUMES, str(tmp_path), logger=null_logger(), has_offset_file=True, tuning=tuning
    ):
        pass
    assert handed == [tuning]
