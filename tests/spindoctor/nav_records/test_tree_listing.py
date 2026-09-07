"""What the walk of a results tree finds, and what it refuses to walk past.

A listing says what is there without opening a single document, which is what
makes it worth so much more than the names it carries: each entry brings the
size and modification time that decide whether that document has changed.

The refusals are the point of most of this.  A directory nobody can list is a
directory whose documents nobody read, and a pass that finished around one would
report itself clean while covering less than the tree, so it stops.  A directory
reached a second way is not a gap and is declined rather than refused.  And a
restriction a listing cannot honour is refused rather than ignored, because a
listing of the whole root handed back as a listing of one mission is a wrong
answer rather than a missing feature.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from filecache import FCPath

from spindoctor.nav_records import (
    METADATA_SUFFIX,
    ListedRecord,
    Selection,
    TreeRecordSource,
    TreeTuning,
    UnlistableDirectoryError,
    UnlistableRootError,
)
from spindoctor.nav_records import walk as walk_module

from .conftest import (
    FIRST_STUB,
    MISSION,
    SECOND_STUB,
    UNLISTABLE_ERRORS,
    document,
    stubs_of,
    tree_source,
    two_volume_tree,
    unlistable_root,
    unlistable_subdirectory,
    write_document,
)

# ---------------------------------------------------------------------------
# What a listing finds
# ---------------------------------------------------------------------------


def test_a_root_is_listed_whole(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """Both volumes' documents, from one walk of the root."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    assert sorted(stubs_of(source.listing(Selection()))) == [FIRST_STUB, SECOND_STUB]


def test_a_file_that_is_not_a_document_is_no_part_of_the_listing(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A name that ends in the document suffix is what says which files are read.

    Neither half of that is enough on its own: a file that merely ends in
    ``.json`` is not a navigation document, and one that merely contains the
    suffix yields a stub with the suffix's length cut off the end of a longer
    name, naming nothing, which a pass then retrieves and fails on forever.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document())
    (root / 'VOL1' / 'index.json').write_text('{}')
    (root / 'VOL1' / f'N1454725799_1_CALIB{METADATA_SUFFIX}.bak').write_text('{}')
    source = tree_source(root, quiet_logger)
    assert stubs_of(source.listing(Selection())) == [FIRST_STUB]


def test_a_listed_document_carries_the_size_the_listing_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The metric comes from the entry, and it has to be the file's own."""
    root = tmp_path / 'results'
    written = write_document(root, FIRST_STUB, document())
    listed = list(tree_source(root, quiet_logger).listing(Selection()))
    assert listed[0].size_bytes == written.stat().st_size


def test_a_listed_document_carries_the_modification_time_the_listing_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The other half of what decides whether a document has changed."""
    root = tmp_path / 'results'
    written = write_document(root, FIRST_STUB, document())
    listed = list(tree_source(root, quiet_logger).listing(Selection()))
    assert listed[0].mtime_ns == round(written.stat().st_mtime * 1_000_000_000)


def test_a_listed_document_reporting_both_metrics_can_be_compared_later(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A local listing supplies both, so a later pass can tell what changed."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document())
    listed = list(tree_source(root, quiet_logger).listing(Selection()))
    assert listed[0].has_metrics is True


def test_a_listing_that_reported_no_modification_time_cannot_be_compared_later() -> None:
    """One metric is not enough: a rewritten file keeps its size more often than not."""
    listed = ListedRecord(stub='x', path=FCPath('x_metadata.json'), mtime_ns=None, size_bytes=12)
    assert listed.has_metrics is False


def test_a_listing_that_reported_no_size_cannot_be_compared_later() -> None:
    """The other half of the same rule, which a one-sided test would miss."""
    listed = ListedRecord(stub='x', path=FCPath('x_metadata.json'), mtime_ns=17, size_bytes=None)
    assert listed.has_metrics is False


# ---------------------------------------------------------------------------
# A directory nobody can list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('error', UNLISTABLE_ERRORS)
def test_a_directory_that_cannot_be_listed_stops_the_walk(
    error: type[OSError],
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pass that finished around the gap would answer wrongly and go on doing so.

    Parameters:
        error: The exception type the unlistable directory raises.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the walk reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    unlistable_subdirectory(monkeypatch, error)
    with pytest.raises(UnlistableDirectoryError, match='could not be listed'):
        list(source.listing(Selection()))


def test_the_refusal_names_the_directory_that_would_not_list(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one thing an operator has to go and fix is the one thing it must say."""
    root = two_volume_tree(tmp_path)
    source = tree_source(root, quiet_logger)
    unlistable_subdirectory(monkeypatch, PermissionError)
    with pytest.raises(UnlistableDirectoryError) as excinfo:
        list(source.listing(Selection()))
    assert (root / 'VOL2').as_posix() in str(excinfo.value)


def test_a_record_stream_stops_at_the_directory_too(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mission read out of part of a tree is a mission nothing can tell is short."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    unlistable_subdirectory(monkeypatch, PermissionError)
    with pytest.raises(UnlistableDirectoryError, match='could not be listed'):
        list(source.records(Selection()))


@pytest.mark.parametrize('error', UNLISTABLE_ERRORS)
def test_a_root_that_cannot_be_listed_at_all_is_refused_as_a_root(
    error: type[OSError],
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its own refusal, because a pass over several roots accounts for each.

    Parameters:
        error: The exception type the root raises.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the walk reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    unlistable_root(monkeypatch, error)
    with pytest.raises(UnlistableRootError, match='could not be listed'):
        list(source.listing(Selection()))


def test_the_refusal_of_a_root_says_what_an_operator_should_check(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mistyped root is the commonest thing an operator types."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    unlistable_root(monkeypatch, FileNotFoundError)
    with pytest.raises(UnlistableRootError) as excinfo:
        list(source.listing(Selection()))
    assert 'check the spelling of the root' in str(excinfo.value)


def test_a_root_nobody_can_list_is_a_directory_nobody_can_list(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every consumer that does not account per root lets it end the run.

    A caller that catches the directory case and not this one would carry on
    past a root it read nothing from, which is the gap both refusals exist to
    close.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    unlistable_root(monkeypatch, PermissionError)
    with pytest.raises(UnlistableDirectoryError):
        list(source.listing(Selection()))


@pytest.mark.skipif(os.geteuid() == 0, reason='the superuser reads a directory of mode 000')
def test_a_directory_the_filesystem_will_not_open_stops_the_walk(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The same thing again, against a real directory rather than a stand-in."""
    root = two_volume_tree(tmp_path)
    closed = root / 'VOL2'
    closed.chmod(0o000)
    try:
        with pytest.raises(UnlistableDirectoryError, match='could not be listed'):
            list(tree_source(root, quiet_logger).listing(Selection()))
    finally:
        closed.chmod(0o755)


# ---------------------------------------------------------------------------
# A directory reached a second way
# ---------------------------------------------------------------------------


def _tree_that_links_back_to_its_own_root(tmp_path: Path) -> Path:
    """Write a results tree holding a link from a volume back to the root.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document())
    (root / 'VOL1' / 'up').symlink_to(root)
    return root


def test_a_directory_reached_a_second_way_is_listed_once(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Descending twice would report one document under a second set of stubs.

    Nothing stops such a walk except the filesystem's own limit on how many
    links it will follow, so one document under a tree that links back to itself
    becomes as many entries as the limit allows, each under a stub no consumer
    will ever ask about.
    """
    root = _tree_that_links_back_to_its_own_root(tmp_path)
    assert stubs_of(tree_source(root, quiet_logger).listing(Selection())) == [FIRST_STUB]


def test_a_directory_reached_a_second_way_does_not_stop_the_walk(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """It is not a gap: its documents are in the listing under the path met first."""
    root = _tree_that_links_back_to_its_own_root(tmp_path)
    assert len(list(tree_source(root, quiet_logger).listing(Selection()))) == 1


def test_a_directory_reached_a_second_way_is_said_to_have_been_declined(
    tmp_path: Path, loud_logger: pdslogger.PdsLogger, capsys: pytest.CaptureFixture[str]
) -> None:
    """A skip nobody said anything about reads exactly like a directory nobody saw."""
    root = _tree_that_links_back_to_its_own_root(tmp_path)
    list(tree_source(root, loud_logger).listing(Selection()))
    assert 'reached a second way and already listed' in capsys.readouterr().out


def test_a_volume_that_is_a_link_to_somewhere_else_is_listed(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A link the walk has not been down before is a directory like any other."""
    root = tmp_path / 'results'
    root.mkdir()
    elsewhere = tmp_path / 'elsewhere'
    write_document(elsewhere, 'N1454725799_1_CALIB', document())
    (root / 'VOL1').symlink_to(elsewhere)
    assert stubs_of(tree_source(root, quiet_logger).listing(Selection())) == [
        'VOL1/N1454725799_1_CALIB'
    ]


def test_a_cloud_directory_is_not_stat_ed_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recognizing a directory already walked costs a cloud root nothing.

    A cloud location has no links for a walk to go round in, and asking a bucket
    about a prefix is a round trip per directory per run. The identity is
    therefore taken only where a loop is possible, and the check that decides is
    a string test that reaches no backend.
    """

    def forbidden(self: FCPath, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError('the walk asked a cloud backend about a directory')

    monkeypatch.setattr(FCPath, 'stat', forbidden)
    assert walk_module._directory_identity(FCPath('gs://rms-nav/nav-offset-results')) is None


# ---------------------------------------------------------------------------
# Listing several directories at once
# ---------------------------------------------------------------------------


def _wide_tree(tmp_path: Path, volumes: int) -> tuple[Path, list[str]]:
    """Write a results tree of one document in each of several volumes.

    Parameters:
        tmp_path: Directory the tree lives under.
        volumes: How many volumes to write, each holding one document.

    Returns:
        The results root, and the stubs written under it in name order.
    """
    root = tmp_path / 'results'
    stubs = [f'VOL{index:02d}/N145472{index:04d}_1_CALIB' for index in range(volumes)]
    for stub in stubs:
        write_document(root, stub, document())
    return root, stubs


@pytest.mark.parametrize('at_once', [1, 2, 3, 32])
def test_a_tree_wider_than_one_round_is_still_listed_whole(
    at_once: int,
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
) -> None:
    """A round is bounded, so a wide tree takes several rounds to drain.

    A round that dropped the directories it could not fit, or one that stopped
    when the first round emptied, would report a short tree as a whole one,
    which is the one answer a listing must never give.

    Parameters:
        at_once: How many directories one round takes off the frontier.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the walk reports through.
    """
    root, stubs = _wide_tree(tmp_path, 7)
    source = tree_source(
        root, quiet_logger, TreeTuning(walk_threads=1, walk_directories_at_once=at_once)
    )
    assert sorted(stubs_of(source.listing(Selection()))) == sorted(stubs)


def test_no_directory_is_listed_twice_across_rounds(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory carried between rounds could otherwise be listed by each."""
    root, _ = _wide_tree(tmp_path, 7)
    listed: list[str] = []
    real_iterdir = FCPath.iterdir_metadata

    def recording(self: FCPath) -> Any:
        listed.append(self.as_posix())
        yield from real_iterdir(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', recording)
    source = tree_source(root, quiet_logger, TreeTuning(walk_threads=1, walk_directories_at_once=2))
    list(source.listing(Selection()))
    assert sorted(listed) == sorted(set(listed))


def test_a_refusal_names_the_same_directory_however_the_threads_finish(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Or an operator would be sent to a different directory on each run.

    Two directories of one round both refuse, and the one earlier in the
    frontier is made the slower of the two.  A walk reporting whichever thread
    raised first would name the other, and would name a different one whenever
    the timing came out differently.
    """
    root, _ = _wide_tree(tmp_path, 4)
    real_iterdir = FCPath.iterdir_metadata

    def refusing(self: FCPath) -> Any:
        if self.name == 'VOL00':
            time.sleep(0.2)
            raise PermissionError(self.as_posix())
        if self.name == 'VOL03':
            raise PermissionError(self.as_posix())
        # The root is listed in name order, so VOL00 is the earlier of the two
        # on the frontier whatever order the filesystem hands its entries back
        # in.
        yield from sorted(real_iterdir(self), key=lambda entry: entry[0].name)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', refusing)
    with pytest.raises(UnlistableDirectoryError) as excinfo:
        list(tree_source(root, quiet_logger).listing(Selection()))
    assert (root / 'VOL00').as_posix() in str(excinfo.value)


def test_the_walk_lists_a_round_of_directories_at_the_same_time(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which is the whole of what the parallel walk buys on a cloud root.

    A listing there is one round trip and no bandwidth, so a tree of them run
    one after another is latency and nothing else.  Overlap is what is under
    test, not a rate: the volumes are made slow to list and the assertion is
    that more than one was inside its listing at once.
    """
    root, _ = _wide_tree(tmp_path, 4)
    inside = 0
    most_at_once = 0
    guard = threading.Lock()
    real_iterdir = FCPath.iterdir_metadata

    def slow(self: FCPath) -> Any:
        nonlocal inside, most_at_once
        if self.name.startswith('VOL'):
            with guard:
                inside += 1
                most_at_once = max(most_at_once, inside)
            time.sleep(0.1)
            with guard:
                inside -= 1
        yield from real_iterdir(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', slow)
    list(tree_source(root, quiet_logger).listing(Selection()))
    assert most_at_once > 1


def test_a_round_of_one_directory_is_listed_without_a_pool(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tree one directory wide should not pay for a thread pool per round."""

    class Forbidden:
        """A pool the walk must not build for a round holding one directory."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError('the walk built a thread pool for one directory')

    monkeypatch.setattr('spindoctor.nav_records.walk.ThreadPoolExecutor', Forbidden)
    root, stubs = _wide_tree(tmp_path, 1)
    assert stubs_of(tree_source(root, quiet_logger).listing(Selection())) == stubs


def test_the_root_is_listed_before_any_pool_is_built(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which is what warms the backend source cache before threads reach it.

    A walk begins with one directory, so its first listing takes the unpooled
    path however wide the tree is.  filecache builds and caches its backend
    source there, in a module-level dict filled by an unlocked check-then-set,
    so a pool that ran first would have every thread build a client and all but
    one throw it away.
    """
    order: list[str] = []
    real_iterdir = FCPath.iterdir_metadata

    def recording(self: FCPath) -> Any:
        order.append(f'list {self.name}')
        yield from real_iterdir(self)

    class Recording(ThreadPoolExecutor):
        """A pool that says when it was built."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            order.append('pool')
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', recording)
    monkeypatch.setattr('spindoctor.nav_records.walk.ThreadPoolExecutor', Recording)
    root, _ = _wide_tree(tmp_path, 4)
    list(tree_source(root, quiet_logger).listing(Selection()))
    assert order[0] == f'list {root.name}'
    assert order[1] == 'pool'


# ---------------------------------------------------------------------------
# What a listing will not answer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('selection', 'named'),
    [
        pytest.param(Selection(instrument=MISSION), 'instrument', id='instrument'),
        pytest.param(Selection(start_et=0.0), 'start_et', id='start_et'),
        pytest.param(Selection(stop_et=1.0), 'stop_et', id='stop_et'),
    ],
)
def test_a_listing_refuses_a_restriction_it_cannot_honour(
    selection: Selection,
    named: str,
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
) -> None:
    """It opens no document, so it can answer nothing a document says.

    Ignoring the restriction would hand a caller a listing of the whole root as
    though it were a listing of the selection, which is a wrong answer rather
    than a missing feature.

    Parameters:
        selection: A selection carrying one restriction a listing cannot answer.
        named: The restriction the refusal has to name.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the walk reports through.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    with pytest.raises(ValueError, match=named):
        list(source.listing(selection))


def test_a_listing_refusal_names_every_restriction_it_was_given(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A refusal naming one of three would be corrected three times."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    selection = Selection(instrument=MISSION, start_et=0.0, stop_et=1.0)
    with pytest.raises(ValueError) as excinfo:
        list(source.listing(selection))
    assert 'instrument, start_et, stop_et' in str(excinfo.value)


def test_a_listing_refuses_before_it_walks_anything(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A usage error belongs where it was made, not partway through a caller's loop."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    with pytest.raises(ValueError, match='instrument'):
        source.listing(Selection(instrument=MISSION))


# ---------------------------------------------------------------------------
# Subtrees, and the roots a selection narrows to
# ---------------------------------------------------------------------------


def test_a_listing_restricted_to_a_subtree_holds_only_that_subtree(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The whole of what a run restricted to one top-level directory should see."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    assert stubs_of(source.listing(Selection(subtrees=('VOL1',)))) == [FIRST_STUB]


def test_a_subtree_the_root_does_not_hold_is_refused(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run restricted to a directory that is not there would report a clean pass over nothing."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    with pytest.raises(UnlistableDirectoryError, match='could not be listed'):
        list(source.listing(Selection(subtrees=('VOL9',))))


def test_two_roots_are_listed_in_the_order_the_source_holds_them(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A selection naming roots narrows; it does not reorder."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_document(first, 'VOL1/N1454725799_1_CALIB', document())
    write_document(second, 'VOL1/N1454725800_1_CALIB', document())
    source = TreeRecordSource([str(first), str(second)], logger=quiet_logger)
    selection = Selection(roots=(str(second), str(first)))
    assert stubs_of(source.listing(selection)) == [
        'VOL1/N1454725799_1_CALIB',
        'VOL1/N1454725800_1_CALIB',
    ]


def test_a_root_spelled_a_second_way_is_the_root_the_source_holds(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The whole reason a root is normalized before it is compared."""
    root = two_volume_tree(tmp_path)
    source = tree_source(root, quiet_logger)
    selection = Selection(roots=(f'{root.as_posix()}/',))
    assert sorted(stubs_of(source.listing(selection))) == [FIRST_STUB, SECOND_STUB]


def test_a_root_the_source_does_not_hold_is_refused(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Answering for a root nobody bound would read another run's tree."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    with pytest.raises(ValueError, match='does not hold'):
        list(source.listing(Selection(roots=('/data/elsewhere',))))


def test_the_refusal_of_an_unheld_root_names_what_is_held(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The reader has to be able to see which spelling they meant."""
    root = two_volume_tree(tmp_path)
    source = tree_source(root, quiet_logger)
    with pytest.raises(ValueError) as excinfo:
        list(source.listing(Selection(roots=('/data/elsewhere',))))
    assert root.as_posix() in str(excinfo.value)


def test_a_source_over_no_root_at_all_is_refused(quiet_logger: pdslogger.PdsLogger) -> None:
    """A source that held no root would answer every question with nothing."""
    with pytest.raises(ValueError, match='at least one results root'):
        TreeRecordSource([], logger=quiet_logger)
