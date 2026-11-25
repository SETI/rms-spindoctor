"""What a listing of named documents answers, and which call it answers with.

A selection that names its stubs asks about those files and no others, which is
what a caller enumerating candidate images asks: which of the ones this run might
still keep has a document.  Nothing about the answer depends on how it was
reached, and everything about what it costs does, so the source picks the call
that is cheap on the root it is answering about -- a check per file where a check
is a syscall, a walk of the directories they lie in where it is a paid round
trip.

Both branches are exercised here against the same stated answers, and each is
held to using its own call and not the other's: a test that only ever ran the
local branch would pass for a source that had no other, and the branch nobody
can reach on a workstation is the one a cloud run takes every time.  The remote
branch is driven through a results root the storage layer answers about as a
remote location while serving it out of a directory, which is the whole of what
the choice turns on.

The two do not answer alike about everything, and the difference is stated
rather than papered over: only a walk reads a directory entry, so only a walk can
report an entry's size and modification time.
"""

from pathlib import Path
from typing import Any

import pdslogger
import pytest
from filecache import FCPath
from tests.packaged_sections import packaged_tree_tuning

from spindoctor.nav_records import (
    ListedRecord,
    Selection,
    TreeRecordSource,
    UnlistableDirectoryError,
    document_path,
)

from .conftest import (
    FIRST_STUB,
    SECOND_STUB,
    RemoteTree,
    document,
    stubs_of,
    tree_source,
    two_volume_tree,
    write_document,
)

BARE_STUB = 'N1454725801_1_CALIB'
"""A document directly under a results root, with no subtree above it.

The simulated dataset's scene basenames are these, and the only walk that
reaches one is a walk of the root itself.
"""

MISSING_STUB = 'VOL1/N1454725900_1_CALIB'
"""A stub under a directory the root holds, for which there is no document."""

MISSING_SUBTREE_STUB = 'VOL9/N1454725901_1_CALIB'
"""A stub under a directory the root does not hold at all."""


def remote_source(remote: RemoteTree, logger: pdslogger.PdsLogger) -> TreeRecordSource:
    """Build a source over one remote results root holding two volumes.

    Parameters:
        remote: The remote root and the directory backing it.
        logger: Logger the walk reports a declined directory through.

    Returns:
        The source, over a root already holding the two-volume tree's documents.
    """
    write_document(remote.backing, FIRST_STUB, document())
    write_document(remote.backing, SECOND_STUB, document())
    return TreeRecordSource([remote.url], logger=logger, tuning=packaged_tree_tuning())


def forbid_walking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any directory listing fail the test it happens in.

    Parameters:
        monkeypatch: Fixture the listing is wrapped through.
    """

    def forbidden(self: FCPath) -> Any:
        raise AssertionError(f'a listing of named documents walked {self.as_posix()}')

    monkeypatch.setattr(FCPath, 'iterdir_metadata', forbidden)


def forbid_checking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any per-file existence check fail the test it happens in.

    Parameters:
        monkeypatch: Fixture the check is wrapped through.
    """

    def forbidden(self: FCPath, sub_path: Any = None, **kwargs: Any) -> Any:
        raise AssertionError(f'a listing of named documents checked {sub_path!r}')

    monkeypatch.setattr(FCPath, 'exists', forbidden)


def count_listings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every directory a walk listed, without changing what listing does.

    Parameters:
        monkeypatch: Fixture the listing is wrapped through.

    Returns:
        A list that grows by one entry per directory listed.
    """
    listed: list[str] = []
    real_iterdir = FCPath.iterdir_metadata

    def counting(self: FCPath) -> Any:
        listed.append(self.as_posix())
        yield from real_iterdir(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', counting)
    return listed


def entries_of(found: Any) -> dict[str, ListedRecord]:
    """Return what a listing yielded, keyed by stub.

    Parameters:
        found: What the listing yielded.

    Returns:
        The entries.
    """
    return {entry.stub: entry for entry in found}


# ---------------------------------------------------------------------------
# The answer, which is the same from either branch
# ---------------------------------------------------------------------------


def test_a_local_listing_covers_the_stubs_it_named(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """One of the two volumes' documents, asked for by name.

    Parameters:
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the source reports through.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    assert stubs_of(source.listing(Selection(stubs=(FIRST_STUB,)))) == [FIRST_STUB]


def test_a_remote_listing_covers_the_stubs_it_named(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The same answer over a root that is not on this filesystem.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
    """
    source = remote_source(remote_tree, quiet_logger)
    assert stubs_of(source.listing(Selection(stubs=(FIRST_STUB,)))) == [FIRST_STUB]


def test_a_local_listing_leaves_out_a_stub_with_no_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Absence is the answer an enumeration reads as "nothing navigated this".

    Parameters:
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the source reports through.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    selection = Selection(stubs=(FIRST_STUB, MISSING_STUB))
    assert stubs_of(source.listing(selection)) == [FIRST_STUB]


def test_a_remote_listing_leaves_out_a_stub_with_no_document(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The same absence, from the walk.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
    """
    source = remote_source(remote_tree, quiet_logger)
    selection = Selection(stubs=(FIRST_STUB, MISSING_STUB))
    assert stubs_of(source.listing(selection)) == [FIRST_STUB]


def test_a_local_listing_answers_in_the_order_it_was_asked(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Naming images is asking about them in that order, which is what a task carries.

    Parameters:
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the source reports through.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    selection = Selection(stubs=(SECOND_STUB, FIRST_STUB))
    assert stubs_of(source.listing(selection)) == [SECOND_STUB, FIRST_STUB]


def test_a_remote_listing_answers_in_the_order_it_was_asked(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The walk finds them in whatever order the directories return them.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
    """
    source = remote_source(remote_tree, quiet_logger)
    selection = Selection(stubs=(SECOND_STUB, FIRST_STUB))
    assert stubs_of(source.listing(selection)) == [SECOND_STUB, FIRST_STUB]


def test_a_local_listing_names_the_document_it_found(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An entry names a file an operator can open, whichever call found it.

    Parameters:
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the source reports through.
    """
    root = two_volume_tree(tmp_path)
    source = tree_source(root, quiet_logger)
    found = entries_of(source.listing(Selection(stubs=(FIRST_STUB,))))
    assert found[FIRST_STUB].path == document_path(FCPath(str(root)), FIRST_STUB)


def test_a_remote_listing_names_the_document_it_found(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The same file, under the root as the storage layer spells it.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
    """
    source = remote_source(remote_tree, quiet_logger)
    found = entries_of(source.listing(Selection(stubs=(FIRST_STUB,))))
    assert found[FIRST_STUB].path == document_path(FCPath(remote_tree.url), FIRST_STUB)


def test_a_local_listing_of_a_stub_with_no_subtree_finds_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A document directly under the root is a document like any other.

    Parameters:
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the source reports through.
    """
    root = two_volume_tree(tmp_path)
    write_document(root, BARE_STUB, document())
    source = tree_source(root, quiet_logger)
    assert stubs_of(source.listing(Selection(stubs=(BARE_STUB,)))) == [BARE_STUB]


def test_a_remote_listing_of_a_stub_with_no_subtree_finds_it(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The only walk that reaches one is a walk of the root, and it is made.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
    """
    source = remote_source(remote_tree, quiet_logger)
    write_document(remote_tree.backing, BARE_STUB, document())
    assert stubs_of(source.listing(Selection(stubs=(BARE_STUB,)))) == [BARE_STUB]


def test_a_remote_walk_of_the_root_answers_about_a_volume_too(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A walk of the root covers every subtree, so nothing is walked twice for one.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
    """
    source = remote_source(remote_tree, quiet_logger)
    write_document(remote_tree.backing, BARE_STUB, document())
    selection = Selection(stubs=(BARE_STUB, FIRST_STUB))
    assert stubs_of(source.listing(selection)) == [BARE_STUB, FIRST_STUB]


# ---------------------------------------------------------------------------
# Which call each branch makes
# ---------------------------------------------------------------------------


def test_a_local_listing_of_named_stubs_lists_no_directory(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the local branch: ten of fifty thousand cost ten calls.

    A walk here would answer correctly and pay for a volume to say something
    about a handful of files in it, which is the cost this branch exists to
    avoid and which no assertion about the answer can see.

    Parameters:
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the source reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    forbid_walking(monkeypatch)
    assert stubs_of(source.listing(Selection(stubs=(FIRST_STUB,)))) == [FIRST_STUB]


def test_a_remote_listing_of_named_stubs_checks_no_file(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the remote branch: a check there is a paid round trip.

    One listing returns about a thousand entries with their metrics, so a check
    per file would answer correctly and pay a round trip per image of the scan.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
        monkeypatch: Fixture the check is wrapped through.
    """
    source = remote_source(remote_tree, quiet_logger)
    forbid_checking(monkeypatch)
    assert stubs_of(source.listing(Selection(stubs=(FIRST_STUB,)))) == [FIRST_STUB]


def test_one_remote_walk_answers_every_batch_of_a_scan(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scan asks in batches, so a walk per batch is a walk per batch of a volume.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    source = remote_source(remote_tree, quiet_logger)
    listed = count_listings(monkeypatch)
    list(source.listing(Selection(stubs=(FIRST_STUB,))))
    already = len(listed)
    list(source.listing(Selection(stubs=(FIRST_STUB,))))
    assert len(listed) == already


def test_a_batch_under_a_root_already_walked_walks_nothing_more(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A walk of the root covered every volume, so no volume is walked again for one.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    source = remote_source(remote_tree, quiet_logger)
    write_document(remote_tree.backing, BARE_STUB, document())
    listed = count_listings(monkeypatch)
    list(source.listing(Selection(stubs=(BARE_STUB,))))
    already = len(listed)
    assert stubs_of(source.listing(Selection(stubs=(FIRST_STUB,)))) == [FIRST_STUB]
    assert len(listed) == already


def test_a_second_batch_naming_another_volume_walks_that_volume(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What is remembered is what was walked, not that anything was.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    source = remote_source(remote_tree, quiet_logger)
    listed = count_listings(monkeypatch)
    list(source.listing(Selection(stubs=(FIRST_STUB,))))
    already = len(listed)
    assert stubs_of(source.listing(Selection(stubs=(SECOND_STUB,)))) == [SECOND_STUB]
    assert len(listed) > already


def test_a_remote_walk_covers_only_the_volumes_the_stubs_lie_in(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A walk of the whole root to answer about one volume is a walk of the archive.

    The volume the stub does lie in is asserted first, so that the absence of
    the other one is read off a recording that spells a walked directory the way
    this one looks for it.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    source = remote_source(remote_tree, quiet_logger)
    listed = count_listings(monkeypatch)
    list(source.listing(Selection(stubs=(FIRST_STUB,))))
    assert f'{remote_tree.url}/VOL1' in listed
    assert f'{remote_tree.url}/VOL2' not in listed


def test_closing_a_source_releases_what_its_walks_found(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A mission's walk held after the run that made it is a mission held in memory.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
    """
    with remote_source(remote_tree, quiet_logger) as source:
        list(source.listing(Selection(stubs=(FIRST_STUB,))))
    assert source._walked == {}


# ---------------------------------------------------------------------------
# What only a walk can report
# ---------------------------------------------------------------------------


def test_a_checked_entry_reports_no_modification_time(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A check says whether the file is there and nothing else about it.

    Parameters:
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the source reports through.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    found = entries_of(source.listing(Selection(stubs=(FIRST_STUB,))))
    assert found[FIRST_STUB].mtime_ns is None


def test_a_checked_entry_reports_no_size(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """The other half of the same absence.

    Parameters:
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the source reports through.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    found = entries_of(source.listing(Selection(stubs=(FIRST_STUB,))))
    assert found[FIRST_STUB].size_bytes is None


def test_a_checked_entry_says_it_has_no_metrics(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A consumer asks rather than reading a stand-in as the file's own.

    Parameters:
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the source reports through.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    found = entries_of(source.listing(Selection(stubs=(FIRST_STUB,))))
    assert found[FIRST_STUB].has_metrics is False


def test_a_walked_entry_reports_its_modification_time(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The walk read a directory entry, so it has what the entry said.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
    """
    source = remote_source(remote_tree, quiet_logger)
    found = entries_of(source.listing(Selection(stubs=(FIRST_STUB,))))
    assert found[FIRST_STUB].mtime_ns is not None


def test_a_walked_entry_reports_the_size_the_listing_gave(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The size a later pass compares to decide whether the document changed.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
    """
    source = remote_source(remote_tree, quiet_logger)
    found = entries_of(source.listing(Selection(stubs=(FIRST_STUB,))))
    written = remote_tree.backing / f'{FIRST_STUB}_metadata.json'
    assert found[FIRST_STUB].size_bytes == written.stat().st_size


def test_a_walked_entry_says_it_has_metrics(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Which is what tells a consumer it may skip a document that has not changed.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
    """
    source = remote_source(remote_tree, quiet_logger)
    found = entries_of(source.listing(Selection(stubs=(FIRST_STUB,))))
    assert found[FIRST_STUB].has_metrics is True


# ---------------------------------------------------------------------------
# A directory that is not there, and one that will not be read
# ---------------------------------------------------------------------------


def test_a_local_listing_of_a_stub_under_a_missing_directory_finds_nothing(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A volume nobody has navigated has no directory, and holds no documents.

    Parameters:
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the source reports through.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    assert stubs_of(source.listing(Selection(stubs=(MISSING_SUBTREE_STUB,)))) == []


def test_a_remote_listing_of_a_stub_under_a_missing_directory_finds_nothing(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The walk answers as the check does: the root holds no such file.

    A listing of a whole root refuses a directory it cannot list, because it
    infers absence from what it did not see.  A listing that named this file
    asked about the file, and its directory not being there answers about it.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
    """
    source = remote_source(remote_tree, quiet_logger)
    assert stubs_of(source.listing(Selection(stubs=(MISSING_SUBTREE_STUB,)))) == []


def test_a_remote_listing_absorbs_a_named_directory_the_root_does_not_hold(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend that refuses to list a directory that is not there says the same thing.

    Some backends report a directory the root does not hold as an empty listing
    and some refuse to list it, and the two must not answer a question about a
    named file differently.  Either way the root holds no such directory and
    therefore none of the documents under it, which is what a check of one of
    those files answers.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    source = remote_source(remote_tree, quiet_logger)
    real_iterdir = FCPath.iterdir_metadata

    def refusing(self: FCPath) -> Any:
        if self.name == 'VOL9':
            raise FileNotFoundError(self.as_posix())
        yield from real_iterdir(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', refusing)
    assert stubs_of(source.listing(Selection(stubs=(MISSING_SUBTREE_STUB,)))) == []


def test_a_remote_listing_still_refuses_a_directory_that_will_not_be_read(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory nobody read may hold the document, so its absence is not an answer.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    source = remote_source(remote_tree, quiet_logger)
    real_iterdir = FCPath.iterdir_metadata

    def refusing(self: FCPath) -> Any:
        if self.name == 'VOL1':
            raise PermissionError(self.as_posix())
        yield from real_iterdir(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', refusing)
    with pytest.raises(UnlistableDirectoryError, match='could not be listed'):
        list(source.listing(Selection(stubs=(FIRST_STUB,))))


def test_a_remote_listing_refuses_a_directory_below_the_one_it_walked(
    remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory that is not there answers only where it is the one asked about.

    One further down that vanished is a gap in what the walk saw of a directory
    it did list, and reading the stubs it did find as the whole of that
    directory is exactly the wrong answer the walk refuses to give.

    Parameters:
        remote_tree: The remote root and the directory backing it.
        quiet_logger: Logger the source reports through.
        monkeypatch: Fixture the listing is wrapped through.
    """
    source = remote_source(remote_tree, quiet_logger)
    write_document(remote_tree.backing, 'VOL1/deeper/N1454725902_1_CALIB', document())
    real_iterdir = FCPath.iterdir_metadata

    def refusing(self: FCPath) -> Any:
        if self.name == 'deeper':
            raise FileNotFoundError(self.as_posix())
        yield from real_iterdir(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', refusing)
    with pytest.raises(UnlistableDirectoryError, match='could not be listed'):
        list(source.listing(Selection(stubs=(FIRST_STUB,))))


# ---------------------------------------------------------------------------
# One stub is a key under every root there is
# ---------------------------------------------------------------------------


def test_a_remote_listing_answers_out_of_the_root_it_was_asked_about(
    remote_tree: RemoteTree, second_remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A walk remembered without its root would answer one root out of another.

    Parameters:
        remote_tree: The root under test.
        second_remote_tree: A second root holding a stub the first does not.
        quiet_logger: Logger the source reports through.
    """
    write_document(remote_tree.backing, FIRST_STUB, document())
    write_document(second_remote_tree.backing, FIRST_STUB, document())
    write_document(second_remote_tree.backing, MISSING_STUB, document())
    source = TreeRecordSource(
        [remote_tree.url, second_remote_tree.url],
        logger=quiet_logger,
        tuning=packaged_tree_tuning(),
    )
    list(source.listing(Selection(roots=(second_remote_tree.url,), stubs=(MISSING_STUB,))))
    selection = Selection(roots=(remote_tree.url,), stubs=(FIRST_STUB, MISSING_STUB))
    assert stubs_of(source.listing(selection)) == [FIRST_STUB]


def test_a_remote_listing_of_the_second_root_answers_for_that_root(
    remote_tree: RemoteTree, second_remote_tree: RemoteTree, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The other side of the same key: the first root's walk is not this one's answer.

    Parameters:
        remote_tree: The first root.
        second_remote_tree: The root under test.
        quiet_logger: Logger the source reports through.
    """
    write_document(remote_tree.backing, FIRST_STUB, document())
    write_document(second_remote_tree.backing, FIRST_STUB, document())
    write_document(second_remote_tree.backing, MISSING_STUB, document())
    source = TreeRecordSource(
        [remote_tree.url, second_remote_tree.url],
        logger=quiet_logger,
        tuning=packaged_tree_tuning(),
    )
    list(source.listing(Selection(roots=(remote_tree.url,), stubs=(FIRST_STUB,))))
    selection = Selection(roots=(second_remote_tree.url,), stubs=(FIRST_STUB, MISSING_STUB))
    assert stubs_of(source.listing(selection)) == [FIRST_STUB, MISSING_STUB]


def test_a_listing_of_stubs_under_two_roots_is_refused(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A stub is a key under a root, so a selection of keys names one root.

    Parameters:
        tmp_path: Directory the trees live under.
        quiet_logger: Logger the source reports through.
    """
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_document(first, FIRST_STUB, document())
    write_document(second, FIRST_STUB, document())
    source = TreeRecordSource(
        [str(first), str(second)], logger=quiet_logger, tuning=packaged_tree_tuning()
    )
    with pytest.raises(ValueError, match='under one root'):
        list(source.listing(Selection(stubs=(FIRST_STUB,))))
