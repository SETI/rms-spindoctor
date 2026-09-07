"""What the record-seam tests build their results trees out of.

A results tree is documents under a root, so the helpers here write one and hand
back the root.  Nothing here reaches a database: the half of the seam these tests
cover is the half that has none.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from filecache import FCPath
from filecache import file_cache as file_cache_module
from filecache.file_cache_source import FileCacheSourceFake

from spindoctor.nav_records import (
    METADATA_SUFFIX,
    NavRecord,
    TreeRecordSource,
    TreeTuning,
    UnreadableFile,
    read_document,
)
from spindoctor.nav_records import document as document_module

MISSION = 'coiss'
"""The instrument identity the mission-filtered reads below keep."""

OTHER_MISSION = 'vgiss'
"""An instrument identity of another mission's documents in the same tree."""


@pytest.fixture
def quiet_logger() -> pdslogger.PdsLogger:
    """Return a logger that discards everything written to it.

    Returns:
        The logger, for the tests that are not about what was logged.
    """
    return pdslogger.NullLogger()


@pytest.fixture
def loud_logger() -> pdslogger.PdsLogger:
    """Return a logger writing to standard output, which ``capsys`` captures.

    ``pdslogger`` writes through its own stream handler rather than through the
    logging module's, so what it writes is read back off the captured stream
    rather than out of ``caplog``.

    Returns:
        The logger, for the tests that are about what was logged.
    """
    return pdslogger.PdsLogger('nav_records_test', lognames=False)


def document(**overrides: Any) -> dict[str, Any]:
    """Build one navigation document, in the shape the navigator writes.

    Parameters:
        overrides: Top-level fields to replace or add.

    Returns:
        The document.
    """
    built: dict[str, Any] = {
        'status': 'success',
        'observation': {'instrument': MISSION, 'image_name': 'N1454725799_1.IMG'},
        'navigation_result': {'times': {'midtime_et': 100.0}},
    }
    built.update(overrides)
    return built


def timed_document(midtime: Any) -> dict[str, Any]:
    """Build a document recording one exposure midtime.

    Parameters:
        midtime: The value to record, of any type.

    Returns:
        The document.
    """
    return document(navigation_result={'times': {'midtime_et': midtime}})


def write_document(root: Path, stub: str, contents: dict[str, Any]) -> Path:
    """Write one navigation document under a results root.

    Parameters:
        root: The results root, which is created if it is not there.
        stub: The image's results path stub.
        contents: The document to write.

    Returns:
        The file that was written.
    """
    return write_text(root, stub, json.dumps(contents))


def write_text(root: Path, stub: str, text: str) -> Path:
    """Write one file where a navigation document belongs, whatever it holds.

    Parameters:
        root: The results root, which is created if it is not there.
        stub: The image's results path stub.
        text: What to write, which need not be a document at all.

    Returns:
        The file that was written.
    """
    path = root / f'{stub}{METADATA_SUFFIX}'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def stubs_of(found: Any) -> list[str]:
    """Return the stubs of everything a stream yielded, in the order it came.

    Parameters:
        found: What the stream yielded, each carrying a stub.

    Returns:
        The stubs.
    """
    return [entry.stub for entry in found]


FIRST_STUB = 'VOL1/N1454725799_1_CALIB'
"""The document of the first volume of the two-volume tree below."""

SECOND_STUB = 'VOL2/N1454725800_1_CALIB'
"""The document of its second volume."""

UNLISTABLE_ERRORS = [
    pytest.param(FileNotFoundError, id='not-there'),
    pytest.param(NotADirectoryError, id='not-a-directory'),
    pytest.param(PermissionError, id='this-user-may-not-read-it'),
    pytest.param(TimeoutError, id='the-share-stopped-answering'),
]
"""Every way a real tree refuses to list a directory.

They are one thing to the walk -- it can see no document there, which is not
evidence that there is none -- and enumerating only some of them lets the others
through as an empty directory.  A permission error is the commonest of all on a
shared tree.
"""


@dataclass(frozen=True)
class RemoteTree:
    """A results root the storage layer answers about as a remote one.

    Parameters:
        url: The root, as a source is given it.
        backing: The local directory the storage layer serves it out of, which
            is where a document is written to put it under the root.
    """

    url: str
    backing: Path


@pytest.fixture
def remote_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RemoteTree:
    """Return a results root that is not on the local filesystem.

    The storage layer serves a ``fake://`` URL out of a local directory while
    answering about it as a remote location, and a remote location is the whole
    of what the choice between listing named files and walking for them turns
    on: ``FCPath.is_local()`` is false for one.  So a source over this root takes
    every branch a cloud root takes, without a bucket, a network or a credential.

    Parameters:
        tmp_path: Directory the storage lives under.
        monkeypatch: Fixture the storage directory is installed through, which
            is process-wide and has to be put back.  So is the storage layer's
            memory of the backends it has built: one is built once per location
            and keeps the storage directory it was built with, so a second test
            in the same process would otherwise read the first one's tree and
            pass on documents it did not write.

    Returns:
        The root and the directory backing it.
    """
    storage = tmp_path / 'remote'
    storage.mkdir()
    monkeypatch.setattr(FileCacheSourceFake, '_DEFAULT_STORAGE_DIR', storage)
    monkeypatch.setattr(file_cache_module, '_SOURCE_CACHE', {})
    return RemoteTree(url='fake://bucket/results', backing=storage / 'bucket' / 'results')


@pytest.fixture
def second_remote_tree(remote_tree: RemoteTree) -> RemoteTree:
    """Return a second remote results root beside the first.

    One stub is a key under every root there is, so anything keyed on a stub
    alone answers about one root out of another's documents.  A second root
    holding the same stubs is what makes that visible.

    Parameters:
        remote_tree: The first root, whose storage directory this one shares.

    Returns:
        The root and the directory backing it.
    """
    return RemoteTree(
        url='fake://bucket/other-results',
        backing=remote_tree.backing.parent / 'other-results',
    )


def two_volume_tree(tmp_path: Path) -> Path:
    """Write a results tree with one document in each of two volumes.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document())
    write_document(root, SECOND_STUB, document())
    return root


def tree_source(
    root: Path, logger: pdslogger.PdsLogger, tuning: TreeTuning | None = None
) -> TreeRecordSource:
    """Build a source over one results root.

    Parameters:
        root: The results root.
        logger: Logger the walk reports a declined directory through.
        tuning: How much of a pass runs at once, or None for the defaults.
            A test wanting a small bound passes one rather than rebinding a
            module global, which is what a caller does.

    Returns:
        The source.
    """
    return TreeRecordSource([str(root)], logger=logger, tuning=tuning)


def _unlistable_directory(monkeypatch: pytest.MonkeyPatch, error: type[OSError], name: str) -> None:
    """Make one directory of a results tree refuse to be listed.

    Parameters:
        monkeypatch: Fixture the listing is wrapped through.
        error: The exception type that directory raises.
        name: The name of the directory that refuses.
    """
    real_iterdir = FCPath.iterdir_metadata

    def refusing(self: FCPath) -> Any:
        if self.name == name:
            raise error(self.as_posix())
        yield from real_iterdir(self)

    monkeypatch.setattr(FCPath, 'iterdir_metadata', refusing)


def unlistable_subdirectory(monkeypatch: pytest.MonkeyPatch, error: type[OSError]) -> None:
    """Make the second volume of a two-volume tree refuse to be listed.

    Parameters:
        monkeypatch: Fixture the listing is wrapped through.
        error: The exception type that directory raises.
    """
    _unlistable_directory(monkeypatch, error, 'VOL2')


def unlistable_root(monkeypatch: pytest.MonkeyPatch, error: type[OSError]) -> None:
    """Make the results root itself refuse to be listed.

    Parameters:
        monkeypatch: Fixture the listing is wrapped through.
        error: The exception type the root raises.
    """
    _unlistable_directory(monkeypatch, error, 'results')


def count_retrievals(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record how many files each batched retrieval was asked for.

    Only the calls naming several files are recorded.  A backend serving a batch
    reaches its own per-file retrieval underneath, and counting those would
    measure the backend rather than what this source asked it for.

    Parameters:
        monkeypatch: Fixture the retrieval is wrapped through.

    Returns:
        A list that grows by one entry per batched retrieval, holding its size.
    """
    calls: list[int] = []
    real_retrieve = FCPath.retrieve

    def counting(self: FCPath, sub_path: Any = None, **kwargs: Any) -> Any:
        if isinstance(sub_path, list):
            calls.append(len(sub_path))
        return real_retrieve(self, sub_path, **kwargs)

    monkeypatch.setattr(FCPath, 'retrieve', counting)
    return calls


def failing_retrievals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every retrieval report that it delivered nothing.

    Parameters:
        monkeypatch: Fixture the retrieval is wrapped through.
    """

    def never_delivers(self: FCPath, sub_path: Any = None, **kwargs: Any) -> Any:
        if not isinstance(sub_path, list):
            # A batch is what the stream asks for, but this patch covers every
            # retrieval the test reaches -- and reading one document reaches a
            # retrieval of itself, with no sub-path at all.  Refusing that the
            # way the storage layer refuses it keeps such a call a file that
            # never arrived rather than a fake that could not be called.
            raise FileNotFoundError(self.as_posix() if sub_path is None else str(sub_path))
        return [FileNotFoundError(str(one)) for one in sub_path]

    monkeypatch.setattr(FCPath, 'retrieve', never_delivers)


def count_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record which documents were opened, without changing what reading does.

    Parameters:
        monkeypatch: Fixture the reader is wrapped through.

    Returns:
        A list that grows by one entry per document read.
    """
    read: list[str] = []
    # Wrapped where the reading happens rather than where the walk asks for it,
    # which is the one function every reader of a document goes through.
    real_read = read_document

    def counting(path: FCPath) -> dict[str, Any]:
        read.append(path.as_posix())
        return real_read(path)

    monkeypatch.setattr(document_module, 'read_document', counting)
    return read


def reasons_of(found: list[NavRecord | UnreadableFile]) -> list[str]:
    """Return the reason of every unreadable file in a stream's output.

    Parameters:
        found: What the stream yielded.

    Returns:
        The reasons, in the order they arrived.
    """
    return [entry.reason for entry in found if isinstance(entry, UnreadableFile)]


def records_of(found: list[NavRecord | UnreadableFile]) -> list[NavRecord]:
    """Return every record in a stream's output.

    Parameters:
        found: What the stream yielded.

    Returns:
        The records, in the order they arrived.
    """
    return [entry for entry in found if isinstance(entry, NavRecord)]
