"""What the record source reads, in what batches, and what it will not read.

One stub names one document.  A stream reads what the listing found, in batches,
and hands back one record at a time -- batched underneath because on a cloud root
each file is a round trip, lazy on top because the caller should not have to hold
a mission to read the first record of one.

A file that yielded no record is reported into the stream rather than raised on:
it names no image, so there is nothing for a run to omit and nothing an omission
reason could be recorded against.  What is passed over silently is what is simply
not this run's business -- another mission's document, and a record outside the
selection's span.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdslogger
import pytest
from filecache import FCPath
from tests.conftest import child_interpreter_environment

from spindoctor.nav_records import (
    METADATA_SUFFIX,
    NAMES_NO_INSTRUMENT,
    RECORDS_NO_MIDTIME,
    Selection,
    TreeRecordSource,
    TreeTuning,
    UnreadableFile,
    read_document,
)
from spindoctor.nav_records import document as document_module

from .conftest import (
    FIRST_STUB,
    MISSION,
    OTHER_MISSION,
    SECOND_STUB,
    count_reads,
    count_retrievals,
    document,
    failing_retrievals,
    reasons_of,
    records_of,
    stubs_of,
    timed_document,
    tree_source,
    two_volume_tree,
    unlistable_subdirectory,
    write_document,
    write_text,
)

_BATCH = TreeTuning().retrieve_batch_size
"""The default retrieval batch, which these fixtures are sized against.

Derived rather than written out, so raising the default moves the tests
with it instead of leaving them quietly inside one batch.
"""

# ---------------------------------------------------------------------------
# One image by its stub
# ---------------------------------------------------------------------------


def test_one_stub_reads_its_own_document(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """The per-image shape: one stub, one file read, one record."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    assert source.record(FIRST_STUB).metadata['status'] == 'success'


def test_one_stub_carries_its_own_identity_back(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A record is paired with the stub it answers for, whatever the document says."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    assert source.record(FIRST_STUB).stub == FIRST_STUB


def test_a_stub_with_no_document_raises_the_error_a_caller_distinguishes(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An unnavigated image is told apart from an unreadable document."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    with pytest.raises(FileNotFoundError):
        source.record('VOL1/N0000000000_1_CALIB')


def test_a_stub_that_escapes_its_root_reads_as_an_image_with_no_record(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A key is not a path, and a key holding ``..`` names a file outside the root."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    with pytest.raises(FileNotFoundError, match='does not name a navigation document'):
        source.record('../../elsewhere/N1454725799')


def test_a_stub_that_escapes_its_root_is_told_which_rule_refused_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Two things mean "no readable record here", and the message says which."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    with pytest.raises(FileNotFoundError) as excinfo:
        source.record('../../elsewhere/N1454725799')
    assert 'names a parent directory' in str(excinfo.value)


def test_a_stub_that_resolves_back_inside_its_root_is_refused_too(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """It names a document this root does hold, under a key no consumer stores.

    A stub is the identity of an image, so two spellings of one document are two
    identities: read under this one, the record comes back where an index keyed
    on the canonical spelling says the image was never navigated.
    """
    root = tmp_path / 'results'
    write_document(root, 'BARESCENE', document())
    source = tree_source(root, quiet_logger)
    with pytest.raises(FileNotFoundError, match='names a parent directory'):
        source.record('VOL1/../BARESCENE')


def test_a_stub_spelled_as_an_absolute_path_reads_no_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Joining an absolute fragment discards the root, so it would read anything."""
    planted = tmp_path / 'elsewhere'
    write_document(planted, 'STOLEN', document())
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    stub = (planted / 'STOLEN').as_posix()
    with pytest.raises(FileNotFoundError, match='fragment is absolute'):
        source.record(stub)


def test_a_document_that_is_not_a_json_object_refuses_the_record(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Reading a field off an array would fail later and further away."""
    root = tmp_path / 'results'
    write_text(root, FIRST_STUB, '[1, 2]')
    with pytest.raises(ValueError, match='not a JSON object'):
        tree_source(root, quiet_logger).record(FIRST_STUB)


def test_one_stub_is_refused_by_a_source_holding_more_than_one_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A stub is a key under a root, and says nothing about which root."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_document(first, FIRST_STUB, document())
    write_document(second, FIRST_STUB, document())
    source = TreeRecordSource([str(first), str(second)], logger=quiet_logger)
    with pytest.raises(ValueError, match='key under one root'):
        source.record(FIRST_STUB)


def test_the_refusal_of_a_bare_stub_names_every_root_the_source_holds(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Naming one of two would leave the reader unable to see the ambiguity."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_document(first, FIRST_STUB, document())
    write_document(second, FIRST_STUB, document())
    source = TreeRecordSource([str(first), str(second)], logger=quiet_logger)
    with pytest.raises(ValueError) as excinfo:
        source.record(FIRST_STUB)
    assert first.as_posix() in str(excinfo.value)
    assert second.as_posix() in str(excinfo.value)


# ---------------------------------------------------------------------------
# The stream, and what it does in batches
# ---------------------------------------------------------------------------


def _many_documents(tmp_path: Path, count: int) -> Path:
    """Write a results tree holding more documents than one retrieval batch.

    Parameters:
        tmp_path: Directory the tree lives under.
        count: How many documents to write.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    for index in range(count):
        write_document(root, f'VOL1/N{index:010d}_1_CALIB', document())
    return root


def test_every_record_of_a_tree_larger_than_one_batch_comes_back(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Batching must not lose the tail of a batch or the last part-batch."""
    count = _BATCH * 3 + 1
    source = tree_source(_many_documents(tmp_path, count), quiet_logger)
    assert len(list(source.records(Selection()))) == count


def test_every_record_of_a_tree_larger_than_one_batch_is_distinct(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A count alone would pass on a batch handed back twice."""
    count = _BATCH * 3 + 1
    source = tree_source(_many_documents(tmp_path, count), quiet_logger)
    assert len(set(stubs_of(source.records(Selection())))) == count


def test_a_tree_larger_than_one_batch_is_retrieved_in_batches(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One retrieval per batch, not one per file: that is what a cloud root pays."""
    count = _BATCH * 3 + 1
    source = tree_source(_many_documents(tmp_path, count), quiet_logger)
    calls = count_retrievals(monkeypatch)
    list(source.records(Selection()))
    assert calls == [_BATCH, _BATCH, _BATCH, 1]


def test_the_first_record_of_a_stream_costs_one_document_read(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stream is lazy on top, so a caller does not have to hold a mission."""
    source = tree_source(_many_documents(tmp_path, _BATCH * 3), quiet_logger)
    read = count_reads(monkeypatch)
    stream = source.records(Selection())
    next(stream)
    assert len(read) == 1


def test_the_first_record_of_a_stream_costs_one_retrieval(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batched underneath: one batch is fetched, not the whole tree and not one file."""
    source = tree_source(_many_documents(tmp_path, _BATCH * 3), quiet_logger)
    calls = count_retrievals(monkeypatch)
    stream = source.records(Selection())
    next(stream)
    assert calls == [_BATCH]


def test_a_stream_of_named_stubs_reads_exactly_those(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """What a queue task carries: a worker reads the files it was handed."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    found = list(source.records(Selection(stubs=(SECOND_STUB,))))
    assert stubs_of(found) == [SECOND_STUB]


def test_a_stream_of_named_stubs_reads_them_in_the_order_they_were_named(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """They name images outright rather than narrowing to them."""
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    found = list(source.records(Selection(stubs=(SECOND_STUB, FIRST_STUB))))
    assert stubs_of(found) == [SECOND_STUB, FIRST_STUB]


def test_a_stream_of_named_stubs_lists_nothing(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker holding its own list must not pay for a walk of the root."""
    root = two_volume_tree(tmp_path)
    source = tree_source(root, quiet_logger)
    unlistable_subdirectory(monkeypatch, PermissionError)
    assert stubs_of(source.records(Selection(stubs=(FIRST_STUB,)))) == [FIRST_STUB]


def test_named_stubs_are_refused_by_a_source_holding_more_than_one_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A stub is a key under a root, however many of them are named at once."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_document(first, FIRST_STUB, document())
    write_document(second, FIRST_STUB, document())
    source = TreeRecordSource([str(first), str(second)], logger=quiet_logger)
    with pytest.raises(ValueError, match='under one root'):
        list(source.records(Selection(stubs=(FIRST_STUB,))))


def test_named_stubs_are_read_when_the_selection_narrows_to_one_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Narrowing a two-root source to one is what makes a bare stub a key again."""
    first = tmp_path / 'first'
    second = tmp_path / 'second'
    write_document(first, FIRST_STUB, document())
    write_document(second, FIRST_STUB, document())
    source = TreeRecordSource([str(first), str(second)], logger=quiet_logger)
    selection = Selection(roots=(str(second),), stubs=(FIRST_STUB,))
    assert stubs_of(source.records(selection)) == [FIRST_STUB]


# ---------------------------------------------------------------------------
# A file that yielded no record
# ---------------------------------------------------------------------------


def test_a_file_that_never_arrived_is_reported_rather_than_raised(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One file that would not download must not cost the rest of the pass.

    Two reasons that read alike say nothing about which files they came from,
    so the files are named as well: one file reported twice and the other lost
    is the same two reasons.
    """
    source = tree_source(two_volume_tree(tmp_path), quiet_logger)
    failing_retrievals(monkeypatch)
    found = list(source.records(Selection()))
    assert reasons_of(found) == ['could not be retrieved', 'could not be retrieved']
    assert sorted(stubs_of(found)) == sorted([FIRST_STUB, SECOND_STUB])


def test_one_named_record_whose_file_never_arrives_raises_rather_than_returning(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stream reports such a file, and a call asked for one record has nothing to report with.

    This reaches the storage layer the other way round from the stream above --
    one file asked for by itself rather than a batch asked of the root -- and it
    raises the error a caller distinguishes an unnavigated image by.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document())
    source = tree_source(root, quiet_logger)
    failing_retrievals(monkeypatch)
    with pytest.raises(FileNotFoundError, match=FIRST_STUB):
        source.record(FIRST_STUB)


NON_ASCII_TEXT = 'phase angle 170\u00b0'
"""One document's text, holding a character no ASCII reader can spell.

Written as an escape so that this file itself is ASCII, and read back through
the seam to say that the character came out of the document rather than out of
the machine that read it.
"""

FOREIGN_ENCODING_ENVIRONMENT = {
    'LC_ALL': 'C',
    'PYTHONUTF8': '0',
    'PYTHONCOERCECLOCALE': '0',
}
"""What gives a child interpreter a preferred encoding that is not UTF-8.

Naming the C locale is not enough on its own: an interpreter told to use it
coerces it to a UTF-8 locale and turns on its own UTF-8 mode, so both of those
are turned off as well.  What is left is the ASCII that locale names, which is
the one encoding other than UTF-8 every machine this suite runs on can prefer:
the Western European encodings, which decode a document's bytes into some other
document rather than refusing them, need locales a machine has only if somebody
generated them.
"""

FOREIGN_ENCODING_PROBE = """
import codecs
import json
import locale
import sys

import pdslogger

from spindoctor.nav_records import NavRecord, Selection, TreeRecordSource, UnreadableFile

found = list(TreeRecordSource([sys.argv[1]], logger=pdslogger.NullLogger()).records(Selection()))
print(
    json.dumps(
        {
            'encoding': codecs.lookup(locale.getpreferredencoding(False)).name,
            'reasons': [one.reason for one in found if isinstance(one, UnreadableFile)],
            'texts': [
                one.metadata['status_reason'] for one in found if isinstance(one, NavRecord)
            ],
        }
    )
)
"""
"""Read a results tree in a fresh interpreter and report what came out of it."""

PROBE_TIMEOUT_S = 120.0
"""A probe that hangs would otherwise stall the run with no failure to read."""


@dataclass(frozen=True)
class ForeignEncodingRead:
    """What a reader running under a foreign preferred encoding made of one tree.

    Parameters:
        encoding: The preferred encoding that reader ran under, under the name
            its codec is registered by rather than the one the machine spells
            it: one encoding has several spellings and they differ by platform.
        reasons: Why each file it would not read is not a record.
        texts: The one text field of each record it read.
    """

    encoding: str
    reasons: list[str]
    texts: list[str]


@pytest.fixture(scope='module')
def read_under_a_foreign_encoding(
    tmp_path_factory: pytest.TempPathFactory,
) -> ForeignEncodingRead:
    """Read a tree of two files in an interpreter whose preferred encoding is not UTF-8.

    The reader is a real subprocess because a preferred encoding is settled
    when an interpreter starts and cannot be changed underneath one that is
    already running.  Its environment names this checkout, so it answers for
    this code rather than for an installed copy.

    The tree holds a document whose bytes are UTF-8 and a file whose bytes are
    not text at all, which are the two answers this reader has to get right
    whatever machine it runs on.

    Parameters:
        tmp_path_factory: Fixture the tree is written under, at module scope so
            that one subprocess answers for every assertion about it.

    Returns:
        What the reader reported.
    """
    root = tmp_path_factory.mktemp('foreign_encoding') / 'results'
    write_text(
        root, FIRST_STUB, json.dumps(document(status_reason=NON_ASCII_TEXT), ensure_ascii=False)
    )
    not_text = root / f'{SECOND_STUB}{METADATA_SUFFIX}'
    not_text.parent.mkdir(parents=True, exist_ok=True)
    not_text.write_bytes(b'\xff\xfe\x00\x01')
    completed = subprocess.run(
        [sys.executable, '-c', FOREIGN_ENCODING_PROBE, str(root)],
        capture_output=True,
        text=True,
        check=False,
        timeout=PROBE_TIMEOUT_S,
        env={**child_interpreter_environment(), **FOREIGN_ENCODING_ENVIRONMENT},
    )
    # check=False so a failing probe reports its own stderr instead of a bare
    # CalledProcessError that hides why the reader stopped.
    assert completed.returncode == 0, completed.stderr
    return ForeignEncodingRead(**json.loads(completed.stdout))


def test_the_probe_ran_under_an_encoding_that_is_not_utf_8(
    read_under_a_foreign_encoding: ForeignEncodingRead,
) -> None:
    """The claims below are about a reader whose machine prefers another encoding.

    An interpreter that settled on UTF-8 after all would make both of them pass
    for the reason every reader on this machine passes them, which is the
    machine's answer rather than this code's.
    """
    assert read_under_a_foreign_encoding.encoding == 'ascii'


def test_a_file_whose_bytes_will_not_read_as_text_is_reported(
    read_under_a_foreign_encoding: ForeignEncodingRead,
) -> None:
    """It names no image, so there is nothing for a run to omit and nothing to raise."""
    assert read_under_a_foreign_encoding.reasons == ['unreadable']


def test_a_document_says_the_same_thing_to_a_reader_of_any_encoding(
    read_under_a_foreign_encoding: ForeignEncodingRead,
) -> None:
    """A document is JSON and JSON is UTF-8, so what it holds is not the reader's to decide.

    A reader that decodes by what its machine prefers reads another document out
    of these bytes or refuses them, and the ingest reading the same tree beside
    it decodes as UTF-8 and reads this one -- so the two halves of the seam would
    disagree about what the tree holds and about which of its files are
    documents at all.
    """
    assert read_under_a_foreign_encoding.texts == [NON_ASCII_TEXT]


def test_a_file_that_is_not_valid_json_is_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A results tree holds files that are not per-image navigation documents."""
    root = tmp_path / 'results'
    write_text(root, FIRST_STUB, '{not json')
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert reasons_of(found) == ['not valid JSON']


NO_VALUE_CAME_OUT = [
    pytest.param(RecursionError, id='the-decoder-gave-up-on-the-nesting'),
    pytest.param(MemoryError, id='the-decoder-ran-out-of-memory'),
]
"""Every way a decoder ends with no value and no decoding error to report it by.

Both are driven through the reader rather than provoked with a document,
because how much nesting a decoder will follow and how much memory it will ask
for are that decoder's own business: one interpreter recurses once per level of
nesting and gives up part way down, another does not recurse at all and parses
the same file to a value.  A document written to trigger either one measures
the interpreter that reads it, where what is under test here is what this code
does once the reader has raised.
"""


def _reader_failing_on(
    monkeypatch: pytest.MonkeyPatch, failure: type[BaseException], stub: str
) -> None:
    """Make the document reader raise on one stub's document and read every other.

    Parameters:
        monkeypatch: Fixture the reader is replaced through.
        failure: What reading that one document raises.
        stub: Stub of the document that will not read.
    """
    # Replaced where the reading happens rather than where the walk asks for
    # it, which is the one function every reader of a document goes through.
    real_read = read_document

    def failing(path: FCPath) -> dict[str, Any]:
        if path.as_posix().endswith(f'{stub}{METADATA_SUFFIX}'):
            raise failure('no value came out of it')
        return real_read(path)

    monkeypatch.setattr(document_module, 'read_document', failing)


@pytest.mark.parametrize('failure', NO_VALUE_CAME_OUT)
def test_a_reader_that_produces_no_value_reports_the_file_it_was_reading(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    """The decoder fails in more ways than one, and only one of them is a decoding error.

    What happened in all of them is the thing the reason states: no value came
    out of the file, and the file it names is the one that would not read: a
    reason on its own reads the same when the document beside it is the one
    reported, which is a pass that names a tree's good document as its bad one.
    """
    root = two_volume_tree(tmp_path)
    _reader_failing_on(monkeypatch, failure, FIRST_STUB)
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert reasons_of(found) == ['not valid JSON']
    assert [entry.stub for entry in found if isinstance(entry, UnreadableFile)] == [FIRST_STUB]


@pytest.mark.parametrize('failure', NO_VALUE_CAME_OUT)
def test_a_reader_that_produces_no_value_does_not_end_the_pass(
    tmp_path: Path,
    quiet_logger: pdslogger.PdsLogger,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    """The whole point of naming them: one file the decoder gave up on costs only itself."""
    root = two_volume_tree(tmp_path)
    _reader_failing_on(monkeypatch, failure, FIRST_STUB)
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert stubs_of(records_of(found)) == [SECOND_STUB]


def test_a_fault_in_the_reader_itself_is_not_reported_as_a_bad_document(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed document is a property of the file; a broken reader is not.

    Reported as a bad document it would say every file in the tree was
    malformed, and an operator would go looking at a tree that is fine.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document())

    def broken(path: FCPath) -> dict[str, Any]:
        raise TypeError('the reader was changed and no longer works')

    monkeypatch.setattr(document_module, 'read_document', broken)
    with pytest.raises(TypeError, match='no longer works'):
        list(tree_source(root, quiet_logger).records(Selection()))


def test_a_file_holding_json_that_is_not_an_object_is_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Valid JSON that is not a document is unreadable for the same reason."""
    root = tmp_path / 'results'
    write_text(root, FIRST_STUB, '[1, 2]')
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert reasons_of(found) == ['not a JSON object']


def test_an_unreadable_file_still_carries_the_stub_it_would_have_answered_for(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A stub comes from where the file is rather than from what it says."""
    root = tmp_path / 'results'
    write_text(root, FIRST_STUB, '{not json')
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert stubs_of(found) == [FIRST_STUB]


def test_one_unreadable_file_does_not_cost_the_records_beside_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The whole reason it is a value in the stream rather than an exception."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document())
    write_text(root, SECOND_STUB, '{not json')
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert stubs_of(records_of(found)) == [FIRST_STUB]


# ---------------------------------------------------------------------------
# One mission of several
# ---------------------------------------------------------------------------


def _tree_of_two_missions(tmp_path: Path) -> Path:
    """Write a results tree holding this mission's document, another's, and a mute one.

    Parameters:
        tmp_path: Directory the tree lives under.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    write_document(root, 'VOL1/A_mine', document())
    write_document(root, 'VOL1/B_theirs', document(observation={'instrument': OTHER_MISSION}))
    write_document(root, 'VOL1/C_mute', document(observation={'image_name': 'N1.IMG'}))
    return root


def test_only_this_missions_document_becomes_a_record(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run is per mission, and another mission's images are not its business."""
    source = tree_source(_tree_of_two_missions(tmp_path), quiet_logger)
    found = list(source.records(Selection(instrument=MISSION)))
    assert stubs_of(records_of(found)) == ['VOL1/A_mine']


def test_a_document_of_another_mission_is_passed_over_silently(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """It is passed over rather than reported: nothing about it went wrong.

    Reporting it would put a line about every image of every other mission into
    a run log, and would make a per-mission pass over a shared tree read as a
    pass that could not read most of what it found.
    """
    source = tree_source(_tree_of_two_missions(tmp_path), quiet_logger)
    found = list(source.records(Selection(instrument=MISSION)))
    assert 'VOL1/B_theirs' not in stubs_of(found)


def test_a_document_naming_no_instrument_is_reported_rather_than_passed_over(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Only a document that names a mission can be another mission's.

    One with no readable instrument is unreadable, not foreign: skipping it
    silently would let a truncated document vanish from every mission's run
    without a trace.
    """
    source = tree_source(_tree_of_two_missions(tmp_path), quiet_logger)
    found = list(source.records(Selection(instrument=MISSION)))
    assert reasons_of(found) == ['names no instrument to attribute it to a mission']


def test_the_mute_document_is_the_one_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A count of one refusal would pass if the wrong file were the one refused."""
    source = tree_source(_tree_of_two_missions(tmp_path), quiet_logger)
    found = list(source.records(Selection(instrument=MISSION)))
    assert [entry.stub for entry in found if isinstance(entry, UnreadableFile)] == ['VOL1/C_mute']


@pytest.mark.parametrize(
    'observation',
    [
        pytest.param(None, id='no-observation'),
        pytest.param('later', id='observation-not-a-block'),
        pytest.param({'image_name': 'A_CALIB'}, id='no-instrument'),
        pytest.param({'instrument': None}, id='instrument-null'),
    ],
)
def test_every_way_a_document_can_name_no_instrument_is_reported(
    observation: Any, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Enumerating only some of them lets the others through as another mission's.

    Parameters:
        observation: What the document records where its observation belongs.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the walk reports through.
    """
    root = tmp_path / 'results'
    contents = document(status='error')
    if observation is None:
        del contents['observation']
    else:
        contents['observation'] = observation
    write_document(root, FIRST_STUB, contents)
    found = list(tree_source(root, quiet_logger).records(Selection(instrument=MISSION)))
    assert reasons_of(found) == ['names no instrument to attribute it to a mission']


def test_a_document_naming_no_instrument_is_a_record_when_no_mission_is_asked_for(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A run reading every mission has nothing to attribute it to and nothing to refuse."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document(observation={'image_name': 'N1.IMG'}))
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert stubs_of(records_of(found)) == [FIRST_STUB]


# ---------------------------------------------------------------------------
# One span of time
# ---------------------------------------------------------------------------


def test_a_record_inside_the_range_is_kept(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The ordinary case the exclusions below are exceptions to."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(0.5))
    found = list(tree_source(root, quiet_logger).records(Selection(start_et=0.0, stop_et=1.0)))
    assert stubs_of(found) == [FIRST_STUB]


@pytest.mark.parametrize('midtime', [pytest.param(1.0, id='start'), pytest.param(3.0, id='stop')])
def test_a_record_at_the_range_edge_is_kept(
    midtime: float, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Both bounds are inclusive, so an exposure exactly on one is inside.

    Parameters:
        midtime: The midtime to record, which is one of the two bounds.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the walk reports through.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(midtime))
    found = list(tree_source(root, quiet_logger).records(Selection(start_et=1.0, stop_et=3.0)))
    assert stubs_of(records_of(found)) == [FIRST_STUB]


@pytest.mark.parametrize('midtime', [pytest.param(0.5, id='before'), pytest.param(3.5, id='after')])
def test_a_record_outside_the_range_is_passed_over(
    midtime: float, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """It is not this run's business, so it is not this run's refusal either.

    Parameters:
        midtime: The midtime to record, which is outside the range.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the walk reports through.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(midtime))
    found = list(tree_source(root, quiet_logger).records(Selection(start_et=1.0, stop_et=3.0)))
    assert found == []


@pytest.mark.parametrize(
    'midtime',
    [
        pytest.param(float('nan'), id='nan'),
        pytest.param(float('inf'), id='inf'),
        pytest.param(float('-inf'), id='minus-inf'),
        pytest.param(None, id='null'),
        pytest.param(True, id='boolean'),
        pytest.param('later', id='text'),
    ],
)
def test_a_record_with_an_unusable_midtime_is_reported_rather_than_passed_over(
    midtime: Any, tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Only a document that can be placed in time can be shown to be outside a span.

    A NaN would otherwise fall inside every range at once, and every one of
    these would otherwise vanish from every time-bounded run without a trace --
    which is exactly what the mission filter refuses to do with a document that
    names no mission.

    Parameters:
        midtime: The value to record, of any type.
        tmp_path: Directory the tree lives under.
        quiet_logger: Logger the walk reports through.
    """
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(midtime))
    found = list(tree_source(root, quiet_logger).records(Selection(start_et=0.0, stop_et=1.0)))
    assert reasons_of(found) == [RECORDS_NO_MIDTIME]


def test_an_unplaceable_record_is_reported_the_way_a_mission_less_one_is(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The two filters are one rule, and a run reading both must see both shortfalls."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, document(observation={'image_name': 'N1.IMG'}))
    write_document(root, SECOND_STUB, timed_document(None))
    selection = Selection(instrument=MISSION, start_et=0.0, stop_et=1.0)
    found = list(tree_source(root, quiet_logger).records(selection))
    assert sorted(reasons_of(found)) == sorted([NAMES_NO_INSTRUMENT, RECORDS_NO_MIDTIME])


def test_a_record_with_an_unusable_midtime_is_kept_when_no_bound_is_given(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """With no range to satisfy there is nothing to place the image against."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(float('nan')))
    found = list(tree_source(root, quiet_logger).records(Selection()))
    assert stubs_of(records_of(found)) == [FIRST_STUB]


def test_a_lower_bound_alone_still_drops_a_record_before_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A half-bounded range is a range, and a two-sided test would miss it."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(0.5))
    found = list(tree_source(root, quiet_logger).records(Selection(start_et=1.0)))
    assert found == []


def test_an_upper_bound_alone_still_drops_a_record_after_it(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The other half of the same rule."""
    root = tmp_path / 'results'
    write_document(root, FIRST_STUB, timed_document(2.0))
    found = list(tree_source(root, quiet_logger).records(Selection(stop_et=1.0)))
    assert found == []


# ---------------------------------------------------------------------------
# What the source says about itself
# ---------------------------------------------------------------------------


def test_the_source_names_its_roots_for_the_run_log(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The run log has to say which storage answered, and where."""
    root = two_volume_tree(tmp_path)
    assert root.as_posix() in tree_source(root, quiet_logger).describe()


def test_the_source_is_a_context_manager(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> None:
    """A caller closes every source the same way, whatever that source holds open."""
    root = two_volume_tree(tmp_path)
    with tree_source(root, quiet_logger) as source:
        assert sorted(stubs_of(source.listing(Selection()))) == [FIRST_STUB, SECOND_STUB]


def test_two_spellings_of_one_root_bind_one_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A source holding one root twice would answer for every image twice."""
    root = two_volume_tree(tmp_path)
    source = TreeRecordSource([root.as_posix(), f'{root.as_posix()}/'], logger=quiet_logger)
    assert len(source.roots) == 1
