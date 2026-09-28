"""Tests for holding a pass down to what the open-file limit allows.

A batch of documents retrieved from a cloud root costs an open file apiece
while the batch is in flight, so the limit the process runs under bounds how
large a batch may be.  What is tested here is
:func:`spindoctor.nav_records.within_open_file_limit`, which raises that limit
as far as the process is allowed to raise it and then sizes the pass against
what it got.

The limit itself is read from the operating system, so every test that is about
a particular limit says what it is rather than depending on the machine the
suite runs on.  Where a test is about the headroom the module keeps back, it
writes the number out rather than reading the module's own constant, because an
assertion made out of the value under test moves with it and cannot fail.
"""

import resource
import sys
import uuid
from pathlib import Path

import pytest
from pdslogger import PdsLogger

from spindoctor.nav_records import TreeRecordSource, TreeTuning, within_open_file_limit
from spindoctor.nav_records import descriptors as descriptors_module
from spindoctor.nav_records.descriptors import RESERVED_DESCRIPTORS, UNBOUNDED_SOFT_LIMIT

pytestmark = pytest.mark.skipif(
    sys.platform == 'win32', reason='the open-file limit these tests are about is Unix-only'
)

_REMOTE = ['gs://a-bucket/a-results-root']
"""A root whose documents are downloaded, and so cost descriptors."""


def _at_limit(monkeypatch: pytest.MonkeyPatch, limit: int | None) -> None:
    """Run the rest of a test as though the process were under one open-file limit.

    Parameters:
        monkeypatch: Fixture the replacement is installed through.
        limit: The soft limit to answer with, or None for a platform that has
            none to answer with.
    """
    monkeypatch.setattr(descriptors_module, 'raise_open_file_limit', lambda: limit)


def _peak(tuning: TreeTuning) -> int:
    """Return how many open files a pass at this tuning holds at its peak.

    Parameters:
        tuning: The tuning the pass runs at.

    Returns:
        One descriptor per document in the batch, plus the file and the
        connection each download thread holds.
    """
    return tuning.retrieve_batch_size + 2 * tuning.retrieve_threads


def _logger() -> PdsLogger:
    """Build a logger no other test shares.

    Returns:
        The logger, named uniquely so that tests stay independent of each other.
    """
    return PdsLogger(f'descriptors_test_{uuid.uuid4().hex}', lognames=False)


def test_a_limit_that_covers_the_tuning_leaves_it_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pass the machine can run is the pass the operator configured."""
    _at_limit(monkeypatch, 65536)
    assert within_open_file_limit(TreeTuning(), _REMOTE) == TreeTuning()


def test_a_limit_too_low_for_the_batch_shrinks_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stock login-shell limit is what the shipped batch size cannot run under."""
    _at_limit(monkeypatch, 1024)
    assert within_open_file_limit(TreeTuning(), _REMOTE).retrieve_batch_size < 1024


def test_the_pass_leaves_room_for_the_files_it_does_not_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The index, the logs and the streams are open too, and none of them is in the sum."""
    _at_limit(monkeypatch, 1024)
    assert _peak(within_open_file_limit(TreeTuning(), _REMOTE)) <= 768


def test_the_threads_are_charged_for_the_files_they_hold_as_well(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A download holds a temporary file and a connection, so a thread costs two."""
    _at_limit(monkeypatch, 1024)
    bounded = within_open_file_limit(TreeTuning(), _REMOTE)
    assert bounded.retrieve_batch_size <= 1024 - RESERVED_DESCRIPTORS - 2 * bounded.retrieve_threads


def test_a_thread_count_the_limit_cannot_cover_is_held_down_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pool larger than the limit would exhaust it before a single batch was locked."""
    _at_limit(monkeypatch, 1024)
    bounded = within_open_file_limit(
        TreeTuning(retrieve_threads=512, retrieve_batch_size=4096), _REMOTE
    )
    assert bounded.retrieve_threads < 512


def test_the_threads_never_take_the_whole_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch left with nothing is a pass that retrieves one document at a time."""
    _at_limit(monkeypatch, 1024)
    bounded = within_open_file_limit(
        TreeTuning(retrieve_threads=512, retrieve_batch_size=4096), _REMOTE
    )
    assert bounded.retrieve_batch_size > bounded.retrieve_threads


def test_a_shrunken_pass_still_fills_its_download_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch below the pool size leaves threads idle, which no limit is a reason for."""
    _at_limit(monkeypatch, 258)
    bounded = within_open_file_limit(TreeTuning(), _REMOTE)
    assert bounded.retrieve_batch_size >= bounded.retrieve_threads


def test_a_limit_below_the_reserve_still_yields_a_pass_that_can_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A limit under the reserve leaves nothing to divide, and zero is no batch."""
    _at_limit(monkeypatch, 64)
    assert within_open_file_limit(TreeTuning(), _REMOTE).retrieve_batch_size >= 1


def test_a_platform_with_no_limit_to_read_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where nothing bounds the open files, nothing bounds the batch either."""
    _at_limit(monkeypatch, None)
    assert within_open_file_limit(TreeTuning(), _REMOTE) == TreeTuning()


def test_a_pass_over_local_roots_is_left_entirely_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A local file is read where it lies, so no lock and no descriptor is spent on it."""
    _at_limit(monkeypatch, 1024)
    assert within_open_file_limit(TreeTuning(), [str(tmp_path)]) == TreeTuning()


def test_a_pass_over_local_roots_does_not_raise_the_process_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Building a source over a directory is no reason to change the whole process."""
    asked: list[bool] = []

    def noting() -> int:
        asked.append(True)
        return 1024

    monkeypatch.setattr(descriptors_module, 'raise_open_file_limit', noting)
    within_open_file_limit(TreeTuning(), [str(tmp_path)])
    assert asked == []


def test_one_remote_root_among_local_ones_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pass reads every root it was given, so one that downloads settles it for all."""
    _at_limit(monkeypatch, 1024)
    bounded = within_open_file_limit(TreeTuning(), ['/tmp', 'gs://a-bucket/a-root'])
    assert bounded != TreeTuning()


def test_holding_a_pass_back_is_said_out_loud(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pass slower than it was configured to be is not a thing to discover later."""
    _at_limit(monkeypatch, 1024)
    within_open_file_limit(TreeTuning(), _REMOTE, logger=_logger())
    assert 'Open file limit of 1024' in capsys.readouterr().out


def test_the_line_names_the_limit_that_would_have_to_be_raised(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The soft limit is already at the hard one, so ulimit -n would advise nothing."""
    _at_limit(monkeypatch, 1024)
    within_open_file_limit(TreeTuning(), _REMOTE, logger=_logger())
    assert 'ulimit -Hn' in capsys.readouterr().out


def test_a_pass_the_limit_covers_says_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ordinary run would otherwise carry a line about a limit that never applied."""
    _at_limit(monkeypatch, 65536)
    within_open_file_limit(TreeTuning(), _REMOTE, logger=_logger())
    assert 'Open file limit' not in capsys.readouterr().out


def test_the_soft_limit_is_raised_to_the_hard_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run behaves the same whichever shell started it, unasked."""
    monkeypatch.setattr(resource, 'getrlimit', lambda _which: (1024, 65536))
    raised: list[tuple[int, int]] = []
    monkeypatch.setattr(resource, 'setrlimit', lambda _which, limits: raised.append(limits))
    assert descriptors_module.raise_open_file_limit() == 65536
    assert raised == [(65536, 65536)]


def test_an_unbounded_hard_limit_is_raised_to_rather_than_given_up_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The soft limit is what a run is held to, and raising it here always works."""
    monkeypatch.setattr(resource, 'getrlimit', lambda _which: (256, resource.RLIM_INFINITY))
    raised: list[tuple[int, int]] = []
    monkeypatch.setattr(resource, 'setrlimit', lambda _which, limits: raised.append(limits))
    assert descriptors_module.raise_open_file_limit() == UNBOUNDED_SOFT_LIMIT
    assert raised == [(UNBOUNDED_SOFT_LIMIT, resource.RLIM_INFINITY)]


def test_a_soft_limit_already_unbounded_bounds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing to be held to means nothing to hold the pass down to."""
    unlimited = resource.RLIM_INFINITY
    monkeypatch.setattr(resource, 'getrlimit', lambda _which: (unlimited, unlimited))
    assert descriptors_module.raise_open_file_limit() is None


def test_a_soft_limit_already_at_the_hard_one_is_not_set_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run started from a shell that already raised it asks the system for nothing."""
    monkeypatch.setattr(resource, 'getrlimit', lambda _which: (65536, 65536))
    raised: list[tuple[int, int]] = []
    monkeypatch.setattr(resource, 'setrlimit', lambda _which, limits: raised.append(limits))
    assert descriptors_module.raise_open_file_limit() == 65536
    assert raised == []


def test_a_limit_that_cannot_be_raised_is_the_one_the_pass_is_sized_against(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused raise is not a failure: the pass runs smaller under the limit it has."""
    monkeypatch.setattr(resource, 'getrlimit', lambda _which: (1024, 65536))

    def refusing(_which: int, _limits: tuple[int, int]) -> None:
        raise ValueError('not permitted')

    monkeypatch.setattr(resource, 'setrlimit', refusing)
    assert descriptors_module.raise_open_file_limit() == 1024


def test_a_source_over_a_remote_root_runs_at_the_bounded_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound reaches the retrieval, rather than being computed and dropped."""
    _at_limit(monkeypatch, 1024)
    source = TreeRecordSource(_REMOTE, tuning=TreeTuning())
    assert source._tuning.retrieve_batch_size < TreeTuning().retrieve_batch_size


def test_a_source_over_a_local_root_runs_at_the_tuning_it_was_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A local pass is the one the open-file limit has nothing to say about."""
    _at_limit(monkeypatch, 1024)
    source = TreeRecordSource([str(tmp_path)], tuning=TreeTuning())
    assert source._tuning == TreeTuning()
