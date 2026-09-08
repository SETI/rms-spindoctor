"""Tests for how much of a pass over a results tree runs at once.

These are the library's own rules about a tuning, whoever built it.  How a
configuration section becomes one, and what the shipped section says, is tested
beside the configuration helper that does it.
"""

from dataclasses import FrozenInstanceError, fields
from typing import Any

import pytest

from spindoctor.nav_records import TreeTuning

_SETTINGS = [field.name for field in fields(TreeTuning)]


@pytest.mark.parametrize('setting', _SETTINGS)
@pytest.mark.parametrize('value', [0, -1, 1.5, True, '8', None])
def test_a_setting_that_is_not_a_positive_integer_is_refused(setting: str, value: Any) -> None:
    """A pass tuned to zero threads does not run slowly; it does not run.

    Parameters:
        setting: The setting to give the bad value to.
        value: A value that is not a count of things.
    """
    with pytest.raises(ValueError, match=setting):
        TreeTuning(**{setting: value})


def test_a_retrieval_batch_smaller_than_its_pool_is_refused() -> None:
    """Threads with nothing to fetch are idle at every setting, so it is said early."""
    with pytest.raises(ValueError, match='retrieve_batch_size must be at least retrieve_threads'):
        TreeTuning(retrieve_threads=64, retrieve_batch_size=8)


def test_a_walk_round_smaller_than_its_pool_is_refused() -> None:
    """The same relation holds for the listing pool, and is refused the same way."""
    with pytest.raises(ValueError, match='walk_directories_at_once must be at least walk_threads'):
        TreeTuning(walk_threads=32, walk_directories_at_once=4)


@pytest.mark.parametrize(
    ('pool', 'work'),
    [
        ('retrieve_threads', 'retrieve_batch_size'),
        ('walk_threads', 'walk_directories_at_once'),
    ],
)
def test_a_round_equal_to_its_pool_is_allowed(pool: str, work: str) -> None:
    """The bound is what cannot fill the pool, not what fills it exactly once.

    What is under test is that construction succeeds; the assertion is the
    witness that the value arrived, since a refusal would have raised first.

    Parameters:
        pool: The thread-count setting.
        work: The setting that says how much work one round hands that pool.
    """
    tuning = TreeTuning(**{pool: 8, work: 8})
    assert getattr(tuning, work) == 8


def test_the_commit_chunk_is_a_multiple_of_the_retrieval_batch() -> None:
    """Or a transaction could be smaller than the batch it is retrieved in."""
    tuning = TreeTuning(retrieve_threads=4, retrieve_batch_size=16, ingest_commit_batches=3)
    assert tuning.ingest_commit_chunk_size == 48


@pytest.mark.parametrize('batch', [64, 1024, 16384])
def test_a_larger_batch_carries_the_chunk_with_it(batch: int) -> None:
    """The reason the chunk is a multiple rather than a number of its own.

    A fixed chunk beside a configurable batch would let an operator raise the
    batch past it, and every download would be quietly cut back down to the
    chunk with nothing to say why.

    Parameters:
        batch: A configured retrieval batch size.
    """
    tuning = TreeTuning(retrieve_batch_size=batch)
    assert tuning.ingest_commit_chunk_size >= batch


def test_a_tuning_cannot_be_changed_once_built() -> None:
    """It is passed down through every layer of a pass, and none of them may edit it."""
    tuning = TreeTuning()
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'walk_threads'"):
        # The assignment is the thing under test, so the type checker's
        # objection to it is the expected one.
        tuning.walk_threads = 1  # type: ignore[misc]
