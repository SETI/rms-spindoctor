"""Tests for how much of a pass over a results tree runs at once.

These are the rules a tuning holds whoever built it.  Each setting's type and range is
the schema core's, and how the configuration section parses is tested beside the
configuration.
"""

from dataclasses import FrozenInstanceError

import pytest
from tests.packaged_sections import packaged_tree_tuning

from spindoctor.schema import RuleError


@pytest.mark.parametrize(
    ('pool', 'work'),
    [
        ('retrieve_threads', 'retrieve_batch_size'),
        ('walk_threads', 'walk_directories_at_once'),
    ],
)
def test_a_round_smaller_than_its_pool_is_refused(pool: str, work: str) -> None:
    """Threads with nothing to do are idle at every setting, so it is said early.

    Parameters:
        pool: The thread-count setting.
        work: The setting that says how much work one round hands that pool.
    """
    with pytest.raises(RuleError, match=f'must be at least {pool}') as excinfo:
        packaged_tree_tuning(**{pool: 8, work: 7})
    assert excinfo.value.at == (work,)


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
    tuning = packaged_tree_tuning(**{pool: 8, work: 8})
    assert getattr(tuning, work) == 8


def test_the_commit_chunk_is_a_multiple_of_the_retrieval_batch() -> None:
    """Or a transaction could be smaller than the batch it is retrieved in."""
    tuning = packaged_tree_tuning(
        retrieve_threads=4, retrieve_batch_size=16, ingest_commit_batches=3
    )
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
    tuning = packaged_tree_tuning(retrieve_threads=64, retrieve_batch_size=batch)
    assert tuning.ingest_commit_chunk_size >= batch


def test_a_tuning_cannot_be_changed_once_built() -> None:
    """It is passed down through every layer of a pass, and none of them may edit it."""
    tuning = packaged_tree_tuning()
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'walk_threads'"):
        # The assignment is the thing under test, so the type checker's
        # objection to it is the expected one.
        tuning.walk_threads = 1  # type: ignore[misc]
