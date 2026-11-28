"""Tests for what the provenance configuration digest covers.

The digest answers one question -- was this result produced by the same
configuration as that one -- so it has to cover everything that could change a
result and nothing that could not. A setting that cannot move an offset but
still changes the digest makes every result of a run compare as differently
configured against the archive it belongs to.
"""

from collections.abc import Mapping
from typing import Any

import pytest
from tests.packaged_sections import NavConfigWith

from spindoctor.config.config import HASH_EXCLUDED_SECTIONS, Config


def _hash(nav_config_with: NavConfigWith, override: Mapping[str, Any] | None = None) -> str:
    """Return the resolved digest of the shipped configuration plus an override.

    Parameters:
        nav_config_with: Builds a configuration from an override.
        override: The override; ``None`` for the defaults alone.

    Returns:
        The digest.
    """
    config = Config() if override is None else nav_config_with(override)
    return str(config.resolved_config_hash())


@pytest.mark.parametrize(
    'override',
    [
        {'logging': {'image': 'DEBUG'}},
        {'logging': {'main': 'WARNING'}},
        {'logging': {'techniques': {'titan_haze': 'DEBUG'}}},
        {'logging': {'models': {'rings': 'NONE'}}},
        {'logging': {'other': {'annotate': 'DEBUG'}}},
        {'logging': {'programs': {'sd_mosaic': {'main': 'DEBUG'}}}},
        {'logging': {'strict_scope': True}},
    ],
)
def test_a_logging_setting_does_not_change_the_digest(
    nav_config_with: NavConfigWith, override: dict[str, Any]
) -> None:
    """Looking harder at a run does not make its results differently configured.

    Raising one component's level is an everyday thing to do while
    investigating, and it would otherwise re-stamp every result of that run.
    """
    assert _hash(nav_config_with, override) == _hash(nav_config_with)


@pytest.mark.parametrize(
    'override',
    [
        {'environment': {'nav_results_root': '/somewhere/else'}},
        {'environment': {'pds3_holdings_root': '/mnt/other/holdings'}},
        {'environment': {'results_index_db': 'sqlite:////data/nav-results/index.sqlite3'}},
        {'environment': {'results_index_db': 'postgresql+psycopg://user@host/spindoctor'}},
    ],
)
def test_an_environment_setting_does_not_change_the_digest(
    nav_config_with: NavConfigWith, override: dict[str, Any]
) -> None:
    """Where a deployment keeps its files is not part of how it navigates.

    Moving a results directory, pointing at a different holdings mirror, or
    naming a database to index the results with cannot move an offset by a pixel,
    so a digest that shifted would report every result of that run as differently
    configured from the archive it belongs to.
    """
    assert _hash(nav_config_with, override) == _hash(nav_config_with)


def test_a_setting_that_can_change_a_result_does_change_the_digest(
    nav_config_with: NavConfigWith,
) -> None:
    """The exclusion is narrow: anything the pipeline reads still counts."""
    override = {'offset': {'correlation_fft_upsample_factor': 256}}
    assert _hash(nav_config_with, override) != _hash(nav_config_with)


def test_the_digest_is_stable_across_repeated_resolution(nav_config_with: NavConfigWith) -> None:
    """The same configuration resolves to the same digest every time."""
    assert _hash(nav_config_with) == _hash(nav_config_with)


@pytest.mark.parametrize(
    'override',
    [
        {'results_tree': {'walk_threads': 8}},
        {'results_tree': {'retrieve_threads': 8, 'retrieve_batch_size': 16}},
        {'results_tree': {'ingest_commit_batches': 1}},
    ],
)
def test_a_results_tree_setting_does_not_change_the_digest(
    nav_config_with: NavConfigWith, override: dict[str, Any]
) -> None:
    """How many requests a pass makes at once decides how long it takes and nothing else.

    A machine tuning the walk to its own link would otherwise re-stamp every
    result it navigates as differently configured from the archive it belongs
    to, over a setting no document read by the pass can see.
    """
    assert _hash(nav_config_with, override) == _hash(nav_config_with)


def test_only_the_argued_sections_are_excluded() -> None:
    """The excluded set is exactly what was argued for, not a growing list.

    Every section added to it stops being able to distinguish two results, so
    growth wants the same argument made again rather than a quiet append.
    """
    assert sorted(HASH_EXCLUDED_SECTIONS) == ['environment', 'logging', 'results_tree']


@pytest.mark.parametrize('section', ['environment', 'logging', 'results_tree'])
def test_an_excluded_section_is_still_present_in_the_configuration(section: str) -> None:
    """Excluding a section from the digest does not remove it from the config."""
    config = Config()
    config.read_config()
    assert getattr(config, section) is not None
