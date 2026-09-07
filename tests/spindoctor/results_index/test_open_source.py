"""Tests for the opener that picks a run's record source.

A run naming an index reads rows and a run naming none reads documents.  What
is pinned here is what the opener hands each of them: the tuning a program
resolved travels through it to the tree source unchanged, and to nothing else,
since a source over rows never walks a tree.
"""

from pathlib import Path
from typing import Any

import pdslogger
import pytest
from tests.spindoctor.conftest import index_url, ingest_tree, metadata_document, write_metadata

from spindoctor.nav_records import TreeRecordSource, TreeTuning
from spindoctor.results_index import open_record_source
from spindoctor.results_index import open_source as open_source_module


def _recording_tree_source(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Stand in for the tree source and note the tuning it is built with.

    Parameters:
        monkeypatch: Fixture the stand-in is installed through.

    Returns:
        The list each construction appends its tuning to.
    """
    handed: list[Any] = []

    class Recording(TreeRecordSource):
        """A tree source that notes what it was built with."""

        def __init__(self, roots: Any, **kwargs: Any) -> None:
            handed.append(kwargs.get('tuning'))
            super().__init__(roots, **kwargs)

    monkeypatch.setattr(open_source_module, 'TreeRecordSource', Recording)
    return handed


def test_the_tuning_reaches_the_tree_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opener passes on what the program resolved, rather than a value of its own."""
    handed = _recording_tree_source(monkeypatch)
    configured = TreeTuning(walk_threads=3, walk_directories_at_once=3)
    with open_record_source([tmp_path], tuning=configured):
        pass
    assert handed == [configured]


def test_a_run_that_reads_rows_builds_no_tree_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source over rows never walks a tree, so the tuning has nothing to reach."""
    handed = _recording_tree_source(monkeypatch)
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1454725799_1_CALIB', metadata_document())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=pdslogger.NullLogger())
    with open_record_source([root], results_index_db_url=url, tuning=TreeTuning()):
        pass
    assert handed == []
