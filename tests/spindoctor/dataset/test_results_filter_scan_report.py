"""What the run log is told about the scan the filter made.

An enumeration a user is never shown answers nobody's question about the
selection they got, so every scan says what it did.  There are two shapes of
scan and therefore two shapes of line: one that lists the selected volumes when
the filter is built knows what the root holds before a single candidate is
offered, and one that asks about its candidates as it meets them knows only when
it is over.  Both are covered here over both storages, because a line that goes
missing over one of them goes missing for the operator reading that run's log.
"""

from pathlib import Path
from typing import Any

import pytest
from tests.spindoctor.dataset.conftest import (
    CANDIDATES,
    NO_RESULT,
    SUCCESS,
    VOLUMES,
    candidate_files,
    index_of_two_roots,
    reporting_logger,
    select_from,
    write_tree,
)

from spindoctor.dataset.results_filter import ResultsFilter


def _tree(tmp_path: Path) -> Path:
    """Write the fixture results tree and return its root.

    Parameters:
        tmp_path: Directory the root is written under.

    Returns:
        The results root under test.
    """
    root = tmp_path / 'results'
    write_tree(root)
    return root


def _scanned(root: Path, *, results_index_db_url: str | None, **flags: Any) -> None:
    """Run every candidate of the fixture tree through one filter and close it.

    Parameters:
        root: The results root under test.
        results_index_db_url: The index to answer from, or None to read the tree.
        flags: The selection flags to apply.
    """
    with ResultsFilter(
        VOLUMES,
        str(root),
        logger=reporting_logger(),
        results_index_db_url=results_index_db_url,
        **flags,
    ) as results_filter:
        select_from(results_filter, candidate_files(root))


def test_a_scan_that_listed_the_volumes_says_what_the_tree_holds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The count is what the listing produced, and it is the reason to take one.

    Parameters:
        tmp_path: Directory the tree is written under.
        capsys: Fixture the logger's own stream is read back through.
    """
    root = _tree(tmp_path)
    _scanned(root, results_index_db_url=None, has_offset_file=True)
    found = f'Results scan found {len(CANDIDATES) - 1} offset metadata files'
    assert found in capsys.readouterr().out


def test_a_scan_that_listed_the_volumes_says_what_the_index_holds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same count out of the other storage, beside the age of its answer.

    Parameters:
        tmp_path: Directory the tree and the index are written under.
        capsys: Fixture the logger's own stream is read back through.
    """
    root = _tree(tmp_path)
    _scanned(root, results_index_db_url=index_of_two_roots(tmp_path, root), has_offset_file=True)
    held = f'Results index holds {len(CANDIDATES) - 1} offset metadata files'
    assert held in capsys.readouterr().out


def test_a_scan_that_names_its_candidates_says_so_before_it_starts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """It has counted nothing yet, so it says what it is about to do instead.

    Parameters:
        tmp_path: Directory the tree is written under.
        capsys: Fixture the logger's own stream is read back through.
    """
    root = _tree(tmp_path)
    _scanned(root, results_index_db_url=None, has_no_offset_file=True)
    assert 'Results scan will ask' in capsys.readouterr().out


def test_a_scan_that_names_its_candidates_still_reports_the_age_of_an_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An index answers as of its last pass whichever question is put to it.

    A run selecting the images nothing has been written for is exactly the run a
    stale answer misleads, so the age travels with it even where no count does.

    Parameters:
        tmp_path: Directory the tree and the index are written under.
        capsys: Fixture the logger's own stream is read back through.
    """
    root = _tree(tmp_path)
    _scanned(root, results_index_db_url=index_of_two_roots(tmp_path, root), has_no_offset_file=True)
    assert 'Results index answers about the images under' in capsys.readouterr().out


def test_a_closed_scan_says_how_many_candidates_it_asked_about(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """What stands in the place of the count a construction listing reports.

    Parameters:
        tmp_path: Directory the tree is written under.
        capsys: Fixture the logger's own stream is read back through.
    """
    root = _tree(tmp_path)
    _scanned(root, results_index_db_url=None, has_no_offset_file=True)
    assert f'Results scan asked about {len(CANDIDATES)} images' in capsys.readouterr().out


def test_a_closed_scan_says_how_many_of_them_were_already_navigated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every candidate but the one nothing was written for.

    Parameters:
        tmp_path: Directory the tree is written under.
        capsys: Fixture the logger's own stream is read back through.
    """
    root = _tree(tmp_path)
    _scanned(root, results_index_db_url=None, has_no_offset_file=True)
    found = len(CANDIDATES) - 1
    assert f'{found} of which had a navigation document' in capsys.readouterr().out


def test_a_scan_counts_one_document_once_for_each_image_that_has_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both halves of the line count images, so they can be read against each other.

    A batch naming one stub twice holds two candidates and one document, and a
    line counting the documents would report fewer images as already navigated
    than were asked about.

    Parameters:
        tmp_path: Directory the tree is written under.
        capsys: Fixture the logger's own stream is read back through.
    """
    root = _tree(tmp_path)
    documented = [image for image in candidate_files(root) if image.results_path_stub == SUCCESS]
    with ResultsFilter(
        VOLUMES, str(root), logger=reporting_logger(), has_no_offset_file=True
    ) as results_filter:
        results_filter.filter_batch(documented * 2)
    assert '2 of which had a navigation document' in capsys.readouterr().out


def test_a_scan_that_asked_about_nothing_says_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An enumeration that accepted no candidate has no tally to report.

    Parameters:
        tmp_path: Directory the tree is written under.
        capsys: Fixture the logger's own stream is read back through.
    """
    root = _tree(tmp_path)
    with ResultsFilter(
        VOLUMES, str(root), logger=reporting_logger(), has_no_offset_file=True
    ) as results_filter:
        results_filter.filter_batch([])
    assert 'Results scan asked about' not in capsys.readouterr().out


def test_a_scan_closed_twice_reports_its_tally_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A filter closed again has no second scan to report.

    Parameters:
        tmp_path: Directory the tree is written under.
        capsys: Fixture the logger's own stream is read back through.
    """
    root = _tree(tmp_path)
    results_filter = ResultsFilter(
        VOLUMES, str(root), logger=reporting_logger(), has_no_offset_file=True
    )
    select_from(results_filter, candidate_files(root))
    results_filter.close()
    capsys.readouterr()
    results_filter.close()
    assert 'Results scan asked about' not in capsys.readouterr().out


def test_the_images_a_closed_scan_counted_are_the_ones_it_kept_and_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The tally is of the whole scan, so the selection it describes is stated too.

    Parameters:
        tmp_path: Directory the tree is written under.
        capsys: Fixture the logger's own stream is read back through.
    """
    root = _tree(tmp_path)
    with ResultsFilter(
        VOLUMES, str(root), logger=reporting_logger(), has_no_offset_file=True
    ) as results_filter:
        kept = select_from(results_filter, candidate_files(root))
    capsys.readouterr()
    assert kept == [NO_RESULT]
