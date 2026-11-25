"""What the filter says about the pass whose answer it is handing on.

An index answers as of the pass that filled it and detects no change since, so
one thing travels with the answer: when that pass finished, which is what a
reader compares against what they know they have navigated.

It is read from the newest run row of the root being enumerated, and every test
here builds a second root whose pass ran afterwards and recorded a time of its
own, so an answer taken from the newest run of the table rather than of the root
says so.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.packaged_sections import packaged_tree_tuning
from tests.spindoctor.dataset.conftest import (
    OTHER_ROOT_NAME,
    VOLUMES,
    index_of_two_roots,
    one_image_tree,
    reported_line,
    reporting_logger,
    stamp_run,
)

from spindoctor.dataset.results_filter import ResultsFilter


def test_the_report_says_how_old_the_index_answer_is(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The index detects no change since its pass, so its age is what says it is usable.

    An exported URL makes a snapshot answer a resume idiom on every machine that
    exports it, and how long ago that snapshot was taken is the fact that
    decides whether this run is affected by it.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_of_two_roots(tmp_path, root)
    stamp_run(url, root, finished_utc=(datetime.now(UTC) - timedelta(days=2)).isoformat())
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=reporting_logger(),
        results_index_db_url=url,
        has_no_offset_file=True,
        tuning=packaged_tree_tuning(),
    )
    assert '2 days ago' in capsys.readouterr().out


def test_the_report_names_the_moment_as_well_as_the_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The interval is what a reader compares against; the stamp names the pass to re-run."""
    root, _images = one_image_tree(tmp_path)
    stamp = '2026-02-03T04:05:06+00:00'
    url = index_of_two_roots(tmp_path, root)
    stamp_run(url, root, finished_utc=stamp)
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=reporting_logger(),
        results_index_db_url=url,
        has_no_offset_file=True,
        tuning=packaged_tree_tuning(),
    )
    assert stamp in capsys.readouterr().out


def test_the_age_is_that_of_this_roots_pass_and_not_another(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The second root was passed over afterwards, and says nothing about this answer."""
    root, _images = one_image_tree(tmp_path)
    url = index_of_two_roots(tmp_path, root)
    stamp_run(url, root, finished_utc='2026-02-03T04:05:06+00:00')
    stamp_run(url, tmp_path / OTHER_ROOT_NAME, finished_utc='2026-03-04T05:06:07+00:00')
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=reporting_logger(),
        results_index_db_url=url,
        has_no_offset_file=True,
        tuning=packaged_tree_tuning(),
    )
    assert '2026-03-04T05:06:07+00:00' not in capsys.readouterr().out


def test_a_finish_time_that_will_not_parse_is_reported_as_it_stands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reader can act on a value the index really holds and not on a fiction.

    Nothing this pipeline writes puts an unreadable stamp in that column, and an
    index restored from somewhere else is exactly where one would come from.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_of_two_roots(tmp_path, root)
    stamp_run(url, root, finished_utc='whenever it was')
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=reporting_logger(),
        results_index_db_url=url,
        has_no_offset_file=True,
        tuning=packaged_tree_tuning(),
    )
    assert 'ingested whenever it was' in capsys.readouterr().out


def test_a_finish_time_in_the_future_is_reported_as_it_stands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two machines disagreeing by seconds is ordinary, and is not an interval.

    The pass is finished by whichever machine ran the ingest and the stamp is
    read by another, so a workstation a few seconds behind a cloud worker reads
    a moment that has not happened yet.  Reporting one as "less than a minute
    ago" would state an interval that is not one.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_of_two_roots(tmp_path, root)
    stamp = (datetime.now(UTC) + timedelta(days=2)).isoformat()
    stamp_run(url, root, finished_utc=stamp)
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=reporting_logger(),
        results_index_db_url=url,
        has_no_offset_file=True,
        tuning=packaged_tree_tuning(),
    )
    assert reported_line(capsys.readouterr().out).endswith(stamp)


def test_a_finish_time_with_no_offset_is_read_as_utc(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An index restored from elsewhere is where a stamp with no offset comes from.

    Every pass this pipeline runs writes an offset, so this is the same input
    the unreadable stamp is: a column filled by something else.  Without a
    reading for it, subtracting it raises out of the constructor, which is a
    crash of the enumeration rather than an answer about the index.
    """
    root, _images = one_image_tree(tmp_path)
    url = index_of_two_roots(tmp_path, root)
    naive = (datetime.now(UTC) - timedelta(days=2)).replace(tzinfo=None).isoformat()
    stamp_run(url, root, finished_utc=naive)
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=reporting_logger(),
        results_index_db_url=url,
        has_no_offset_file=True,
        tuning=packaged_tree_tuning(),
    )
    assert reported_line(capsys.readouterr().out).endswith(f'{naive} (2 days ago)')


@pytest.mark.parametrize('blank', ['', '   '])
def test_a_recorded_finish_time_of_nothing_is_reported_as_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], blank: str
) -> None:
    """The column is not null, so an empty string is a value a consumer meets.

    It says nothing about when the pass finished, and the report says that
    rather than naming a moment or leaving the sentence unfinished.  Spaces say
    no more than nothing does, and reported as they stand they end the line in a
    blank where a moment should be.

    """
    root, _images = one_image_tree(tmp_path)
    url = index_of_two_roots(tmp_path, root)
    stamp_run(url, root, finished_utc=blank)
    ResultsFilter(
        VOLUMES,
        str(root),
        logger=reporting_logger(),
        results_index_db_url=url,
        has_no_offset_file=True,
        tuning=packaged_tree_tuning(),
    )
    assert 'at a time this index does not record' in capsys.readouterr().out
