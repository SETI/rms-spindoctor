"""What a report covers, root by root, and what it says about a root it did not.

One index serves several results roots and one report may span them, so the root
is half of every key the report reads and the whole of what it can be narrowed
to without opening a document.  Two things follow, and both are measured here
over two roots that differ in the value under test: a report handed two roots
covers both of them, and a report that could be bound to only some of the roots
it was pointed at says which ones those were.

A root whose newest ingest run never finished is the second case.  Its rows are
in the index and mean nothing -- absence under it is absence of a pass rather
than absence of an image -- so the report covers none of it, and a narrowing
nobody could see would read as an answer about the whole index.
"""

from pathlib import Path

import pdslogger
import pytest
from tests.packaged_sections import packaged_tree_tuning
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    write_metadata,
    write_refusal,
)

from spindoctor.cli.stats.report import build_report, main_report
from spindoctor.nav_records import TreeRecordSource
from spindoctor.results_index import INGEST_RUNS, SCHEMA_VERSION, open_index

from .conftest import index_source

_COVERED_STUB = 'VOL/N1000000001_1_CALIB'
"""The image the root a report can be bound to holds."""

_DROPPED_STUB = 'VGISS_5101/C1385455_GEOMED'
"""The image the other root holds, of another mission and another numbering.

Different in every value the report groups by, so a report that covered the
wrong root, or only one of two, differs from one that covered both in more than
a total.
"""


def _two_roots(tmp_path: Path) -> tuple[Path, Path]:
    """Write two results roots, each holding one image the other does not.

    Parameters:
        tmp_path: Directory the roots are written under.

    Returns:
        The two roots.
    """
    first = tmp_path / 'primary'
    second = tmp_path / 'rescue'
    write_metadata(first, _COVERED_STUB, metadata_document())
    write_metadata(
        second,
        _DROPPED_STUB,
        metadata_document(
            image_name='C1385455_GEOMED.IMG',
            instrument='vgiss',
            camera=None,
            image_shape=[800, 800],
        ),
    )
    return first, second


# ---------------------------------------------------------------------------
# Two roots, both covered
# ---------------------------------------------------------------------------


def test_a_report_over_two_named_roots_counts_the_images_of_both(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """``--root`` accumulates, so naming two roots asks about two roots.

    A selection built from the first of them alone would report one image and
    say nothing about having narrowed itself: the header would still name both.
    """
    first, second = _two_roots(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first, second], logger=quiet_logger)
    with index_source(url, [first, second]) as source:
        text = build_report(
            source, tmp_path / 'report', roots=[first.as_posix(), second.as_posix()]
        ).read_text(encoding='utf-8')
    assert 'Total images: 2' in text


def test_a_report_over_two_named_roots_names_both_of_them(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The header says what was covered, and covering two roots is two names.

    Compared against the whole line, since a header naming the first root alone
    is a prefix of one naming both.
    """
    first, second = _two_roots(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [first, second], logger=quiet_logger)
    with index_source(url, [first, second]) as source:
        text = build_report(
            source, tmp_path / 'report', roots=[first.as_posix(), second.as_posix()]
        ).read_text(encoding='utf-8')
    expected = f'Filters: root in {first.as_posix()}, {second.as_posix()}'
    assert expected in text.splitlines()


# ---------------------------------------------------------------------------
# A root the report could not be bound to
# ---------------------------------------------------------------------------


def _index_with_one_unfinished_root(
    tmp_path: Path, logger: pdslogger.PdsLogger, *, dropped_holds_a_refusal: bool = False
) -> tuple[str, Path, Path]:
    """Ingest two roots and leave the second one's newest run unfinished.

    Both roots' rows are in the index, which is the whole difficulty: the rows
    are there and mean nothing, because the pass that would have written the
    rest of them died.

    Parameters:
        tmp_path: Directory the roots and the index live under.
        logger: Logger the ingest reports through.
        dropped_holds_a_refusal: Whether the second root also holds a file no
            record can be read out of.

    Returns:
        The index URL, the root with a completed ingest, and the root without
        one.
    """
    covered, dropped = _two_roots(tmp_path)
    if dropped_holds_a_refusal:
        write_refusal(dropped, 'VGISS_5101/edges')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [covered, dropped], logger=logger)
    engine = open_index(url)
    try:
        with engine.begin() as connection:
            connection.execute(
                INGEST_RUNS.insert().values(
                    root_url=dropped.as_posix(),
                    started_utc='2026-08-20T00:00:00+00:00',
                    finished_utc=None,
                    schema_version=SCHEMA_VERSION,
                )
            )
    finally:
        engine.dispose()
    return url, covered, dropped


def _report_over(url: str, out: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run the driver against an index it was named, naming no root of its own.

    Naming no root is the case the narrowing happens in: a run that names its
    roots is refused outright when one of them has no completed ingest.

    Parameters:
        url: The index URL.
        out: Directory receiving the report.
        monkeypatch: Fixture the ambient index variable is cleared through, so
            that what the run reads is what it was handed.

    Returns:
        The Markdown the report wrote.
    """
    monkeypatch.delenv('NAV_RESULTS_INDEX_DB', raising=False)
    main_report(['--results-index-db', url, '--output-dir', str(out)])
    return (out / 'report.md').read_text(encoding='utf-8')


def test_a_report_that_dropped_a_root_names_the_roots_it_covered(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A narrowing nobody can see is one an operator reads as an answer about everything.

    The index holds rows under both roots and the report covers one of them, so
    a header still saying that every image was read would be false about the
    half of the index it never looked at.
    """
    url, covered, _dropped = _index_with_one_unfinished_root(tmp_path, quiet_logger)
    text = _report_over(url, tmp_path / 'report', monkeypatch)
    assert f'Filters: root in {covered.as_posix()}' in text.splitlines()


def test_a_report_that_dropped_a_root_names_the_root_it_dropped(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming what was covered is half of it; the other half is what to ingest.

    The root that was left out is the one thing the operator has to act on, and
    the report is where they are reading.
    """
    url, _covered, dropped = _index_with_one_unfinished_root(tmp_path, quiet_logger)
    text = _report_over(url, tmp_path / 'report', monkeypatch)
    assert f'Roots dropped: {dropped.as_posix()}' in text.splitlines()


def test_a_report_covering_every_root_names_no_dropped_root(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A line that appeared over a whole index would say a run had covered less.

    The paragraph is printed because a root was left out, so an index with a
    completed ingest of everything it holds prints neither it nor the roots
    under it.
    """
    covered, dropped = _two_roots(tmp_path)
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [covered, dropped], logger=quiet_logger)
    text = _report_over(url, tmp_path / 'report', monkeypatch)
    assert 'Roots dropped:' not in text


def test_a_dropped_root_contributes_no_images(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading rows under a root whose pass died is reading absence with no license to.

    The dropped root holds one image, and the index holds its row.  Counting it
    would report a number nothing stands behind: whatever else that root holds
    was never walked.
    """
    url, _covered, _dropped = _index_with_one_unfinished_root(tmp_path, quiet_logger)
    text = _report_over(url, tmp_path / 'report', monkeypatch)
    assert 'Total images: 1' in text


def test_a_dropped_root_contributes_no_refusals_either(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-covered root is worse than an uncovered one, so it is not covered at all.

    A refusal is the one thing under a dropped root a filter cannot exclude, so
    it is the count that would leak.  Reporting a root's refused files while
    reporting none of its images would describe a root the report never read.
    """
    url, _covered, _dropped = _index_with_one_unfinished_root(
        tmp_path, quiet_logger, dropped_holds_a_refusal=True
    )
    text = _report_over(url, tmp_path / 'report', monkeypatch)
    assert 'Files that yielded no record: 0' in text


def test_a_dropped_roots_refusal_is_in_the_index_that_left_it_out(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The absence above is an absence of something, not of an empty fixture.

    Ingested with both roots completed, the same index reports the refusal, so
    the zero beside it is the dropped root being dropped rather than a file that
    was never refused.
    """
    covered, dropped = _two_roots(tmp_path)
    write_refusal(dropped, 'VGISS_5101/edges')
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [covered, dropped], logger=quiet_logger)
    text = _report_over(url, tmp_path / 'report', monkeypatch)
    assert 'Files that yielded no record: 1' in text


def test_the_tree_reports_over_every_root_it_was_named(tmp_path: Path) -> None:
    """No index means no ingest to be incomplete, so a tree drops no root.

    A results tree is read as it is now, so both roots are covered and neither
    is named as one the report left out.
    """
    first, second = _two_roots(tmp_path)
    out = tmp_path / 'report'
    out.mkdir(parents=True, exist_ok=True)
    with TreeRecordSource(
        [first.as_posix(), second.as_posix()], tuning=packaged_tree_tuning()
    ) as source:
        build_report(source, out)
    assert 'Total images: 2' in (out / 'report.md').read_text(encoding='utf-8')
