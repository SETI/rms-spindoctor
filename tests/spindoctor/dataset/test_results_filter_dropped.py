"""What a scan says about the candidates it could read no navigation record out of.

An error filter asks what a document records, so a file that yields no facts
satisfies none of them, the one phrased in the negative included.  A document
written to an earlier metadata schema is such a file, and a results root holding
nothing else therefore answers every error filter with no image at all --
including the filter whose whole purpose is finding the images to run again.

That is an answer a run has to be told about, because a selection short for this
reason is otherwise indistinguishable from a root holding no such image.  So the
scan counts the candidates it read nothing out of and names one of them with the
reason.  It says it once, when the enumeration is done with the filter, since a
root holding hundreds of them would otherwise bury the selection the run asked
for under the report of what it passed over.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from filecache import FCPath
from tests.packaged_sections import packaged_tree_tuning
from tests.spindoctor.conftest import metadata_document, write_metadata
from tests.spindoctor.dataset.conftest import (
    VOLUMES,
    reporting_logger,
    select_from,
)

from spindoctor.dataset.dataset import ImageFile
from spindoctor.dataset.results_filter import SPICE_STATUS_ERROR, ResultsFilter

EARLIER_SCHEMA = 'COISS_2001/data/a/N1000000001_1_CALIB'
"""A document naming no instrument, which is the shape an earlier schema wrote."""

SECOND_EARLIER_SCHEMA = 'COISS_2001/data/a/N1000000002_1_CALIB'
"""A second of them, so that the count reported is a count and not a constant."""

CURRENT_SCHEMA = 'COISS_2001/data/a/N1000000003_1_CALIB'
"""A current-schema document recording the same fatal error as the others."""

DROPPED = 'yielded no navigation record'
"""The phrase the line reporting what a scan read nothing out of carries."""


def _spice_error(stub: str, *, instrument: str | None) -> dict[str, Any]:
    """Build a document recording a fatal SPICE error for one image.

    Every document in this module records the same outcome, so that what a
    filter answers turns on whether the document can be read at all rather than
    on what it says.

    Parameters:
        stub: The image's results path stub, which names the image.
        instrument: The recorded instrument, or None for the document that
            carries the observation fields alone.

    Returns:
        The document.
    """
    return metadata_document(
        image_name=f'{Path(stub).name}.IMG',
        instrument=instrument,
        status='error',
        status_error=SPICE_STATUS_ERROR,
        offset=None,
    )


def _write_root(tmp_path: Path, earlier: Sequence[str], current: Sequence[str] = ()) -> Path:
    """Write a results root holding documents of the two metadata schemas.

    Parameters:
        tmp_path: Directory the root is written under.
        earlier: Stubs whose document names no instrument.
        current: Stubs whose document is a current-schema navigation document.

    Returns:
        The results root.
    """
    root = tmp_path / 'results'
    for stub in earlier:
        write_metadata(root, stub, _spice_error(stub, instrument=None))
    for stub in current:
        write_metadata(root, stub, _spice_error(stub, instrument='coiss'))
    return root


def _candidate_files(root: Path, stubs: Sequence[str]) -> list[ImageFile]:
    """Build the images an enumeration would offer the filter.

    Parameters:
        root: The results root, only so the stand-in URLs point somewhere.
        stubs: The candidates, in enumeration order.

    Returns:
        One image per stub, in that order.
    """
    return [
        ImageFile(
            image_file_url=FCPath(root / f'{stub}.IMG'),
            label_file_url=FCPath(root / f'{stub}.LBL'),
            results_path_stub=stub,
        )
        for stub in stubs
    ]


def _scan(root: Path, stubs: Sequence[str]) -> list[str]:
    """Run a SPICE-error filter over a root exactly as an enumeration does.

    The filter is left the way an enumeration leaves it, because the report of
    what the scan read nothing out of is what the end of the scan produces.

    Parameters:
        root: The results root to answer from.
        stubs: The candidates, in enumeration order.

    Returns:
        The stubs the filter selected, in enumeration order.
    """
    with ResultsFilter(
        VOLUMES,
        str(root),
        logger=reporting_logger(),
        results_index_db_url=None,
        has_offset_spice_error=True,
        tuning=packaged_tree_tuning(),
    ) as under_test:
        return select_from(under_test, _candidate_files(root, stubs))


def _dropped_lines(out: str) -> list[str]:
    """Return every line reporting candidates a scan read no record out of.

    Parameters:
        out: Everything the filter wrote.

    Returns:
        Those lines, in the order they were written.
    """
    return [line for line in out.splitlines() if DROPPED in line]


def test_a_root_of_earlier_schema_documents_satisfies_no_error_filter(tmp_path: Path) -> None:
    """This is the selection the report exists for, so it is stated rather than assumed.

    Every assertion below would hold just as well of a filter that selected
    these images, and the operator would then have no need of a report at all.
    """
    root = _write_root(tmp_path, [EARLIER_SCHEMA, SECOND_EARLIER_SCHEMA])
    assert _scan(root, [EARLIER_SCHEMA, SECOND_EARLIER_SCHEMA]) == []


def test_a_current_schema_document_is_still_selected_beside_them(tmp_path: Path) -> None:
    """A file that cannot be read costs itself and not the rest of the scan."""
    root = _write_root(tmp_path, [EARLIER_SCHEMA], [CURRENT_SCHEMA])
    assert _scan(root, [EARLIER_SCHEMA, CURRENT_SCHEMA]) == [CURRENT_SCHEMA]


@pytest.mark.parametrize(
    ('earlier', 'counted'),
    [
        pytest.param([EARLIER_SCHEMA], '1 file under', id='one'),
        pytest.param([EARLIER_SCHEMA, SECOND_EARLIER_SCHEMA], '2 files under', id='several'),
    ],
)
def test_the_scan_says_how_many_candidates_it_read_no_record_out_of(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    earlier: Sequence[str],
    counted: str,
) -> None:
    """How much of the selection went this way is what says whether it explains the answer.

    Without the number a reader cannot tell one stray file in a campaign from a
    whole root the filters can say nothing about.

    Parameters:
        tmp_path: Directory the results root is written under.
        capsys: Fixture the filter's own output is read back through.
        earlier: Stubs whose document names no instrument.
        counted: How the line names that many files.
    """
    root = _write_root(tmp_path, earlier)
    _scan(root, earlier)
    assert counted in capsys.readouterr().out


def test_the_file_named_is_the_first_one_read_and_not_the_last(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Which file is named may not turn on where an enumeration happened to stop.

    A run cut short by a limit, and the same run without one, read different
    numbers of documents; named the last, the two would send an operator to
    different files for one root's answer.

    The file that should be named is asserted first, so that the absence of the
    other one is read off a line that named a file at all rather than off a
    report the scan never made.
    """
    earlier = [EARLIER_SCHEMA, SECOND_EARLIER_SCHEMA]
    root = _write_root(tmp_path, earlier)
    _scan(root, earlier)
    reported = capsys.readouterr().out
    assert f'{EARLIER_SCHEMA}_metadata.json' in reported
    assert f'{SECOND_EARLIER_SCHEMA}_metadata.json' not in reported


def test_the_scan_names_one_of_the_files_it_read_no_record_out_of(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A number alone sends a reader looking; a file is something they can open."""
    root = _write_root(tmp_path, [EARLIER_SCHEMA], [CURRENT_SCHEMA])
    _scan(root, [EARLIER_SCHEMA, CURRENT_SCHEMA])
    assert f'{EARLIER_SCHEMA}_metadata.json' in capsys.readouterr().out


def test_the_scan_says_why_no_record_came_out_of_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The reason is what tells a whole root of earlier-schema documents apart from a fault.

    Named, it says the documents need rewriting; unnamed, an operator meets a
    count of files their tools will not read and no way to tell why.
    """
    root = _write_root(tmp_path, [EARLIER_SCHEMA], [CURRENT_SCHEMA])
    _scan(root, [EARLIER_SCHEMA, CURRENT_SCHEMA])
    assert 'no observation.instrument' in capsys.readouterr().out


def test_the_scan_says_it_once_however_many_files_it_read_nothing_out_of(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real results root holds hundreds of them, and a line apiece buries the answer."""
    earlier = [EARLIER_SCHEMA, SECOND_EARLIER_SCHEMA]
    root = _write_root(tmp_path, earlier)
    _scan(root, earlier)
    assert len(_dropped_lines(capsys.readouterr().out)) == 1


def test_a_scan_that_read_every_candidate_says_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing went unread, so a line about it would be noise in every ordinary run."""
    root = _write_root(tmp_path, [], [CURRENT_SCHEMA])
    _scan(root, [CURRENT_SCHEMA])
    assert _dropped_lines(capsys.readouterr().out) == []


def test_a_filter_closed_again_does_not_report_a_second_scan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Closing twice costs nothing, so it may not say a scan dropped these files twice.

    An enumeration abandoned part way is closed as it is torn down and closed
    again by whatever was holding it, which is the ordinary way a filter is
    closed more than once.
    """
    root = _write_root(tmp_path, [EARLIER_SCHEMA])
    under_test = ResultsFilter(
        VOLUMES,
        str(root),
        logger=reporting_logger(),
        results_index_db_url=None,
        has_offset_spice_error=True,
        tuning=packaged_tree_tuning(),
    )
    select_from(under_test, _candidate_files(root, [EARLIER_SCHEMA]))
    under_test.close()
    under_test.close()
    assert len(_dropped_lines(capsys.readouterr().out)) == 1
