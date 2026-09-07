"""Reporting over a results tree, with the report from an index as the control.

One pass of accumulators answers every section, and the pass reads the record
seam rather than a storage, so the same report comes out of a results tree and
out of an index ingested from that tree.  That parity is the acceptance
criterion, and it is measured here over the same frozen fixture tree and the same
two invocations the report's frozen output is measured over, so the comparison
covers every section the report writes.

Three things the frozen tree cannot show are tested over trees written for them.
It is deliberately all readable, so the count of files that yielded no record is
zero over it.  It holds one root, so the tie two roots holding one basename make
is unreachable in it.  And it records no structured JSON column beyond a
covariance, so what the export writes for a matrix or a kernel list is pinned
over a document carrying every one of them.

One thing it does hold is a document written for an image that never loaded, and
so records no epoch, which is what the date bounds are measured against here: an
image that cannot be placed in time is inside no range, and each bound has to say
so on its own.
"""

import csv
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdslogger
import pytest
import sqlalchemy
from tests.spindoctor.conftest import (
    index_url,
    ingest_tree,
    metadata_document,
    technique,
    write_metadata,
    write_refusal,
)

from spindoctor.cli.stats import report as report_module
from spindoctor.cli.stats.report import build_report, main_report
from spindoctor.nav_records import (
    ImageFacts,
    Selection,
    TreeRecordSource,
    TreeTuning,
    UnreadableFile,
)
from spindoctor.results_index import IMAGES, INGEST_RUNS, SCHEMA_VERSION, open_index

from .conftest import (
    GOLDEN_DIR,
    GOLDEN_VARIANTS,
    RESULTS_TREE,
    ReplayedFacts,
    index_source,
    report_from_the_index,
    report_from_the_tree,
)

_A_TREE_IMAGE = 'N1294561202'
"""An image name the fixture tree holds, for telling a report from an empty one."""


# ---------------------------------------------------------------------------
# The same report, out of either storage
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def reports(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple[Path, Path]]:
    """Write each golden variant twice, once over the tree and once from an index.

    Built once for the whole module: every comparison below only reads, and
    building per test would ingest the same tree once per comparison.

    Parameters:
        tmp_path_factory: Factory the indexes and the outputs live under.

    Returns:
        Per variant name, the tree-read report directory and the index-read one.
    """
    logger = pdslogger.PdsLogger(f'stats_parity_{uuid.uuid4().hex}')
    logger.set_level('ERROR')
    root = tmp_path_factory.mktemp('parity')
    written = {}
    for variant, options in GOLDEN_VARIANTS.items():
        from_tree = report_from_the_tree(root / f'{variant}-tree', **options)
        from_index = report_from_the_index(
            index_url(root / f'{variant}.sqlite3'),
            root / f'{variant}-index',
            logger=logger,
            **options,
        )
        written[variant] = (from_tree, from_index)
    return written


@pytest.mark.parametrize('variant', sorted(GOLDEN_VARIANTS))
def test_the_report_is_byte_identical_to_one_from_an_index(
    reports: dict[str, tuple[Path, Path]], variant: str
) -> None:
    """The acceptance criterion: one set of statements answers either way.

    Byte-identical rather than equivalent, because the report is deterministic
    by design and a difference of any kind is a difference in the data or a
    defect.

    Parameters:
        reports: The two report directories per variant.
        variant: Which invocation is being compared.
    """
    from_tree, from_index = reports[variant]
    assert (from_tree / 'report.md').read_bytes() == (from_index / 'report.md').read_bytes()


@pytest.mark.parametrize('variant', sorted(GOLDEN_VARIANTS))
def test_the_report_is_not_empty(reports: dict[str, tuple[Path, Path]], variant: str) -> None:
    """Two empty reports would be byte-identical and would prove nothing.

    The comparison above is an equality, so it would pass over a tree that
    yielded no record at all.  This is what says the tree was read.

    Parameters:
        reports: The two report directories per variant.
        variant: Which invocation is being read.
    """
    from_tree, _from_index = reports[variant]
    assert _A_TREE_IMAGE in (from_tree / 'report.md').read_text(encoding='utf-8')


@pytest.mark.parametrize('variant', sorted(GOLDEN_VARIANTS))
def test_the_tree_reproduces_the_frozen_report(
    reports: dict[str, tuple[Path, Path]], variant: str
) -> None:
    """The frozen output is written from the rows, and the documents reproduce it.

    Held here as well as against the index, so the frozen file and the report it
    pins are not both produced by one route: a defect the two storages shared
    would pass the comparison against each other and this one would still see it.

    Parameters:
        reports: The two report directories per variant.
        variant: Which invocation is being compared.
    """
    from_tree, _from_index = reports[variant]
    frozen = (GOLDEN_DIR / variant / 'report.md').read_text(encoding='utf-8')
    assert (from_tree / 'report.md').read_text(encoding='utf-8') == frozen


def test_the_report_covers_every_image_the_tree_holds(
    reports: dict[str, tuple[Path, Path]],
) -> None:
    """And the count is the tree's, not whatever the walk reached first.

    The export writes one row per image plus its header, so an unfiltered run
    over the tree exports as many rows as the tree holds documents.
    """
    from_tree, _from_index = reports['full']
    rows = (from_tree / 'images.csv').read_text(encoding='utf-8').splitlines()
    assert len(rows) == len(list(RESULTS_TREE.rglob('*_metadata.json'))) + 1


def _csv_rows(path: Path) -> list[str]:
    """Read the data rows of a CSV export as text, without its header.

    Parameters:
        path: The CSV file.

    Returns:
        One string per data row.
    """
    return path.read_text(encoding='utf-8').splitlines()[1:]


def _csv_header(path: Path) -> str:
    """Read the header row of a CSV export.

    Parameters:
        path: The CSV file.

    Returns:
        The header line.
    """
    return path.read_text(encoding='utf-8').splitlines()[0]


@pytest.mark.parametrize('variant', sorted(GOLDEN_VARIANTS))
def test_the_csv_export_holds_the_same_rows(
    reports: dict[str, tuple[Path, Path]], variant: str
) -> None:
    """The export covers the same images with the same values, out of either storage.

    Compared as sorted rows rather than as bytes, and deliberately: the export
    writes a row where the pass reads it rather than collecting and sorting every
    one of them, which is what makes it a streaming write.  The seam promises no
    order, a walk finds documents in directory order and a query returns them in
    the server's collation of the key, so the file says which images were
    exported and not in what sequence.  The export leads with the column an
    operator sorts on for exactly that reason.

    Parameters:
        reports: The two report directories per variant.
        variant: Which invocation is being compared.
    """
    from_tree, from_index = reports[variant]
    assert sorted(_csv_rows(from_tree / 'images.csv')) == sorted(
        _csv_rows(from_index / 'images.csv')
    )


@pytest.mark.parametrize('variant', sorted(GOLDEN_VARIANTS))
def test_the_csv_export_headers_match(reports: dict[str, tuple[Path, Path]], variant: str) -> None:
    """The header is a header, not a row, so it is compared as one.

    Sorting the whole file would let a header that differed between the two
    storages sort into the middle of the rows and pass unnoticed.  What the
    header holds, and in what order, is pinned against a written-out list in
    ``test_report_regression``.

    Parameters:
        reports: The two report directories per variant.
        variant: Which invocation is being compared.
    """
    from_tree, from_index = reports[variant]
    assert _csv_header(from_tree / 'images.csv') == _csv_header(from_index / 'images.csv')


@pytest.mark.parametrize('variant', sorted(GOLDEN_VARIANTS))
def test_the_csv_export_writes_its_header_once(
    reports: dict[str, tuple[Path, Path]], variant: str
) -> None:
    """A header repeated among the rows is a file every reader mis-parses.

    The comparison above reads line one and the comparison before it reads the
    rest, so a header written a second time among the rows would be compared as
    a row by one of them and never as a header.

    Parameters:
        reports: The two report directories per variant.
        variant: Which invocation is being read.
    """
    from_tree, _from_index = reports[variant]
    assert _csv_header(from_tree / 'images.csv') not in _csv_rows(from_tree / 'images.csv')


@pytest.mark.parametrize('variant', sorted(GOLDEN_VARIANTS))
def test_every_filelist_is_written_alike(
    reports: dict[str, tuple[Path, Path]], variant: str
) -> None:
    """The drill-down lists feed back into re-runs, so they are output too.

    Parameters:
        reports: The two report directories per variant.
        variant: Which invocation is being compared.
    """
    from_tree, from_index = reports[variant]
    written = {
        path.name: path.read_text(encoding='utf-8')
        for path in sorted(from_tree.glob('filelists/*.txt'))
    }
    control = {
        path.name: path.read_text(encoding='utf-8')
        for path in sorted(from_index.glob('filelists/*.txt'))
    }
    assert written == control


def test_the_full_variant_wrote_filelists(reports: dict[str, tuple[Path, Path]]) -> None:
    """Otherwise the comparison above is two empty mappings agreeing."""
    from_tree, _from_index = reports['full']
    assert len(list(from_tree.glob('filelists/*.txt'))) > 0


# ---------------------------------------------------------------------------
# Files that yielded no record
# ---------------------------------------------------------------------------

_NO_RECORD_LINE = 'Files that yielded no record: '
"""How the line naming the count opens, for reading the number off a report."""


def _no_record_count(out: Path) -> int:
    """Read the count of files that yielded no record out of a written report.

    Parameters:
        out: The directory the report was written into.

    Returns:
        The number the report printed.

    Raises:
        ValueError: If the report printed no such line, which is a section that
            stopped being printed rather than a count of zero.
    """
    for line in (out / 'report.md').read_text(encoding='utf-8').splitlines():
        if line.startswith(_NO_RECORD_LINE):
            return int(line[len(_NO_RECORD_LINE) :])
    raise ValueError(f'{out / "report.md"} prints no line opening {_NO_RECORD_LINE!r}')


def test_the_frozen_tree_yields_a_record_from_every_file(
    reports: dict[str, tuple[Path, Path]],
) -> None:
    """The line is printed at zero, so a reader can tell it from a report that never looked.

    Parameters:
        reports: The two report directories per variant.
    """
    from_tree, _from_index = reports['full']
    assert _no_record_count(from_tree) == 0


# ---------------------------------------------------------------------------
# A bound over the image that records no epoch
# ---------------------------------------------------------------------------

_DATED_TREE_IMAGES = 7
"""How many of the frozen tree's images record an epoch a date bound can compare.

One of its documents was written for an image that never loaded, so it records
no provenance and no epoch.  A bound placed on the date is therefore satisfied
by every other image of the tree and by none of that one.
"""


def test_a_start_date_leaves_out_the_image_that_records_no_epoch(tmp_path: Path) -> None:
    """An image that cannot be placed in time cannot be shown to be inside a bound.

    The bound here is earlier than every epoch the tree records, so the only
    image it may exclude is the one recording none.  A report that read an
    absent date as inside the range would count that image and report eight.
    """
    out = report_from_the_tree(tmp_path / 'report', start_date='1900-01-01')
    assert f'Total images: {_DATED_TREE_IMAGES}' in (out / 'report.md').read_text(encoding='utf-8')


def test_an_end_date_leaves_out_the_image_that_records_no_epoch(tmp_path: Path) -> None:
    """The upper bound is the same rule, and a one-sided test would miss half of it.

    The bound here is later than every epoch the tree records, so again the only
    image it may exclude is the one recording none.
    """
    out = report_from_the_tree(tmp_path / 'report', end_date='2100-01-01')
    assert f'Total images: {_DATED_TREE_IMAGES}' in (out / 'report.md').read_text(encoding='utf-8')


_REFUSALS_PER_ROOT = 3
"""How many files that yield no record each root of the fixture below holds."""


def _two_roots_holding_refusals(tmp_path: Path) -> list[Path]:
    """Write two results roots, each holding one image and three files that are not one.

    Two roots holding the same stubs, because the count is of files rather than
    of keys: a count that read the stub alone would report half of them.  The
    three refusals differ in kind, since all three are ordinary in a real tree:
    one is not JSON at all, one is JSON that is not a navigation document, and
    one is JSON that names a mission and still holds no image to attribute to
    it.  The last names a different mission under each root, so a count that
    read the mission out of the document rather than out of the facts the
    document failed to yield would keep one of the two and drop the other.

    Parameters:
        tmp_path: Directory the roots are written under.

    Returns:
        The two roots.
    """
    roots = [tmp_path / 'primary', tmp_path / 'rescue']
    for root, mission in zip(roots, ('vgiss', 'coiss'), strict=True):
        write_metadata(root, 'VOL/N1294561202_1_CALIB', metadata_document())
        write_refusal(root, 'VOL/edges')
        (root / 'VOL' / 'notes_metadata.json').write_text('{not json', encoding='utf-8')
        write_metadata(root, 'VOL/earlier_schema', {'observation': {'instrument': mission}})
    return roots


def _report_over_roots(roots: Sequence[Path], out: Path, **options: Any) -> Path:
    """Write one report over the documents of the named roots.

    Parameters:
        roots: The results roots to read.
        out: Directory receiving the report.
        options: Report options, passed through to ``build_report``.

    Returns:
        The directory the report was written into.
    """
    out.mkdir(parents=True, exist_ok=True)
    with TreeRecordSource([root.as_posix() for root in roots]) as source:
        build_report(source, out, **options)
    return out


def _report_over_an_index_of(
    roots: Sequence[Path], out: Path, url: str, logger: pdslogger.PdsLogger, **options: Any
) -> Path:
    """Ingest the named roots and write one report from the rows.

    Parameters:
        roots: The results roots to ingest and read.
        out: Directory receiving the report.
        url: The index URL to create and ingest into.
        logger: Logger the ingest reports through.
        options: Report options, passed through to ``build_report``.

    Returns:
        The directory the report was written into.
    """
    ingest_tree(url, list(roots), logger=logger)
    out.mkdir(parents=True, exist_ok=True)
    with index_source(url, roots) as source:
        build_report(source, out, **options)
    return out


def test_a_file_that_yielded_no_record_is_counted(tmp_path: Path) -> None:
    """A report that quietly covered less than the tree is worse than one that says how much less.

    Two roots hold three such files each, so a count keyed on the stub rather
    than on the file would report three where six files were refused.
    """
    roots = _two_roots_holding_refusals(tmp_path)
    out = _report_over_roots(roots, tmp_path / 'report')
    assert _no_record_count(out) == len(roots) * _REFUSALS_PER_ROOT


def test_the_two_storages_agree_on_the_count(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """An ingest records what it refused, so an index answers this as the tree does."""
    roots = _two_roots_holding_refusals(tmp_path)
    over_the_tree = _no_record_count(_report_over_roots(roots, tmp_path / 'from-tree'))
    over_an_index = _no_record_count(
        _report_over_an_index_of(
            roots, tmp_path / 'from-index', index_url(tmp_path / 'index.sqlite3'), quiet_logger
        )
    )
    assert over_the_tree == over_an_index


def test_the_refused_files_are_not_counted_as_images(tmp_path: Path) -> None:
    """They record no instrument, so a per-instrument denominator must not hold them."""
    roots = _two_roots_holding_refusals(tmp_path)
    out = _report_over_roots(roots, tmp_path / 'report')
    assert 'Total images: 2' in (out / 'report.md').read_text(encoding='utf-8')


def test_an_instrument_filter_does_not_narrow_the_count(tmp_path: Path) -> None:
    """A file that yielded no record stays counted whatever mission it happens to name.

    One refusal under each root is a JSON object that names a mission and still
    holds no image, and the two roots name different missions.  A mission filter
    read out of the document rather than out of the facts the document failed to
    yield would drop the one naming the other mission, and this report would
    count five files where six were refused.
    """
    roots = _two_roots_holding_refusals(tmp_path)
    out = _report_over_roots(roots, tmp_path / 'report', instrument='coiss')
    assert _no_record_count(out) == len(roots) * _REFUSALS_PER_ROOT


def test_the_two_storages_agree_on_the_count_under_a_mission_filter(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """A row and a document are narrowed on the same values, so the two counts are one count.

    The index cannot narrow a refusal at all -- a refusal row records no mission
    -- so this is the comparison a walk that applied the filter to the document
    would fail.
    """
    roots = _two_roots_holding_refusals(tmp_path)
    over_the_tree = _no_record_count(
        _report_over_roots(roots, tmp_path / 'from-tree', instrument='coiss')
    )
    over_an_index = _no_record_count(
        _report_over_an_index_of(
            roots,
            tmp_path / 'from-index',
            index_url(tmp_path / 'index.sqlite3'),
            quiet_logger,
            instrument='coiss',
        )
    )
    assert over_the_tree == over_an_index


def test_a_date_filter_does_not_narrow_the_count(tmp_path: Path) -> None:
    """Nor a date: the file was never parsed, so it records no epoch to compare."""
    roots = _two_roots_holding_refusals(tmp_path)
    out = _report_over_roots(roots, tmp_path / 'report', start_date='1999-01-01')
    assert _no_record_count(out) == len(roots) * _REFUSALS_PER_ROOT


def test_an_image_number_filter_does_not_narrow_the_count(tmp_path: Path) -> None:
    """Nor an image range, for the same reason: there is no image name to number."""
    roots = _two_roots_holding_refusals(tmp_path)
    out = _report_over_roots(roots, tmp_path / 'report', min_image='N9999999999')
    assert _no_record_count(out) == len(roots) * _REFUSALS_PER_ROOT


def test_a_filter_that_selects_nothing_still_reports_the_count(tmp_path: Path) -> None:
    """The count is of the roots, so it survives a selection that empties the report."""
    roots = _two_roots_holding_refusals(tmp_path)
    out = _report_over_roots(roots, tmp_path / 'report', min_image='N9999999999')
    assert 'No images match the filters.' in (out / 'report.md').read_text(encoding='utf-8')
    assert _no_record_count(out) == len(roots) * _REFUSALS_PER_ROOT


# ---------------------------------------------------------------------------
# The stream promises no order
# ---------------------------------------------------------------------------


_TIED_STUB = 'VOL/N1294561202_1_CALIB'
"""The stub two roots of the shuffled fixture both hold, which is the tie to break."""

_ORDERS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (3, 2, 1, 0),
    (1, 0, 3, 2),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (0, 3, 1, 2),
)
"""The orders the same facts are fed to the report in.

Written out rather than drawn at random, so a failure is reproducible and so
that the two orders that matter are certainly among them.  One root's copy of
:data:`_TIED_STUB` arrives before the other's in some of them and after it in
the rest, which is the tie a sort keyed on an image name alone leaks the input
order through.  And the four confidences reach the mean in an order whose
left-to-right sum rounds to 0.709 in some of them and to 0.710 in the rest,
which is the drift an accumulated sum leaks the input order through.
"""


def _shuffled_fixture(root: Path) -> list[Path]:
    """Write two roots that between them make the report's ordering assumptions visible.

    Both roots hold :data:`_TIED_STUB`, under one image name, with offsets that
    reach the same fraction of the search limit and print differently.  The two
    are therefore one row apart in the suspect table on a key that reads the
    image name alone, and the row that comes first is whichever the stream
    yielded first.

    Their four confidences are chosen so that adding them up left to right
    rounds to two different numbers depending on the order they arrive in.

    Parameters:
        root: Directory the two roots are written under.

    Returns:
        The two roots.
    """
    primary = root / 'primary'
    rescue = root / 'rescue'
    # |dV| is 0.98 of the NAC 1024 CALIB margin of (50, 140) either way round.
    write_metadata(
        primary,
        _TIED_STUB,
        metadata_document(
            image_name='N1294561202_1_CALIB.IMG',
            image_shape=[1024, 1024],
            offset=[49.0, 10.0],
            per_technique=[technique('BodyLimbNav', (1.0, 1.0), confidence=0.949)],
        ),
    )
    write_metadata(
        primary,
        'VOL/N1294562000_1_CALIB',
        metadata_document(
            image_name='N1294562000_1_CALIB.IMG',
            image_shape=[1024, 1024],
            offset=[1.0, 1.0],
            per_technique=[technique('BodyLimbNav', (1.0, 1.0), confidence=0.924)],
        ),
    )
    write_metadata(
        rescue,
        _TIED_STUB,
        metadata_document(
            image_name='N1294561202_1_CALIB.IMG',
            image_shape=[1024, 1024],
            offset=[-49.0, 10.0],
            per_technique=[technique('BodyLimbNav', (1.0, 1.0), confidence=0.778)],
        ),
    )
    write_metadata(
        rescue,
        'VOL/N1294563000_1_CALIB',
        metadata_document(
            image_name='N1294563000_1_CALIB.IMG',
            image_shape=[1024, 1024],
            offset=[2.0, 2.0],
            per_technique=[technique('BodyLimbNav', (1.0, 1.0), confidence=0.187)],
        ),
    )
    return [primary, rescue]


@dataclass(frozen=True)
class _ShuffledRun:
    """What replaying one set of facts in several orders produced.

    Parameters:
        facts_read: How many facts the fixture tree yielded, so a comparison
            over a stream shorter than the fixture is recognizable.
        reports: The bytes of ``report.md`` from each order, in the order the
            orders are written down.
    """

    facts_read: int
    reports: tuple[bytes, ...]


@pytest.fixture(scope='module')
def shuffled_reports(tmp_path_factory: pytest.TempPathFactory) -> _ShuffledRun:
    """Write the same facts into a report once per order, and return the reports.

    Parameters:
        tmp_path_factory: Factory the trees and the outputs live under.

    Returns:
        What each order produced, and how many facts were replayed into it.
    """
    root = tmp_path_factory.mktemp('shuffled')
    roots = _shuffled_fixture(root)
    with TreeRecordSource([one.as_posix() for one in roots]) as source:
        found = list(source.facts(Selection()))
    written = []
    for index, order in enumerate(_ORDERS):
        out = root / f'order-{index}'
        out.mkdir()
        build_report(ReplayedFacts([found[place] for place in order]), out, top_n=5)
        written.append((out / 'report.md').read_bytes())
    return _ShuffledRun(facts_read=len(found), reports=tuple(written))


def test_every_document_of_the_shuffled_fixture_is_replayed(
    shuffled_reports: _ShuffledRun,
) -> None:
    """An order over fewer facts than the fixture holds leaves the tie out of some of them."""
    assert shuffled_reports.facts_read == len(_ORDERS[0])


def test_every_order_produces_one_report(shuffled_reports: _ShuffledRun) -> None:
    """A comparison over one report passes for the wrong reason."""
    assert len(shuffled_reports.reports) == len(_ORDERS)


def test_the_report_does_not_depend_on_the_order_the_facts_arrive_in(
    shuffled_reports: _ShuffledRun,
) -> None:
    """The seam promises no order, so a report that depended on one would depend on the storage.

    Every section either counts, reduces to a minimum, sums exactly, or sorts on
    a key that carries the pair identifying an image; an image name alone is not
    unique across roots, and Python's sort is stable, so a key that stopped at
    the name would print two tied rows in whatever order the stream yielded them.
    """
    assert len(set(shuffled_reports.reports)) == 1


def test_the_shuffled_reports_hold_both_tied_rows(shuffled_reports: _ShuffledRun) -> None:
    """Otherwise the tie the orders are built around is not in the output at all.

    Two rows of one image name in the one always-on table that prints a row per
    image, reaching the same fraction of the search limit and printing different
    offsets, is the whole of what a name-only sort key would order by arrival.
    """
    text = shuffled_reports.reports[0].decode('utf-8')
    suspects = text.split('## Suspect offsets')[1].split('## BOTSIM')[0]
    assert suspects.count('| N1294561202 | coiss | ') == 2


def test_the_tied_rows_print_differently(shuffled_reports: _ShuffledRun) -> None:
    """Two identical rows would swap places invisibly and prove nothing."""
    text = shuffled_reports.reports[0].decode('utf-8')
    suspects = text.split('## Suspect offsets')[1].split('## BOTSIM')[0]
    offsets = [
        line.split('|')[3].strip()
        for line in suspects.splitlines()
        if line.startswith('| N1294561202 | coiss | ')
    ]
    assert offsets == ['49.000', '-49.000']


def test_the_shuffled_reports_hold_the_exact_mean_confidence(
    shuffled_reports: _ShuffledRun,
) -> None:
    """The four confidences average to 0.7095, which rounds up only when summed exactly.

    Added left to right in some of these orders they come to 0.7094999999999999
    instead, which prints one digit lower.  The mean is computed over the values
    rather than accumulated as they arrive, so the printed number is the one the
    values have whichever order they came in.
    """
    text = shuffled_reports.reports[0].decode('utf-8')
    assert '| BodyLimbNav | coiss | 4 (100.0%) | 4 (100.0%) | 0.710 |' in text


# ---------------------------------------------------------------------------
# What the export writes for a structured column
# ---------------------------------------------------------------------------

_COVARIANCE = [[0.0961, 0.01, 0.0025], [0.01, 0.0784, -0.005], [0.0025, -0.005, 0.0009]]
"""A 3x3 covariance of the kind a twist-fitted result records."""

_CMATRIX = [
    0.955336489125606,
    -0.29552020666133955,
    0.0,
    0.29552020666133955,
    0.955336489125606,
    0.0,
    0.0,
    0.0,
    1.0,
]
"""A rotation whose terms need every digit a double carries to be written back."""

_CMATRIX_ORIGINAL = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
"""The attitude the kernels gave, before the navigation corrected it."""

_KERNELS = ['naif0012.tls', 'cas00172.tsc', 'cas_v40.tf']
"""A kernel list, which is the one column holding a list of plain text."""

_EXCLUDED = ['StarRefineNav', 'BodyBlobNav']
"""An exclusion set, recorded in the order the ensemble wrote it."""

_DIAGNOSTICS = {'peak': 0.9531, 'iterations': 7, 'converged': True}
"""A technique's diagnostics, which is the one column holding a mapping."""

_JSON_CELLS = {
    'covariance_px2': '[[0.0961, 0.01, 0.0025], [0.01, 0.0784, -0.005], [0.0025, -0.005, 0.0009]]',
    'excluded_from_consensus': '["StarRefineNav", "BodyBlobNav"]',
    'spice_kernels': '["naif0012.tls", "cas00172.tsc", "cas_v40.tf"]',
    'cmatrix': '[0.955336489125606, -0.29552020666133955, 0.0, 0.29552020666133955, '
    '0.955336489125606, 0.0, 0.0, 0.0, 1.0]',
    'cmatrix_original': '[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]',
}
"""What the export writes into each structured column of the document below.

Written out rather than produced by encoding the values again, because that is
the whole question: a container reaches the export as a container whichever
storage answered, and what a reader gets back is whatever the export makes of
it.  Neither the shortest round-trip form of a float nor the order of a
mapping's keys is guaranteed by anything, so both are stated here.
"""


def _document_with_every_json_column() -> dict[str, Any]:
    """Build a document populating every column the export encodes as JSON.

    Returns:
        The document.
    """
    document = metadata_document(
        image_name='N1294561202_1_CALIB.IMG',
        image_shape=[1024, 1024],
        excluded=_EXCLUDED,
        per_technique=[technique('BodyLimbNav', (1.0, 1.0))],
        times={'midtime_et': 136576860.1724845},
        pointing={
            'camera_frame': 'CASSINI_ISS_NAC',
            'camera_frame_id': -82360,
            'ck_frame_id': -82000,
            'cmatrix': _CMATRIX,
            'cmatrix_original': _CMATRIX_ORIGINAL,
        },
    )
    document['navigation_result']['covariance_px2'] = _COVARIANCE
    document['navigation_result']['provenance']['spice_kernels'] = _KERNELS
    document['navigation_result']['per_technique'][0]['diagnostics'] = _DIAGNOSTICS
    return document


@pytest.fixture
def json_columns(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> dict[str, dict[str, str]]:
    """Export the same structured document out of each storage and read the cells back.

    Parameters:
        tmp_path: Directory the tree, the index and the exports live under.
        quiet_logger: Logger the ingest reports through.

    Returns:
        Per storage name, the one exported row keyed by column name.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1294561202_1_CALIB', _document_with_every_json_column())
    from_tree = _report_over_roots([root], tmp_path / 'from-tree', csv_export=True)
    from_index = _report_over_an_index_of(
        [root],
        tmp_path / 'from-index',
        index_url(tmp_path / 'index.sqlite3'),
        quiet_logger,
        csv_export=True,
    )
    return {
        name: next(
            iter(csv.DictReader((out / 'images.csv').read_text(encoding='utf-8').splitlines()))
        )
        for name, out in (('tree', from_tree), ('index', from_index))
    }


def test_the_export_writes_each_structured_column_as_the_json_it_holds(
    json_columns: dict[str, dict[str, str]],
) -> None:
    """A CSV carrying a Python container's repr is one nothing else can read back."""
    written = {column: json_columns['tree'][column] for column in _JSON_CELLS}
    assert written == _JSON_CELLS


def test_the_two_storages_write_the_same_json(
    json_columns: dict[str, dict[str, str]],
) -> None:
    """One storage hands back stored text and the other a decoded container, and the file agrees."""
    from_tree = {column: json_columns['tree'][column] for column in _JSON_CELLS}
    from_index = {column: json_columns['index'][column] for column in _JSON_CELLS}
    assert from_tree == from_index


def test_a_technique_carries_its_diagnostics_through_either_storage(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """The export writes the image's own columns, so this is read at the seam instead.

    Diagnostics are a per-technique value and no column of ``images.csv``, so
    what pins them is that both storages hand back the mapping the document
    recorded.
    """
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1294561202_1_CALIB', _document_with_every_json_column())
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [root], logger=quiet_logger)
    with TreeRecordSource([root.as_posix()]) as walked:
        from_tree = list(walked.facts(Selection()))
    with index_source(url, [root]) as indexed:
        from_index = list(indexed.facts(Selection()))
    assert _diagnostics_of(from_tree) == _diagnostics_of(from_index)


def _diagnostics_of(found: Sequence[ImageFacts | UnreadableFile]) -> list[Any]:
    """Read every technique's diagnostics out of a stream of facts.

    Parameters:
        found: What a source yielded.

    Returns:
        One diagnostics mapping per technique entry, in stream order.
    """
    return [
        entry['diagnostics']
        for facts in found
        if isinstance(facts, ImageFacts)
        for entry in facts.techniques
    ]


def test_the_diagnostics_reaching_the_seam_are_the_recorded_ones(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Otherwise the comparison above is two storages agreeing on an empty mapping."""
    root = tmp_path / 'results'
    write_metadata(root, 'VOL/N1294561202_1_CALIB', _document_with_every_json_column())
    with TreeRecordSource([root.as_posix()]) as walked:
        found = list(walked.facts(Selection()))
    assert _diagnostics_of(found) == [_DIAGNOSTICS]


def test_the_covariance_reaching_the_export_is_the_recorded_one() -> None:
    """The pinned cell has to be what encoding the recorded matrix produces."""
    assert _JSON_CELLS['covariance_px2'] == json.dumps(_COVARIANCE)


# ---------------------------------------------------------------------------
# The command line, over a tree
# ---------------------------------------------------------------------------


def _from_the_tree_at(root: Path, out: Path, *options: str) -> int:
    """Run the report over one named tree.

    Parameters:
        root: The results root to read.
        out: Directory receiving the report.
        options: Further command-line options.

    Returns:
        The exit code.
    """
    return main_report(['--nav-results-root', str(root), '--output-dir', str(out), *options])


def test_naming_a_root_with_no_index_is_refused(tmp_path: Path) -> None:
    """``--root`` selects among the roots an index holds, and there is no index.

    Read as a second spelling of ``--nav-results-root`` it would quietly report
    on a tree the operator did not name; refused, it says which flag to use.
    """
    with pytest.raises(SystemExit) as caught:
        main_report(['--root', str(RESULTS_TREE), '--output-dir', str(tmp_path / 'report')])
    assert caught.value.code == 2


def test_the_refusal_names_the_flag_that_does_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusal that does not say what to type instead is one nobody can act on.

    Asserted against the sentence the refusal is made of rather than against the
    flag alone: argparse prints its usage line before any message it is given,
    and that line names every flag this program takes, so a bare flag name is
    satisfied by any refusal at all.
    """
    with pytest.raises(SystemExit):
        main_report(['--root', str(RESULTS_TREE), '--output-dir', str(tmp_path / 'report')])
    assert 'Name the trees to report on with --nav-results-root' in capsys.readouterr().err


def test_a_root_that_cannot_be_listed_fails_the_run(tmp_path: Path) -> None:
    """A report over a tree nobody could list would cover less and not say so."""
    assert _from_the_tree_at(tmp_path / 'absent', tmp_path / 'report') == 1


def test_a_root_that_cannot_be_listed_writes_no_report(tmp_path: Path) -> None:
    """The exit status and the output agree: nothing was reported on."""
    out = tmp_path / 'report'
    _from_the_tree_at(tmp_path / 'absent', out)
    assert not (out / 'report.md').exists()


def test_a_root_that_cannot_be_listed_says_which(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An operator with several roots named needs to know which of them failed."""
    _from_the_tree_at(tmp_path / 'absent', tmp_path / 'report')
    assert 'absent' in capsys.readouterr().err


def test_the_run_says_which_tree_it_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One full read of every document is worth announcing before it starts."""
    _from_the_tree_at(RESULTS_TREE, tmp_path / 'report')
    assert str(RESULTS_TREE) in capsys.readouterr().out


def test_two_trees_are_reported_on_together(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """``--nav-results-root`` repeats, as it does for the ingest.

    Held against an index ingested from both, so the second root is read rather
    than merely accepted.
    """
    second = tmp_path / 'second'
    write_metadata(
        second, 'VOL/N1666666666_1_CALIB', metadata_document(image_name='N1666666666_1_CALIB.IMG')
    )
    from_tree = tmp_path / 'from-tree'
    from_index = tmp_path / 'from-index'
    exit_code = main_report(
        [
            '--nav-results-root',
            str(RESULTS_TREE),
            '--nav-results-root',
            str(second),
            '--output-dir',
            str(from_tree),
        ]
    )
    url = index_url(tmp_path / 'control.sqlite3')
    ingest_tree(url, [RESULTS_TREE, second], logger=quiet_logger)
    main_report(['--results-index-db', url, '--output-dir', str(from_index)])
    assert exit_code == 0
    assert (from_tree / 'report.md').read_bytes() == (from_index / 'report.md').read_bytes()


def test_the_second_tree_reaches_the_report(tmp_path: Path) -> None:
    """Otherwise the comparison above holds two reports of the first root."""
    second = tmp_path / 'second'
    write_metadata(
        second, 'VOL/N1666666666_1_CALIB', metadata_document(image_name='N1666666666_1_CALIB.IMG')
    )
    out = tmp_path / 'report'
    main_report(
        [
            '--nav-results-root',
            str(RESULTS_TREE),
            '--nav-results-root',
            str(second),
            '--output-dir',
            str(out),
        ]
    )
    assert 'N1666666666' in (out / 'report.md').read_text(encoding='utf-8')


# ---------------------------------------------------------------------------
# An index answers for the roots it finished ingesting
# ---------------------------------------------------------------------------


def _index_over_two_roots_one_half_ingested(
    tmp_path: Path, logger: pdslogger.PdsLogger
) -> tuple[str, Path, Path]:
    """Ingest two roots, then leave the second one's newest run unfinished.

    Parameters:
        tmp_path: Directory the roots and the index live under.
        logger: Logger the ingest reports through.

    Returns:
        The index URL, the root whose ingest completed, and the root whose
        newest run did not.
    """
    finished = tmp_path / 'finished'
    unfinished = tmp_path / 'unfinished'
    write_metadata(
        finished,
        'VOL/N1294561202_1_CALIB',
        metadata_document(image_name='N1294561202_1_CALIB.IMG'),
    )
    write_metadata(
        unfinished,
        'VOL/N1294562000_1_CALIB',
        metadata_document(image_name='N1294562000_1_CALIB.IMG'),
    )
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [finished, unfinished], logger=logger)
    engine = open_index(url)
    with engine.begin() as connection:
        connection.execute(
            INGEST_RUNS.insert().values(
                root_url=unfinished.as_posix(),
                started_utc='2026-08-07T00:00:00+00:00',
                finished_utc=None,
                schema_version=SCHEMA_VERSION,
            )
        )
    engine.dispose()
    return url, finished, unfinished


def test_a_report_over_every_root_leaves_out_a_half_ingested_one(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Under a root whose newest ingest died, absence of a row says nothing.

    A report that counted such a root would be reading absence it has no license
    to read, so the roots reported over are the roots the index holds a completed
    ingest of.
    """
    url, _finished, _unfinished = _index_over_two_roots_one_half_ingested(tmp_path, quiet_logger)
    out = tmp_path / 'report'
    main_report(['--results-index-db', url, '--output-dir', str(out)])
    assert 'N1294562000' not in (out / 'report.md').read_text(encoding='utf-8')


def test_the_completed_root_is_still_reported(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Otherwise the narrowing above is a report that covered nothing at all."""
    url, _finished, _unfinished = _index_over_two_roots_one_half_ingested(tmp_path, quiet_logger)
    out = tmp_path / 'report'
    main_report(['--results-index-db', url, '--output-dir', str(out)])
    assert 'N1294561202' in (out / 'report.md').read_text(encoding='utf-8')


def test_the_half_ingested_root_holds_rows_to_leave_out(
    tmp_path: Path, quiet_logger: pdslogger.PdsLogger
) -> None:
    """Otherwise the narrowing is an empty root being left out of an empty report.

    The row is there and the report does not read it, which is the whole of the
    ruling: a root whose newest ingest died is one nothing may read absence from,
    and the rows an earlier pass did write under it are not the root.
    """
    url, _finished, unfinished = _index_over_two_roots_one_half_ingested(tmp_path, quiet_logger)
    engine = open_index(url)
    try:
        with engine.connect() as connection:
            held = [
                str(row[0])
                for row in connection.execute(
                    sqlalchemy.select(IMAGES.c.image_name).where(
                        IMAGES.c.root_url == unfinished.as_posix()
                    )
                )
            ]
    finally:
        engine.dispose()
    assert held == ['N1294562000_1_CALIB.IMG']


def test_the_report_reads_the_tree_at_the_tuning_the_configuration_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A report over a tree is a front end onto it like any other, and honors the section."""
    configured = TreeTuning(walk_threads=3, walk_directories_at_once=3)
    handed: list[Any] = []

    class Recording(TreeRecordSource):
        """A source that notes the tuning it was built with."""

        def __init__(self, roots: Any, **kwargs: Any) -> None:
            handed.append(kwargs.get('tuning'))
            super().__init__(roots, **kwargs)

    monkeypatch.setattr(report_module, 'get_results_tree_tuning', lambda config: configured)
    monkeypatch.setattr(report_module, 'TreeRecordSource', Recording)
    assert _from_the_tree_at(RESULTS_TREE, tmp_path / 'report') == 0
    assert handed == [configured]
