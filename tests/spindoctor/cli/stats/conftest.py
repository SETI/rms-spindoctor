"""Tree and index helpers for the statistics report tests.

The document factories the trees here are built out of are shared with the rest
of the subtree and live in ``tests/spindoctor/conftest.py``, and the pass that
writes an index is defined once beside the tests of that pass, in
``tests/spindoctor/cli/results_index/conftest.py``.  What is left here is the
frozen fixture tree, the report invocations measured against it, and the
sources a report is handed.

:class:`ReplayedFacts` is here because a source promises no order, and the way
to measure that a report does not depend on one is to hand it the same facts in
an order the test chose.  Two modules do that -- one over a whole report, one
over a section whose reduction has two candidates to choose between -- so the
stand-in source is written once.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import pdslogger
import pytest
from tests.packaged_sections import packaged_tree_tuning
from tests.spindoctor.cli.results_index.conftest import (
    build_tree,
    complete,
    fan_out,
    postgres_decoy_schema,
    postgres_schema,
    postgres_server_url,
    postgres_url,
    quiet_logger,
    reported,
    run_rows,
    write_metadata_in_each,
)
from tests.spindoctor.conftest import index_url, ingest_tree

from spindoctor.cli.stats.report import build_report
from spindoctor.nav_records import (
    ImageFacts,
    ListedRecord,
    NavRecord,
    RecordSource,
    Selection,
    TreeRecordSource,
    UnreadableFile,
)
from spindoctor.results_index import IndexRecordSource, open_index

# The report is measured over an index something ingested and, on the postgres
# tier, over a schema of its own; both are defined beside the tests of the pass
# that writes them, so what a report test needs of either is re-exported here
# rather than restated.
__all__ = [
    'build_tree',
    'complete',
    'fan_out',
    'postgres_decoy_schema',
    'postgres_schema',
    'postgres_server_url',
    'postgres_url',
    'quiet_logger',
    'reported',
    'run_rows',
    'write_metadata_in_each',
]


DATA_DIR = Path(__file__).resolve().parent / 'data'
"""Directory holding the fixture results tree and the frozen report output."""

RESULTS_TREE = DATA_DIR / 'results_tree'
"""Fixture results tree the report regression is measured over."""

GOLDEN_DIR = DATA_DIR / 'golden'
"""Frozen report and CSV output this tree produces, which a change must reproduce."""

GOLDEN_VARIANTS: dict[str, dict[str, Any]] = {
    'full': {'top_n': 5, 'filelists': True, 'csv_export': True},
    'filtered': {
        'instrument': 'coiss',
        'min_image': '1294561202',
        'max_image': '1294563000',
        'top_n': 3,
        'csv_export': True,
    },
}
"""The two report invocations the frozen output under :data:`GOLDEN_DIR` holds.

Between them they cover the report with every drill-down on and with a narrowing
that leaves one instrument, so the parity of the two storages is measured over
the same two invocations the frozen output pins.
"""


class ReplayedFacts:
    """A record source handing back facts a test already read, in an order it chose.

    The seam promises no order, so the way to measure that the report does not
    depend on one is to give it the same facts several times over in several
    orders.  Reading them out of a real source once and replaying them is what
    makes the orders comparable: two storages differ in more than their order,
    and a difference in the output would then say nothing about which.

    Parameters:
        facts: What to yield, in the order to yield it.
    """

    def __init__(self, facts: Sequence[ImageFacts | UnreadableFile]) -> None:
        self._facts = tuple(facts)

    def record(self, stub: str) -> NavRecord:
        """Refuse the per-image lookup, which nothing being measured here asks for.

        Parameters:
            stub: The image's results path stub.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError('replayed facts answer no per-image lookup')

    def records(self, selection: Selection) -> Iterator[NavRecord | UnreadableFile]:
        """Refuse the record stream, which nothing being measured here asks for.

        Parameters:
            selection: Which records were asked for.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError('replayed facts answer no record stream')

    def facts(self, selection: Selection) -> Iterator[ImageFacts | UnreadableFile]:
        """Yield the held facts, in the order this source was built with.

        Parameters:
            selection: Which images were asked for; the facts were already
                narrowed when they were read, so this narrows nothing further.

        Returns:
            The facts, one at a time.
        """
        return iter(self._facts)

    def listing(self, selection: Selection) -> Iterator[ListedRecord]:
        """Refuse the listing, which nothing being measured here asks for.

        Parameters:
            selection: Which files were asked for.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError('replayed facts answer no listing')

    def describe(self) -> str:
        """Say where these records came from, for a run log.

        Returns:
            A phrase naming the replay rather than a storage.
        """
        return 'facts replayed in an order chosen by a test'

    def close(self) -> None:
        """Release what this source holds open, which is nothing."""

    def __enter__(self) -> RecordSource:
        """Enter a run's use of this source.

        Returns:
            The source itself.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave a run's use of this source.

        Parameters:
            exc_type: The exception's class, when the run is leaving on one.
            exc: The exception, when the run is leaving on one.
            traceback: Its traceback, when the run is leaving on one.
        """
        self.close()


def index_source(url: str, roots: Sequence[Path]) -> IndexRecordSource:
    """Open a record source over the rows an index holds for the given roots.

    Parameters:
        url: The index URL, which must already carry the schema and the rows.
        roots: The results roots whose rows the source answers about.

    Returns:
        The open source, which the caller closes when it is done with it.
    """
    return IndexRecordSource(open_index(url), [root.as_posix() for root in roots], url, ())


@pytest.fixture
def indexed_tree(tmp_path: Path, quiet_logger: pdslogger.PdsLogger) -> Iterator[RecordSource]:
    """Yield a record source over an index built from the frozen fixture tree.

    Parameters:
        tmp_path: Directory the index file is written into.
        quiet_logger: Logger the ingest reports through.

    Yields:
        The open source, reading rows.
    """
    url = index_url(tmp_path / 'index.sqlite3')
    ingest_tree(url, [RESULTS_TREE], logger=quiet_logger)
    with index_source(url, [RESULTS_TREE]) as source:
        yield source


@pytest.fixture
def walked_tree() -> Iterator[RecordSource]:
    """Yield a record source over the documents of the frozen fixture tree.

    The other half of every parity comparison: the same records, read out of the
    files themselves rather than out of an index ingested from them.

    Yields:
        The open source, reading documents.
    """
    with TreeRecordSource([RESULTS_TREE.as_posix()], tuning=packaged_tree_tuning()) as source:
        yield source


def report_from_the_index(
    url: str, out: Path, *, logger: pdslogger.PdsLogger, **options: Any
) -> Path:
    """Ingest the fixture tree into an index and write one report from its rows.

    One definition of the whole cycle -- ingest, open, build, dispose -- so that
    a change to the report's signature is made once rather than once per backend.

    Parameters:
        url: The index URL to create and ingest into.
        out: Directory receiving the report.
        logger: Logger the ingest reports through.
        options: Report options, passed through to ``build_report``.

    Returns:
        The directory the report was written into.
    """
    ingest_tree(url, [RESULTS_TREE], logger=logger)
    out.mkdir(parents=True, exist_ok=True)
    with index_source(url, [RESULTS_TREE]) as source:
        build_report(source, out, **options)
    return out


def report_from_the_tree(out: Path, **options: Any) -> Path:
    """Write one report over the documents of the fixture tree, opening no index.

    Parameters:
        out: Directory receiving the report.
        options: Report options, passed through to ``build_report``.

    Returns:
        The directory the report was written into.
    """
    out.mkdir(parents=True, exist_ok=True)
    with TreeRecordSource([RESULTS_TREE.as_posix()], tuning=packaged_tree_tuning()) as source:
        build_report(source, out, **options)
    return out
