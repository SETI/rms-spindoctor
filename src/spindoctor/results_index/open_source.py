"""Opening whichever half of the record seam a run resolved.

A program reading navigation records names an index or does not, and that is the
whole of the choice: with no index the records are the documents, and with one
they are the rows an ingest pass wrote.  Neither half decides it, because a
source that could return the other one would be a source with two storages
behind it; the choice is made once, here, and every program makes it by calling
this.

It lives beside the half that reads rows rather than beside the half that reads
documents, for the reason that split exists: choosing needs the index layer, and
:mod:`spindoctor.nav_records` may not acquire one.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import sqlalchemy
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.nav_records import RecordSource, TreeRecordSource, TreeTuning, distinct_roots
from spindoctor.results_index.record_source import IndexRecordSource
from spindoctor.results_index.roots import open_index_for_roots

__all__ = ['open_record_source']


def open_record_source(
    roots: Sequence[str | Path | FCPath],
    *,
    results_index_db_url: str | None = None,
    columns: Sequence[sqlalchemy.Column[Any]] = (),
    logger: PdsLogger | None = None,
    tuning: TreeTuning | None = None,
) -> RecordSource:
    """Open the source a run reads its navigation records through.

    With no index URL the source reads documents, which is every program's
    default.  With one, the index is opened and every root is checked against its
    ingest bookkeeping before anything is read: a root the index has not fully
    ingested cannot say what it holds, so it is refused rather than read short.

    Parameters:
        roots: The results roots to read, in the order questions are answered
            about them.  Two spellings of one root are one root.
        results_index_db_url: Connection URL of the results index, or None to
            read the documents.
        columns: The columns of ``images`` a consumer's *records* are rebuilt
            from.  Ignored when the documents are read, which carry every field
            whatever is selected; ignored by a stream of facts, which is the
            whole row over either storage; and needed by no caller that asks
            only for a listing.
        logger: The caller's own logger, lent to the source for the one line it
            has to say: that it declined to descend a directory it had already
            listed under another name.  None says nothing at all.  Nothing here
            constructs one or reaches for a program's own, because a layer with
            a voice its caller did not configure would report a run's work
            somewhere the run does not control.
        tuning: How much of a pass over the documents runs at once, which the
            program resolves from its configuration and passes down the way it
            passes its logger; None is the library's own defaults.  A run that
            reads rows never walks a tree, so it is ignored there.

    Returns:
        The source, which the caller closes when it is done with it and which is
        usable as a context manager.

    Raises:
        ValueError: If no root is named, or one of them is not a location; or if
            the index cannot be opened, is not an index, or was written by
            another version of the schema; or if a named root has no completed
            ingest run in it.
    """
    root_urls = distinct_roots(roots)
    if not root_urls:
        raise ValueError('a record source needs at least one results root to read')
    if results_index_db_url is None:
        return TreeRecordSource(root_urls, logger=logger, tuning=tuning)
    engine = open_index_for_roots(results_index_db_url, root_urls)
    return IndexRecordSource(engine, root_urls, results_index_db_url, columns)
