"""One ingest pass over each of the named results roots.

Each root is walked once; the files the index has already read are dropped from
what the pass reads; the rest are ingested in chunks; and the root is pruned of
the rows whose documents have left the tree.

What a row is keyed by
----------------------

``(root_url, results_path_stub)``.  The stub is derived from the document's own
location -- its path under the ingest root with ``_metadata.json`` removed --
and never from anything inside the document, because that is precisely how the
navigator chose the path it wrote to.  The root is the ``nav_results_root``,
normalized once by :func:`~spindoctor.results_index.normalize_root_url` so that
a consumer's spelling of the same root matches.  Ingesting a subdirectory of a
results root would produce stubs no consumer's lookup can match, so the root
identity is part of the contract rather than a convenience.

What is read again
------------------

A file whose recorded ``(mtime_ns, size_bytes)`` still matches the listing is
not read at all.  A backend whose listing supplies neither metric cannot answer
that question, so such a root is re-read in full, with a warning saying so.

Those two metrics are everything a listing supplies, so a document rewritten in
place that kept both of them is skipped, and its row goes on recording what the
document before it said.  Reading the file to find out whether it needs reading
is the retrieval this skip exists to avoid, so ``force`` is the answer to that
rather than a finer comparison, and the consequence for a consumer is stated
with the rest of what the index answers differently in
:mod:`spindoctor.dataset.results_filter`.

What leaving the tree costs
---------------------------

Presence has to mean what absence means.  A document deleted from the tree
leaves a row that would answer for an image the tree no longer holds, so the
rows of one root whose stub the walk did not find are deleted with it.  That is
sound only for a pass that listed the whole root: a worker handed a share of a
root has no evidence about the stubs outside its share, and deleting on that
evidence would delete its peers' work.  The prune therefore takes a listing of
a whole root and nothing else, and a share of a root is a list of files that
cannot become one.

A walk supplies one or it stops: a directory it cannot list ends the pass where
it finds it, so a run that reaches the prune at all listed every directory
under its root.  A completed pass therefore prunes unless it was told not to,
which is what keeps one unreadable subdirectory from holding a root's stale
rows across any number of passes that finished.

What declining to prune costs
-----------------------------

A pass may be told to leave those rows alone, and then presence stops implying
that the document is still there: a row outlives its document, so a consumer
asking whether an image has been navigated is answered yes for one whose result
the tree no longer holds.

Absence is untouched, because skipping a delete adds no row.  Every answer read
from absence -- "this image was never navigated", the selection filter that
asks for images with no result, the refusal of a root nobody has ingested --
therefore means exactly what it meant.  That is sound only because a pass covers
a whole root: there is no ingest of part of one, so a pass that reaches the
prune listed everything, and the rows it declines to remove are exactly the rows
it would have removed.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import sqlalchemy
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.results_index.chunks import _batched, _ingest_chunk
from spindoctor.cli.results_index.counts import IngestCounts
from spindoctor.cli.results_index.runs import _finish_run, _start_run
from spindoctor.cli.results_index.store import _recorded_files, _RecordedFile
from spindoctor.nav_records import (
    ListedRecord,
    Selection,
    TreeRecordSource,
    TreeTuning,
    UnlistableRootError,
    distinct_roots,
)
from spindoctor.results_index import FAILED_FILES, IMAGES, STUBS_PER_STATEMENT

__all__ = ['ingest_metadata_files']


@dataclass(frozen=True)
class _RootListing:
    """Everything one listing of a whole results root found.

    Only :func:`_listing_of_root` builds one, and it builds one only for a root
    it listed entirely, which is what makes this type the prune's license: a
    share of a root is a list of files and can never become one of these, so a
    worker cannot reach for evidence it does not have.  The root travels with
    the documents for the same reason -- a prune is evidence about the root that
    was listed and about no other, and the rows it deletes carry that root as
    half of their key.

    Parameters:
        root_url: Normalized URL of the root this is a listing of.
        documents: The navigation documents found under it, in stub order.
        has_file_metrics: Whether every one of them reported both a size and a
            modification time.  A listing that reports neither cannot answer
            "has this changed", so such a root is re-read in full.
    """

    root_url: str
    documents: tuple[ListedRecord, ...]
    has_file_metrics: bool


def _listing_of_root(
    root_url: str, *, logger: PdsLogger, tuning: TreeTuning
) -> _RootListing | None:
    """List one results root whole, or report that nothing under it was listed.

    The listing itself is
    :class:`~spindoctor.nav_records.TreeRecordSource`'s, because every program
    that reads a results tree needs the same one and a rule about which
    directories were listed is not a rule while two readers hold different
    versions of it.  What is here is the accounting a pass keeps around it: the
    stream collected whole, because a pass compares it against the rows the
    index already holds and then prunes on the strength of holding all of it.

    The documents are put in stub order, which the stream does not promise: a
    fan-out cuts its tasks out of this listing in the order it holds them, and a
    task file is an operator-visible artifact that should describe the same
    tree the same way however the storage layer happened to enumerate it.

    Parameters:
        root_url: Normalized URL of the results root to list.
        logger: Logger for the scan summary, the degraded-listing warning and
            the root that could not be listed at all.
        tuning: How much of the listing runs at once.

    Returns:
        What the listing found, or None when the root itself could not be
        listed.  A root that is not there is a different thing from a root that
        is empty, and only the second one has been ingested when the pass ends,
        so the two are told apart by the type rather than by a count: there is
        no listing of the first for a prune to act on.  The root is also the one
        directory whose refusal is reported rather than raised, because a pass
        over several roots accounts for each of them separately and a mistyped
        root is the commonest thing an operator types.

    Raises:
        UnlistableDirectoryError: If a directory under the root could not be
            listed, which ends the whole pass rather than this root's part of
            it.  A directory nobody enumerated holds documents nobody recorded,
            and absence of a row is exactly what every consumer reads as "this
            image was never navigated", so a pass that finished around the gap
            would stamp that reading as an answer.
    """
    source = TreeRecordSource([root_url], logger=logger, tuning=tuning)
    try:
        documents = sorted(source.listing(Selection()), key=lambda listed: listed.stub)
    except UnlistableRootError:
        logger.error(
            'Results root %s could not be listed, so nothing under it has been ingested: '
            'check the spelling of the root',
            root_url,
        )
        return None
    logger.info('Results scan found %d metadata file(s) under %s', len(documents), root_url)
    listing = _RootListing(
        root_url=root_url,
        documents=tuple(documents),
        has_file_metrics=all(listed.has_metrics for listed in documents),
    )
    if not listing.has_file_metrics:
        logger.warning(
            'Listing of %s reports no size or modification time, so every document '
            'is re-read: this root cannot be ingested incrementally',
            root_url,
        )
    return listing


def _is_unchanged(listed: ListedRecord, recorded: _RecordedFile | None) -> bool:
    """Whether a listed file is exactly what the index already read.

    The comparison is the two metrics the listing supplies, and they are
    compared the same way for a refused file as for an ingested one, since
    both tables record them and both kinds of file are skipped unchanged.

    A file the listing reported no metric for has not been shown to be
    unchanged, whatever the pass says about its listing as a whole.  A whole-root
    walk reports metrics for every file or for none of them, so this only
    separates them for a share, whose claim to have them travels in the task
    beside the entries it is a claim about: an entry carrying neither metric
    beneath a task claiming both would otherwise compare equal to a row that
    recorded neither, and be skipped on that evidence by every pass that ever
    reached it.

    Parameters:
        listed: The file as this walk saw it.
        recorded: What the index holds about it, or None when it holds nothing.

    Returns:
        True when the file need not be read again.
    """
    if recorded is None:
        return False
    if listed.mtime_ns is None or listed.size_bytes is None:
        return False
    return (recorded.mtime_ns, recorded.size_bytes) == (listed.mtime_ns, listed.size_bytes)


def _files_to_read(
    files: Sequence[ListedRecord],
    recorded: dict[str, _RecordedFile],
    *,
    force: bool,
    has_file_metrics: bool,
) -> list[ListedRecord]:
    """Select the metadata files this pass has to read.

    A file the last pass refused is skipped on the same evidence as one it
    ingested: it has not changed, so reading it produces the same refusal.
    ``force`` re-reads both.

    Written over the files rather than over a whole-root listing, because a pass
    over a share of a root selects from its share by exactly this rule and must
    not be able to reach for anything a complete listing would have carried.

    Parameters:
        files: The metadata files this pass is responsible for.
        recorded: Stub to what the index already holds about it.
        force: Whether to re-read every document regardless.
        has_file_metrics: Whether the listing reported a size and modification
            time for every one of those files.  A listing that reports neither
            cannot answer "has this changed", so all of them are read.

    Returns:
        The files to read, in the order given.
    """
    if force or not has_file_metrics:
        return list(files)
    return [listed for listed in files if not _is_unchanged(listed, recorded.get(listed.stub))]


def _reads_recorded_rows(*, prune: bool, force: bool, has_file_metrics: bool) -> bool:
    """Whether a pass has any use for what the index already holds about a root.

    Two things read it.  The skip rule compares each listed file against the
    metrics recorded for it, and has nothing to compare when the pass is reading
    every document anyway -- because ``force`` was given, or because the listing
    supplies no metric.  The prune needs every stub the root has a row under,
    whatever the skip rule is doing.  With neither asking, the query is one
    statement over every row of a root that may hold several hundred thousand,
    for an answer nothing looks at.

    Parameters:
        prune: Whether the rows of documents that have left the tree go.
        force: Whether every document is read regardless of what is recorded.
        has_file_metrics: Whether the listing reported both a size and a
            modification time for every file.

    Returns:
        True when the recorded rows have to be read.
    """
    return prune or (not force and has_file_metrics)


def _prune_missing(
    engine: sqlalchemy.Engine,
    listing: _RootListing,
    recorded: dict[str, _RecordedFile],
    *,
    logger: PdsLogger,
) -> int:
    """Delete the rows of one root whose document has left the tree.

    Absence of an ``images`` row is what a consumer reads as "this image was
    never navigated", so presence has to mean the tree still holds the result.
    A re-navigation that renames or removes documents otherwise leaves rows
    that answer confidently for images nothing produced.

    Rows may only be removed on the evidence of a listing of a whole root, and
    a :class:`_RootListing` is the only thing that is one: a pass over a share
    of a root knows nothing about the stubs outside its share and would delete
    another worker's rows on that evidence, and it holds a list of files that
    this will not take.  The root deleted under is the listing's own for the
    same reason, since the rows carry it as half of their key and a listing of
    one root is no evidence at all about another.

    Parameters:
        engine: The open index.
        listing: The listing this prune is entitled to act on, and the root it
            was a listing of.
        recorded: What the index held about the root before this pass.
        logger: Logger for the count removed.

    Returns:
        How many image rows were deleted.
    """
    root_url = listing.root_url
    found = {listed.stub for listed in listing.documents}
    gone = sorted(stub for stub in recorded if stub not in found)
    if not gone:
        return 0
    removed = sum(1 for stub in gone if recorded[stub].from_images)
    logger.info('Removing %d row(s) under %s whose document has left the tree', removed, root_url)
    # One transaction per statement's worth of stubs: the batch here is the
    # number of bind parameters a statement may carry, not a tuning choice.
    for batch in _batched(gone, STUBS_PER_STATEMENT):
        with engine.begin() as connection:
            connection.execute(
                IMAGES.delete().where(
                    IMAGES.c.root_url == root_url,
                    IMAGES.c.results_path_stub.in_(batch),
                )
            )
            connection.execute(
                FAILED_FILES.delete().where(
                    FAILED_FILES.c.root_url == root_url,
                    FAILED_FILES.c.results_path_stub.in_(batch),
                )
            )
    return removed


def ingest_metadata_files(
    engine: sqlalchemy.Engine,
    roots: list[str],
    *,
    force: bool = False,
    prune: bool = True,
    logger: PdsLogger,
    tuning: TreeTuning,
) -> IngestCounts:
    """Ingest every metadata document under the given results roots.

    Each root is walked once, and each file whose recorded size and
    modification time still match the walk is skipped without being read,
    whether the last pass ingested it or refused it.  A document that cannot be
    read as a current-schema navigation document is counted against its own
    file and the run continues.

    Each root walked is also pruned: the rows of documents the tree no longer
    holds are deleted, so that presence of a row means what absence of one
    means.  A root the walk could not list is left alone entirely, and its
    ingest run is deliberately not completed, because a mistyped or unmounted
    root is not an empty one.

    A pass told not to prune leaves those rows in place, and a row may then
    outlive its document.  Absence keeps its meaning either way, since nothing
    is added by skipping a delete.  The recorded rows are read when either the
    skip rule or the prune wants them, so such a pass still reads them to decide
    what to skip and stops reading them altogether under ``force``.

    Two spellings of one root are one root, and are walked once.

    Each root's pass closes by reporting how many refused documents the index
    then holds under it.  That is the root's own total rather than this pass's:
    a file refused by an earlier pass and unchanged since is skipped without
    being read, so the pass that follows the one that refused it refuses
    nothing, and only the total says what an error filter answered from this
    index will pass over.

    Parameters:
        engine: The open index, which must already carry the schema.
        roots: Navigation results roots -- local directories or any URL the
            ``filecache`` layer accepts.  Each is normalized to the form the
            rows record and consumers compare against.
        force: Re-read every document, ignoring the recorded file metrics.
        prune: Delete the rows of one root whose documents the walk did not
            find.  False keeps them, which relaxes what presence of a row means
            and leaves what absence of one means alone.
        logger: Logger for the per-root scan summary and per-file failures.
        tuning: How much of the pass runs at once and how many documents one
            transaction covers, which the program resolved from its
            configuration.

    Returns:
        What the pass did, summed over every root.

    Raises:
        UnlistableDirectoryError: If a directory under any root could not be
            listed.  The whole pass ends there: the roots already walked keep
            what they ingested and their completed runs, this root and every
            root named after it keep no completed run at all, and no consumer
            reads absence under one of those as an answer.
    """
    total = IngestCounts()
    # The normalized form is what is walked, not the string as typed.  It is
    # the same location, absolute and spelled once: walking the typed form
    # would record a relative source_file beside an absolute root_url, and a
    # relative local root is one the storage layer refuses outright.  Two
    # spellings of one root are one root here as they are at a fan-out, so the
    # same command line means the same thing in every mode.
    for root_url in distinct_roots(roots):
        root = FCPath(root_url)
        counts = IngestCounts()
        run_id = _start_run(engine, root_url)
        logger.info('Ingesting %s', root_url)
        listing = _listing_of_root(root_url, logger=logger, tuning=tuning)
        if listing is None:
            # The run row keeps its NULL finish time, so every consumer treats
            # this root as one nobody has ingested rather than as one that
            # holds nothing.
            counts.roots_unreadable = 1
            total.add(counts)
            continue
        counts.files_seen = len(listing.documents)
        recorded: dict[str, _RecordedFile] = {}
        if _reads_recorded_rows(
            prune=prune, force=force, has_file_metrics=listing.has_file_metrics
        ):
            with engine.connect() as connection:
                recorded = _recorded_files(connection, root_url)
        to_read = _files_to_read(
            listing.documents,
            recorded,
            force=force,
            has_file_metrics=listing.has_file_metrics,
        )
        counts.files_skipped = counts.files_seen - len(to_read)
        for chunk in _batched(to_read, tuning.ingest_commit_chunk_size):
            _ingest_chunk(
                engine,
                root,
                chunk,
                root_url=root_url,
                counts=counts,
                logger=logger,
                tuning=tuning,
            )
        if prune:
            counts.files_removed = _prune_missing(engine, listing, recorded, logger=logger)
        _finish_run(engine, run_id, counts)
        logger.info(
            'Ingested %d, skipped %d unchanged, failed %d, removed %d of %d file(s) under %s',
            counts.files_ingested,
            counts.files_skipped,
            counts.files_failed,
            counts.files_removed,
            counts.files_seen,
            root_url,
        )
        total.add(counts)
    return total
