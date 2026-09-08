"""Retrieving a chunk of metadata files, reading each one, and writing them.

Retrieval is batched because a cloud backend downloads a batch in parallel: the
tuning says how many downloads run at once and how large a batch is handed to
them, and a batch has to be at least the thread count or the pool never fills.
It is ``retrieve()`` that is called rather than
``get_local_path()``, which on a cloud root names a file the cache would hold
and downloads nothing.

What a failed document costs
----------------------------

A results tree holds ``*_metadata.json`` files that are not per-image
navigation documents -- products of other tools, and documents written by an
older metadata schema.  Each is counted as an error for its own file and no
more: the run continues, and the closing summary tallies the failures by
reason, so several hundred documents that were never navigation results read as
exactly that rather than as a broken ingest.

A refused file is recorded in ``failed_files`` with everything the walk knows
about it and nothing the document would have said: the same two metrics an
ingested one records, so an unchanged refusal is skipped on the next pass
instead of being downloaded and parsed again forever, plus the subtree it lives
under, which is what a selection filter asks of a file it never opens.  A
retrieval that never delivered the file is counted without being recorded: it
says nothing about the file that will still be true next pass, and a recorded
refusal is skipped for as long as the file does not change.

Two families of reason
----------------------

A reason says which of two things happened.
:data:`~spindoctor.nav_records.document.UNREADABLE`,
:data:`~spindoctor.nav_records.document.NOT_VALID_JSON` and
:data:`~spindoctor.nav_records.document.NOT_A_JSON_OBJECT` are the file yielding
no JSON object at all; :data:`~spindoctor.nav_records.facts.NOT_A_NAVIGATION_DOCUMENT`
is a JSON object this schema will not accept.  A selection answers no error
filter for either family whichever storage it reads, because a reader of
documents narrows on the facts a document yields and neither family yields any,
so the reason says what to fix rather than what a consumer will be short by.

Neither family is decided here.  The first is
:func:`~spindoctor.nav_records.document.document_or_refusal`'s, which is what
the tree path reads through as well, and the second is
:func:`~spindoctor.nav_records.facts.facts_from_document`'s, which is what
builds the rows either storage answers with; so the two never state one file's
fault two different ways.  Neither turns a fault in this code into a reason: a
refusal is written with the file's own modification time and size, so the next
pass skips it, and a defect cached that way would outlive its own fix while
every run after it reported itself clean.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import sqlalchemy
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.results_index.counts import IngestCounts
from spindoctor.cli.results_index.store import _write_chunk
from spindoctor.nav_records import (
    COULD_NOT_RETRIEVE,
    METADATA_SUFFIX,
    DocumentOrigin,
    ImageFacts,
    ListedRecord,
    MetadataDocumentError,
    TreeTuning,
    document_or_refusal,
    facts_from_document,
    subtree_of,
)

_Item = TypeVar('_Item')
"""What one slice of a batched sequence holds."""


def _batched(items: Sequence[_Item], size: int) -> Iterator[Sequence[_Item]]:
    """Yield consecutive slices of a sequence.

    Parameters:
        items: The sequence to slice.
        size: Maximum slice length.

    Yields:
        The slices, in order.
    """
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _read_document(local_path: Path, source: DocumentOrigin) -> ImageFacts:
    """Read one retrieved metadata file into rows.

    Parameters:
        local_path: Local path the retrieval produced.
        source: Where the document came from.

    Returns:
        The rows the document becomes.

    Raises:
        MetadataDocumentError: If the file cannot be read, does not parse as
            JSON, does not parse to a JSON object, or is not a current-schema
            navigation document.  The reason is the one the tree path states for
            the same file, because both come from the same reader.
    """
    parsed = document_or_refusal(local_path)
    if isinstance(parsed, str):
        raise MetadataDocumentError(parsed, source_file=source.source_file)
    return facts_from_document(parsed, source)


def _ingest_chunk(
    engine: sqlalchemy.Engine,
    root: FCPath,
    chunk: Sequence[ListedRecord],
    *,
    root_url: str,
    counts: IngestCounts,
    logger: PdsLogger,
    tuning: TreeTuning,
) -> None:
    """Retrieve, read and write one chunk of metadata files.

    The whole chunk is one transaction, so a crash costs one chunk rather than
    a whole archive-scale run, and no writer holds a lock for the length of one.

    Parameters:
        engine: The open index.
        root: The results root, which the retrieval is relative to.
        chunk: The files to ingest.
        root_url: Normalized URL of the root, as the rows record it.
        counts: Accumulator this chunk's outcomes are added to.
        logger: Logger for per-file failures.
        tuning: How many documents are retrieved at once, and in what batches.
    """
    pending: list[ImageFacts] = []
    refused: list[dict[str, Any]] = []
    for batch in _batched(chunk, tuning.retrieve_batch_size):
        sub_paths: list[str | Path] = [f'{listed.stub}{METADATA_SUFFIX}' for listed in batch]
        # retrieve() rather than get_local_path(): on a cloud root the latter
        # names a file it never downloads.  exception_on_fail=False keeps one
        # unreadable file from ending the run.
        local_paths = cast(
            list[Path | Exception],
            root.retrieve(sub_paths, exception_on_fail=False, nthreads=tuning.retrieve_threads),
        )
        for listed, local_path in zip(batch, local_paths, strict=True):
            source = DocumentOrigin(
                root_url=root_url,
                results_path_stub=listed.stub,
                source_file=(root / f'{listed.stub}{METADATA_SUFFIX}').as_posix(),
                mtime_ns=listed.mtime_ns,
                size_bytes=listed.size_bytes,
            )
            if isinstance(local_path, BaseException):
                # Nothing was read, so nothing is known about the file beyond
                # the listing.  A retrieval that failed once is worth trying
                # again, so no refusal is recorded for it.
                counts.record_failure(COULD_NOT_RETRIEVE, source.source_file)
                logger.debug('Skipping %s: %s', source.source_file, COULD_NOT_RETRIEVE)
                continue
            try:
                pending.append(_read_document(local_path, source))
            except MetadataDocumentError as exc:
                counts.record_failure(exc.reason, source.source_file)
                logger.debug('Skipping %s', exc)
                refused.append(
                    {
                        'root_url': root_url,
                        'results_path_stub': listed.stub,
                        'reason': exc.reason,
                        # The walk knows this whatever the file says, and a
                        # selection filter asks about the file rather than
                        # about its contents: a refused document is one the
                        # tree still holds, under a subtree.  The subtree is
                        # derived by the same function the images row uses, so
                        # the two tables can never disagree about which subtree
                        # a stub is under.
                        'subtree': subtree_of(listed.stub),
                        'mtime_ns': listed.mtime_ns,
                        'size_bytes': listed.size_bytes,
                    }
                )
    if not pending and not refused:
        return
    counts.files_ingested += _write_chunk(engine, pending, refused, logger=logger)
