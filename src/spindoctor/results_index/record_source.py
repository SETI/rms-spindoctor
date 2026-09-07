"""The navigation records as rows of an ingested results index.

A navigation pass writes one document per image, and an ingest pass copies what
a consumer reads into one row per image and its child rows.  Reading a document
is one file read per image, which on a cloud root is one paid round trip per
image and a Cassini-scale root holds several hundred thousand; reading a row is
one query per run.  This module is the half of the seam that reads rows; the
choice of which half a run gets is
:func:`spindoctor.results_index.open_record_source`.

The other half -- what a record is, what a document is named, what a caller is
asking for, the protocol, and the implementation over the documents themselves
-- is :mod:`spindoctor.nav_records`, which imports no database layer because the
packages that reach it must not acquire one.  Neither half decides anything
about a record it hands back: the classification that reads a pointing out of
one, the eligibility that reads a status, and the arithmetic that reads a matrix
are the caller's, unchanged from one storage to the other.  What makes that safe
is that both storages answer in one shape, rebuilt through the one
column-to-field correspondence in :mod:`spindoctor.results_index.rebuild`.

What differs between the two storages, and what may not
-------------------------------------------------------

**A record read from a document carries every field the document has; one
rebuilt from a row carries the columns its consumer selected.**  That is what
makes a row cheap, and it is why a consumer names its columns rather than
selecting the whole table: nobody reads forty fields.  A consumer that reads a
field it did not select reads it as absent, which is why the columns a consumer
names are pinned by a test rather than left to be noticed in production.

**The facts do not narrow, whatever columns a consumer named.**  A record is
defined as looking like the document it stands for, so selecting fewer columns
gives a smaller record; the facts are defined as the whole row, so a selection
of them would answer a different question.
:meth:`IndexRecordSource.facts` therefore reads every column of ``images`` and
merges on every child row, and the two storages hand back the same mappings.

**A selection naming its own stubs reads no listing, so the facts built from
the documents carry neither file metric.**  The two metrics are what a walk saw
of a file, and naming an image walks nothing; the index carries what the walk
that ingested the root saw.  A run that has to compare them asks for a listing,
which is the call that answers what is there.

**The order of one image's child rows is the document's on one storage and
undefined on the other.**  A reader of documents builds the technique entries in
the order the document wrote them; the index stores no ordinal for them, and the
statement that reads them sorts on the image key alone, so rows of one key
arrive in whatever order the server answers that sort in -- in practice the
order of the index the sort is answered off, which puts the techniques of one
image in name order rather than the document's.  Nothing in the shape depends on
it -- a technique is identified by its name and a feature source by its type and
source -- so a consumer that wants an order sorts on that identity.

**A value the ingest could not store is rebuilt as absent**, which is the one
class of difference the seam cannot close.  It belongs to what a column can
hold; :mod:`spindoctor.results_index.rebuild` states it, and each consumer's
documentation says what it does about it.

**There is no image an ingest left out of both tables.**  The class above is a
value one column could not hold inside a row that was written.  A document whose
rows the database will not take at all ends the ingest where it happened rather
than being counted and passed over, and the root's ingest run keeps its NULL
finish time, so every consumer says the root has no completed ingest instead of
reading absence under it as an answer.  What an index with a completed run holds
is therefore every document the tree held: a stub with no row in either table is
an image nothing navigated.

**A file the ingest refused is not a row.**  It is recorded in ``failed_files``,
and it is neither an image that was never navigated nor one whose record can be
read: the document may well record a perfectly good pointing, and reading the
refusal as absence would build a corrected product from the documents and an
uncorrected one from the index without saying so.  All three calls therefore
report a refusal as a refusal.  :meth:`IndexRecordSource.record` fails the
image, naming the stub, the index and the reason the ingest recorded.
:meth:`IndexRecordSource.records` yields every refused file the selection covers
as an unreadable file carrying that reason, which is what the walk does with a
document it cannot attribute to a mission.  And :meth:`IndexRecordSource.listing`
counts it, because a document the ingest refused is a file that exists and a
listing answers what is there.

**A refused file is reported here under every mission filter and every time
bound, and a stream of facts over the documents reports it under both too.**  A
``failed_files`` row holds no mission and no epoch, so no filter can exclude it
and every stream yields it.  The walk refuses a file before any filter is
applied to it and then narrows on the facts it read, which are the values the
rows here are narrowed on, so the same files are refused and the same images
selected whichever storage answered.

A stream of *records* is the one that differs, and deliberately: a record is the
document, so the walk reads the mission and the midtime out of the document
itself.  A document it can attribute to another mission, or place outside the
span, is passed over and never refused -- including a file this half holds a
refusal for -- and one it cannot place at all is reported under the walk's own
reason for what the filter found rather than under the reason an ingest of the
same file records.  A run that has to compare the two streams of records under a
filter reads the refusals unfiltered from both.

**A row that cannot be placed in time is not a refusal here.**  ``instrument``
is not nullable and a document naming none is refused before an ingest could
write a row for it, so every row names a mission; ``midtime_et`` is nullable,
and a row recording none satisfies no bound and is simply not selected, exactly
as a row outside the span is not.  A stream of facts over the documents reads
such an image the same way, because the bound is applied to the midtime the
facts carry.  A stream of records does not: the walk reports a document
recording no usable midtime, since a record it cannot place cannot be shown to
be another run's.  A run that has to account for those images reads them under
no bound.

**A stub both tables record is one file, read as the record it is.**  The ingest
writes the two tables independently and a pass divided into shares can leave a
stale refusal beside a record, so all three calls prefer the record: the
per-image lookup reads both halves of the key, the stream naming its own stubs
puts the records in last, and the whole-root stream and the listing exclude a
refusal whose key also carries a record.

**A subtree the root does not hold is refused by the walk and is empty here.**
The walk descends the directory a selection named and stops if it is not there,
so a run restricted to a directory nobody created is refused rather than
reported as a clean pass over nothing.  A query has no directory to fail on: a
subtree nothing was ever ingested under and one holding no images are the same
absence of rows.

**A stub the index has never heard of yields nothing**, where a selection naming
that stub over the documents yields an unreadable file saying the document never
arrived.  Under a root with a completed ingest, no row in either table means no
file was there to read, which is an image nothing navigated rather than a file
that failed to arrive; the walk cannot know that, because it was handed the name
of a file rather than asked what the root holds.

**Nothing falls back.**  A URL that cannot be opened, or a root nobody has
ingested, fails the run rather than quietly reading the tree instead.

The order rows arrive in
------------------------

None of the four calls promises a total order.  A server sorts text under its
own collation, and a locale collation orders a separator against an underscore
differently from the codepoint order a walk produces, so an ``ORDER BY`` on a
stub or a path hands back one order from SQLite and another from PostgreSQL for
the same tree.  A caller that needs a total order sorts the stream it received,
which is the one key both storages share.

The per-image lookup, the listing and the stream of records therefore order on
nothing at all.  The stream of facts orders on the key, because its three
statements have to be merged onto one another and adjacent rows are what makes
that possible; that is safe for the one reason a text sort ever is, which is
that all three orders are the same server's and are compared against nothing but
each other.  It also reads them from one snapshot of the index, since an order
three statements share is no help while they answer about three states of what
they are ordering.  :mod:`spindoctor.results_index.facts_stream` states what the
merge does and does not depend on.

A stream over rows yields the images the selection covers and then the files the
ingest refused, each in the order the server returned them.  A stream naming its
own stubs is the exception: it yields them in the order they were named, because
naming an image is not a narrowing and a queue task's report has to line up with
the task it was given.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import sqlalchemy
from filecache import FCPath
from sqlalchemy.engine import Connection, Engine

from spindoctor.nav_records import (
    ImageFacts,
    ListedRecord,
    NavRecord,
    Selection,
    UnreadableFile,
    distinct_roots,
    document_path,
    in_batches,
    refuse_what_a_listing_cannot_answer,
    root_for_stub,
    root_for_stubs,
    selected_roots,
)
from spindoctor.results_index.engine import STUBS_PER_STATEMENT, reporting_a_failed_read
from spindoctor.results_index.facts_stream import facts_stream, reading_one_snapshot
from spindoctor.results_index.masking import masked_url
from spindoctor.results_index.rebuild import record_from_row
from spindoctor.results_index.schema import FAILED_FILES, IMAGES

__all__ = ['IndexRecordSource']

_ROW_FETCH_SIZE = 1000
"""How many rows a streamed query brings back from the server at a time.

Not the batch size a tree retrieves documents in: that one bounds a parallel
download of files, this one bounds the buffer behind a server-side cursor, and
the two are paid on different storages for different reasons.  Its only effect
is on peak memory against the number of fetches, since the caller sees one row
at a time whatever it is.
"""


class IndexRecordSource:
    """One or more results roots' navigation records, as rows of a results index.

    The columns are the consumer's, because a row is only cheaper than a
    document while it carries less: a per-image lookup that dragged back every
    JSON column would spend on the matrix and the kernel list what it saved on
    the round trip.

    Parameters:
        engine: The open index, which this source disposes of when it is closed.
        roots: The results roots whose rows to read, in the order questions are
            answered about them.  Two spellings of one root are one root, and
            the source holds each of them once.  Every statement filters on
            them: one index serves several roots, and a query asking about a
            stub alone would answer with another root's images.
        url: The index URL, kept for the messages that name it.
        columns: The columns of ``images`` a consumer's *records* are rebuilt
            from.  Each must be a column :mod:`spindoctor.results_index.rebuild`
            knows a place for, or the rebuilt record silently lacks the field it
            was selected for.  :meth:`facts` reads every column and ignores
            this: the facts are the whole row, so a subset of them is a
            different question rather than a cheaper answer to that one.

    Raises:
        ValueError: If no root is given, or if one of them is not a location.
    """

    def __init__(
        self,
        engine: Engine,
        roots: Sequence[str | Path | FCPath],
        url: str,
        columns: Sequence[sqlalchemy.Column[Any]],
    ) -> None:
        held = distinct_roots(roots)
        if not held:
            raise ValueError('a record source over the results index needs at least one root')
        self._engine = engine
        self._roots = tuple(held)
        self._url = masked_url(url)
        self._raw_url = url
        self._columns = tuple(columns)
        # An index is a snapshot of its last ingest, so a row can be absent
        # because nothing navigated the image or because the image was navigated
        # after that ingest.  Neither the row nor its absence can say which, so
        # the message says what was searched and leaves the reader able to tell.
        self._storage = (
            f'the results index {self._url}, a snapshot of its last ingest of '
            f'{", ".join(self._roots)}'
        )

    def __enter__(self) -> 'IndexRecordSource':
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
        """Leave a run's use of this source, closing it.

        Parameters:
            exc_type: The exception's class, when the run is leaving on one.
            exc: The exception, when the run is leaving on one.
            traceback: Its traceback, when the run is leaving on one.
        """
        self.close()

    @property
    def roots(self) -> tuple[str, ...]:
        """The roots this source answers about, in the order it holds them.

        Returns:
            The normalized root URLs.
        """
        return self._roots

    def record(self, stub: str) -> NavRecord:
        """Rebuild one image's navigation record from its row.

        Parameters:
            stub: The image's results path stub.

        Returns:
            The record, paired with the document the ingest recorded reading.

        Raises:
            FileNotFoundError: If the index holds no row for this stub under this
                root, naming both and the index, and saying that the index is a
                snapshot: the row is absent either because nothing navigated the
                image or because it was navigated after the last ingest, and the
                message names the snapshot so the two can be told apart.  A
                missing document raises the same way, so the caller reports both
                the same way.
            ValueError: If this source holds more than one root, naming them all,
                since a bare stub does not say which of them it is a key under.
                Also if the index records the document for this stub as one the
                ingest refused.  Deliberately not the same exception as an
                absent row: a caller reports a missing record as an image
                nothing navigated, and this image was navigated -- the index
                simply cannot say what it recorded.
        """
        root_url = root_for_stub(self._roots, stub)
        row = self._row(root_url, stub)
        if row.record_stub is None:
            self._refuse_a_document_the_ingest_refused(stub, row.refusal_reason)
            raise FileNotFoundError(
                f'{stub}: no navigation record for this image in {self._storage}'
            )
        return NavRecord(
            path=self._path_of(root_url, stub, row.source_file),
            stub=stub,
            metadata=record_from_row(row),
        )

    def records(self, selection: Selection) -> Iterator[NavRecord | UnreadableFile]:
        """Return the records the selection covers, yielded one at a time.

        The rows are streamed in server-side chunks rather than fetched whole,
        so a caller holds one record where a mission's worth of them would
        otherwise be held between the server and the run.  What the selection
        asks for is checked before anything is read.

        Parameters:
            selection: Which records to yield.  A selection naming stubs reads
                exactly those, in the order it names them; one naming none takes
                every row the selection's roots, subtrees, mission and time
                bounds cover.

        Returns:
            One record per row that carries one, and one
            :class:`~spindoctor.nav_records.record.UnreadableFile` per file the
            ingest refused, carrying the reason it recorded.

        Raises:
            ValueError: If the selection names stubs without resolving to
                exactly one root, since a stub is a key under a root; if it
                names a root this source does not hold; or if the index cannot
                be read, naming it with any password masked.  The last of those
                is raised as the caller reads, since that is when the query runs.
        """
        roots = selected_roots(self._roots, selection.roots)
        if not selection.stubs:
            return self._records_of(roots, selection)
        return self._records_of_stubs(root_for_stubs(roots, selection.stubs), selection)

    def facts(self, selection: Selection) -> Iterator[ImageFacts | UnreadableFile]:
        """Return what every image the selection covers says about itself.

        Every column of ``images``, whatever columns this source was opened
        with.  Those narrow :meth:`records`, where a record is defined as
        looking like the document it stands for and nobody selects forty columns
        to read five; the facts are by definition the whole row, so a subset of
        them would be a different question rather than a cheaper answer to this
        one.  The per-technique and per-feature rows are merged on afterwards
        from their own tables, which is what a reader of the documents gets for
        nothing out of the same file.

        All three statements are streamed in server-side chunks and merged as
        they arrive, so a stream over a root holds one image's facts at a time
        where a root's worth of rows would otherwise be held between the server
        and the run.  A selection naming stubs holds one batch of them instead,
        since the answers to a batch are put back into the order it named.

        Parameters:
            selection: Which images to yield.  A selection naming stubs reads
                exactly those, in the order it names them; one naming none takes
                every row the selection's roots, subtrees, mission and time
                bounds cover.

        Returns:
            One set of facts per image row, and one
            :class:`~spindoctor.nav_records.record.UnreadableFile` per file the
            ingest refused, carrying the reason it recorded.  No set of facts
            carries a record: a record is rebuilt from the columns its consumer
            declares, which are not the columns the facts hold, so a consumer
            wanting both asks this source twice.  Handing back a record built
            out of every column would be a second rebuild of a row beside
            :mod:`spindoctor.results_index.rebuild`'s, which is how the two
            storages came to disagree before there was one.

        Raises:
            ValueError: If the selection names stubs without resolving to
                exactly one root, since a stub is a key under a root; if it
                names a root this source does not hold; or if the index cannot
                be read, naming it with any password masked.  The last of those
                is raised as the caller reads, since that is when the query runs.
        """
        roots = selected_roots(self._roots, selection.roots)
        if not selection.stubs:
            return self._facts_of(roots, selection)
        return self._facts_of_stubs(root_for_stubs(roots, selection.stubs), selection)

    def listing(self, selection: Selection) -> Iterator[ListedRecord]:
        """Return every file the index records under the selection, in one query.

        Both tables record a file, so both are listed: a document the ingest
        refused is a file that exists, and a listing answers what is there
        rather than what a document said.  A stub both tables record is one
        file, listed once, as the record it is; a listing of a tree finds one
        file there and the two would otherwise disagree about how many there
        are.  Nothing here reads a document or a record field, which is what
        makes it the cheap call over either storage.

        A selection naming stubs is answered about those keys and no others,
        which is the question a caller enumerating candidate images asks: which
        of the ones this run might still keep is recorded here.  It is asked in
        batches, for the reason every keyed read here is, and each batch's
        answers are put back into the order the batch named.

        Parameters:
            selection: Which files to list.  ``instrument``, ``start_et`` and
                ``stop_et`` may not be set.  A selection naming stubs names the
                files outright, and its ``subtrees`` narrow nothing further,
                exactly as they narrow nothing for a stream of records or of
                facts.

        Returns:
            One entry per recorded file, carrying its stub, where the ingest
            found it, and the two metrics that say whether it has changed since.
            A named stub the index records no file for yields nothing: under a
            root with a completed ingest that is an image nothing navigated.

        Raises:
            ValueError: If the selection carries ``instrument``, ``start_et`` or
                ``stop_et``, naming which.  The index could answer those from
                its columns and deliberately does not: a call that meant one
                thing over one storage and another over the next would not be a
                seam, and a listing that ignored a restriction would answer for
                the whole root as though it were the selection.  Also if the
                selection names a root this source does not hold, if it names
                stubs without resolving to exactly one root, or if the index
                cannot be read.
        """
        refuse_what_a_listing_cannot_answer(selection)
        roots = selected_roots(self._roots, selection.roots)
        if not selection.stubs:
            return self._listing_of(roots, selection.subtrees)
        return self._listing_of_stubs(root_for_stubs(roots, selection.stubs), selection.stubs)

    def describe(self) -> str:
        """Return the roots and the index the records were read out of.

        Returns:
            The roots, followed by the index URL with any password masked.
        """
        return f'{", ".join(self._roots)} in the results index {self._url}'

    def close(self) -> None:
        """Dispose of the engine, closing every connection it pooled."""
        self._engine.dispose()

    def _listing_of(self, roots: Sequence[str], subtrees: Sequence[str]) -> Iterator[ListedRecord]:
        """Stream one query over both tables, yielding an entry per recorded file.

        Parameters:
            roots: The normalized roots to list, which every arm filters on.
            subtrees: The top-level directories to keep, or empty for all of
                them.

        Yields:
            One entry per recorded file, in the order the server returned them.
        """
        yield from self._listed(
            self._scope(IMAGES, roots, subtrees), self._scope(FAILED_FILES, roots, subtrees)
        )

    def _listing_of_stubs(self, root_url: str, stubs: Sequence[str]) -> Iterator[ListedRecord]:
        """List the files a selection named outright, in the order it named them.

        Asked in batches for the reason every keyed read here is: a caller is
        free to name a mission's worth of keys, and a statement binding every one
        of them at once is one a driver refuses somewhere above its own parameter
        limit.  One batch's answers are held so they can be put back into the
        order the batch named, which is the order naming a file means.

        Parameters:
            root_url: The one normalized root those keys are under, which both
                arms filter on: the index is keyed by root and stub together,
                and a term asking about the stub alone would answer with another
                root's files.
            stubs: The stubs to answer about, in the order to answer them.

        Yields:
            One entry per named stub the index records a file for, in the order
            named.  A stub it records none for yields nothing.
        """
        for batch in in_batches(iter(stubs), STUBS_PER_STATEMENT):
            found = {
                entry.stub: entry
                for entry in self._listed(
                    [IMAGES.c.root_url == root_url, IMAGES.c.results_path_stub.in_(batch)],
                    [
                        FAILED_FILES.c.root_url == root_url,
                        FAILED_FILES.c.results_path_stub.in_(batch),
                    ],
                )
            }
            for stub in batch:
                if stub in found:
                    yield found[stub]

    def _listed(
        self,
        images: Sequence[sqlalchemy.ColumnElement[bool]],
        refusals: Sequence[sqlalchemy.ColumnElement[bool]],
    ) -> Iterator[ListedRecord]:
        """Stream one query over both tables, yielding an entry per recorded file.

        Both tables record a file, so both are read, whichever way the caller
        narrowed them: the two conditions arrive already built because a listing
        of a root narrows on its subtrees and a listing of named files narrows on
        their keys, and everything after that is the same query.

        Parameters:
            images: What restricts the image rows, which is never empty: every
                arm filters on the root.
            refusals: What restricts the rows of the files the ingest refused,
                to which the exclusion of a stub the images also record is
                added.

        Yields:
            One entry per recorded file, in the order the server returned them.
        """
        documents = sqlalchemy.select(
            IMAGES.c.root_url,
            IMAGES.c.results_path_stub,
            IMAGES.c.source_file,
            IMAGES.c.mtime_ns,
            IMAGES.c.size_bytes,
        ).where(*images)
        # The refusal table records no path, because the ingest refused the file
        # rather than reading it, so the arm supplies a typed absence and the
        # path is rebuilt from the key.  The cast is what keeps a union of the
        # two arms a union of two text columns on a server that types them.
        refused = sqlalchemy.select(
            FAILED_FILES.c.root_url,
            FAILED_FILES.c.results_path_stub,
            sqlalchemy.cast(sqlalchemy.null(), sqlalchemy.Text).label('source_file'),
            FAILED_FILES.c.mtime_ns,
            FAILED_FILES.c.size_bytes,
        ).where(*refusals, _has_no_record_row())
        with reporting_a_failed_read(self._raw_url), self._streaming() as connection:
            for row in connection.execute(documents.union_all(refused)):
                root_url = str(row.root_url)
                stub = str(row.results_path_stub)
                yield ListedRecord(
                    stub=stub,
                    path=self._path_of(root_url, stub, row.source_file),
                    mtime_ns=None if row.mtime_ns is None else int(row.mtime_ns),
                    size_bytes=None if row.size_bytes is None else int(row.size_bytes),
                )

    def _records_of(
        self, roots: Sequence[str], selection: Selection
    ) -> Iterator[NavRecord | UnreadableFile]:
        """Stream the rows the selection covers, then the files the ingest refused.

        Two statements on one connection rather than one union: the two carry
        different columns, and a consumer's column list is the whole of what a
        record read costs.  Running them independently is what makes the refusal
        arm carry the exclusion: a stub in both tables is a record with a stale
        refusal beside it and must be read as the record it is, and two
        statements that did not know about each other would yield one image as
        a record and as a shortfall in one pass.

        The exclusion is on the key alone rather than on the selection, because
        a record the selection filtered out is still a record: the refusal
        beside it is stale whichever mission or span is being read.

        Parameters:
            roots: The normalized roots to read, which both statements filter on.
            selection: The selection, for the subtrees, the mission and the time
                bounds.

        Yields:
            One record per image row, then one unreadable file per refusal.
        """
        images = sqlalchemy.select(*self._bulk_columns()).where(
            *self._scope(IMAGES, roots, selection.subtrees),
            *self._what_a_document_says(selection),
        )
        refusals = sqlalchemy.select(
            FAILED_FILES.c.root_url, FAILED_FILES.c.results_path_stub, FAILED_FILES.c.reason
        ).where(
            *self._scope(FAILED_FILES, roots, selection.subtrees),
            _has_no_record_row(),
        )
        with reporting_a_failed_read(self._raw_url), self._streaming() as connection:
            for row in connection.execute(images):
                yield self._record_of(row)
            for row in connection.execute(refusals):
                yield self._refusal_of(row)

    def _records_of_stubs(
        self, root_url: str, selection: Selection
    ) -> Iterator[NavRecord | UnreadableFile]:
        """Read the images a selection named outright, in the order it named them.

        Asked in batches rather than in one statement: a queue task carries
        hundreds of keys and a caller is free to name a mission's worth, and a
        statement binding every one of them at once is one a driver refuses
        somewhere above its own parameter limit.  Each batch's rows are put back
        into the order the batch named, which is the order naming an image means.

        Parameters:
            root_url: The one normalized root those keys are under.
            selection: The selection, for the stubs it names and the mission and
                time bounds a row still has to satisfy.

        Yields:
            One record per named stub the index holds a row for, and one
            unreadable file per named stub it holds a refusal for.  A stub it
            holds neither for yields nothing: under a root with a completed
            ingest that is an image nothing navigated.
        """
        conditions = self._what_a_document_says(selection)
        with reporting_a_failed_read(self._raw_url), self._streaming() as connection:
            for batch in in_batches(iter(selection.stubs), STUBS_PER_STATEMENT):
                images = sqlalchemy.select(*self._bulk_columns()).where(
                    IMAGES.c.root_url == root_url,
                    IMAGES.c.results_path_stub.in_(batch),
                    *conditions,
                )
                refusals = sqlalchemy.select(
                    FAILED_FILES.c.root_url,
                    FAILED_FILES.c.results_path_stub,
                    FAILED_FILES.c.reason,
                ).where(
                    FAILED_FILES.c.root_url == root_url,
                    FAILED_FILES.c.results_path_stub.in_(batch),
                )
                found: dict[str, NavRecord | UnreadableFile] = {
                    str(row.results_path_stub): self._refusal_of(row)
                    for row in connection.execute(refusals)
                }
                # A stub in both tables is a record with a stale refusal beside
                # it, and must be read as the record it is, so the records are
                # placed second and win.
                found.update(
                    (str(row.results_path_stub), self._record_of(row))
                    for row in connection.execute(images)
                )
                for stub in batch:
                    if stub in found:
                        yield found[stub]

    def _facts_of(
        self, roots: Sequence[str], selection: Selection
    ) -> Iterator[ImageFacts | UnreadableFile]:
        """Stream the images the selection covers, then the files the ingest refused.

        The refusals come last and separately, exactly as they do for a stream
        of records and for the same reasons: they carry different columns, and
        a stub in both tables is a record with a stale refusal beside it and
        must be read as the image it is.

        Parameters:
            roots: The normalized roots to read, which every statement filters
                on.
            selection: The selection, for the subtrees, the mission and the time
                bounds.

        Yields:
            One set of facts per image row, then one unreadable file per
            refusal.
        """
        conditions = [
            *self._scope(IMAGES, roots, selection.subtrees),
            *self._what_a_document_says(selection),
        ]
        refusals = sqlalchemy.select(
            FAILED_FILES.c.root_url, FAILED_FILES.c.results_path_stub, FAILED_FILES.c.reason
        ).where(*self._scope(FAILED_FILES, roots, selection.subtrees), _has_no_record_row())
        with reporting_a_failed_read(self._raw_url), self._reading_one_snapshot() as connection:
            yield from facts_stream(connection, conditions)
            for row in connection.execute(refusals):
                yield self._refusal_of(row)

    def _facts_of_stubs(
        self, root_url: str, selection: Selection
    ) -> Iterator[ImageFacts | UnreadableFile]:
        """Read the images a selection named outright, in the order it named them.

        Asked in batches for the reason :meth:`_records_of_stubs` is: a caller is
        free to name a mission's worth of keys, and a statement binding every one
        of them at once is one a driver refuses somewhere above its own parameter
        limit.  One batch's answers are held so they can be put back into the
        order the batch named, which is the order naming an image means.

        Parameters:
            root_url: The one normalized root those keys are under.
            selection: The selection, for the stubs it names and the mission and
                time bounds a row still has to satisfy.

        Yields:
            One set of facts per named stub the index holds a row for, and one
            unreadable file per named stub it holds a refusal for.  A stub it
            holds neither for yields nothing.
        """
        bounds = self._what_a_document_says(selection)
        with reporting_a_failed_read(self._raw_url), self._reading_one_snapshot() as connection:
            for batch in in_batches(iter(selection.stubs), STUBS_PER_STATEMENT):
                refusals = sqlalchemy.select(
                    FAILED_FILES.c.root_url,
                    FAILED_FILES.c.results_path_stub,
                    FAILED_FILES.c.reason,
                ).where(
                    FAILED_FILES.c.root_url == root_url,
                    FAILED_FILES.c.results_path_stub.in_(batch),
                )
                found: dict[str, ImageFacts | UnreadableFile] = {
                    str(row.results_path_stub): self._refusal_of(row)
                    for row in connection.execute(refusals)
                }
                # As for a stream of records: a stub in both tables is an image
                # with a stale refusal beside it, so the images are placed
                # second and win.
                found.update(
                    (str(one.image['results_path_stub']), one)
                    for one in facts_stream(
                        connection,
                        [
                            IMAGES.c.root_url == root_url,
                            IMAGES.c.results_path_stub.in_(batch),
                            *bounds,
                        ],
                    )
                )
                for stub in batch:
                    if stub in found:
                        yield found[stub]

    def _streaming(self) -> Connection:
        """Open a connection whose results arrive in server-side chunks.

        Opened inside the caller's own translation of a database failure, so
        that a connection lost between the open and the last row is reported as
        the refusal every consumer already catches rather than as the database
        layer's own exception type.

        Returns:
            The connection, which the caller closes.
        """
        return self._engine.connect().execution_options(yield_per=_ROW_FETCH_SIZE)

    def _reading_one_snapshot(self) -> Connection:
        """Open a streamed connection whose statements answer about one index.

        What the merge behind :meth:`facts` needs beyond a streamed read: its
        three statements are put back together by key, and a key present in one
        of them and absent from another is a pass that hands back images with
        none of their own child rows.

        Returns:
            The connection, which the caller closes.
        """
        return reading_one_snapshot(self._streaming())

    @staticmethod
    def _scope(
        table: sqlalchemy.Table, roots: Sequence[str], subtrees: Sequence[str]
    ) -> list[sqlalchemy.ColumnElement[bool]]:
        """Return what restricts one table to the roots and subtrees selected.

        The root is never optional.  The index is keyed by root and stub
        together and one database serves several roots, so a term that asked
        about the stub alone would answer with another root's files.

        Parameters:
            table: The table being restricted, which carries both columns.
            roots: The normalized roots to keep.
            subtrees: The top-level directories to keep, or empty for all of
                them.  A stub with no subtree above it -- a bare scene name --
                is under no named directory and is matched by none of them,
                because SQL's ``IN`` is false for NULL.

        Returns:
            The conditions, root first.
        """
        conditions: list[sqlalchemy.ColumnElement[bool]] = [table.c.root_url.in_(list(roots))]
        if subtrees:
            conditions.append(table.c.subtree.in_(list(subtrees)))
        return conditions

    @staticmethod
    def _what_a_document_says(selection: Selection) -> list[sqlalchemy.ColumnElement[bool]]:
        """Return what restricts the image rows to what their documents recorded.

        The epoch is the recorded exposure midtime rather than the epoch the
        ingest derived for its own grouping, because that is the field the walk
        reads a document's midtime out of; a filter over the derived column
        would keep a different set of images from the same tree.

        Parameters:
            selection: The selection.

        Returns:
            The conditions, empty when the selection restricts neither.  A row
            with no recorded midtime satisfies no bound, since a comparison with
            NULL is not true -- which is how a stream of facts over the
            documents reads one recording none.
        """
        conditions: list[sqlalchemy.ColumnElement[bool]] = []
        if selection.instrument is not None:
            conditions.append(IMAGES.c.instrument == selection.instrument)
        if selection.start_et is not None:
            conditions.append(IMAGES.c.midtime_et >= selection.start_et)
        if selection.stop_et is not None:
            conditions.append(IMAGES.c.midtime_et <= selection.stop_et)
        return conditions

    def _bulk_columns(self) -> tuple[sqlalchemy.ColumnElement[Any], ...]:
        """Return the columns a streamed read of the image rows selects.

        The key and the recorded source file are added to the consumer's own
        columns rather than asked of it: a stream hands back records paired with
        where each is kept, so it needs all three whatever the consumer reads.

        Returns:
            The columns, with the three added ones first and none of them
            repeated if the consumer named it too.
        """
        added = (IMAGES.c.root_url, IMAGES.c.results_path_stub, IMAGES.c.source_file)
        names = {column.name for column in added}
        return (*added, *(column for column in self._columns if column.name not in names))

    def _row(self, root_url: str, stub: str) -> sqlalchemy.Row[Any]:
        """Read what the index holds about one image, from both of its tables.

        One query rather than a record lookup followed by a refusal lookup: an
        image with no record is the common case on a partially navigated root,
        and it is the case that would pay the second round trip -- against the
        stage whose whole purpose is removing one per image.  The key is selected
        as a row of its own and both tables are joined onto it, so exactly one
        row comes back whether the index holds a record, a refusal or neither.

        Parameters:
            root_url: The normalized root the stub is a key under.
            stub: The image's results path stub.

        Returns:
            The row.  ``record_stub`` carries the stub when the index holds a
            navigation record for it and nothing otherwise, and
            ``refusal_reason`` carries the recorded reason when the index holds a
            refusal for it and nothing otherwise; a stub the index knows nothing
            about answers to neither.  Both halves are read rather than one,
            because a stub in both tables is a record with a stale refusal beside
            it and must be read as the record it is.

        Raises:
            ValueError: If the index cannot be read at all -- a lost connection,
                a table the account may not read, a partially restored database.
                Translated here for the same reason the selection seam translates
                it: a caller of this module reports the failure against one
                image, and the database layer's own exception types are ones it
                cannot name.
        """
        key = sqlalchemy.select(
            sqlalchemy.literal(root_url, sqlalchemy.Text).label('root_url'),
            sqlalchemy.literal(stub, sqlalchemy.Text).label('results_path_stub'),
        ).subquery()
        statement = (
            sqlalchemy.select(
                IMAGES.c.results_path_stub.label('record_stub'),
                IMAGES.c.source_file,
                FAILED_FILES.c.reason.label('refusal_reason'),
                # Never twice: a consumer is free to read the recorded path, and
                # this statement already selects it to pair the record with the
                # document it stands for.
                *(column for column in self._columns if column.name != 'source_file'),
            )
            .select_from(key)
            .outerjoin(
                IMAGES,
                sqlalchemy.and_(
                    IMAGES.c.root_url == key.c.root_url,
                    IMAGES.c.results_path_stub == key.c.results_path_stub,
                ),
            )
            .outerjoin(
                FAILED_FILES,
                sqlalchemy.and_(
                    FAILED_FILES.c.root_url == key.c.root_url,
                    FAILED_FILES.c.results_path_stub == key.c.results_path_stub,
                ),
            )
        )
        with reporting_a_failed_read(self._raw_url), self._engine.connect() as connection:
            row = connection.execute(statement).first()
        # The key is selected as a row of its own and both tables are joined onto
        # it, so the statement answers with one row for every stub, including one
        # neither table knows.
        assert row is not None
        return row

    def _refuse_a_document_the_ingest_refused(self, stub: str, reason: Any) -> None:
        """Fail an image whose document the ingest recorded as unreadable.

        Parameters:
            stub: The stub whose record was not found.
            reason: What the index records as the reason it could not read that
                image's document, or None when it records no refusal for it.

        Raises:
            ValueError: If the index records this stub as a document the ingest
                refused, naming the stub, the index and the recorded reason.  A
                refusal means the index cannot answer for this image, which is a
                different fact from nothing having navigated it, and reading the
                one as the other builds a product from the document under one
                storage and from uncorrected pointing under the other.
        """
        if reason is None:
            return
        raise ValueError(
            f'{stub}: {self._storage} records the navigation document for this image as '
            f'one the ingest could not read ({reason}), so the index cannot say what it '
            f'recorded. Read the navigation documents instead, or fix the document and '
            f'ingest that root again.'
        )

    @staticmethod
    def _path_of(root_url: str, stub: str, source_file: Any) -> FCPath:
        """Return the document one row stands for.

        Parameters:
            root_url: The normalized root the row belongs to, which is read off
                the row rather than taken from the source: a source holding two
                roots would otherwise name every document under the first.
            stub: The image's results path stub.
            source_file: The path the ingest recorded reading, or None for a row
                that records none.

        Returns:
            The path the ingest recorded reading, so a message about this record
            names the file an operator would open, falling back to where the
            stub says the document lives.
        """
        if source_file is None:
            return document_path(root_url, stub)
        return FCPath(str(source_file))

    def _record_of(self, row: sqlalchemy.Row[Any]) -> NavRecord:
        """Rebuild one row's record, with the document it stands for.

        Parameters:
            row: One row of ``images``, carrying the consumer's columns and the
                three a streamed read adds.

        Returns:
            The record.
        """
        stub = str(row.results_path_stub)
        return NavRecord(
            path=self._path_of(str(row.root_url), stub, row.source_file),
            stub=stub,
            metadata=record_from_row(row),
        )

    def _refusal_of(self, row: sqlalchemy.Row[Any]) -> UnreadableFile:
        """Rebuild one refusal row as the file it says the ingest could not read.

        Parameters:
            row: One row of ``failed_files``, carrying its key and its reason.

        Returns:
            The unreadable file, naming where the document is and what the
            ingest said about it.
        """
        stub = str(row.results_path_stub)
        return UnreadableFile(
            path=self._path_of(str(row.root_url), stub, None),
            stub=stub,
            reason=str(row.reason),
        )


def _has_no_record_row() -> sqlalchemy.ColumnElement[bool]:
    """Return what keeps a refused file out of a stream that also carries its record.

    A stub in both tables is a record with a stale refusal beside it: the ingest
    refused the file on one pass and read it on another, and the tables are
    written independently.  The module states twice that such a stub is read as
    the record it is, and a whole-root stream runs the two statements
    separately, so the refusal arm has to say what it is not.

    Returns:
        The condition, matched on the whole key: the root as well as the stub,
        since one index serves several roots and another root's record is no
        evidence about this one's refusal.
    """
    return ~sqlalchemy.exists().where(
        IMAGES.c.root_url == FAILED_FILES.c.root_url,
        IMAGES.c.results_path_stub == FAILED_FILES.c.results_path_stub,
    )
