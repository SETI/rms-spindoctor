"""The navigation records as documents under one or more results roots.

The authoritative storage: the documents as navigation left them, complete and
always current.  Everything a program asks of this source is answered by looking
at the tree, which is why a question about every image costs a read of every
file, and why the listing is worth so much more than the names it carries.

Finding the documents is :mod:`spindoctor.nav_records.walk`'s, which lists a
directory at a time and carries each entry's size and modification time out of
the listing that reported them.

A listing of named documents, and where a syscall costs nothing
---------------------------------------------------------------

A selection that names its stubs asks about those files and no others, and there
are two ways to answer it.  Checking each named file directly is one call per
file; walking the directories they lie under is one call per directory for as
many entries as it holds.  Which is cheaper is not a ratio of the two counts, it
is what one call costs: on a local root a check is a syscall, and checking ten
files beats walking a volume of fifty thousand documents by three orders of
magnitude, a fifth of the volume by two and a half times, and loses only where
very nearly every document in the volume is named -- and there by about half, on
a call the whole of which takes under a second.  On a cloud root a check is a
paid round trip per file, against one round trip per directory for a listing
that returns about a thousand entries with their metrics, so the walk wins above
roughly a thousandth of the root.

So the choice is made on whether the root is local, and it is made here rather
than by each caller: a caller has one way to ask what a root holds, and two
shapes in the callers would be two answers to maintain.  A walk made to answer
one batch answers every later batch of the same run from what it already found,
because a run asks in batches and a walk per batch would be a walk per batch of
the whole scan.

What a check cannot report is the size and the modification time, which come
from a directory entry: an entry answered by a check carries neither, and says
so through :attr:`~spindoctor.nav_records.ListedRecord.has_metrics`.

Batched underneath, lazy on top
-------------------------------

Documents are retrieved in groups and yielded one at a time.  Batched because on
a cloud root each file is a separate round trip and a backend downloads a batch
in parallel; lazy because the caller should not have to hold a mission in memory
to read the first record of one.

Read once, answered twice
-------------------------

The records and the per-image facts come off one pass over one batch: a document
is retrieved once, parsed once, and either handed back as the record it is or
flattened into the facts it holds.  The two file metrics the facts carry are the
listing's own, threaded through from the walk that found the document, so
nothing here stats a file and nothing walks a root a second time to answer the
second question about it.

Each of the two narrows on what it read.  A record is the document, so a stream
of records reads the mission and the exposure midtime out of the document, and a
document recording neither is reported rather than passed over, which is the
only thing such a stream can say about one.  The facts are the row an ingest of
the same file writes, so a stream of facts narrows on the facts: a file that
yields none is an unreadable file whatever the selection asks for, and a file
that yields facts is compared on the same two values an index compares its rows
on.  That is what makes one selection over one tree cover the same images out of
either storage.
"""

from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import cast

from filecache import FCPath
from pdslogger import NullLogger, PdsLogger

from spindoctor.nav_records.document import (
    COULD_NOT_RETRIEVE,
    METADATA_SUFFIX,
    document_or_refusal,
    document_path,
    read_document,
    stub_refusal,
)
from spindoctor.nav_records.facts import (
    DocumentOrigin,
    ImageFacts,
    MetadataDocumentError,
    facts_from_document,
    subtree_of,
)
from spindoctor.nav_records.record import ListedRecord, NavRecord, UnreadableFile
from spindoctor.nav_records.roots import distinct_roots
from spindoctor.nav_records.selection import Selection
from spindoctor.nav_records.source import (
    in_batches,
    refuse_what_a_listing_cannot_answer,
    root_for_stub,
    root_for_stubs,
    selected_roots,
)
from spindoctor.nav_records.tuning import TreeTuning
from spindoctor.nav_records.walk import (
    UnlistableDirectoryError,
    UnlistableRootError,
    walk_from,
)
from spindoctor.support.nav_record import record_midtime_et

__all__ = [
    'NAMES_NO_INSTRUMENT',
    'RECORDS_NO_MIDTIME',
    'TreeRecordSource',
]

NAMES_NO_INSTRUMENT = 'names no instrument to attribute it to a mission'
"""Why a document that read perfectly well is still not one mission's record.

Not one of the reasons in :mod:`spindoctor.nav_records.document`, which are the
ways a file yields no document at all.  This one is a document, read and parsed,
that a mission-filtered stream of records cannot place: only a document that
*names* a mission can be another mission's, so one naming none is reported
rather than passed over silently, which is how a truncated or corrupted document
would otherwise vanish from every mission's run without a trace.
"""

RECORDS_NO_MIDTIME = 'records no exposure midtime to place it in a span of time'
"""Why a document that read perfectly well is still not one span's record.

The time filter's half of :data:`NAMES_NO_INSTRUMENT`, reported by the same
stream and for the same reason: only a document that records a usable midtime
can be shown to lie outside a span, so one recording none is reported rather
than passed over.  A truncated document, or one whose midtime is a NaN that
would otherwise satisfy every bound at once, would vanish from every
time-bounded run without a trace.
"""


class TreeRecordSource:
    """The navigation records as documents under one or more results roots.

    Parameters:
        roots: The results roots to read, in the order questions are answered
            about them.  Two spellings of one root are one root, and the source
            holds each of them once.
        logger: Logger for what the walk declines to descend into.  Nothing else
            here logs: a file that yielded no record is yielded to the caller,
            which reports it in its own terms.  None says nothing at all, which
            is what a caller holding no logger of its own needs -- a layer that
            must not acquire a voice its own caller did not configure has none
            to lend.
        tuning: How much of a pass over the documents runs at once.  None is
            the library's own defaults, for a caller with no configuration to
            consult; a program resolves its configuration once and passes what
            it says, the way it passes its logger.

    Raises:
        ValueError: If no root is given, or if one of them is not a location.
    """

    def __init__(
        self,
        roots: Sequence[str | Path | FCPath],
        *,
        logger: PdsLogger | None = None,
        tuning: TreeTuning | None = None,
    ) -> None:
        held = distinct_roots(roots)
        if not held:
            raise ValueError('a record source over the documents needs at least one results root')
        self._roots = tuple(held)
        self._logger = NullLogger() if logger is None else logger
        self._tuning = TreeTuning() if tuning is None else tuning
        # What a walk answering a listing of named documents found, keyed by the
        # root and the top-level directory walked.  A scan asks in batches, so a
        # walk made for one batch answers every later batch of the same scan.
        # The root is half the key because a source holds several of them and
        # one stub is a key under each: a memory keyed on the directory alone
        # would hand one root's answer back for another.
        self._walked: dict[tuple[str, str], dict[str, ListedRecord]] = {}

    def __enter__(self) -> 'TreeRecordSource':
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
        """Read one image's document.

        Parameters:
            stub: The image's results path stub.

        Returns:
            The record, paired with the document it was read from.

        Raises:
            ValueError: If this source holds more than one root, naming them
                all: a stub is a key under a root and says nothing about which
                root it is a key under, so there is no single document to read.
                Also if the file is not valid JSON, or is valid JSON that is not
                an object.
            FileNotFoundError: If the document is not there, or if the stub is
                not a key under a root at all.  Both mean the same thing to a
                caller -- this image has no readable record here -- and the
                message says which of the two it was.
        """
        root = FCPath(root_for_stub(self._roots, stub))
        refusal = stub_refusal(stub)
        if refusal is not None:
            raise FileNotFoundError(
                f'{stub}: does not name a navigation document under '
                f'{root} ({refusal}), so none can be read for this image'
            )
        path = document_path(root, stub)
        return NavRecord(path=path, stub=stub, metadata=read_document(path))

    def listing(self, selection: Selection) -> Iterator[ListedRecord]:
        """Return every document the selection covers, without opening one.

        A selection that names no stubs walks: the roots in the order this
        source holds them, and each root's documents in an order the walk does
        not define, since it lists several directories at once and no consumer
        reads an order from it.  One that names stubs asks about those files and
        no others, and is answered by whichever call is cheap on the root they
        are under -- a check per file on a local root, where a check is a
        syscall, and a walk of the directories they lie in on a remote one,
        where it is a paid round trip.  Which of the two answered is not
        something a caller has to know: they answer the same question, and only
        the walk can report an entry's metrics.

        What the selection asks for is checked before anything is walked or
        checked, so a selection this source cannot honour is refused where it is
        asked rather than partway through a caller's loop.

        Parameters:
            selection: Which documents to list.  ``instrument``, ``start_et``
                and ``stop_et`` may not be set.  A selection naming stubs names
                the documents outright, and its ``subtrees`` narrow nothing
                further, exactly as they narrow nothing for a stream of records
                or of facts.

        Returns:
            One entry per document that is there, produced as the answer comes.
            An entry a walk found carries the size and modification time its
            directory listing reported; one a check answered carries neither,
            and says so through
            :attr:`~spindoctor.nav_records.ListedRecord.has_metrics`.  Stubs are
            answered in the order the selection named them, and a named stub the
            root holds no document for yields nothing.

        Raises:
            ValueError: If the selection carries ``instrument``, ``start_et`` or
                ``stop_et``, naming which.  A listing opens no document, so it
                cannot answer what a document says, and a restriction silently
                ignored is a wrong answer rather than a missing feature: a
                caller would read a listing of the whole root as a listing of
                one mission.  Also if the selection names a root this source
                does not hold, or names stubs without resolving to exactly one
                root, since a stub is a key under a root.
            ~spindoctor.nav_records.UnlistableDirectoryError: If a directory
                under a selected root could not be listed, or
                :class:`~spindoctor.nav_records.UnlistableRootError` if a
                selected root could not be listed at all.  Raised from the
                walk, so it reaches the caller as it reads.  A listing of named
                stubs is answered about files rather than about a whole root,
                so a directory of the tree that is not there holds none of them
                instead of ending the listing, which is the answer a check of
                one of those files gives.
        """
        refuse_what_a_listing_cannot_answer(selection)
        roots = selected_roots(self._roots, selection.roots)
        if not selection.stubs:
            return self._listing_of(roots, selection.subtrees)
        return self._listing_of_named(root_for_stubs(roots, selection.stubs), selection.stubs)

    def records(self, selection: Selection) -> Iterator[NavRecord | UnreadableFile]:
        """Return the records the selection covers, yielded one at a time.

        Documents are retrieved in batches and yielded singly, so a caller
        holds one record where the source holds one batch.  What the selection
        asks for is checked before anything is read.

        Parameters:
            selection: Which records to yield.  A selection naming stubs reads
                exactly those, in the order it names them; one naming none takes
                its stubs from the listing of the selected roots.  Either way
                every stub is a key under the root: a named one was checked
                where the selection was written, and a listed one came from a
                document under the root itself.

        Returns:
            One record per document that yielded one, and one
            :class:`~spindoctor.nav_records.record.UnreadableFile` per file that
            did not.  A document of another mission is passed over silently when
            the selection names one, and so is a record outside the selection's
            time bounds; a document that names no mission, or that records no
            midtime a bound could be applied to, is reported instead, because
            only a document that says which it is can be shown not to be this
            run's.

        Raises:
            ValueError: If the selection names stubs without resolving to
                exactly one root, since a stub is a key under a root; or if it
                names a root this source does not hold.
            ~spindoctor.nav_records.UnlistableDirectoryError: If a directory
                under a selected root could not be listed, or
                :class:`~spindoctor.nav_records.UnlistableRootError` if a
                selected root could not be listed at all.  Raised from the walk
                as the caller reads, and not at all when the selection names
                its own stubs, which lists nothing.
        """
        return self._selected_records(self._found(selection), selection)

    def facts(self, selection: Selection) -> Iterator[ImageFacts | UnreadableFile]:
        """Return what every document the selection covers says about its image.

        One pass over the same batched retrieval :meth:`records` makes, so a
        document is read once whichever of the two a caller asked for.  The two
        file metrics the facts carry come from the walk's own listing entries,
        which is where the walk already had them: asking the storage layer for
        them again would be one round trip per image on a cloud root, against
        one per directory for the listing that reported them in the first place.
        A selection naming its own stubs did no walk, so those images carry
        neither metric.

        Every set of facts carries the record it was read out of, since this
        source has one in hand by the time it has any facts at all.  A consumer
        that narrows on what a document says and then wants that document reads
        it once.

        Parameters:
            selection: Which images to yield.  A selection naming stubs reads
                exactly those, in the order it names them; one naming none takes
                its stubs from the listing of the selected roots.

        Returns:
            One set of facts per document that yielded one, and one
            :class:`~spindoctor.nav_records.record.UnreadableFile` per file that
            did not.  A file that is no navigation document is one of those,
            carrying the reason the document reader refuses it by, which is the
            reason an ingest of the same tree records for it.  The mission
            filter and the time bounds are then applied to the facts, which are
            the values an index filters its rows on, so a file that yields no
            facts is always an unreadable file and a file that yields facts is
            narrowed on exactly what the other storage narrows on.  An image
            whose facts record no midtime is therefore not selected under a
            time bound, in place of being reported: a row recording none
            satisfies no bound either.

        Raises:
            ValueError: If the selection names stubs without resolving to
                exactly one root, or names a root this source does not hold.
            ~spindoctor.nav_records.UnlistableDirectoryError: If a directory
                under a selected root could not be listed, or
                :class:`~spindoctor.nav_records.UnlistableRootError` if a
                selected root could not be listed at all.
        """
        return self._selected_facts(self._found(selection), selection)

    def _found(
        self, selection: Selection
    ) -> Iterator[tuple[DocumentOrigin, NavRecord | UnreadableFile]]:
        """Read the documents the selection names a place for, with where each came from.

        The one stream both public reads are built on, and the reason each of
        them applies the selection's restrictions itself: the two read different
        things out of one document, and each narrows on what it read.  What the
        selection asks for is checked here rather than inside the generator, so
        a selection this source cannot honour is refused where a caller asked
        rather than partway through its loop.

        Parameters:
            selection: Which documents to read, for the roots, the subtrees and
                the stubs it names.

        Returns:
            One pair per document read: where it came from, and the record or
            the unreadable file it produced.

        Raises:
            ValueError: If the selection names stubs without resolving to
                exactly one root, or names a root this source does not hold.
        """
        roots = selected_roots(self._roots, selection.roots)
        if not selection.stubs:
            return self._found_of(roots, selection)
        root_url = root_for_stubs(roots, selection.stubs)
        root = FCPath(root_url)
        listed = (
            ListedRecord(stub=stub, path=document_path(root, stub), mtime_ns=None, size_bytes=None)
            for stub in selection.stubs
        )
        return self._found_of_root(root, root_url, listed)

    def _selected_records(
        self,
        found: Iterator[tuple[DocumentOrigin, NavRecord | UnreadableFile]],
        selection: Selection,
    ) -> Iterator[NavRecord | UnreadableFile]:
        """Yield the records of the documents the selection covers.

        Parameters:
            found: What was read, document by document.
            selection: The selection, for the restrictions a document answers.

        Yields:
            One record per document the selection covers, and one unreadable
            file per document that yielded none or that cannot be placed
            against a restriction the selection makes.
        """
        for _origin, one in found:
            selected = self._document_selected(one, selection)
            if selected is not None:
                yield selected

    def _selected_facts(
        self,
        found: Iterator[tuple[DocumentOrigin, NavRecord | UnreadableFile]],
        selection: Selection,
    ) -> Iterator[ImageFacts | UnreadableFile]:
        """Yield the facts of the documents the selection covers.

        Parameters:
            found: What was read, document by document.
            selection: The selection, for the restrictions the facts answer.

        Yields:
            One set of facts per document the selection covers, and one
            unreadable file per document that yielded none.
        """
        for origin, one in found:
            facts = self._facts_of(origin, one)
            if isinstance(facts, UnreadableFile) or self._facts_selected(facts, selection):
                yield facts

    @staticmethod
    def _facts_of(
        origin: DocumentOrigin, found: NavRecord | UnreadableFile
    ) -> ImageFacts | UnreadableFile:
        """Flatten one document that was read, or pass on the file that was not.

        Parameters:
            origin: Where the document came from and what the walk saw of it.
            found: The record read out of it, or the unreadable file it was
                instead.

        Returns:
            The facts, carrying the record they were read out of, or the
            unreadable file.  A document that is no current-schema navigation
            document becomes one of those, carrying the reason the document
            reader states for it, so this source and an index ingested from the
            same tree refuse the same files for the same reason.

            The record travels with the facts because it is already read and
            already parsed: a consumer that narrows on what a document says and
            then wants the document itself would otherwise retrieve and parse it
            a second time, which on a cloud root is a second download.  This is
            the one place the field is filled, so a set of facts carrying a
            record came from a document and one carrying none came from a row.
        """
        if isinstance(found, UnreadableFile):
            return found
        try:
            facts = facts_from_document(found.metadata, origin)
        except MetadataDocumentError as exc:
            return UnreadableFile(path=found.path, stub=found.stub, reason=exc.reason)
        return replace(facts, record=found)

    def _listing_of_named(self, root_url: str, stubs: Sequence[str]) -> Iterator[ListedRecord]:
        """Say which of the named documents are there, by whichever call is cheap.

        The whole of the decision, made here so that no caller makes it: one
        caller asking two ways would be two answers to keep true of each other,
        and the question is the same one either way.

        Parameters:
            root_url: The one normalized root those keys are under.
            stubs: The stubs to answer about, in the order to answer them.

        Returns:
            One entry per named document that is there, in the order the stubs
            were named.

        Raises:
            UnlistableDirectoryError: From the walk a remote root is answered
                from, for a directory under a named stub's own that would not be
                listed.
        """
        root = FCPath(root_url)
        if root.is_local():
            return self._checked(root, stubs)
        return self._found_in_a_walk(root, root_url, stubs)

    def _checked(self, root: FCPath, stubs: Sequence[str]) -> Iterator[ListedRecord]:
        """Ask the filesystem about each named document, a batch of paths at a time.

        The call that costs a syscall per file, which is what makes it the cheap
        one on a local root wherever the question is worth asking: measured over
        a volume of fifty thousand documents, ten files named beat a walk of
        their directories by three orders of magnitude, a fifth of the volume by
        two and a half times, and only naming very nearly every document in it
        does the walk come back ahead, by about half.  Batched so that a caller
        naming a mission's worth of keys does not put every one of them in one
        list.

        Parameters:
            root: The results root the stubs are keys under, which is local.
            stubs: The stubs to answer about, in the order to answer them.

        Yields:
            One entry per named document that is there, carrying neither metric.
            A check says whether the file is there and reports nothing else
            about it, so a consumer that needs the metrics reads
            :attr:`~spindoctor.nav_records.ListedRecord.has_metrics` and finds
            them absent, rather than being handed a stand-in for them.
        """
        for batch in in_batches(iter(stubs), self._tuning.retrieve_batch_size):
            sub_paths: list[str | Path] = [f'{stub}{METADATA_SUFFIX}' for stub in batch]
            there = cast(list[bool], root.exists(sub_paths, nthreads=self._tuning.retrieve_threads))
            for stub, found in zip(batch, there, strict=True):
                if found:
                    yield ListedRecord(
                        stub=stub, path=document_path(root, stub), mtime_ns=None, size_bytes=None
                    )

    def _found_in_a_walk(
        self, root: FCPath, root_url: str, stubs: Sequence[str]
    ) -> Iterator[ListedRecord]:
        """Answer about the named documents from a walk of the directories they lie in.

        The call that costs one round trip per directory for about a thousand
        entries rather than one round trip per file, which is what makes it the
        cheap one on a cloud root above roughly a thousandth of the root.  What
        a walk found is kept for the rest of the run: a scan asks in batches,
        and a walk per batch would list one directory once for every batch of
        the images under it.

        Parameters:
            root: The results root the stubs are keys under, which is remote.
            root_url: The same root normalized, which is half of what a walk is
                remembered under.
            stubs: The stubs to answer about, in the order to answer them.

        Yields:
            One entry per named document a walk of this root found, in the order
            the stubs were named, carrying the two metrics the listing reported.

        Raises:
            UnlistableDirectoryError: If a directory under one of the walked
                ones would not be listed.
        """
        self._walk_for(root, root_url, stubs)
        for stub in stubs:
            found = self._walked_entry(root_url, stub)
            if found is not None:
                yield found

    def _walk_for(self, root: FCPath, root_url: str, stubs: Sequence[str]) -> None:
        """Walk whatever of this root the named stubs are not already answered from.

        Parameters:
            root: The results root, which is remote.
            root_url: The same root normalized, which is half of the key.
            stubs: The stubs to be answered about.
        """
        if (root_url, '') in self._walked:
            # The whole root is in hand, so every stub under it is answered.
            return
        wanted = {subtree_of(stub) or '' for stub in stubs}
        # A stub with no subtree above it names a document directly under the
        # root, and the only walk that reaches one is a walk of the root -- which
        # covers every other stub's directory too, so it is the only walk left
        # to make.
        for scope in sorted({''} if '' in wanted else wanted):
            key = (root_url, scope)
            if key not in self._walked:
                self._walked[key] = self._walk_of(root, scope)

    def _walk_of(self, root: FCPath, scope: str) -> dict[str, ListedRecord]:
        """List one top-level directory of a root and everything under it, by stub.

        Parameters:
            root: The results root.
            scope: The top-level directory to walk, or empty for the whole root.

        Returns:
            Every document found, keyed by its stub.  Empty when that directory
            is not there at all: the root holds no such directory, so it holds
            none of the documents under it, which is the answer a check of one
            of those files gives on a local root.  Every other way a directory
            refuses to be listed is a directory whose documents nobody saw, and
            is refused rather than read as an absence.

        Raises:
            UnlistableDirectoryError: If the directory is there and would not be
                listed, or if any directory under it would not be.
        """
        directory = root if scope == '' else root / scope
        unlistable = UnlistableRootError if scope == '' else UnlistableDirectoryError
        try:
            return {
                entry.stub: entry
                for entry in walk_from(
                    directory,
                    '' if scope == '' else f'{scope}/',
                    {},
                    unlistable=unlistable,
                    logger=self._logger,
                    tuning=self._tuning,
                )
            }
        except UnlistableDirectoryError as exc:
            if exc.directory != directory.as_posix() or not isinstance(
                exc.__cause__, (FileNotFoundError, NotADirectoryError)
            ):
                raise
        return {}

    def _walked_entry(self, root_url: str, stub: str) -> ListedRecord | None:
        """Return what a walk of this root found for one stub, if anything did.

        Parameters:
            root_url: The normalized root the stub is a key under, which is what
                keeps one root's answer out of another's.
            stub: The stub.

        Returns:
            The listing entry, or None when no walk of this root found it.
        """
        whole = self._walked.get((root_url, ''))
        if whole is not None:
            return whole.get(stub)
        return self._walked.get((root_url, subtree_of(stub) or ''), {}).get(stub)

    def _listing_of(self, roots: Sequence[str], subtrees: Sequence[str]) -> Iterator[ListedRecord]:
        """Walk each root in turn, yielding its documents as they are found.

        Parameters:
            roots: The normalized roots to walk, in the order to walk them.
            subtrees: The top-level directories of each to descend, or empty for
                the whole of each root.

        Yields:
            One entry per document found.
        """
        for root_url in roots:
            yield from self._listing_of_root(FCPath(root_url), subtrees)

    def _found_of(
        self, roots: Sequence[str], selection: Selection
    ) -> Iterator[tuple[DocumentOrigin, NavRecord | UnreadableFile]]:
        """Read each root's documents in turn, taking its listing as it goes.

        The listing entries are carried through rather than reduced to their
        stubs, because they hold the two file metrics the facts record and the
        walk is the only thing that knows them.

        Parameters:
            roots: The normalized roots to read, in the order to read them.
            selection: The selection, for the subtrees to walk.

        Yields:
            One pair per document under those subtrees.
        """
        for root_url in roots:
            root = FCPath(root_url)
            listed = self._listing_of_root(root, selection.subtrees)
            yield from self._found_of_root(root, root_url, listed)

    def describe(self) -> str:
        """Return where these records came from, for the run log.

        Returns:
            The roots the documents were read under, in the order this source
            holds them.
        """
        return f'the navigation documents under {", ".join(self._roots)}'

    def close(self) -> None:
        """Release what a listing of named documents walked, and nothing else.

        Reading documents holds nothing open, so this is the whole of it: a walk
        made to answer one batch of named stubs is held for every later batch of
        the same scan, and a run that is done with the source is done with it.
        """
        self._walked.clear()

    def _listing_of_root(self, root: FCPath, subtrees: Sequence[str]) -> Iterator[ListedRecord]:
        """Yield one root's documents, restricted to the named top-level directories.

        Parameters:
            root: The results root to walk.
            subtrees: The top-level directories to descend, or empty for the
                whole root.

        Yields:
            One entry per document found.

        Raises:
            UnlistableRootError: If the whole root was asked for and could not
                be listed.
            UnlistableDirectoryError: If any directory under it could not be
                listed, a named subtree among them.  A subtree the root does not
                hold is refused rather than passed over: a run restricted to a
                directory that is not there would otherwise report a clean pass
                over nothing.
        """
        # One record of where this walk has been, shared by every subtree of the
        # root, so that two subtrees that are one directory reached two ways are
        # listed once between them rather than once each.
        visited: dict[tuple[int, int], str] = {}
        if not subtrees:
            yield from walk_from(
                root,
                '',
                visited,
                unlistable=UnlistableRootError,
                logger=self._logger,
                tuning=self._tuning,
            )
            return
        for subtree in subtrees:
            yield from walk_from(
                root / subtree,
                f'{subtree}/',
                visited,
                unlistable=UnlistableDirectoryError,
                logger=self._logger,
                tuning=self._tuning,
            )

    def _found_of_root(
        self, root: FCPath, root_url: str, listed: Iterator[ListedRecord]
    ) -> Iterator[tuple[DocumentOrigin, NavRecord | UnreadableFile]]:
        """Retrieve one root's documents in batches and yield them one at a time.

        Every document read is yielded.  What a selection restricts is applied
        to what was read out of a document, and the two public reads read
        different things out of one document, so nothing here narrows.

        Parameters:
            root: The results root the stubs are keys under.
            root_url: The same root normalized, which is how a document records
                where it came from.
            listed: The documents to read, which arrive lazily when they come
                from a listing.

        Yields:
            One pair per listed document, read and unfiltered.
        """
        for batch in in_batches(listed, self._tuning.retrieve_batch_size):
            sub_paths: list[str | Path] = [f'{entry.stub}{METADATA_SUFFIX}' for entry in batch]
            # retrieve() rather than get_local_path(): on a cloud root the
            # latter names a file it never downloads.  exception_on_fail=False
            # keeps one file that never arrived from ending the pass.
            local_paths = cast(
                list[Path | Exception],
                root.retrieve(
                    sub_paths, exception_on_fail=False, nthreads=self._tuning.retrieve_threads
                ),
            )
            for entry, local_path in zip(batch, local_paths, strict=True):
                found = self._read_of(root, entry.stub, local_path)
                yield self._origin_of(root_url, entry, found), found

    @staticmethod
    def _origin_of(
        root_url: str, listed: ListedRecord, found: NavRecord | UnreadableFile
    ) -> DocumentOrigin:
        """Return where one document came from and what the walk saw of it.

        Parameters:
            root_url: The normalized results root the document is under.
            listed: The listing entry the document came from, which carries the
                two metrics.
            found: What was read out of it, for the path it names.

        Returns:
            The origin.  The file it names is the one the record or the
            unreadable file names, so a message about an image and the
            provenance recorded for it name one file.
        """
        return DocumentOrigin(
            root_url=root_url,
            results_path_stub=listed.stub,
            source_file=found.path.as_posix(),
            mtime_ns=listed.mtime_ns,
            size_bytes=listed.size_bytes,
        )

    @staticmethod
    def _read_of(
        root: FCPath, stub: str, local_path: Path | Exception
    ) -> NavRecord | UnreadableFile:
        """Read one retrieved document, or say why it yielded no record.

        Parameters:
            root: The results root the stub is a key under.
            stub: The image's results path stub.
            local_path: What the retrieval produced for it: a local file, or the
                exception that says it never arrived.

        Returns:
            The record the document holds, or the unreadable file it is instead.
        """
        path = document_path(root, stub)
        if isinstance(local_path, BaseException):
            return UnreadableFile(path=path, stub=stub, reason=COULD_NOT_RETRIEVE)
        metadata = document_or_refusal(FCPath(local_path))
        if isinstance(metadata, str):
            return UnreadableFile(path=path, stub=stub, reason=metadata)
        return NavRecord(path=path, stub=stub, metadata=metadata)

    def _document_selected(
        self, found: NavRecord | UnreadableFile, selection: Selection
    ) -> NavRecord | UnreadableFile | None:
        """Apply the selection's restrictions to one document, as a record.

        A record is the document, so the mission and the span are read out of
        the document itself: what a stream of records hands back is what the
        file says, and the field a filter compares is the field the file
        records.

        Parameters:
            found: The record read out of the document, or the unreadable file
                it was instead, which no restriction can be applied to and which
                every run reports.
            selection: The selection, for the restrictions a document has to be
                opened to answer.

        Returns:
            The record, or the unreadable file it is instead, or None when the
            document is outside the selection -- another mission's, or outside
            its time bounds.  Those two are passed over rather than reported,
            because neither is this run's business.  A document that cannot be
            placed at all -- naming no mission under a mission filter, recording
            no usable midtime under a time bound -- is an unreadable file
            instead: it cannot be shown to be another run's, so passing over it
            would drop it out of every run there is.
        """
        if isinstance(found, UnreadableFile):
            return found
        metadata = found.metadata
        if selection.instrument is not None:
            observation = metadata.get('observation')
            instrument = observation.get('instrument') if isinstance(observation, dict) else None
            if not isinstance(instrument, str):
                return UnreadableFile(path=found.path, stub=found.stub, reason=NAMES_NO_INSTRUMENT)
            if instrument != selection.instrument:
                return None
        if selection.bounded_in_time:
            midtime = record_midtime_et(metadata)
            if midtime is None:
                return UnreadableFile(path=found.path, stub=found.stub, reason=RECORDS_NO_MIDTIME)
            if not self._within(midtime, selection):
                return None
        return found

    def _facts_selected(self, facts: ImageFacts, selection: Selection) -> bool:
        """Whether one document's facts are inside the selection's restrictions.

        The mission and the midtime are read off the facts, which are the values
        a row carries and the values a query over rows compares, so the two
        storages narrow one tree to the same images.  A document that carries
        neither is refused before it gets here -- the facts hold an instrument
        or they are not a navigation document's -- so the only value that can be
        absent is the midtime, and an image with none is not selected under a
        bound, exactly as a row with none satisfies no comparison.

        Parameters:
            facts: What the document says about its image.
            selection: The selection, for the mission and the time bounds.

        Returns:
            True when the image is inside every restriction the selection makes.
        """
        if selection.instrument is not None and facts.image['instrument'] != selection.instrument:
            return False
        if not selection.bounded_in_time:
            return True
        midtime = facts.image['midtime_et']
        return midtime is not None and self._within(float(midtime), selection)

    @staticmethod
    def _within(midtime: float, selection: Selection) -> bool:
        """Whether one exposure midtime lies inside the selection's time bounds.

        Parameters:
            midtime: The record's exposure midtime, which the caller has already
                read and found usable.
            selection: The selection, which places at least one bound.

        Returns:
            True when the midtime is inside every bound the selection places.
            Both bounds are inclusive.
        """
        if selection.start_et is not None and midtime < selection.start_et:
            return False
        return not (selection.stop_et is not None and midtime > selection.stop_et)
