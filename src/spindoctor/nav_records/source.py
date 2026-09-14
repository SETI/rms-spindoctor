"""The seam every program reads its navigation records through.

A navigation pass writes one document per image, and a results index holds one
row per image built from those documents.  The programs downstream want the
record and do not care which of the two answered, so they ask through this
protocol and each storage implements it.

Four questions, because the programs ask four things.  One image, by its stub,
asked inside a per-image loop.  A stream of records, asked once per run by a
program that summarizes or sweeps, and asked with an explicit list of stubs by a
queue worker that must read exactly the files it was handed.  A listing, which
answers what is there without opening a single document, for the two consumers
that need no more than that.  And a stream of per-image facts, for a program that
reads every field of every image rather than the few a record carries.

A record and a per-image fact set are different shapes because the two storages
can supply different things.  A record looks like the document it stands for, so
the index rebuilds one out of the columns its consumer selected and refuses to
invent a field the document did not have; that is what makes a record cheap.
The facts are the whole row -- everything the index column set holds about one
image, in the shape both storages hold it in -- so a consumer that reads
everything reads one shape rather than one storage.  What that column set leaves
out of a document it leaves out of the facts as well.

What every implementation owes a caller
---------------------------------------

**A stream yields.**  A caller that wants a list writes ``list(...)`` and owns
that decision.  Nothing is accumulated on the caller's behalf, so a program that
sweeps a mission holds one record at a time rather than the mission.

**Nothing promises an order.**  Each implementation yields in the order it finds
records and documents what that order is; no implementation promises a total
order over the stream, because neither storage can supply one without giving up
the streaming.  A caller that needs a total order calls ``sorted()``.

**The roots bind to the source.**  :meth:`RecordSource.record` takes no root, so
the source has to know one, and a selection naming no root covers every root the
source holds.  A source holding more than one root cannot answer for a bare stub
and refuses to.

**The source owns what it opened.**  A stream may hold a connection or a cursor,
and a caller that walks away mid-loop must not leak it, so a source is closed
when a run is done with it and every implementation is usable as a context
manager.

**A failure arrives from ``next()``.**  A file that could not be read is yielded
into the stream rather than raised, so one of them costs itself and not the rest
of the pass; but a refusal that ends a pass -- a directory nobody can list, an
index that stops answering -- surfaces in the middle of the caller's loop rather
than up front.  A program that writes files as it goes can therefore be
interrupted half-written, so a program using the stream finishes its pass before
it writes its output.

The rules an implementation applies before it answers
-----------------------------------------------------

Which roots a selection covers, which single root a stub is a key under, and
what a listing may not be asked are decisions about the seam rather than about a
storage, so they are made here and both implementations call them.  Written out
twice they would be two answers: one storage would accept a selection the other
refused, and the difference would surface as a run that read a different set of
records depending on where its records were kept.
"""

from collections.abc import Iterator, Sequence
from itertools import islice
from types import TracebackType
from typing import Protocol, TypeVar

from spindoctor.nav_records.facts import ImageFacts
from spindoctor.nav_records.record import ListedRecord, NavRecord, UnreadableFile
from spindoctor.nav_records.roots import distinct_roots
from spindoctor.nav_records.selection import Selection

__all__ = [
    'RecordSource',
    'in_batches',
    'refuse_what_a_listing_cannot_answer',
    'root_for_stub',
    'root_for_stubs',
    'selected_roots',
]

_Item = TypeVar('_Item')


class RecordSource(Protocol):
    """One or more results roots' navigation records, however they are stored."""

    def record(self, stub: str) -> NavRecord:
        """Return the navigation record of one image.

        A stub is a key under a root, so this asks a source that holds exactly
        one root.  It is the shape a per-image loop asks in, so it is the one
        call an implementation answers in a single round trip.

        Parameters:
            stub: The image's results path stub, which is its identity under the
                root.

        Returns:
            The record, paired with the document it stands for.

        Raises:
            FileNotFoundError: If nothing recorded this image.
            ValueError: If the source holds more than one root, naming them all,
                since a bare stub does not say which of them it belongs to; or
                if something recorded this image and the source cannot say what
                it recorded.
        """
        ...

    def records(self, selection: Selection) -> Iterator[NavRecord | UnreadableFile]:
        """Yield the records a selection covers, one at a time.

        Every file the selection covers is accounted for: one that yielded a
        record is yielded as a record, and one that yielded none is yielded as
        an :class:`~spindoctor.nav_records.record.UnreadableFile` saying why,
        rather than raised on.  A file that names no image is one no run can
        report an omission against, so a pass that passed over it in silence
        would report itself clean while covering less than the selection.

        The order is whatever order the implementation finds records in, and it
        promises no more than that.

        Parameters:
            selection: Which records to yield.  A selection naming stubs names
                the images outright, and they are read in the order it names
                them.

        Yields:
            One record, or one unreadable file, per file the selection covers.

        Raises:
            ValueError: If the selection cannot be honored by this source --
                naming a root it does not hold, or naming stubs without
                resolving to the single root a stub is a key under.
        """
        ...

    def facts(self, selection: Selection) -> Iterator[ImageFacts | UnreadableFile]:
        """Yield what every image the selection covers says about itself.

        The question :meth:`records` cannot answer.  A record is defined as
        looking like the document it stands for, so a source reading rows hands
        back the fields its consumer's columns hold and nothing else; a consumer
        that reads every field of every image would have to name every column
        and would still get no per-technique or per-feature detail, since
        neither lives in a record.  So a program that summarizes a whole run
        asks for the facts instead, and gets the same shape from either storage.

        Every file the selection covers is accounted for: one that yielded no
        facts is yielded as an
        :class:`~spindoctor.nav_records.record.UnreadableFile` saying why,
        rather than raised on or passed over.  Which files those are is not the
        set :meth:`records` refuses.  A record is what a document says, so
        anything that parses to a JSON object is one; the facts are the fields
        an image is summarized by, so a document naming no image, or holding a
        field of the schema in some other shape, yields none.  Such a file is a
        record from :meth:`records` and an unreadable file here, and the reason
        given here is the one an ingest of the same tree records for it.

        The order is whatever order the implementation finds images in, and it
        promises no more than that.

        Parameters:
            selection: Which images to yield.  A selection naming stubs names
                the images outright, and they are answered in the order it names
                them.

        Yields:
            One set of facts per image the selection covers, or one unreadable
            file per file no facts could be read out of.

        Raises:
            ValueError: If the selection cannot be honored by this source --
                naming a root it does not hold, or naming stubs without
                resolving to the single root a stub is a key under.
        """
        ...

    def listing(self, selection: Selection) -> Iterator[ListedRecord]:
        """Yield what the source holds, without opening a document.

        Much cheaper than :meth:`records`: on a cloud root a directory listing
        returns up to a thousand entries with their metrics in one round trip,
        where reading their documents is one round trip apiece.  What it cannot
        do is answer anything a document says, so a selection restricting on
        what a document says is refused rather than partly honored.

        A selection naming stubs asks about those files and no others, which is
        what a caller enumerating candidates asks: which of the images this run
        might still keep has a document.  It is not a restriction on what a
        document says -- a stub is the identity of a file -- so each storage
        answers it by whatever means is cheap on the storage it is, and neither
        opens anything.

        The order is whatever order the implementation finds documents in, and
        the order the selection named them in when it named stubs.

        Parameters:
            selection: Which documents to list.  Only the restrictions that can
                be answered without opening a document may be set.

        Yields:
            One entry per document, carrying its stub, its location and the two
            metrics that say whether it has changed.  An implementation that
            answered by checking named files reports neither metric, since a
            check reports nothing about a file but whether it is there, and says
            so through
            :attr:`~spindoctor.nav_records.ListedRecord.has_metrics`.

        Raises:
            ValueError: If the selection carries a restriction a listing cannot
                answer, naming which; if it names a root the source does not
                hold; or if it names stubs without resolving to the single root
                a stub is a key under.
        """
        ...

    def describe(self) -> str:
        """Return where these records came from, for the run log.

        Returns:
            The storage that answered, in terms an operator can act on: the
            roots, and the index the records were read out of when they were
            read out of one.  Any password in an index URL is masked.
        """
        ...

    def close(self) -> None:
        """Release whatever the source holds open.

        Called once, when a run is done with the source.  An implementation
        holding nothing open releases nothing, and closing twice costs nothing.
        """
        ...

    def __enter__(self) -> 'RecordSource':
        """Enter a run's use of this source.

        Declared on the protocol rather than left to each implementation: a
        stream may hold a connection or a cursor, so a caller that walks away
        mid-loop must be able to close the source without knowing which storage
        answered.

        Returns:
            The source itself.
        """
        ...

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
        ...


def selected_roots(held: Sequence[str], named: Sequence[str]) -> tuple[str, ...]:
    """Return the roots a selection covers, in the order the source holds them.

    The order is the source's rather than the selection's, so that two
    selections naming the same roots two ways are answered the same way.

    Parameters:
        held: The normalized roots the source holds, in its own order.
        named: The roots the selection named, in whatever spelling it named
            them.  Naming none covers every root the source holds.

    Returns:
        The normalized root URLs.

    Raises:
        ValueError: If the selection names a root the source does not hold,
            naming both it and the roots that are held.  A root spelled a
            second way is the same root and is accepted, which is the whole
            reason a root is normalized before it is compared.
    """
    if not named:
        return tuple(held)
    wanted = distinct_roots(named)
    unheld = [root for root in wanted if root not in held]
    if unheld:
        raise ValueError(
            f'this record source does not hold {", ".join(unheld)}. It holds: {", ".join(held)}.'
        )
    covered = set(wanted)
    return tuple(root for root in held if root in covered)


def root_for_stub(held: Sequence[str], stub: str) -> str:
    """Return the one root a bare stub is a key under.

    Parameters:
        held: The normalized roots the source holds.
        stub: The image's results path stub, named so the refusal says which
            image could not be answered for.

    Returns:
        The single root the source holds.

    Raises:
        ValueError: If the source holds any number of roots other than one,
            naming them all: a stub is a key under a root and says nothing
            about which root it is a key under, so there is no one record to
            hand back.
    """
    if len(held) != 1:
        raise ValueError(
            f'{stub}: a results path stub is a key under one root, and this source holds '
            f'{len(held)}: {", ".join(held)}. Name the root this image is '
            f'under and ask that source for it.'
        )
    return held[0]


def root_for_stubs(roots: Sequence[str], stubs: Sequence[str]) -> str:
    """Return the one root a selection naming stubs is a selection of keys under.

    Parameters:
        roots: The roots the selection resolved to, which is every root the
            source holds when the selection named none.
        stubs: The stubs the selection named, counted in the refusal.

    Returns:
        The single root those keys belong to.

    Raises:
        ValueError: If the selection resolves to any number of roots other than
            one, naming them.  Reading the same stubs under two roots would
            hand a caller two records per stub, which is not what naming an
            image means.
    """
    if len(roots) != 1:
        raise ValueError(
            f'a selection naming {len(stubs)} stub(s) is a selection of keys '
            f'under one root, and this one resolves to {len(roots)}: '
            f'{", ".join(roots) or "(none)"}'
        )
    return roots[0]


def refuse_what_a_listing_cannot_answer(selection: Selection) -> None:
    """Refuse a selection restricting on anything a document has to be opened for.

    A listing opens no document, so it cannot honor a restriction on what a
    document says.  The index could answer some of them from its columns, and
    deliberately does not: a call meaning one thing over one storage and another
    over the next is not a seam, and a caller would read a listing of the whole
    root as a listing of one mission.

    A selection naming stubs is not one of those.  A stub is the identity of a
    file rather than something the file says, so both storages answer it without
    opening anything: a source over the documents checks the named files or
    finds them in a walk of the directories they lie in, and a source over the
    index reads the keys out of the tables that record them.

    Parameters:
        selection: The selection to check.

    Raises:
        ValueError: If it carries ``instrument``, ``start_et`` or ``stop_et``,
            naming every one of them it carries.
    """
    carried: list[str] = []
    if selection.instrument is not None:
        carried.append('instrument')
    if selection.start_et is not None:
        carried.append('start_et')
    if selection.stop_et is not None:
        carried.append('stop_et')
    if carried:
        raise ValueError(
            f'a listing opens no document, so it cannot honor {", ".join(carried)}. '
            f'Ask records() for what a document says, or drop the restriction: a listing '
            f'that ignored one would answer for the whole root as though it were the '
            f'selection.'
        )


def in_batches(items: Iterator[_Item], size: int) -> Iterator[list[_Item]]:
    """Yield consecutive batches of an unbounded stream.

    Both storages batch what they ask for: a tree retrieves a group of documents
    in one parallel download, and an index asks for a group of keys in one
    statement rather than binding a whole mission's worth of them into a single
    one.  Neither may hold the stream it is batching, so the batches come off it
    lazily.

    Parameters:
        items: The stream to batch, which a walk or a caller produces lazily.
        size: Maximum batch length.

    Yields:
        The batches, in order, none of them empty.
    """
    while True:
        batch = list(islice(items, size))
        if not batch:
            return
        yield batch
